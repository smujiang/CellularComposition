from PIL import Image
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, roc_curve, auc
sys.path.append(os.path.abspath('../Topological'))
from cell_profiler import GetCellsFromMask, GetCellsFromQuPathMeasurements

sys.path.append(os.path.abspath('../Evaluation'))
from label_csv_manager import label_color_CSVManager
import openslide
import platform
import pickle
from sklearn.neighbors import KernelDensity


def get_coordinates_from_fn(filename):
    filename = os.path.split(filename)[1]
    l = filename.index("_")
    r = filename.rindex("_")
    x = filename[l + 1:r]
    e = filename.index(".")
    y = filename[r + 1:e]
    return np.array([int(x), int(y)])


def get_patch_from_WSI_by_pixel_location(wsi_obj, location, box_size=(50, 50)):
    loc = [int(location[0]), int(location[1])]
    cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, box_size)
    return cell_img


def get_cell_from_WSI_by_pixel_location(wsi_obj, location, box_size=(50, 50)):
    loc = [int(location[0] - box_size[0] / 2), int(location[1] - box_size[1] / 2)]
    cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, box_size)
    return cell_img


def get_cell_features_for_patch(img_name, cells):
    Img = Image.open(img_name, 'r')
    loc = get_coordinates_from_fn(img_name)

    ps = Img.size
    cells_in_patch = []
    for c in cells:
        if (loc[0] < c.loc[0] < loc[0] + ps[0]) and (loc[1] < c.loc[1] < loc[1] + ps[1]):
            cells_in_patch.append(c)
    return cells_in_patch

def f_importances(coef, names, save_to, fig_sz=(8, 12), top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))
    # Show all features
    if top == -1:
        top = len(names)
    plt.figure(figsize=fig_sz)
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()
    return imp, names

def get_img_fn_list(img_data_root, case_id):
    fn_list = os.listdir(os.path.join(img_data_root, case_id))
    img_fn_list = []
    for fn in fn_list:
        if '.jpg' in fn:
            img_fn_list.append(os.path.join(img_data_root, case_id, fn))
    return img_fn_list

if "Linux" in platform.platform():
    machine = "infodev2"
    DEBUG = False
else:
    machine = "Mac"
    DEBUG = True

if __name__ == "__main__":
    if machine == "Mac":
        txt_root = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification/QuPathMeasurements"
        QuPath_csv = "../Evaluation/label_color_table_QuPath_update.csv"
        img_data_root = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification/ROI_Masks_out_norm"
        model_out_root = "/Users/My_LANID/Dataset/OvaryCancer/PatchClassification"
        data_out_file = "/Users/My_LANID/Dataset/OvaryCancer/PatchClassification/train_test_data.npz"
        # mask_fn = "/Users/My_LANID/Dataset/OvaryCancer/512Norm_QBRC_out/OCMC-001/OCMC-001_59333_55264.png"
        wsi_root = "/Users/My_LANID/Dataset/OvaryData/WSIs"
    elif machine == "infodev2":
        txt_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/QuPathMeasurements_for_patchLevel"
        QuPath_csv = "../Evaluation/label_color_table_QuPath_update.csv"
        img_data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/ForPatchClassification_ROI_Masks_out"
        model_out_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification"
        data_out_file = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/train_test_data.npz"
        # mask_fn = "/infodev1/non-phi-data/junjiang/OvaryCancer/CellClassification/ROI_Masks_out_norm_out/OCMC-001/OCMC-001_59333_55264.png"
        wsi_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs"
    else:
        raise Exception("undefined machine to run the code")

    label_color_man = label_color_CSVManager(QuPath_csv)

    PatchFeature_list = []
    PatchLabel_list = []
    PatchLocation = []

    case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                    "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                    "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015",
                    "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                    "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                    "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

    # case_id_list = ["OCMC-001", "OCMC-016"]

    borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                               "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                               "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

    high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                               "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                               "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]

    if not os.path.exists(data_out_file):
        for case_index, case_id in enumerate(case_id_list):
            case_info_txt = os.path.join(model_out_root, "case_info", case_id+"_patch_info.txt")
            info_fp = open(case_info_txt, 'a')
            info_fp.write("patch_fn,eligible,stroma_cnt,tumor_cnt\n")

            eligible_patch = 0
            print("Processing case %s" % case_id)
            img_fn_list = get_img_fn_list(img_data_root, case_id)
            # txt_fn = os.path.join(txt_root, case_id, os.listdir(os.path.join(txt_root, case_id))[0])
            txt_fn = os.path.join(txt_root, case_id, os.path.join(txt_root, case_id, "detections_measurements_predictions.txt"))
            cells_creator_fea = GetCellsFromQuPathMeasurements(txt_fn, label_color_man)
            cells = cells_creator_fea.cell_list
            for img_fn in img_fn_list:
                # #For debug
                # patch_loc = get_coordinates_from_fn(img_fn)
                # wsi_fn = os.path.join(wsi_root, case_id+'.svs')
                # wsi_obj = openslide.OpenSlide(wsi_fn)
                # patch_img = get_patch_from_WSI_by_pixel_location(wsi_obj, (patch_loc[0], patch_loc[1]), box_size=(512, 512))
                # #For debug
                # plt.imshow(patch_img)
                # plt.show()
                # plt.imshow(Image.open(img_fn))
                # plt.show()

                selected_cells = get_cell_features_for_patch(img_fn, cells)
                # print("\t Found %d cells in this patch" % len(selected_cells))
                # PatchFeatures = []

                # calculate patch level features
                tumor_features = []
                stroma_features = []
                tumor_cnt = 0
                stroma_cnt = 0
                tumor_x = []
                tumor_y = []
                stroma_x = []
                stroma_y = []
                for c in selected_cells:
                    # #For debug
                    # cell_img = get_cell_from_WSI_by_pixel_location(wsi_obj, (c.loc[0], c.loc[1]), box_size=(50, 50))
                    # plt.imshow(cell_img)
                    # plt.show()
                    if c.label_id == 1:
                        tumor_cnt += 1
                        tumor_features.append(c.features)
                        tumor_x.append(c.loc[0])
                        tumor_y.append(c.loc[1])
                    else:
                        stroma_cnt += 1
                        stroma_features.append(c.features)
                        stroma_x.append(c.loc[0])
                        stroma_y.append(c.loc[1])

                # if stroma_cnt > 0 and tumor_cnt > 0:

                if stroma_cnt > 10 and tumor_cnt > 10:
                    eligible_patch += 1
                    # print("eligible patch")
                    xy = np.vstack([tumor_x, tumor_y])
                    s_xy = np.vstack([stroma_x, stroma_y]).T
                    kde_skl_1 = KernelDensity(bandwidth=16)
                    kde_skl_1.fit(xy.T)
                    sc_1 = kde_skl_1.score_samples(s_xy)

                    kde_skl_2 = KernelDensity(bandwidth=20)
                    kde_skl_2.fit(xy.T)
                    sc_2 = kde_skl_2.score_samples(s_xy)

                    kde_skl_3 = KernelDensity(bandwidth=24)
                    kde_skl_3.fit(xy.T)
                    sc_3 = kde_skl_3.score_samples(s_xy)

                    kde_skl_4 = KernelDensity(bandwidth=30)
                    kde_skl_4.fit(xy.T)
                    sc_4 = kde_skl_4.score_samples(s_xy)

                    kde_skl_5 = KernelDensity(bandwidth=34)
                    kde_skl_5.fit(xy.T)
                    sc_5 = kde_skl_5.score_samples(s_xy)

                    all_scores = np.vstack([sc_1, sc_2, sc_3, sc_4, sc_5]).T
                    kde_mean = np.mean(all_scores, axis=0)
                    kde_median = np.median(all_scores, axis=0)
                    kde_std = np.median(all_scores, axis=0)
                    kde_amax = np.amax(all_scores, axis=0)
                    kde_amin = np.amin(all_scores, axis=0)
                    kde_q1 = np.quantile(all_scores, .25, axis=0)  # Q1
                    kde_q3 = np.quantile(all_scores, .75, axis=0)  # Q3
                    # z = len(tumor_x) * (kde_skl.score_samples(s_xy))
                    # zz = z[np.where(z != 0)]
                    # patch_score = abs(np.sum(zz) / len(zz))  # TODO: formulize this score

                    stroma_mean = np.mean(stroma_features, axis=0)
                    stroma_median = np.median(stroma_features, axis=0)
                    stroma_std = np.median(stroma_features, axis=0)
                    stroma_amax = np.amax(stroma_features, axis=0)
                    stroma_amin = np.amin(stroma_features, axis=0)
                    stroma_q1 = np.quantile(stroma_features, .25, axis=0)  # Q1
                    stroma_q3 = np.quantile(stroma_features, .75, axis=0)  # Q3

                    tumor_mean = np.mean(tumor_features, axis=0)
                    tumor_median = np.median(tumor_features, axis=0)
                    tumor_std = np.median(tumor_features, axis=0)
                    tumor_amax = np.amax(tumor_features, axis=0)
                    tumor_amin = np.amin(tumor_features, axis=0)
                    tumor_q1 = np.quantile(tumor_features, .25, axis=0)  # Q1
                    tumor_q3 = np.quantile(tumor_features, .75, axis=0)  # Q3

                    patch_x, patch_y = get_coordinates_from_fn(img_fn)
                    PatchFeatures = np.hstack([case_index, patch_x, patch_y, stroma_mean, stroma_median, stroma_std, stroma_amax, stroma_amin, stroma_q1, stroma_q3,
                                               kde_mean, kde_median, kde_std, kde_amin, kde_amin, kde_q1, kde_q3,
                                                tumor_mean, tumor_median, tumor_std, tumor_amax, tumor_amin, tumor_q1, tumor_q3])

                    PatchFeature_list.append(PatchFeatures)
                    if case_id in borderline_case_id_list:
                        PatchLabel_list.append(0)
                    elif case_id in high_grade_case_id_list:
                        PatchLabel_list.append(1)
                    else:
                        raise Exception("undefined case id")
                    info_fp.write("%s,%s, %d,%d\n" % (img_fn, "Y", stroma_cnt, tumor_cnt))
                else:
                    info_fp.write("%s,%s, %d,%d\n" % (img_fn, "N", stroma_cnt, tumor_cnt))
            print("Found %d eligible patches in %d patches" % (eligible_patch, len(img_fn_list)))
            info_fp.close()
        train, test, train_labels, test_labels = train_test_split(np.array(PatchFeature_list),
                                                                  np.array(PatchLabel_list), test_size=0.33,
                                                                  random_state=5)

        np.savez(data_out_file, train, test, train_labels, test_labels)
    else:
        npzfile = np.load(data_out_file)
        train = npzfile['arr_0']
        test = npzfile['arr_1']
        train_labels = npzfile['arr_2']
        test_labels = npzfile['arr_3']

    train_case_idx = train[:, 0]
    test_case_idx = test[:, 0]
    train_patch_xy = train[:, 1:3]
    test_patch_xy = test[:, 1:3]
    train = train[:, 3:]
    test = test[:, 3:]

    pkl_filename = os.path.join(model_out_root, "patch_classification_model.pkl")
    if not os.path.exists(pkl_filename):
        s_w = class_weight.compute_sample_weight('balanced', train_labels)
        clf = SVC(kernel='linear', probability=True)
        model = clf.fit(train, train_labels, s_w)

        # save model to
        s = pickle.dumps(clf)
        pkl_filename = os.path.join(model_out_root, "patch_classification_model.pkl")
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
    else:
        pkl_fp = open(pkl_filename, 'rb')
        clf = pickle.load(pkl_fp)

    ########################################################

    #############ROC curve################
    y_score = clf.decision_function(test)
    fpr, tpr, _ = roc_curve(test_labels, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    ####################score histogram #########################
    plt.figure(2)
    plt.hist(y_score[test_labels >= 0.5], bins=20, histtype="step")
    plt.hist(y_score[test_labels < 0.5], bins=20, histtype="step")
    plt.grid()
    plt.legend(["HGOT", "BOT"])
    plt.show()


    #############Average precision################
    predictions = clf.predict(test)
    print("Over all accuracy: %f" % accuracy_score(test_labels, predictions))

    #############Confusion matrix################
    print("Confusion matrix:")
    cm = confusion_matrix(test_labels, predictions, normalize='true')
    print(cm)
    plt.figure(figsize=[5, 5])
    disp = plot_confusion_matrix(clf, test, test_labels,
                                 display_labels=["borderline", "high grade"],
                                 cmap=plt.get_cmap('Blues'),
                                 normalize='true')
    disp.ax_.set_title("Confusion matrix: BOT vs. HGSOC")
    plt.subplots_adjust(left=0.2)
    plt.show()

    #######################Misclassified cell analysis###########################################
    # case distribution

    # save patches case_id-location-score.jpg


    #################save misclassified patch location###########################
    misclass_fn = os.path.join(model_out_root, 'misclassified_patches_loc.txt')
    fp_misclass = open(misclass_fn, 'a')
    misclassified_cnt_in_case = np.zeros([len(case_id_list), 1])
    for idx, p in enumerate(predictions):
        if not p == test_labels[idx]:
            # update misclassified patch count in a case
            misclassified_cnt_in_case[int(test_case_idx[idx])] = misclassified_cnt_in_case[int(test_case_idx[idx])] +1
            # save locations into a txt file
            wrt_str = case_id_list[int(test_case_idx[idx])] + "," + str(test_patch_xy[idx, 0]) + "," + str(test_patch_xy[idx, 1]) + "," + str(y_score[idx]) + '\n'
            fp_misclass.write(wrt_str)
    fp_misclass.close()

    # get eligible patch count
    total = [302, 440, 381, 363, 318, 202, 397, 532, 501, 485, 442, 200, 581, 572, 730, 284, 284, 276, 160, 360, 259, 120, 138, 289, 480, 218, 272, 280, 258, 347]
    eligible_10 = [215, 253, 267, 214, 180, 14, 318, 323, 287, 314, 361, 132, 428, 361, 558, 167, 5, 137, 70, 199, 130, 99, 114, 216, 366, 157, 151, 188, 73, 188]
    eligible_20 = [172, 164, 186, 173, 120, 1, 246, 223, 255, 254, 302, 96, 378, 208, 510, 135, 0, 104, 52, 147, 92, 81, 104, 157, 239, 125, 86, 142, 47, 72]

    eligible_cnt_in_case = [215,253,267,214,180,14,318,323,287,314,361,132,428,361,558,167,5,137,70,199,130,99,114,216,366,157,151,188,73,188]
    cnt_plt = []
    for i in range(len(case_id_list)):
        cnt_plt.append(misclassified_cnt_in_case[i] / eligible_cnt_in_case[i])
    plt.figure(figsize=[12, 5])
    plt.plot(range(0, int(len(case_id_list)/2)), cnt_plt[0:int(len(case_id_list)/2)])
    plt.plot(range(int(len(case_id_list)/2), len(case_id_list)), cnt_plt[int(len(case_id_list)/2):])
    plt.xticks(range(len(case_id_list)), case_id_list, rotation='vertical')
    plt.ylabel('Misclassified patch Cnt/Eligible patch Cnt')
    plt.xlabel('Case ID')
    for i in range(len(case_id_list)):
        plt.text(i, cnt_plt[i], '%d/%d' % (misclassified_cnt_in_case[i],  eligible_cnt_in_case[i]))

    plt.legend(["HGOT", "BOT"], loc="upper right")
    plt.grid()
    plt.show()



    ############## Feature importance #############
    FeatureImportance = clf.coef_[0]
    feature_names = ['stroma_mean_Nucleus: Area', 'stroma_mean_Nucleus: Perimeter', 'stroma_mean_Nucleus: Circularity',
     'stroma_mean_Nucleus: Max caliper', 'stroma_mean_Nucleus: Min caliper', 'stroma_mean_Nucleus: Eccentricity',
     'stroma_mean_Nucleus: Hematoxylin OD mean', 'stroma_mean_Nucleus: Hematoxylin OD sum',
     'stroma_mean_Nucleus: Hematoxylin OD std dev', 'stroma_mean_Nucleus: Hematoxylin OD max',
     'stroma_mean_Nucleus: Hematoxylin OD min', 'stroma_mean_Nucleus: Hematoxylin OD range',
     'stroma_mean_Nucleus: Eosin OD mean', 'stroma_mean_Nucleus: Eosin OD sum', 'stroma_mean_Nucleus: Eosin OD std dev',
     'stroma_mean_Nucleus: Eosin OD max', 'stroma_mean_Nucleus: Eosin OD min', 'stroma_mean_Nucleus: Eosin OD range',
     'stroma_mean_Cell: Area', 'stroma_mean_Cell: Perimeter', 'stroma_mean_Cell: Circularity',
     'stroma_mean_Cell: Max caliper', 'stroma_mean_Cell: Min caliper', 'stroma_mean_Cell: Eccentricity',
     'stroma_mean_Cell: Hematoxylin OD mean', 'stroma_mean_Cell: Hematoxylin OD std dev',
     'stroma_mean_Cell: Hematoxylin OD max', 'stroma_mean_Cell: Hematoxylin OD min', 'stroma_mean_Cell: Eosin OD mean',
     'stroma_mean_Cell: Eosin OD std dev', 'stroma_mean_Cell: Eosin OD max', 'stroma_mean_Cell: Eosin OD min',
     'stroma_mean_Cytoplasm: Hematoxylin OD mean', 'stroma_mean_Cytoplasm: Hematoxylin OD std dev',
     'stroma_mean_Cytoplasm: Hematoxylin OD max', 'stroma_mean_Cytoplasm: Hematoxylin OD min',
     'stroma_mean_Cytoplasm: Eosin OD mean', 'stroma_mean_Cytoplasm: Eosin OD std dev',
     'stroma_mean_Cytoplasm: Eosin OD max', 'stroma_mean_Cytoplasm: Eosin OD min',
     'stroma_mean_Nucleus/Cell area ratio', 'stroma_median_Nucleus: Area', 'stroma_median_Nucleus: Perimeter',
     'stroma_median_Nucleus: Circularity', 'stroma_median_Nucleus: Max caliper', 'stroma_median_Nucleus: Min caliper',
     'stroma_median_Nucleus: Eccentricity', 'stroma_median_Nucleus: Hematoxylin OD mean',
     'stroma_median_Nucleus: Hematoxylin OD sum', 'stroma_median_Nucleus: Hematoxylin OD std dev',
     'stroma_median_Nucleus: Hematoxylin OD max', 'stroma_median_Nucleus: Hematoxylin OD min',
     'stroma_median_Nucleus: Hematoxylin OD range', 'stroma_median_Nucleus: Eosin OD mean',
     'stroma_median_Nucleus: Eosin OD sum', 'stroma_median_Nucleus: Eosin OD std dev',
     'stroma_median_Nucleus: Eosin OD max', 'stroma_median_Nucleus: Eosin OD min',
     'stroma_median_Nucleus: Eosin OD range', 'stroma_median_Cell: Area', 'stroma_median_Cell: Perimeter',
     'stroma_median_Cell: Circularity', 'stroma_median_Cell: Max caliper', 'stroma_median_Cell: Min caliper',
     'stroma_median_Cell: Eccentricity', 'stroma_median_Cell: Hematoxylin OD mean',
     'stroma_median_Cell: Hematoxylin OD std dev', 'stroma_median_Cell: Hematoxylin OD max',
     'stroma_median_Cell: Hematoxylin OD min', 'stroma_median_Cell: Eosin OD mean',
     'stroma_median_Cell: Eosin OD std dev', 'stroma_median_Cell: Eosin OD max', 'stroma_median_Cell: Eosin OD min',
     'stroma_median_Cytoplasm: Hematoxylin OD mean', 'stroma_median_Cytoplasm: Hematoxylin OD std dev',
     'stroma_median_Cytoplasm: Hematoxylin OD max', 'stroma_median_Cytoplasm: Hematoxylin OD min',
     'stroma_median_Cytoplasm: Eosin OD mean', 'stroma_median_Cytoplasm: Eosin OD std dev',
     'stroma_median_Cytoplasm: Eosin OD max', 'stroma_median_Cytoplasm: Eosin OD min',
     'stroma_median_Nucleus/Cell area ratio', 'stroma_std_Nucleus: Area', 'stroma_std_Nucleus: Perimeter',
     'stroma_std_Nucleus: Circularity', 'stroma_std_Nucleus: Max caliper', 'stroma_std_Nucleus: Min caliper',
     'stroma_std_Nucleus: Eccentricity', 'stroma_std_Nucleus: Hematoxylin OD mean',
     'stroma_std_Nucleus: Hematoxylin OD sum', 'stroma_std_Nucleus: Hematoxylin OD std dev',
     'stroma_std_Nucleus: Hematoxylin OD max', 'stroma_std_Nucleus: Hematoxylin OD min',
     'stroma_std_Nucleus: Hematoxylin OD range', 'stroma_std_Nucleus: Eosin OD mean',
     'stroma_std_Nucleus: Eosin OD sum', 'stroma_std_Nucleus: Eosin OD std dev', 'stroma_std_Nucleus: Eosin OD max',
     'stroma_std_Nucleus: Eosin OD min', 'stroma_std_Nucleus: Eosin OD range', 'stroma_std_Cell: Area',
     'stroma_std_Cell: Perimeter', 'stroma_std_Cell: Circularity', 'stroma_std_Cell: Max caliper',
     'stroma_std_Cell: Min caliper', 'stroma_std_Cell: Eccentricity', 'stroma_std_Cell: Hematoxylin OD mean',
     'stroma_std_Cell: Hematoxylin OD std dev', 'stroma_std_Cell: Hematoxylin OD max',
     'stroma_std_Cell: Hematoxylin OD min', 'stroma_std_Cell: Eosin OD mean', 'stroma_std_Cell: Eosin OD std dev',
     'stroma_std_Cell: Eosin OD max', 'stroma_std_Cell: Eosin OD min', 'stroma_std_Cytoplasm: Hematoxylin OD mean',
     'stroma_std_Cytoplasm: Hematoxylin OD std dev', 'stroma_std_Cytoplasm: Hematoxylin OD max',
     'stroma_std_Cytoplasm: Hematoxylin OD min', 'stroma_std_Cytoplasm: Eosin OD mean',
     'stroma_std_Cytoplasm: Eosin OD std dev', 'stroma_std_Cytoplasm: Eosin OD max',
     'stroma_std_Cytoplasm: Eosin OD min', 'stroma_std_Nucleus/Cell area ratio', 'stroma_amax_Nucleus: Area',
     'stroma_amax_Nucleus: Perimeter', 'stroma_amax_Nucleus: Circularity', 'stroma_amax_Nucleus: Max caliper',
     'stroma_amax_Nucleus: Min caliper', 'stroma_amax_Nucleus: Eccentricity',
     'stroma_amax_Nucleus: Hematoxylin OD mean', 'stroma_amax_Nucleus: Hematoxylin OD sum',
     'stroma_amax_Nucleus: Hematoxylin OD std dev', 'stroma_amax_Nucleus: Hematoxylin OD max',
     'stroma_amax_Nucleus: Hematoxylin OD min', 'stroma_amax_Nucleus: Hematoxylin OD range',
     'stroma_amax_Nucleus: Eosin OD mean', 'stroma_amax_Nucleus: Eosin OD sum', 'stroma_amax_Nucleus: Eosin OD std dev',
     'stroma_amax_Nucleus: Eosin OD max', 'stroma_amax_Nucleus: Eosin OD min', 'stroma_amax_Nucleus: Eosin OD range',
     'stroma_amax_Cell: Area', 'stroma_amax_Cell: Perimeter', 'stroma_amax_Cell: Circularity',
     'stroma_amax_Cell: Max caliper', 'stroma_amax_Cell: Min caliper', 'stroma_amax_Cell: Eccentricity',
     'stroma_amax_Cell: Hematoxylin OD mean', 'stroma_amax_Cell: Hematoxylin OD std dev',
     'stroma_amax_Cell: Hematoxylin OD max', 'stroma_amax_Cell: Hematoxylin OD min', 'stroma_amax_Cell: Eosin OD mean',
     'stroma_amax_Cell: Eosin OD std dev', 'stroma_amax_Cell: Eosin OD max', 'stroma_amax_Cell: Eosin OD min',
     'stroma_amax_Cytoplasm: Hematoxylin OD mean', 'stroma_amax_Cytoplasm: Hematoxylin OD std dev',
     'stroma_amax_Cytoplasm: Hematoxylin OD max', 'stroma_amax_Cytoplasm: Hematoxylin OD min',
     'stroma_amax_Cytoplasm: Eosin OD mean', 'stroma_amax_Cytoplasm: Eosin OD std dev',
     'stroma_amax_Cytoplasm: Eosin OD max', 'stroma_amax_Cytoplasm: Eosin OD min',
     'stroma_amax_Nucleus/Cell area ratio', 'stroma_amin_Nucleus: Area', 'stroma_amin_Nucleus: Perimeter',
     'stroma_amin_Nucleus: Circularity', 'stroma_amin_Nucleus: Max caliper', 'stroma_amin_Nucleus: Min caliper',
     'stroma_amin_Nucleus: Eccentricity', 'stroma_amin_Nucleus: Hematoxylin OD mean',
     'stroma_amin_Nucleus: Hematoxylin OD sum', 'stroma_amin_Nucleus: Hematoxylin OD std dev',
     'stroma_amin_Nucleus: Hematoxylin OD max', 'stroma_amin_Nucleus: Hematoxylin OD min',
     'stroma_amin_Nucleus: Hematoxylin OD range', 'stroma_amin_Nucleus: Eosin OD mean',
     'stroma_amin_Nucleus: Eosin OD sum', 'stroma_amin_Nucleus: Eosin OD std dev', 'stroma_amin_Nucleus: Eosin OD max',
     'stroma_amin_Nucleus: Eosin OD min', 'stroma_amin_Nucleus: Eosin OD range', 'stroma_amin_Cell: Area',
     'stroma_amin_Cell: Perimeter', 'stroma_amin_Cell: Circularity', 'stroma_amin_Cell: Max caliper',
     'stroma_amin_Cell: Min caliper', 'stroma_amin_Cell: Eccentricity', 'stroma_amin_Cell: Hematoxylin OD mean',
     'stroma_amin_Cell: Hematoxylin OD std dev', 'stroma_amin_Cell: Hematoxylin OD max',
     'stroma_amin_Cell: Hematoxylin OD min', 'stroma_amin_Cell: Eosin OD mean', 'stroma_amin_Cell: Eosin OD std dev',
     'stroma_amin_Cell: Eosin OD max', 'stroma_amin_Cell: Eosin OD min', 'stroma_amin_Cytoplasm: Hematoxylin OD mean',
     'stroma_amin_Cytoplasm: Hematoxylin OD std dev', 'stroma_amin_Cytoplasm: Hematoxylin OD max',
     'stroma_amin_Cytoplasm: Hematoxylin OD min', 'stroma_amin_Cytoplasm: Eosin OD mean',
     'stroma_amin_Cytoplasm: Eosin OD std dev', 'stroma_amin_Cytoplasm: Eosin OD max',
     'stroma_amin_Cytoplasm: Eosin OD min', 'stroma_amin_Nucleus/Cell area ratio', 'stroma_q1_Nucleus: Area',
     'stroma_q1_Nucleus: Perimeter', 'stroma_q1_Nucleus: Circularity', 'stroma_q1_Nucleus: Max caliper',
     'stroma_q1_Nucleus: Min caliper', 'stroma_q1_Nucleus: Eccentricity', 'stroma_q1_Nucleus: Hematoxylin OD mean',
     'stroma_q1_Nucleus: Hematoxylin OD sum', 'stroma_q1_Nucleus: Hematoxylin OD std dev',
     'stroma_q1_Nucleus: Hematoxylin OD max', 'stroma_q1_Nucleus: Hematoxylin OD min',
     'stroma_q1_Nucleus: Hematoxylin OD range', 'stroma_q1_Nucleus: Eosin OD mean', 'stroma_q1_Nucleus: Eosin OD sum',
     'stroma_q1_Nucleus: Eosin OD std dev', 'stroma_q1_Nucleus: Eosin OD max', 'stroma_q1_Nucleus: Eosin OD min',
     'stroma_q1_Nucleus: Eosin OD range', 'stroma_q1_Cell: Area', 'stroma_q1_Cell: Perimeter',
     'stroma_q1_Cell: Circularity', 'stroma_q1_Cell: Max caliper', 'stroma_q1_Cell: Min caliper',
     'stroma_q1_Cell: Eccentricity', 'stroma_q1_Cell: Hematoxylin OD mean', 'stroma_q1_Cell: Hematoxylin OD std dev',
     'stroma_q1_Cell: Hematoxylin OD max', 'stroma_q1_Cell: Hematoxylin OD min', 'stroma_q1_Cell: Eosin OD mean',
     'stroma_q1_Cell: Eosin OD std dev', 'stroma_q1_Cell: Eosin OD max', 'stroma_q1_Cell: Eosin OD min',
     'stroma_q1_Cytoplasm: Hematoxylin OD mean', 'stroma_q1_Cytoplasm: Hematoxylin OD std dev',
     'stroma_q1_Cytoplasm: Hematoxylin OD max', 'stroma_q1_Cytoplasm: Hematoxylin OD min',
     'stroma_q1_Cytoplasm: Eosin OD mean', 'stroma_q1_Cytoplasm: Eosin OD std dev', 'stroma_q1_Cytoplasm: Eosin OD max',
     'stroma_q1_Cytoplasm: Eosin OD min', 'stroma_q1_Nucleus/Cell area ratio', 'stroma_q3_Nucleus: Area',
     'stroma_q3_Nucleus: Perimeter', 'stroma_q3_Nucleus: Circularity', 'stroma_q3_Nucleus: Max caliper',
     'stroma_q3_Nucleus: Min caliper', 'stroma_q3_Nucleus: Eccentricity', 'stroma_q3_Nucleus: Hematoxylin OD mean',
     'stroma_q3_Nucleus: Hematoxylin OD sum', 'stroma_q3_Nucleus: Hematoxylin OD std dev',
     'stroma_q3_Nucleus: Hematoxylin OD max', 'stroma_q3_Nucleus: Hematoxylin OD min',
     'stroma_q3_Nucleus: Hematoxylin OD range', 'stroma_q3_Nucleus: Eosin OD mean', 'stroma_q3_Nucleus: Eosin OD sum',
     'stroma_q3_Nucleus: Eosin OD std dev', 'stroma_q3_Nucleus: Eosin OD max', 'stroma_q3_Nucleus: Eosin OD min',
     'stroma_q3_Nucleus: Eosin OD range', 'stroma_q3_Cell: Area', 'stroma_q3_Cell: Perimeter',
     'stroma_q3_Cell: Circularity', 'stroma_q3_Cell: Max caliper', 'stroma_q3_Cell: Min caliper',
     'stroma_q3_Cell: Eccentricity', 'stroma_q3_Cell: Hematoxylin OD mean', 'stroma_q3_Cell: Hematoxylin OD std dev',
     'stroma_q3_Cell: Hematoxylin OD max', 'stroma_q3_Cell: Hematoxylin OD min', 'stroma_q3_Cell: Eosin OD mean',
     'stroma_q3_Cell: Eosin OD std dev', 'stroma_q3_Cell: Eosin OD max', 'stroma_q3_Cell: Eosin OD min',
     'stroma_q3_Cytoplasm: Hematoxylin OD mean', 'stroma_q3_Cytoplasm: Hematoxylin OD std dev',
     'stroma_q3_Cytoplasm: Hematoxylin OD max', 'stroma_q3_Cytoplasm: Hematoxylin OD min',
     'stroma_q3_Cytoplasm: Eosin OD mean', 'stroma_q3_Cytoplasm: Eosin OD std dev', 'stroma_q3_Cytoplasm: Eosin OD max',
     'stroma_q3_Cytoplasm: Eosin OD min', 'stroma_q3_Nucleus/Cell area ratio', 'tumor_mean_Nucleus: Area',
     "kde_1_mean", "kde_1_median", "kde_1_std", "kde_1_amin", "kde_1_amin", "kde_1_q1", "kde_1_q3",
     "kde_2_mean", "kde_2_median", "kde_2_std", "kde_2_amin", "kde_2_amin", "kde_2_q1", "kde_2_q3",
     "kde_3_mean", "kde_3_median", "kde_3_std", "kde_3_amin", "kde_3_amin", "kde_3_q1", "kde_3_q3",
     "kde_4_mean", "kde_4_median", "kde_4_std", "kde_4_amin", "kde_4_amin", "kde_4_q1", "kde_4_q3",
     "kde_5_mean", "kde_5_median", "kde_5_std", "kde_5_amin", "kde_5_amin", "kde_5_q1", "kde_5_q3",
     'tumor_mean_Nucleus: Perimeter', 'tumor_mean_Nucleus: Circularity', 'tumor_mean_Nucleus: Max caliper',
     'tumor_mean_Nucleus: Min caliper', 'tumor_mean_Nucleus: Eccentricity', 'tumor_mean_Nucleus: Hematoxylin OD mean',
     'tumor_mean_Nucleus: Hematoxylin OD sum', 'tumor_mean_Nucleus: Hematoxylin OD std dev',
     'tumor_mean_Nucleus: Hematoxylin OD max', 'tumor_mean_Nucleus: Hematoxylin OD min',
     'tumor_mean_Nucleus: Hematoxylin OD range', 'tumor_mean_Nucleus: Eosin OD mean',
     'tumor_mean_Nucleus: Eosin OD sum', 'tumor_mean_Nucleus: Eosin OD std dev', 'tumor_mean_Nucleus: Eosin OD max',
     'tumor_mean_Nucleus: Eosin OD min', 'tumor_mean_Nucleus: Eosin OD range', 'tumor_mean_Cell: Area',
     'tumor_mean_Cell: Perimeter', 'tumor_mean_Cell: Circularity', 'tumor_mean_Cell: Max caliper',
     'tumor_mean_Cell: Min caliper', 'tumor_mean_Cell: Eccentricity', 'tumor_mean_Cell: Hematoxylin OD mean',
     'tumor_mean_Cell: Hematoxylin OD std dev', 'tumor_mean_Cell: Hematoxylin OD max',
     'tumor_mean_Cell: Hematoxylin OD min', 'tumor_mean_Cell: Eosin OD mean', 'tumor_mean_Cell: Eosin OD std dev',
     'tumor_mean_Cell: Eosin OD max', 'tumor_mean_Cell: Eosin OD min', 'tumor_mean_Cytoplasm: Hematoxylin OD mean',
     'tumor_mean_Cytoplasm: Hematoxylin OD std dev', 'tumor_mean_Cytoplasm: Hematoxylin OD max',
     'tumor_mean_Cytoplasm: Hematoxylin OD min', 'tumor_mean_Cytoplasm: Eosin OD mean',
     'tumor_mean_Cytoplasm: Eosin OD std dev', 'tumor_mean_Cytoplasm: Eosin OD max',
     'tumor_mean_Cytoplasm: Eosin OD min', 'tumor_mean_Nucleus/Cell area ratio', 'tumor_median_Nucleus: Area',
     'tumor_median_Nucleus: Perimeter', 'tumor_median_Nucleus: Circularity', 'tumor_median_Nucleus: Max caliper',
     'tumor_median_Nucleus: Min caliper', 'tumor_median_Nucleus: Eccentricity',
     'tumor_median_Nucleus: Hematoxylin OD mean', 'tumor_median_Nucleus: Hematoxylin OD sum',
     'tumor_median_Nucleus: Hematoxylin OD std dev', 'tumor_median_Nucleus: Hematoxylin OD max',
     'tumor_median_Nucleus: Hematoxylin OD min', 'tumor_median_Nucleus: Hematoxylin OD range',
     'tumor_median_Nucleus: Eosin OD mean', 'tumor_median_Nucleus: Eosin OD sum',
     'tumor_median_Nucleus: Eosin OD std dev', 'tumor_median_Nucleus: Eosin OD max',
     'tumor_median_Nucleus: Eosin OD min', 'tumor_median_Nucleus: Eosin OD range', 'tumor_median_Cell: Area',
     'tumor_median_Cell: Perimeter', 'tumor_median_Cell: Circularity', 'tumor_median_Cell: Max caliper',
     'tumor_median_Cell: Min caliper', 'tumor_median_Cell: Eccentricity', 'tumor_median_Cell: Hematoxylin OD mean',
     'tumor_median_Cell: Hematoxylin OD std dev', 'tumor_median_Cell: Hematoxylin OD max',
     'tumor_median_Cell: Hematoxylin OD min', 'tumor_median_Cell: Eosin OD mean', 'tumor_median_Cell: Eosin OD std dev',
     'tumor_median_Cell: Eosin OD max', 'tumor_median_Cell: Eosin OD min',
     'tumor_median_Cytoplasm: Hematoxylin OD mean', 'tumor_median_Cytoplasm: Hematoxylin OD std dev',
     'tumor_median_Cytoplasm: Hematoxylin OD max', 'tumor_median_Cytoplasm: Hematoxylin OD min',
     'tumor_median_Cytoplasm: Eosin OD mean', 'tumor_median_Cytoplasm: Eosin OD std dev',
     'tumor_median_Cytoplasm: Eosin OD max', 'tumor_median_Cytoplasm: Eosin OD min',
     'tumor_median_Nucleus/Cell area ratio', 'tumor_std_Nucleus: Area', 'tumor_std_Nucleus: Perimeter',
     'tumor_std_Nucleus: Circularity', 'tumor_std_Nucleus: Max caliper', 'tumor_std_Nucleus: Min caliper',
     'tumor_std_Nucleus: Eccentricity', 'tumor_std_Nucleus: Hematoxylin OD mean',
     'tumor_std_Nucleus: Hematoxylin OD sum', 'tumor_std_Nucleus: Hematoxylin OD std dev',
     'tumor_std_Nucleus: Hematoxylin OD max', 'tumor_std_Nucleus: Hematoxylin OD min',
     'tumor_std_Nucleus: Hematoxylin OD range', 'tumor_std_Nucleus: Eosin OD mean', 'tumor_std_Nucleus: Eosin OD sum',
     'tumor_std_Nucleus: Eosin OD std dev', 'tumor_std_Nucleus: Eosin OD max', 'tumor_std_Nucleus: Eosin OD min',
     'tumor_std_Nucleus: Eosin OD range', 'tumor_std_Cell: Area', 'tumor_std_Cell: Perimeter',
     'tumor_std_Cell: Circularity', 'tumor_std_Cell: Max caliper', 'tumor_std_Cell: Min caliper',
     'tumor_std_Cell: Eccentricity', 'tumor_std_Cell: Hematoxylin OD mean', 'tumor_std_Cell: Hematoxylin OD std dev',
     'tumor_std_Cell: Hematoxylin OD max', 'tumor_std_Cell: Hematoxylin OD min', 'tumor_std_Cell: Eosin OD mean',
     'tumor_std_Cell: Eosin OD std dev', 'tumor_std_Cell: Eosin OD max', 'tumor_std_Cell: Eosin OD min',
     'tumor_std_Cytoplasm: Hematoxylin OD mean', 'tumor_std_Cytoplasm: Hematoxylin OD std dev',
     'tumor_std_Cytoplasm: Hematoxylin OD max', 'tumor_std_Cytoplasm: Hematoxylin OD min',
     'tumor_std_Cytoplasm: Eosin OD mean', 'tumor_std_Cytoplasm: Eosin OD std dev', 'tumor_std_Cytoplasm: Eosin OD max',
     'tumor_std_Cytoplasm: Eosin OD min', 'tumor_std_Nucleus/Cell area ratio', 'tumor_amax_Nucleus: Area',
     'tumor_amax_Nucleus: Perimeter', 'tumor_amax_Nucleus: Circularity', 'tumor_amax_Nucleus: Max caliper',
     'tumor_amax_Nucleus: Min caliper', 'tumor_amax_Nucleus: Eccentricity', 'tumor_amax_Nucleus: Hematoxylin OD mean',
     'tumor_amax_Nucleus: Hematoxylin OD sum', 'tumor_amax_Nucleus: Hematoxylin OD std dev',
     'tumor_amax_Nucleus: Hematoxylin OD max', 'tumor_amax_Nucleus: Hematoxylin OD min',
     'tumor_amax_Nucleus: Hematoxylin OD range', 'tumor_amax_Nucleus: Eosin OD mean',
     'tumor_amax_Nucleus: Eosin OD sum', 'tumor_amax_Nucleus: Eosin OD std dev', 'tumor_amax_Nucleus: Eosin OD max',
     'tumor_amax_Nucleus: Eosin OD min', 'tumor_amax_Nucleus: Eosin OD range', 'tumor_amax_Cell: Area',
     'tumor_amax_Cell: Perimeter', 'tumor_amax_Cell: Circularity', 'tumor_amax_Cell: Max caliper',
     'tumor_amax_Cell: Min caliper', 'tumor_amax_Cell: Eccentricity', 'tumor_amax_Cell: Hematoxylin OD mean',
     'tumor_amax_Cell: Hematoxylin OD std dev', 'tumor_amax_Cell: Hematoxylin OD max',
     'tumor_amax_Cell: Hematoxylin OD min', 'tumor_amax_Cell: Eosin OD mean', 'tumor_amax_Cell: Eosin OD std dev',
     'tumor_amax_Cell: Eosin OD max', 'tumor_amax_Cell: Eosin OD min', 'tumor_amax_Cytoplasm: Hematoxylin OD mean',
     'tumor_amax_Cytoplasm: Hematoxylin OD std dev', 'tumor_amax_Cytoplasm: Hematoxylin OD max',
     'tumor_amax_Cytoplasm: Hematoxylin OD min', 'tumor_amax_Cytoplasm: Eosin OD mean',
     'tumor_amax_Cytoplasm: Eosin OD std dev', 'tumor_amax_Cytoplasm: Eosin OD max',
     'tumor_amax_Cytoplasm: Eosin OD min', 'tumor_amax_Nucleus/Cell area ratio', 'tumor_amin_Nucleus: Area',
     'tumor_amin_Nucleus: Perimeter', 'tumor_amin_Nucleus: Circularity', 'tumor_amin_Nucleus: Max caliper',
     'tumor_amin_Nucleus: Min caliper', 'tumor_amin_Nucleus: Eccentricity', 'tumor_amin_Nucleus: Hematoxylin OD mean',
     'tumor_amin_Nucleus: Hematoxylin OD sum', 'tumor_amin_Nucleus: Hematoxylin OD std dev',
     'tumor_amin_Nucleus: Hematoxylin OD max', 'tumor_amin_Nucleus: Hematoxylin OD min',
     'tumor_amin_Nucleus: Hematoxylin OD range', 'tumor_amin_Nucleus: Eosin OD mean',
     'tumor_amin_Nucleus: Eosin OD sum', 'tumor_amin_Nucleus: Eosin OD std dev', 'tumor_amin_Nucleus: Eosin OD max',
     'tumor_amin_Nucleus: Eosin OD min', 'tumor_amin_Nucleus: Eosin OD range', 'tumor_amin_Cell: Area',
     'tumor_amin_Cell: Perimeter', 'tumor_amin_Cell: Circularity', 'tumor_amin_Cell: Max caliper',
     'tumor_amin_Cell: Min caliper', 'tumor_amin_Cell: Eccentricity', 'tumor_amin_Cell: Hematoxylin OD mean',
     'tumor_amin_Cell: Hematoxylin OD std dev', 'tumor_amin_Cell: Hematoxylin OD max',
     'tumor_amin_Cell: Hematoxylin OD min', 'tumor_amin_Cell: Eosin OD mean', 'tumor_amin_Cell: Eosin OD std dev',
     'tumor_amin_Cell: Eosin OD max', 'tumor_amin_Cell: Eosin OD min', 'tumor_amin_Cytoplasm: Hematoxylin OD mean',
     'tumor_amin_Cytoplasm: Hematoxylin OD std dev', 'tumor_amin_Cytoplasm: Hematoxylin OD max',
     'tumor_amin_Cytoplasm: Hematoxylin OD min', 'tumor_amin_Cytoplasm: Eosin OD mean',
     'tumor_amin_Cytoplasm: Eosin OD std dev', 'tumor_amin_Cytoplasm: Eosin OD max',
     'tumor_amin_Cytoplasm: Eosin OD min', 'tumor_amin_Nucleus/Cell area ratio', 'tumor_q1_Nucleus: Area',
     'tumor_q1_Nucleus: Perimeter', 'tumor_q1_Nucleus: Circularity', 'tumor_q1_Nucleus: Max caliper',
     'tumor_q1_Nucleus: Min caliper', 'tumor_q1_Nucleus: Eccentricity', 'tumor_q1_Nucleus: Hematoxylin OD mean',
     'tumor_q1_Nucleus: Hematoxylin OD sum', 'tumor_q1_Nucleus: Hematoxylin OD std dev',
     'tumor_q1_Nucleus: Hematoxylin OD max', 'tumor_q1_Nucleus: Hematoxylin OD min',
     'tumor_q1_Nucleus: Hematoxylin OD range', 'tumor_q1_Nucleus: Eosin OD mean', 'tumor_q1_Nucleus: Eosin OD sum',
     'tumor_q1_Nucleus: Eosin OD std dev', 'tumor_q1_Nucleus: Eosin OD max', 'tumor_q1_Nucleus: Eosin OD min',
     'tumor_q1_Nucleus: Eosin OD range', 'tumor_q1_Cell: Area', 'tumor_q1_Cell: Perimeter',
     'tumor_q1_Cell: Circularity', 'tumor_q1_Cell: Max caliper', 'tumor_q1_Cell: Min caliper',
     'tumor_q1_Cell: Eccentricity', 'tumor_q1_Cell: Hematoxylin OD mean', 'tumor_q1_Cell: Hematoxylin OD std dev',
     'tumor_q1_Cell: Hematoxylin OD max', 'tumor_q1_Cell: Hematoxylin OD min', 'tumor_q1_Cell: Eosin OD mean',
     'tumor_q1_Cell: Eosin OD std dev', 'tumor_q1_Cell: Eosin OD max', 'tumor_q1_Cell: Eosin OD min',
     'tumor_q1_Cytoplasm: Hematoxylin OD mean', 'tumor_q1_Cytoplasm: Hematoxylin OD std dev',
     'tumor_q1_Cytoplasm: Hematoxylin OD max', 'tumor_q1_Cytoplasm: Hematoxylin OD min',
     'tumor_q1_Cytoplasm: Eosin OD mean', 'tumor_q1_Cytoplasm: Eosin OD std dev', 'tumor_q1_Cytoplasm: Eosin OD max',
     'tumor_q1_Cytoplasm: Eosin OD min', 'tumor_q1_Nucleus/Cell area ratio', 'tumor_q3_Nucleus: Area',
     'tumor_q3_Nucleus: Perimeter', 'tumor_q3_Nucleus: Circularity', 'tumor_q3_Nucleus: Max caliper',
     'tumor_q3_Nucleus: Min caliper', 'tumor_q3_Nucleus: Eccentricity', 'tumor_q3_Nucleus: Hematoxylin OD mean',
     'tumor_q3_Nucleus: Hematoxylin OD sum', 'tumor_q3_Nucleus: Hematoxylin OD std dev',
     'tumor_q3_Nucleus: Hematoxylin OD max', 'tumor_q3_Nucleus: Hematoxylin OD min',
     'tumor_q3_Nucleus: Hematoxylin OD range', 'tumor_q3_Nucleus: Eosin OD mean', 'tumor_q3_Nucleus: Eosin OD sum',
     'tumor_q3_Nucleus: Eosin OD std dev', 'tumor_q3_Nucleus: Eosin OD max', 'tumor_q3_Nucleus: Eosin OD min',
     'tumor_q3_Nucleus: Eosin OD range', 'tumor_q3_Cell: Area', 'tumor_q3_Cell: Perimeter',
     'tumor_q3_Cell: Circularity', 'tumor_q3_Cell: Max caliper', 'tumor_q3_Cell: Min caliper',
     'tumor_q3_Cell: Eccentricity', 'tumor_q3_Cell: Hematoxylin OD mean', 'tumor_q3_Cell: Hematoxylin OD std dev',
     'tumor_q3_Cell: Hematoxylin OD max', 'tumor_q3_Cell: Hematoxylin OD min', 'tumor_q3_Cell: Eosin OD mean',
     'tumor_q3_Cell: Eosin OD std dev', 'tumor_q3_Cell: Eosin OD max', 'tumor_q3_Cell: Eosin OD min',
     'tumor_q3_Cytoplasm: Hematoxylin OD mean', 'tumor_q3_Cytoplasm: Hematoxylin OD std dev',
     'tumor_q3_Cytoplasm: Hematoxylin OD max', 'tumor_q3_Cytoplasm: Hematoxylin OD min',
     'tumor_q3_Cytoplasm: Eosin OD mean', 'tumor_q3_Cytoplasm: Eosin OD std dev', 'tumor_q3_Cytoplasm: Eosin OD max',
     'tumor_q3_Cytoplasm: Eosin OD min', 'tumor_q3_Nucleus/Cell area ratio']


    # feature_names = ["stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "stroma_mean", "stroma_median", "stroma_std", "stroma_amax", "stroma_amin", "stroma_q1", "stroma_q3",
    #              "kde_1_mean", "kde_1_median", "kde_1_std", "kde_1_amin", "kde_1_amin", "kde_1_q1", "kde_1_q3",
    #              "kde_2_mean", "kde_2_median", "kde_2_std", "kde_2_amin", "kde_2_amin", "kde_2_q1", "kde_2_q3",
    #              "kde_3_mean", "kde_3_median", "kde_3_std", "kde_3_amin", "kde_3_amin", "kde_3_q1", "kde_3_q3",
    #              "kde_4_mean", "kde_4_median", "kde_4_std", "kde_4_amin", "kde_4_amin", "kde_4_q1", "kde_4_q3",
    #              "kde_5_mean", "kde_5_median", "kde_5_std", "kde_5_amin", "kde_5_amin", "kde_5_q1", "kde_5_q3",
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3",
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3",
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"
    #              "tumor_mean", "tumor_median", "tumor_std", "tumor_amax", "tumor_amin", "tumor_q1", "tumor_q3"]
    save_to = os.path.join(model_out_root, 'patch_classification_feature_importance.png')
    reshaped_feature_importance = np.reshape(FeatureImportance, [87, 7])
    plt.figure(figsize=[20, 5])
    plt.imshow(reshaped_feature_importance.T)
    plt.show()

    from sklearn import linear_model
    # import glmnet_python
    from glmnet_python import glmnet
    from glmnet_python import glmnetPlot
    # from glmnetPrint import glmnetPrint
    # from glmnetCoef import glmnetCoef
    # from glmnetPredict import glmnetPredict
    # from cvglmnet import cvglmnet
    # from cvglmnetCoef import cvglmnetCoef
    # from cvglmnetPlot import cvglmnetPlot
    # from cvglmnetPredict import cvglmnetPredict

    import scipy
    k = np.array(train_labels).astype(np.float64)
    f = isinstance(k, scipy.ndarray)
    f2 = isinstance(train_labels, scipy.ndarray)
    s_w = class_weight.compute_sample_weight('balanced', train_labels)
    fit = glmnet(x=train, y=np.array(train_labels).astype(np.float64), family='gaussian', weights=s_w, alpha=0.2, nlambda=20)
    glmnetPlot(fit, xvar='lambda', label=True)

    '''
    references
    https://glmnet-python.readthedocs.io/en/latest/glmnet_vignette.html#Installation
    '''

    clf = linear_model.Lasso(alpha=0.001)
    model = clf.fit(train, train_labels, s_w)
    FeatureImportance = clf.coef_
    reshaped_feature_importance = np.reshape(FeatureImportance, [87, 7])
    plt.figure(figsize=[20, 5])
    plt.imshow(reshaped_feature_importance.T)
    plt.show()

    # show important features only
    important_f = {}
    for idx, f in enumerate(FeatureImportance):
        if abs(f) > 0.1:
            important_f[feature_names[idx]] = f
    print(important_f)

    predictions = clf.predict(test)
    p_out = []
    for p in predictions:
        if p > 0.5:
            p_out.append(1)
        else:
            p_out.append(0)
    cm = confusion_matrix(test_labels, p_out, normalize='true')
    print(cm)


    print("Done")
