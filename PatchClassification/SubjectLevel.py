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
        txt_root = "/Users/m192500/Dataset/OvaryCancer/CellClassification/QuPathMeasurements"
        QuPath_csv = "../Evaluation/label_color_table_QuPath_update.csv"
        img_data_root = "/Users/m192500/Dataset/OvaryCancer/CellClassification/ROI_Masks_out_norm"
        model_out_root = "/Users/m192500/Dataset/OvaryCancer/PatchClassification"
        data_out_file = "/Users/m192500/Dataset/OvaryCancer/PatchClassification/train_test_data.npz"
        # mask_fn = "/Users/m192500/Dataset/OvaryCancer/512Norm_QBRC_out/OCMC-001/OCMC-001_59333_55264.png"
        wsi_root = "/Users/m192500/Dataset/OvaryData/WSIs"
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

    ####################patch score histogram #########################
    plt.figure(2)
    plt.hist(y_score[test_labels >= 0.5], bins=20, linewidth=2, histtype="step")
    plt.hist(y_score[test_labels < 0.5], bins=20, linewidth=2,histtype="step")
    plt.grid()
    plt.legend(["HGOSC", "BOSC"])
    plt.show()

    #############################################
    # get patch scores in each case
    all_cases_scores = []
    for i in range(len(case_id_list)):
        case_i_scores = []
        for idx, test_case_id in enumerate(test_case_idx):
            if i == test_case_id:
                case_i_scores.append(y_score[idx])
        all_cases_scores.append(case_i_scores)
    from sklearn.utils import resample
    all_HG_cases_resampled_scores = []
    all_B_cases_resampled_scores = []

    for idx, case_scores in enumerate(all_cases_scores):
        for i in range(1000):
            temp = resample(case_scores, n_samples=1000, replace=True, random_state=i)
            print("case index:%d, case score: %.2f" % (idx, np.average(temp)))
            if idx < 15:
                all_HG_cases_resampled_scores.append(np.average(temp))
            else:
                all_B_cases_resampled_scores.append(np.average(temp))
    plt.figure(3)
    plt.hist(all_HG_cases_resampled_scores, bins=5, linewidth=2, histtype="step")
    plt.hist(all_B_cases_resampled_scores, bins=5, linewidth=2, histtype="step")
    plt.legend(["HGOSC", "BOSC"])
    plt.grid()
    plt.show()

    print("Done")
