import numpy as np
import os
import sys
import platform
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, compute_sample_weight
import pandas as pd

sys.path.append(os.path.abspath('../Evaluation'))
from label_csv_manager import label_color_CSVManager
import openslide


def get_feature_names(first_line):
    ele = first_line.split('\t')
    loc = ele[5:7]
    return loc, ele[7:]


def loadQuPathFeatures(line, with_cell_location=True):
    if "Non-lymphocyte stromal" in line:
        print("found one Non-lymphocyte stromal")
    ele = line.split('\t')
    if with_cell_location:
        feature = [float(i) for i in ele[5:]]
    else:
        feature = [float(i) for i in ele[7:]]
    return ele[2], feature  # label and features


def loadQuPathMeasurements_asDF(txt_file_list):
    df_all = pd.DataFrame()
    for txt_fn in txt_file_list:
        case_df = pd.read_csv(txt_fn, sep='\t')
        df_all = pd.concat([df_all, case_df])
    return df_all

def loadQuPathMeasurements(txt_fn_list, class_label_manager, with_cell_location=True):
    fp0 = open(txt_fn_list[0], 'r')
    line = fp0.readline()
    feature_dims = len(get_feature_names(line)[1])
    fp0.close()
    if with_cell_location:
        feature_dims += 2
    features = np.empty((0, feature_dims), np.float)
    label_ids = []
    for txt_fn in txt_fn_list:
        # print("processing %s " % txt_fn)
        with open(txt_fn, 'r') as fp:
            line = fp.readline()
            _, feature_names = get_feature_names(line)
            for line in fp.readlines():
                case_label_txt, case_feature = loadQuPathFeatures(line, with_cell_location)
                label_id = class_label_manager.get_label_id_by_label_text(case_label_txt)
                label_ids.append(label_id)
                features = np.vstack([features, case_feature])
    return features, label_ids, feature_names


def get_txt_file_list(data_root_dir, case_ids):
    txt_file_list = []
    for case_id in case_ids:
        txt_files = os.listdir(os.path.join(data_root_dir, case_id))
        txt_file_list.append(os.path.join(data_root_dir, case_id, txt_files[0]))
        # for txt in txt_files:
        #     txt_file_list.append(os.path.join(data_root_dir, case_id, txt))
    return txt_file_list


def save_to_QuPath_points(color, points, save_to):
    color = -16711936
    wrt_str = "Name\tPathAnnotationObject\nColor\t%d\n" % color
    wrt_str += "Coordinates\t%d\n" % len(points)
    for p in points:
        wrt_str += "%f\t%f\n" % (p[0], p[1])
    csv_fp = open(save_to, 'w')
    csv_fp.write(wrt_str)
    csv_fp.close()


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


def get_cells_by_location(wsi_obj, location, res=0.2523, box_size=(50, 50)):
    loc = [int(location[0]/res-box_size[1]/2), int(location[1]/res-box_size[1]/2)]
    cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, box_size)
    return cell_img

if "Linux" in platform.platform():
    machine = "infodev2"
    DEBUG = False
else:
    machine = "Mac"
    DEBUG = True

if __name__ == "__main__":
    if machine == "Mac":
        measurments_data_root = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification/QuPathMeasurements"
        data_out_root = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification/QuPathMeasurements_analysis"
        wsi_dir = "/Users/My_LANID/Dataset/OvaryData/WSIs"
    elif machine == "infodev2":
        measurments_data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/QuPathMeasurements"
        data_out_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/QuPathMeasurements_analysis"
        wsi_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs"
    else:
        raise Exception("undefined machine to run the code")
    # class_names = ["Tumor", "Lymphocyte", "Stroma"]
    class_names = ["Tumor", "Stroma"]

    GET_CELL_LOCATION = True
    QuPath_csv = "../../Evaluation/label_color_table_QuPath_update.csv"

    LOOP_CNT = 1

    # EvaluationTask_list = ["high_grade", "all", "borderline"]
    # EvaluationTask_list = ["high_grade", "borderline"]
    # EvaluationTask_list = ["borderline", "high_grade"]
    EvaluationTask_list = ["all"]

    for EvaluationTask in EvaluationTask_list:
        if EvaluationTask == "all":
            plt_title = "All Cases"
            case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                            "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020"]
        elif EvaluationTask == "borderline":
            plt_title = "Borderline Cases"
            if not DEBUG:
                case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020"]
            else:
                # case_id_list = ["OCMC-020"]  # for debug
                case_id_list = ["OCMC-016", "OCMC-017"]
                # case_id_list = ["OCMC-016"]

        elif EvaluationTask == "high_grade":
            plt_title = "High Grade Cases"
            if not DEBUG:
                case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005"]
            else:
                case_id_list = ["OCMC-004"]  # for debug
                # case_id_list = ["OCMC-001", "OCMC-004"]
        else:
            raise Exception("Undefined evaluation task")

        importance_dim1_loop_csv = os.path.join(data_out_root, EvaluationTask + "_dim1_loops.csv")
        # importance_dim2_loop_csv = os.path.join(data_out_root, EvaluationTask + "_dim2_loops.csv")

        label_class_lcm = label_color_CSVManager(QuPath_csv)
        label_colors = label_class_lcm.get_color_list()
        label_ids = label_class_lcm.get_label_id_list()

        print("Processing %s" % case_id_list)
        txt_file_list = get_txt_file_list(measurments_data_root, case_id_list)

        df = loadQuPathMeasurements_asDF(txt_file_list)
        ids_txt = df.iloc[:, 2]
        ids = []
        for id_txt in ids_txt:
            cell_class_id = label_class_lcm.get_label_id_by_label_text(id_txt)
            if cell_class_id == 3:
                cell_class_id = 2
            ids.append(cell_class_id)

        for lp in range(LOOP_CNT):
            print("Loop %d" % lp)
            train, test, train_labels, test_labels = train_test_split(df, np.array(ids), test_size=0.33,
                                                                      random_state=lp)
            train_cell_data = np.array(train.iloc[:, 7:])
            train_cell_location = np.array(train.iloc[:, 5:7])
            test_cell_data = np.array(test.iloc[:, 7:])
            test_cell_location = np.array(test.iloc[:, 5:7])
            k = train.iloc[:, 0]
            train_cell_from_wsi_name = list(train.iloc[:, 0])  # WSI name for training
            test_cell_from_wsi_name = list(test.iloc[:, 0])  # WSI name for testing



            # for c_idx in range(len(train_cell_from_wsi_name)):
            #     wf = os.path.join(wsi_dir, test_cell_from_wsi_name[c_idx])
            #     wsi = openslide.OpenSlide(wf)
            #     I = get_cells_by_location(wsi, test_cell_location[c_idx])
            #     plt.imshow(I)
            #     plt.show()

            s_w = class_weight.compute_sample_weight('balanced', train_labels)





            import pickle
            pkl_filename = os.path.join(data_out_root, "cell_classification_model.pkl")
            if os.path.exists(pkl_filename):
                # s = pickle.dumps(clf)
                print("Load model")
                pkl_fp = open(pkl_filename, 'rb')
                clf = pickle.load(pkl_fp)
            else:
                print("Model fitting")
                clf = SVC(kernel='linear')
                model = clf.fit(train_cell_data, train_labels, s_w)
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(model, file)


            pred = clf.predict(test_cell_data)
            #############################
            # get cells correspond to support vectors
            #############################
            print("Get support cells")
            spv_indices = clf.support_
            spv_cell_locations = train_cell_location[spv_indices]
            sup_cell_labels = train_labels[spv_indices]
            spv_cell_from_wsi_name = np.array(train_cell_from_wsi_name)[spv_indices]

            # for cell_idx, cell_train in enumerate(train_cell_data):
            #     for cell_spv in spv:
            #         if np.array_equal(cell_spv, cell_train):
            #             spv_cell_locations.append(train_cell_location[cell_idx, :])
            #             spv_cell_from_wsi_name.append(train_cell_from_wsi_name[cell_idx])

            spv_cell_fn = [""]*len(spv_indices)
            wsi_names = set(spv_cell_from_wsi_name)
            for wn in wsi_names:
                wf = os.path.join(wsi_dir, wn)
                wsi = openslide.OpenSlide(wf)
                for spv_cl_idx, spv_cl in enumerate(spv_cell_locations):
                    if spv_cell_from_wsi_name[spv_cl_idx] == wn:
                        cell_img = get_cells_by_location(wsi, spv_cl)
                        save_to_dir = os.path.join(data_out_root, str(lp)+"_support_cells")
                        if sup_cell_labels[spv_cl_idx] == 1:
                            f_str = "T"
                        else:
                            f_str = "S"
                        if not os.path.exists(save_to_dir):
                            os.makedirs(save_to_dir)
                        sv_fn = os.path.join(save_to_dir, wn[:-4]+"_"+str(int(spv_cl[0]))+"_"+str(int(spv_cl[0]))+ "_" + f_str +".png")
                        spv_cell_fn[spv_cl_idx] = sv_fn
                        cell_img.save(sv_fn)

            # KNN support cells
            print("KNN support cells")
            sup_cell_features = train_cell_data[spv_indices]
            K = 10
            from sklearn.cluster import KMeans
            import shutil

            kmeans = KMeans(n_clusters=K, random_state=0).fit(sup_cell_features)
            for idx, km in enumerate(kmeans.labels_):
                if sup_cell_labels[idx] == 1:
                    f_str = "T"
                else:
                    f_str = "S"
                save_to_dir = os.path.join(data_out_root, "knn_support_cells",
                                           str(km) + "_" + f_str)
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)
                shutil.copy2(spv_cell_fn[idx], save_to_dir)


            ###############################################################
            print("Get misclassified cells")
            misclassified_cell_idx = []
            misclassify_gtruth = []
            for i in range(len(test_labels)):
                if not test_labels[i] == pred[i]:
                    misclassified_cell_idx.append(i)
                    if test_labels[i] == 1:
                        misclassify_gtruth.append("T")
                    else:
                        misclassify_gtruth.append("S")
            misclassified_cell_loc = test_cell_location[misclassified_cell_idx]
            mis_cell_from_wsi_name = np.array(test_cell_from_wsi_name)[misclassified_cell_idx]
            mis_cell_names = [""] * len(misclassified_cell_idx)
            wsi_names = set(mis_cell_from_wsi_name)
            for wn in wsi_names:
                wf = os.path.join(wsi_dir, wn)
                wsi = openslide.OpenSlide(wf)
                for mis_cl_idx, mis_cl in enumerate(misclassified_cell_loc):
                    if mis_cell_from_wsi_name[mis_cl_idx] == wn:
                        cell_img = get_cells_by_location(wsi, mis_cl)
                        save_to_dir = os.path.join(data_out_root, str(lp) + "_misclassified_cells")
                        if not os.path.exists(save_to_dir):
                            os.makedirs(save_to_dir)
                        sv_fn = os.path.join(save_to_dir,
                                             wn[:-4] + "_" + str(int(mis_cl[0])) + "_" + str(int(mis_cl[0])) + "_" + misclassify_gtruth[mis_cl_idx] +".png")
                        mis_cell_names[mis_cl_idx] = sv_fn
                        cell_img.save(sv_fn)

            # KNN misclassified cells
            print("KNN misclassified cells")
            misclassified_cell_features = test_cell_data[misclassified_cell_idx]
            misclassified_cell_labels = test_labels[misclassified_cell_idx]
            K = 10
            from sklearn.cluster import KMeans
            import shutil
            kmeans = KMeans(n_clusters=K, random_state=0).fit(misclassified_cell_features)
            for idx, km in enumerate(kmeans.labels_):
                save_to_dir = os.path.join(data_out_root, "knn_misclassified_cells", str(km)+"_"+misclassify_gtruth[idx])
                if not os.path.exists(save_to_dir):
                    os.makedirs(save_to_dir)
                shutil.copy2(mis_cell_names[idx], save_to_dir)
            break



''' 
        features, I_ids, feature_names = loadQuPathMeasurements(txt_file_list, label_class_lcm, GET_CELL_LOCATION)
        ids = np.array(I_ids)
        ids[np.where(ids == 3)] = 2

        for lp in range(LOOP_CNT):
            print("Loop %d" % lp)
            train, test, train_labels, test_labels = train_test_split(features, np.array(ids), test_size=0.33,
                                                                      random_state=lp)

            if GET_CELL_LOCATION:
                train_cell_location = train[:, 0:2]
                train = train[:, 2:]
                test_cell_location = test[:, 0:2]
                test = test[:, 2:]

            # c_w = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
            s_w = class_weight.compute_sample_weight('balanced', train_labels)

            clf = SVC(kernel='linear')
            # clf = LinearSVC()
            model = clf.fit(train, train_labels, s_w)

            # get support vectors (cell samples)
            spv_cell_locations = []
            spv = clf.support_vectors_
            for cell_idx, cell_train in enumerate(train):
                for cell_spv in spv:
                    if np.array_equal(cell_spv, cell_train):
                        spv_cell_locations.append(train_cell_location[cell_idx, :])

            # model = clf.fit(train, train_labels)
            #############################
            # Feature importance
            #############################
            for idx, f_name in enumerate(feature_names):
                feature_names[idx] = f_name.strip()
            print("Feature importance analysis")
            save_to = os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_importance_0.png")
            # f_importances(abs(clf.coef_[0]), feature_names, save_to)
            f_importances((clf.coef_[0]), feature_names, save_to)
            save_to = os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_importance_0_top10.png")
            # imp, names = f_importances(abs(clf.coef_[0]), feature_names, save_to, fig_sz=(12, 8), top=10)
            imp, names = f_importances((clf.coef_[0]), feature_names, save_to, fig_sz=(12, 8), top=10)
            fp = open(importance_dim1_loop_csv, 'a')
            wrt_str = ', '.join(names)
            wrt_str += '\n'
            wrt_str += str(imp).strip('[]')
            wrt_str += '\n'
            fp.write(wrt_str)
            fp.close()

            # save_to = os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_importance_1.png")
            # f_importances(abs(clf.coef_[1]), feature_names, save_to)
            # save_to = os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_importance_1_top10.png")
            # imp, names = f_importances(abs(clf.coef_[1]), feature_names, save_to, fig_sz=(6, 6), top=10)
            # fp = open(importance_dim2_loop_csv, 'a')
            # wrt_str = ', '.join(names)
            # wrt_str += '\n'
            # wrt_str = str(imp).strip('[]')
            # wrt_str += '\n'
            # fp.write(wrt_str)
            # fp.close()

            # save_to = os.path.join(data_out_root, EvaluationTask+"_importance_2.png")
            # f_importances(abs(clf.coef_[2]), feature_names, save_to)
            # save_to = os.path.join(data_out_root, EvaluationTask + "_importance_2_top10.png")
            # f_importances(abs(clf.coef_[2]), feature_names, save_to, fig_sz=(6, 6), top=10)

            predictions = clf.predict(test)
            print("Over all accuracy: %f" % accuracy_score(test_labels, predictions))

            #############################
            # Confusion matrix
            #############################
            print("Confusion matrix:")
            cm = confusion_matrix(test_labels, predictions, normalize='true')
            print(cm)
            output_cm_csv = os.path.join(data_out_root, EvaluationTask + "_cm.csv")
            fp = open(output_cm_csv, 'a')
            fp.write(str(cm))
            fp.write('\n')
            fp.close()
            plt.figure(figsize=[5, 5])
            disp = plot_confusion_matrix(clf, test, test_labels,
                                         display_labels=class_names,
                                         cmap=plt.get_cmap('Blues'),
                                         normalize='true')
            disp.ax_.set_title(plt_title)
            plt.subplots_adjust(left=0.2)
            plt.savefig(os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_confusion_matrix.png"),
                        dpi=300)
            plt.close()

            #############################
            # misclassified cells' locations
            #############################
            wrt_str = "c_x,c_y\n"
            if GET_CELL_LOCATION:
                for idx, tl in enumerate(test_labels):
                    if not tl == predictions[idx]:
                        wrt_str += "%f,%f\n" % (test_cell_location[idx][0], test_cell_location[idx][1])
            # output_location_csv = os.path.join(data_out_root, selected_case+"_erro_loc.csv")
            output_location_csv = os.path.join(data_out_root, EvaluationTask + "_lp" + str(lp) + "_erro_loc.csv")

            csv_fp = open(output_location_csv, 'w')
            csv_fp.write(wrt_str)
            csv_fp.close()
print("Done")
'''

