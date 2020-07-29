import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, compute_sample_weight
sys.path.append(os.path.abspath('../../Evaluation'))
from label_csv_manager import label_color_CSVManager

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
    return ele[2], feature   # label and features

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
    return features, label_ids

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
    wrt_str = "Name\tPathAnnotationObject\nColor\t%d\n" %color
    wrt_str += "Coordinates\t%d\n" % len(points)
    for p in points:
        wrt_str += "%f\t%f\n" % (p[0], p[1])
    csv_fp = open(save_to, 'w')
    csv_fp.write(wrt_str)
    csv_fp.close()

# EvaluationTask = "borderline"
# EvaluationTask = "high_grade"
EvaluationTask = "all"  # "high_grade" , "borderline" or "all"
if __name__ == "__main__":
    measurments_data_root = "/Users/m192500/Dataset/OvaryCancer/CellClassification/QuPathMeasurements"

    data_out_root = "/Users/m192500/Dataset/OvaryCancer/CellClassification/QuPathMeasurements_analysis"

    if EvaluationTask == "all":
        plt_title = "All Cases"
        case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                        "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020"]
    elif EvaluationTask == "borderline":
        plt_title = "Borderline Cases"
        case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020"]
    elif EvaluationTask == "high_grade":
        plt_title = "High Grade Cases"
        case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005"]
        # case_id_list = ["OCMC-001", "OCMC-002"]  # for debug
        # case_id_list = ["OCMC-004"]  # for debug
    else:
        raise Exception("Undefined evaluation task")

    class_names = ["Tumor", "Lymphocyte", "Stroma"]

    GET_CELL_LOCATION = True

    QuPath_csv = "../../Evaluation/label_color_table_QuPath_update.csv"
    label_class_lcm = label_color_CSVManager(QuPath_csv)
    label_colors = label_class_lcm.get_color_list()
    label_ids = label_class_lcm.get_label_id_list()

    selected_case = case_id_list
    # for selected_case in case_id_list:
    print("Processing %s" % selected_case)
    # txt_file_list = get_txt_file_list(measurments_data_root, [selected_case])
    txt_file_list = get_txt_file_list(measurments_data_root, case_id_list)

    features, ids = loadQuPathMeasurements(txt_file_list, label_class_lcm, GET_CELL_LOCATION)


    # TODO: test just one case, I think accuracy will be higher
    # #######################################################
    # # classify cells
    # #######################################################

    train, test, train_labels, test_labels = train_test_split(features, np.array(ids), test_size=0.33, random_state=42)

    if GET_CELL_LOCATION:
        train_cell_location = train[:, 0:2]
        train = train[:, 2:]
        test_cell_location = test[:, 0:2]
        test = test[:, 2:]

    # c_w = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    s_w = class_weight.compute_sample_weight('balanced', train_labels)

    # gnb = GaussianNB()
    # model = gnb.fit(train, train_labels, s_w)
    # model = gnb.fit(train, train_labels)
    # predictions = gnb.predict(test)

    clf = SVC(gamma='auto')
    model = clf.fit(train, train_labels, s_w)
    # model = clf.fit(train, train_labels)
    predictions = clf.predict(test)

    # tumor_cnt = len(np.where(test_labels==1)[0])
    # lymphocyte_cnt = len(np.where(test_labels==2)[0])
    # stroma_cnt = len(np.where(test_labels==3)[0])
    # print("Ground truth in testing. Tumor:%d, Lymphocyte:%d, Stroma:%d" % (tumor_cnt, lymphocyte_cnt, stroma_cnt))

    # tumor_cnt = len(np.where(predictions == 1)[0])
    # lymphocyte_cnt = len(np.where(predictions == 2)[0])
    # stroma_cnt = len(np.where(predictions == 3)[0])
    # print("Predictions in testing. Tumor:%d, Lymphocyte:%d, Stroma:%d" % (tumor_cnt, lymphocyte_cnt, stroma_cnt))

    print("Over all accuracy: %f" % accuracy_score(test_labels, predictions))

    cm = confusion_matrix(test_labels, predictions)
    print("Confusion matrix:")
    print(cm)
    plt.figure(figsize=[5, 5])
    disp = plot_confusion_matrix(clf, test, test_labels,
                                 display_labels=class_names,
                                 cmap=plt.get_cmap('Blues'),
                                 normalize='true')
    disp.ax_.set_title(plt_title)
    plt.subplots_adjust(left=0.2)
    plt.savefig(os.path.join(data_out_root, EvaluationTask+"_confusion_matrix.png"), dpi=300)

    wrt_str = "c_x,c_y\n"
    if GET_CELL_LOCATION:
        for idx, tl in enumerate(test_labels):
            if not tl == predictions[idx]:
                wrt_str += "%f,%f\n" % (test_cell_location[idx][0], test_cell_location[idx][1])
    # output_location_csv = os.path.join(data_out_root, selected_case+"_erro_loc.csv")
    output_location_csv = os.path.join(data_out_root, EvaluationTask+"_erro_loc.csv")

    csv_fp = open(output_location_csv, 'w')
    csv_fp.write(wrt_str)
    csv_fp.close()
print("Done")





