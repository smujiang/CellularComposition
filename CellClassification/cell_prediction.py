import pickle
import platform
import os, sys
import numpy as np
from FeatureAnalysis import loadQuPathMeasurements_asDF

if "Linux" in platform.platform():
    machine = "infodev2"
    DEBUG = False
else:
    machine = "Mac"
    DEBUG = True

if __name__ == "__main__":
    measurments_data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/QuPathMeasurements"
    model_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/QuPathMeasurements_analysis"
    data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/QuPathMeasurements_for_patchLevel"
    wsi_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/WSIs"

    class_names = ["Tumor", "Stroma"]

    # case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
    #                 "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020"]
    # case_id_list = ["OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
    #                 "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015",
    #                 "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
    #                 "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

    case_id_list = ["S002_VHE_region_046", "S002_VHE_region_122", "S002_VHE_region_253"]

    # case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
    #                 "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
    #                 "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015",
    #                 "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
    #                 "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
    #                 "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

    # Load trained model from file
    pkl_filename = os.path.join(model_root, "cell_classification_model.pkl")
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    for case_id in case_id_list:
        data_txt_dir = os.path.join(measurments_data_root, case_id)
        for fn in os.listdir(data_txt_dir):
            print("Processing %s %s" % (case_id, fn))
            data_txt = os.path.join(data_txt_dir, fn)
            df = loadQuPathMeasurements_asDF([data_txt])
            start_idx = 0
            for idx, col in enumerate(df.columns):
                if "Centroid Y" in col:
                    start_idx = idx+1
                    break
            test_cell_data = np.array(df.iloc[:, start_idx:])
            test_cell_location = np.array(df.iloc[:, start_idx-2:start_idx])

            predictions = pickle_model.predict(test_cell_data)
            for idx, p in enumerate(predictions):
                if p == 1:
                    df.iloc[idx, 2] = "Tumor"
                else:
                    df.iloc[idx, 2] = "Stroma"

            # Save to
            data_out_dir = os.path.join(data_root, case_id)
            if not os.path.exists(data_out_dir):
                os.makedirs(data_out_dir)
            data_out_txt = os.path.join(data_out_dir, fn.replace(".txt", "_prediction.txt"))
            df.to_csv(data_out_txt, index=None, sep='\t')

print("Done")

