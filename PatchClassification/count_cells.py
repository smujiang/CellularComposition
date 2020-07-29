import os

data_root_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/QuPathMeasurements_for_patchLevel"
borderline_case_id_list = ["OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                           "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                           "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

high_grade_case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                           "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                           "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015"]


high_cell_cnt = 0
high_tumor_cell_cnt = 0
bord_cell_cnt = 0
bord_tumor_cell_cnt = 0

for c in high_grade_case_id_list:
    fn = os.path.join(data_root_dir, c, "detections_measurements_predictions.txt")
    fp = open(fn, 'r')
    lines = fp.readlines()
    for l in lines:
        if "Tumor" in l:
            high_tumor_cell_cnt += 1
    high_cell_cnt += len(lines) - 1

for c in borderline_case_id_list:
    fn = os.path.join(data_root_dir, c, "detections_measurements_predictions.txt")
    fp = open(fn, 'r')
    lines = fp.readlines()
    for l in lines:
        if "Tumor" in l:
            bord_tumor_cell_cnt += 1
    bord_cell_cnt += len(lines) - 1

print("High grade cell count: %d, in which %d are tumor cells" %(high_cell_cnt, high_tumor_cell_cnt))
print("Borderline cell count: %d, in which %d are tumor cells" %(bord_cell_cnt, bord_tumor_cell_cnt))

