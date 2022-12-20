import openslide
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_patch_by_location(wsi_obj, location, res=1, patch_size=(512, 512)):
    loc = [int(location[0]/res), int(location[1]/res)]
    cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, patch_size)
    return cell_img



case_id_list = ["OCMC-001", "OCMC-002", "OCMC-003", "OCMC-004", "OCMC-005",
                "OCMC-006", "OCMC-007", "OCMC-008", "OCMC-009", "OCMC-010",
                "OCMC-011", "OCMC-012", "OCMC-013", "OCMC-014", "OCMC-015",
                "OCMC-016", "OCMC-017", "OCMC-018", "OCMC-019", "OCMC-020",
                "OCMC-021", "OCMC-022", "OCMC-023", "OCMC-024", "OCMC-025",
                "OCMC-026", "OCMC-027", "OCMC-028", "OCMC-029", "OCMC-030"]

wsi_dir = "\\\\mfad\\researchmn\\HCPR\\HCPR-GYNECOLOGICALTUMORMICROENVIRONMENT\\WSIs"
model_out_root = "H:\\OvaryCancer\\PatchClassification"
img_out_root = "H:\\OvaryCancer\\PatchClassification\\misclassified_patches"

misclass_fn = os.path.join(model_out_root, 'misclassified_patches_loc_score.txt')
fp_misclass = open(misclass_fn, 'a')

data = pd.read_csv(misclass_fn, sep=',')
for idx, case_id in enumerate(case_id_list):
    print("Saving misclassified patches from %s" % case_id)
    wsi_obj = openslide.OpenSlide(os.path.join(wsi_dir, case_id+'.svs'))
    mis_case_id_list = data.iloc[:, 0]
    for mis_idx, mis_case_id in enumerate(mis_case_id_list):
        if mis_case_id == case_id:
            loc = data.iloc[mis_idx, 1:]
            img = get_patch_by_location(wsi_obj, loc)
            img_fn = data.iloc[mis_idx,0] + "_" + str(data.iloc[mis_idx,1]) + "_" + str(data.iloc[mis_idx,2]) + "_" + str(data.iloc[mis_idx,3])
            save_to = os.path.join(img_out_root, img_fn+".png")
            img.save(save_to)









