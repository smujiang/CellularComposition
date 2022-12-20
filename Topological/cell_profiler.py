import os
import sys
import numpy as np
from PIL import Image
from skimage import measure
from skimage.measure import regionprops
import openslide
sys.path.append(os.path.abspath('../Evaluation'))
from label_csv_manager import label_color_CSVManager


class Cell:
    def __init__(self, loc, img_filename, features=None, label_id=-1, label_txt=""):
        self.img_fn = img_filename
        self.loc = np.array(loc)   # unit pixel
        self.features = features
        self.label_id = label_id
        self.label_txt = label_txt

    # def get_cells_by_location(self, wsi_obj, box_size=(20, 20)):
    #     cell_img = openslide.OpenSlide.read_region(wsi_obj, self.loc, 0, box_size)
    #     return cell_img

    # location: absolute location (in WSI, unit μm)
    def get_cell_from_WSI_by_um_location(self, wsi_obj, location, res=0.2523, box_size=(50, 50)):
        loc = [int(location[0] / res - box_size[0] / 2), int(location[1] / res - box_size[1] / 2)]
        cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, box_size)
        return cell_img

    # location: pixel location (in WSI, unit μm)
    def get_cell_from_WSI_by_pixel_location(self, wsi_obj, location, box_size=(50, 50)):
        loc = [int(location[0] - box_size[0] / 2), int(location[1] - box_size[1] / 2)]
        cell_img = openslide.OpenSlide.read_region(wsi_obj, loc, 0, box_size)
        return cell_img


class GetCellsFromQuPathMeasurements:
    '''
    The exported QuPath Measurements were saved in this data structure
    Image\tName\tClass\tParent ROI\tCentroid X µm\tCentroid Y µm\tFeature_1...Feature_n
    '''
    def __init__(self, QuPathMeasurementsTxt, class_label_manager):
        self.cell_list = []
        fp0 = open(QuPathMeasurementsTxt, 'r')
        self.lines = fp0.readlines()
        self.parse(class_label_manager)
        fp0.close()

    def get_feature_names(self):
        ele = self.lines[0].split('\t')
        return ele[7:]

    @staticmethod
    def parse_line(line):
        ele = line.split('\t')
        locations = [float(i) for i in ele[5:7]]
        features = [float(i) for i in ele[7:]]
        label_txt = ele[2]
        img_filename = ele[0]
        return locations, features, img_filename, label_txt

    def parse(self, class_label_manager, res=0.2523):
        for line in self.lines[1:]:
            loc, features, filename, lb_txt = self.parse_line(line)
            loc = [round((np.array(loc)/res)[0]), round((np.array(loc)/res)[1])]
            # loc = (np.array(loc)).astype(np.int)
            lb_id = class_label_manager.get_label_id_by_label_text(lb_txt)
            self.cell_list.append(Cell(loc, filename, features, lb_id, lb_txt))

    # get cell features according to cell location
    def get_cell_features(self, cell):
        for c in self.cell_list:
            if np.equal(c.loc, cell.loc):
                return c.loc, c.features
        return None, None
        # raise Exception("Can't get cell feature")

    # # get cell features according to cell location (abs_location, unit μm)
    def get_cell_features_from_loc(self, location, delta=2):
        for c in self.cell_list:
            # print(c.loc)
            # print(location)
            # print('---------')
            if abs(c.loc[0] - location[0]) < delta and abs(c.loc[1] - location[1]) < delta:
                print(c.loc)
                print(location)
                print('----found-----')
            # x = np.array(c.loc)
            # y = location
            # if np.equal(x, y):
                return c.loc, c.features
        return None, None
        # raise Exception("Can't get cell feature")



class CellMask:
    def __init__(self, mask, label_id):
        self.mask = mask
        self.id = label_id

class GetCellsFromMask:
    def __init__(self, mask_Img, orig_Img, label_color_manager, GetCellsFromQuPathMeasurements=None):
        '''

        :param mask_Img:  PIL image
        :param orig_Img: PIL image
        :param label_color_manager:
        '''
        self.label_colors = label_color_manager.get_color_list()
        self.label_ids = label_color_manager.get_label_id_list()
        self.offset = self.get_coordinates_from_fn(orig_Img.filename)
        # self.offset = self.get_abs_coordinates_from_fn(orig_Img.filename)
        self.cell_list = []
        masks = self.getCellMasks(np.array(mask_Img), self.label_colors, self.label_ids)
        if GetCellsFromQuPathMeasurements is not None:
            self.getCells_plus(masks, orig_Img, label_color_manager, GetCellsFromQuPathMeasurements)
        else:
            self.getCells(masks, orig_Img, label_color_manager)

    @staticmethod
    def calculate_features(mask, Img):
        blobs_labels, num = measure.label(mask, background=0, return_num=True)
        props = regionprops(blobs_labels)[0]
        location = props.centroid
        features = [props.major_axis_length, props.minor_axis_length]  # add other features
        # print(Img.filename)
        return location, features

    @staticmethod
    def get_coordinates_from_fn(filename):
        filename = os.path.split(filename)[1]
        l = filename.index("_")
        r = filename.rindex("_")
        x = filename[l + 1:r]
        e = filename.index(".")
        y = filename[r + 1:e]
        return np.array([int(x), int(y)])

    @staticmethod  # unit μm
    def get_abs_coordinates_from_fn(filename, res=0.2523):
        filename = os.path.split(filename)[1]
        l = filename.index("_")
        r = filename.rindex("_")
        x = filename[l + 1:r]
        e = filename.index(".")
        y = filename[r + 1:e]
        return np.array([int(x)*res, int(y)*res])

    def get_cell_location(self, location):
        loc = np.array([round(location[0]), round(location[1])])
        # yy = loc + self.offset
        return loc + self.offset

    def get_cell_abs_location(self, location, res=0.2523):
        # yy = np.array(location)*res
        return list((np.array(location)*res) + np.array(self.offset))

    def getCells(self, masks, orig_Img, label_color_manager):
        for m in masks:
            location, features = self.calculate_features(m.mask, orig_Img)
            loc = self.get_cell_location(location)
            label_txt = label_color_manager.get_label_text_by_label_id(m.id)
            self.cell_list.append(Cell(loc, orig_Img.filename, features, m.id, label_txt))

    def get_feature_from_QuPathMeasurements(self, loc, GetCellsFromQuPathMeasurements):
        c_l, cf = GetCellsFromQuPathMeasurements.get_cell_features_from_loc(loc)
        return c_l, cf

    def getCells_plus(self, masks, orig_Img, label_color_manager, GetCellsFromQuPathMeasurements):
        for m in masks:
            blobs_labels, num = measure.label(m.mask, background=0, return_num=True)
            props = regionprops(blobs_labels)[0]
            location = props.centroid
            cell_loc = self.get_cell_location(location)
            loc, features = self.get_feature_from_QuPathMeasurements(cell_loc, GetCellsFromQuPathMeasurements)
            if loc is not None and features is not None:
                label_txt = label_color_manager.get_label_text_by_label_id(m.id)
                self.cell_list.append(Cell(loc, orig_Img.filename, features, m.id, label_txt))

    def get_cell_img(self):
        print()

    @staticmethod
    def getCellMasks(mask_img_arr, color_list, id_list):
        '''
        Get all the single cell masks and labels, with color as identification
        :param mask_img_arr: image numpy array, 3 channel
        :param color_list: colors in the multi-label mask, order sensitive, should be concordant with id_list
        :param id_list: labels in the multi-label mask, order sensitive, should be concordant with color_list
        :return:
        '''
        masks = []
        for idx, color in enumerate(color_list):
            m_cells = np.all(mask_img_arr == np.zeros(mask_img_arr.shape, dtype=np.uint8) + np.array(color), axis=2) * 1
            blobs_labels, num = measure.label(m_cells, background=0, return_num=True)
            for i in range(1, num + 1):
                cell = np.array(blobs_labels == i) * 255
                cm = CellMask(cell, id_list[idx])
                masks.append(cm)
        return masks

if __name__=="__main__":
    # img_fn = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification/ROI_Masks_out_norm/OCMC-001/OCMC-001_63977_28940.jpg"
    # mask_fn = "/Users/My_LANID/Dataset/OvaryCancer/512Norm_QBRC_out/OCMC-001/OCMC-001_63977_28940.png"

    img_fn = "/Users/My_LANID/Dataset/OvaryData/temp/OurAnnotation/ROI_Masks_out/OCMC-004/OCMC-004_15187_59298.jpg"
    mask_fn = "/Users/My_LANID/Dataset/OvaryData/temp/OurAnnotation/ROI_Masks_out/OCMC-004/OCMC-004_15187_59298-mask.png"

    QuPath_csv = "../Evaluation/label_color_table_QuPath_update.csv"
    label_color_man = label_color_CSVManager(QuPath_csv)

    mask_img = Image.open(mask_fn, 'r')
    orig_img = Image.open(img_fn, 'r')

    cells_creator = GetCellsFromMask(mask_img, orig_img, label_color_man)
    temp = cells_creator.cell_list







