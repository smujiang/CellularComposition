import os
import pandas as pd

class label_color_CSVManager:
    def __init__(self, label_color_table=None):
        if label_color_table is None:
            label_color_table = "./label_color_table.csv"
        if not os.path.exists(label_color_table):
            raise Exception("File does not exist.")
        self.df = pd.read_csv(label_color_table)
        self.df_rows = self.df.shape[0]
        self.df_columns = self.df.shape[1]

    def get_label_text_by_color(self, label_color, RGB=True):
        for i in range(self.df_rows):
            label_color_temp = self.df.at[i, 'label_color']
            if label_color == label_color_temp:
                return self.df.at[i, 'label_text']
        raise Exception("label color not found")

    def get_color_by_label_text(self, label_text, RGB=True):
        for i in range(self.df_rows):
            label_text_temp = self.df.at[i, 'label_text']
            if label_text == label_text_temp:
                if RGB:
                    color_str = self.df.at[i, 'label_color']
                    color_str = color_str[2:]
                    return tuple(int(color_str[i:i + 2], 16) for i in (0, 2, 4))
                else:
                    return self.df.at[i, 'label_color']
        raise Exception("label text not found")

    def get_label_text_by_label_id(self, label_id):
        for i in range(self.df_rows):
            label_id_temp = self.df.at[i, 'label_ID']
            if label_id == label_id_temp:
                class_text = self.df.at[i, 'label_text']
                return class_text
        raise Exception("label text %d not found" % label_id)

    def get_label_id_by_label_text(self, label_text):
        for i in range(self.df_rows):
            label_text_temp = self.df.at[i, 'label_text']
            if label_text == label_text_temp:
                class_id = self.df.at[i, 'label_ID']
                return int(class_id)
        raise Exception("label text %s not found" % label_text)

    def get_color_list(self, column_name='label_color', RGB=True):
        color_str_list = []
        if RGB:
            color_str = self.df[column_name].values
            for cl in color_str:
                cl = cl[2:]
                color_str_list.append(tuple(int(cl[i:i + 2], 16) for i in (0, 2, 4)))
            return color_str_list
        else:
            return self.df[column_name].values

    def get_label_id_list(self, column_name='label_ID'):
        return self.df[column_name].values

    def get_label_txt_list(self, column_name='label_text'):
        return self.df[column_name].values


if __name__ == "__main__":
    lcm = label_color_CSVManager()
    print(lcm.get_color_by_label_text("Tumor"))
    print(lcm.get_label_text_by_color("0xFF0000"))

