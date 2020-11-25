import json
import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
import os


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# io: txt <-> list
def write_list_to_txt(a_list, txt_path):
    with open(txt_path, 'w') as f:
        for p in a_list:
            f.write(p + '\n')


def read_txt_as_list(f):
    with open(f, 'r') as f:
        return [p.replace('\n', '') for p in f.readlines()]


# json io
def dump_json(adict, out_path):
    with open(out_path, 'w', encoding='UTF-8') as json_file:
        # 设置缩进，格式化多行保存; ascii False 保存中文
        json_str = json.dumps(adict, indent=2, ensure_ascii=False)
        json_file.write(json_str)


def load_json(in_path):
    with open(in_path, 'rb') as f:
        adict = json.load(f)
        return adict


# pickle io
def dump_pickle(data, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
        print('write data to', out_path)


def load_pickle(in_path):
    with open(in_path, 'rb') as f:
        data = pickle.load(f)  # list
        return data


def get_label_name_colors(csv_path):
    """
    read csv_file and save as label names and colors list
    :param csv_path: csv color file path
    :return: lable name list, label color list
    """
    label_names, label_colors = [], []
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i > 0:  # 跳过第一行
                label_names.append(row[0])
                label_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return label_names, label_colors


def color_code_target(target, label_colors):
    return np.array(label_colors, dtype='uint8')[target.astype(int)]


def plt_img_target(img, target, label_colors=None, title=None):
    f, axs = plt.subplots(nrows=1, ncols=2, dpi=100)
    f.set_size_inches((8, 4))
    ax1, ax2 = axs.flat[0], axs.flat[1]

    # ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('img')

    # ax2.axis('off')
    if label_colors:
        target = color_code_target(target, label_colors)
    ax2.imshow(target)
    ax2.set_title('target')

    if title:
        plt.suptitle(title)

    plt.show()
