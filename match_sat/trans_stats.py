import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.misc import *

root = '/datasets/RS_Dataset/HZ20'

# match result dir
results_dir = '/datasets/RS_Dataset/HZ20/results'


def read_pts(txt_path):
    pts = []
    with open(txt_path) as f:
        for line in f.readlines():
            pt = [int(s.strip()) for s in line.split(',')]
            pts.append(pt)
    return pts


def pts_hw2xy(pts):  # match 匹配结果为 h,w -> 转成 x,y 计算 affine matrix
    return np.array(pts, dtype=np.float32)[:, ::-1]


def save_trans_matrixs():
    trans_matrixs = {}
    angles = []
    scales = []

    tbar = tqdm(sorted(os.listdir(results_dir)))
    for tile in tbar:
        tbar.set_description(tile)
        tile_dir = os.path.join(results_dir, tile)

        sat_pts = read_pts(f'{tile_dir}/correspondence_A_top_5.txt')
        lbl_pts = read_pts(f'{tile_dir}/correspondence_Bt_top_5.txt')

        if len(sat_pts) < 3:
            continue

        M = cv2.estimateAffinePartial2D(pts_hw2xy(lbl_pts), pts_hw2xy(sat_pts))[0]  # >estimateAffine2D
        angle = np.rad2deg(np.arctan(M[1][0] / M[1][1]))  # (-90, 90)
        scale = M[0][0] if angle == 0 else M[1][0] / np.sin(np.deg2rad(angle))  # 28 个 angle=0

        trans_matrixs[tile] = {
            'matrix': M,
            'angle': angle,
            'scale': scale
        }

        angles.append(angle)
        scales.append(scale)

    dump_pickle(trans_matrixs, f'{root}/trans_matrixs.pkl')  # 1712/1755
    np.save(f'{root}/trans_angles.npy', np.array(angles))
    np.save(f'{root}/trans_scales.npy', np.array(scales))


def plt_hist(data, bins=20, title=None):
    hist = np.histogram(data, bins=bins)
    print(hist)

    freq, xticks = hist
    x = np.arange(len(xticks))
    xticks = np.around(xticks, decimals=2)

    plt.figure(figsize=(12 * bins / 20, 5))
    plt.xticks(x, xticks)
    plt.bar(x[:-1] + 0.1, freq, align='edge', width=0.8)  # 与区间左刻度对齐

    # y 轴数字标签
    for a, b in zip(x[:-1], freq):
        plt.text(a + 0.5, b + 0.002, b, ha='center', va='bottom')

    # 关闭 top,right 边界线
    # ax = plt.axes()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    if title:
        plt.title(title)
    plt.show()


def trans_stats():
    # scales = np.load(f'{root}/trans_scales.npy')
    # plt_hist(scales, title='scale factors distribution')

    angles = np.load(f'{root}/trans_angles.npy')
    # plt_hist(angles, title='rotation angles distribution')

    # 进一步查看旋角区间
    hist = np.histogram(angles, bins=20)
    freq, xticks = hist
    idx = np.argmax(freq)
    xleft, xright = xticks[idx], xticks[idx + 2]
    angles = angles[(xleft < angles) & (angles < xright)]

    plt_hist(angles, bins=15)


if __name__ == '__main__':
    # save_trans_matrixs()
    trans_stats()
