import os
import numpy as np
from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
from utils import util
from options import Options
from utils.misc import *
from hz.GridMap import load_gmap
import cv2
import shutil
from constants import *

img_dir = '/datasets/RS_Dataset/HZ20/image'
mask_dir = '/datasets/RS_Dataset/HZ20/mask'
lbl_dir = '/datasets/RS_Dataset/HZ20/label'

label_names, label_colors = get_label_name_colors('hz/hz20.csv')

args = [
    '--imageSize', '160',
    '--k_final', '5',  # 最后1层 k 数量; 计算 AM, choose top3 from 5; 或者直接聚成 3 簇
    '--k_per_level', '15',  # 调大，可预防 k_final cluster 数量不够计算 affine matrix; 15-7, 10-4
    '--results_dir', 'results',
    '--fast'
]
opt = Options().parse(args)
opt.results_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/match_sat/results'


def get_nbbs():
    save_dir = os.path.join(opt.results_dir, opt.name)

    vgg19 = vgg19_model.define_Vgg19(opt)
    nbbs = NBBs.sparse_semantic_correspondence(
        vgg19,  # model
        opt.gpu_ids, opt.tau, opt.border_size, save_dir,
        opt.k_per_level,
        opt.k_final, opt.fast
    )
    print('creat NBB matcher')
    return nbbs


def crop_mask_from_gmap(tile_pts, pad=15):  # pad 太大会引入一些外部区域，造成匹配干扰?
    pts = np.array(tile_pts)
    x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
    y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())

    x_min = max(0, x_min - pad)
    x_max = min(map_w - 1, x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(map_h - 1, y_max + pad)

    # 极点外界矩形
    tile_mask = gmap[y_min:y_max + 1, x_min:x_max + 1]
    return tile_mask


def pts_hw2xy(pts):  # match 匹配结果为 h,w -> 转成 x,y 计算 affine matrix
    return np.array(pts, dtype=np.float32)[:, ::-1]


def top3_pts_from_cluster(top_k_correspondence):
    pts_A, pts_B, score = top_k_correspondence  # score: rank activations
    idxs = sorted(range(len(score)), key=lambda t: score[t], reverse=True)[:3]

    pts_A = np.array([pts_A[i] for i in idxs])
    pts_B = np.array([pts_B[i] for i in idxs])

    return pts_A, pts_B


def nbb_match_tiles():
    nbbs = get_nbbs()
    transform = util.get_transform(opt.imageSize)

    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    dsize = (opt.imageSize, opt.imageSize)

    for idx, (tile, tile_pts) in enumerate(tile_mapping.items()):
        print()
        print(idx, tile)
        save_dir = os.path.join(opt.results_dir, tile)

        tile_mask = crop_mask_from_gmap(tile_pts)
        tile_img = cv2.imread(f'{sat_dir}/{tile}.jpeg')[:, :, ::-1]
        tile_lbl = color_code_target(tile_mask, label_colors)

        A = transform(tile_img).unsqueeze(0)  # 256
        B = transform(tile_lbl).unsqueeze(0)

        # 保存中间结果
        nbbs.save_dir = save_dir
        top_k_correspondence = nbbs.run(A, B)  # pts_A, pts_B

        if len(top_k_correspondence[0]) < 3:  # 不足够
            print('Stop, not enought pts')
            continue

        # sat_pts, lbl_pts = top3_pts_from_cluster(top_k_correspondence)  # get 3 组 score 最高的
        sat_pts, lbl_pts = top_k_correspondence[:2]

        # from lbl to sat
        M = cv2.estimateAffinePartial2D(pts_hw2xy(lbl_pts), pts_hw2xy(sat_pts))[0]  # >estimateAffine2D

        # filter some invalid result
        # [[cos, -sin, tx],
        #  [sin,  cos, ty]]
        angle = np.rad2deg(np.arctan(M[1][0] / M[1][1]))  # (-90, 90)
        print('angle:', angle)  # 1.8842955329981195
        if abs(angle) > 3:  # 旋角过大 错误匹配
            print('Stop, not valid rotation')
            continue

        scale = M[0][0] if angle == 0 else M[1][0] / np.sin(np.deg2rad(angle))
        print('scale:', scale)  # 1.2023725018301632
        if scale < 1 or scale > 1.34:  # lbl -> sat, 且 lbl 包含场景多，scale >1 才正常
            print('Stop, not valid scale')
            continue

        # resize tile_mask to 160, 统一 match_pts 坐标系
        tile_mask = cv2.resize(tile_mask, dsize, interpolation=cv2.INTER_NEAREST)
        aff_mask = cv2.warpAffine(tile_mask, M, dsize=dsize, flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')

        # affine 有效区域占比 >= 0.8, rotate 已经暗含占比
        # if (aff_mask > 0).sum() / aff_mask.size >= 0.8:
        np.save(f'{mask_dir}/{tile}.npy', aff_mask)  # mask
        cv2.imwrite(f'{lbl_dir}/{tile}.png', color_code_target(aff_mask, label_colors)[:, :, ::-1])  # label
        shutil.copy(src=f'{sat_dir}/{tile}.jpeg', dst=img_dir)  # image
        print('save', tile)


if __name__ == '__main__':
    gmap = load_gmap()
    map_h, map_w = 10532, 11153

    nbb_match_tiles()
