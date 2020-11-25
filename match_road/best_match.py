"""
在 match_bf 基础上，添加启发式，迭代边界搜索，大大减少搜索用时
    虽然我们不能控制 for 跳出的时间
    但是我们可以限制 for 的宽度，而启发式地增加边界
    问题：局部最优解有时候出现在边界内部，不易处理!
"""
import sys

sys.path.insert(0, '/nfs/xs/codes/NBBs')
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import numba as nb
from tqdm import tqdm
from pprint import pprint
from constants import *
from hz.GridMap import load_gmap
from utils.misc import *

# map
gmap = load_gmap()
map_h, map_w = 10532, 11153
label_names, label_colors = get_label_name_colors('hz/hz20.csv')

results_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/match_road'
msk_dir = os.path.join(results_dir, 'mask')
lbl_dir = os.path.join(results_dir, 'label')
mkdir(msk_dir)
mkdir(lbl_dir)


def crop_mask_from_gmap(tile_pts, pad=20):  # pad 太大会引入一些外部区域，造成匹配干扰?
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


# 匹配上的前景占比
@nb.jit(nopython=True)
def matched_fg_ratio(base_msk, aff_msk):
    fg_msk = aff_msk > 0
    fg_sum = fg_msk.sum()
    if fg_sum == 0:
        return 0
    else:
        match_fg = fg_msk & (base_msk == aff_msk)
        return match_fg.sum() / fg_sum


def get_iter_vals(min_, max_, precision):
    num = int((max_ - min_) / precision + 1)
    return np.linspace(min_, max_, num)


thetas = get_iter_vals(-1, 3, precision=0.05)
scales = get_iter_vals(1.1, 1.4, precision=0.05)
# dsize 不影响配准 scale (只有 crop pad 影响)，但影响 for 最内层 cv2.warpAffine
dsize = (160, 160)  # 尺寸再小效果就差了


def bf_match(sat_msk, lbl_msk, txs, tys, last_aff=None):
    best_aff = {
        'ratio': 0,
        'theta': 0,
        'scale': 0,
        'tx': 0,
        'ty': 0,
        'M': None,
    } if last_aff is None else last_aff

    for th_deg in tqdm(thetas):
        th = np.pi * th_deg / 180
        for s in scales:
            for tx in txs:
                for ty in tys:
                    R = s * np.array([[np.cos(th), -np.sin(th)],
                                      [np.sin(th), np.cos(th)]])
                    t = np.array([[tx],
                                  [ty]])
                    M = np.hstack((R, t))

                    aff_msk = cv2.warpAffine(lbl_msk, M, dsize=dsize, flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
                    ratio = matched_fg_ratio(sat_msk, aff_msk)

                    if ratio > best_aff['ratio']:
                        best_aff = {
                            'ratio': ratio,
                            'theta': th_deg,
                            'scale': s,
                            'tx': tx,
                            'ty': ty,
                            'M': M,
                        }
    return best_aff


def expand_vals(val, num, precision):
    vals = []
    num = (num - 1) // 2  # half num
    for i in range(-num, num + 1):
        vals.append(val + i * precision)
    return vals


def get_tx_range(tx, tx_left, tx_right, num=5, precision=3):
    if tx == tx_left:  # 继续向左
        txs = get_iter_vals(tx_left - num * precision, tx_left, precision=precision)  # 以 tx 为 right，长度为 5 小范围
    elif tx == tx_right:
        txs = get_iter_vals(tx_right, tx_right + num * precision, precision=precision)  # 以 tx 为 left，长度为 5 小范围
    else:
        txs = expand_vals(tx, num=num, precision=precision)  # 以 tx 为中心，长度为 5 小范围
    return txs


def match_sat_lbl_road():
    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    num_tiles = len(tile_mapping)

    affine_stats = {}

    # tx/ty global range
    global_tx_left, global_tx_right = -18, 18
    global_ty_left, global_ty_right = -18, 18

    xs = get_iter_vals(global_tx_left, global_tx_right, precision=3)  # 160
    ys = get_iter_vals(global_ty_left, global_ty_right, precision=3)

    for idx, (tile, tile_pts) in enumerate(tile_mapping.items()):
        # sat_road
        sat_road_msk = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        if len(sat_road_msk.shape) < 3:  # 13046-3446 直接 (256,256) 值全 = 0，baidu 存在没有 road 标注的图片
            continue

        print(f'{idx}/{num_tiles}', tile)
        sat_road_msk = sat_road_msk[:, :, -1]  # 取末尾 alpha 通道
        sat_road_msk[sat_road_msk > 0] = 255  # 二值化
        sat_road_msk = cv2.resize(sat_road_msk, dsize, interpolation=cv2.INTER_NEAREST)

        # lbl_road
        tile_msk = crop_mask_from_gmap(tile_pts)
        tile_msk = cv2.resize(tile_msk, dsize, interpolation=cv2.INTER_NEAREST)
        lbl_road_msk = np.zeros_like(tile_msk, dtype='uint8')
        lbl_road_msk[tile_msk == 20] = 255

        # match
        tx_left, tx_right = global_tx_left, global_tx_right
        ty_left, ty_right = global_ty_left, global_ty_right
        txs, tys = xs, ys

        aff = None
        last_ratio = 0.
        while True:
            aff = bf_match(sat_road_msk, lbl_road_msk, txs, tys, last_aff=aff)

            # 合法区间内找到，并且由于传入 last_ratio 必然发生了更新 无需再扩展区间
            if tx_left < aff['tx'] < ty_right and ty_left < aff['ty'] < ty_right:
                break
            else:  # 不在合法区间
                if aff['ratio'] == last_ratio:  # 如果无更新，说明还是在原来的边界处取到
                    break

            # 在边界发生了更新，需要扩展边界继续搜索; 更新下次搜索边界
            txs = get_tx_range(aff['tx'], tx_left, tx_right)
            tys = get_tx_range(aff['ty'], ty_left, ty_right)
            tx_left, tx_right = txs[0], txs[-1]
            ty_left, ty_right = tys[0], tys[-1]
            # 更新 ratio; 有两种 break 情况
            last_ratio = aff['ratio']

        if aff['M'] is None:
            continue

        affine_stats[tile] = aff
        pprint(aff)

        # aff_lbl_msk = cv2.warpAffine(tile_msk, aff['M'], dsize, flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
        #
        # aff_lbl_msk = cv2.resize(aff_lbl_msk, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(f'{lbl_dir}/{tile}.png', color_code_target(aff_lbl_msk, label_colors)[:, :, ::-1])
        # np.save(f'{msk_dir}/{tile}.npy', aff_lbl_msk)

    dump_pickle(affine_stats, f'/datasets/rs_segment/AerialCitys/HZ20/matches/road_affine_stats_heuri.pkl')


if __name__ == '__main__':
    match_sat_lbl_road()
