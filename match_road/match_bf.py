"""
迭代 仿射矩阵; # 认为是相似变换，少2个参数
    旋转
    放缩
    平移
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


results_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/match_road'
msk_dir = os.path.join(results_dir, 'mask')
lbl_dir = os.path.join(results_dir, 'label')
mkdir(msk_dir)
mkdir(lbl_dir)

thetas = get_iter_vals(-1, 3, precision=0.05)
scales = get_iter_vals(1.1, 1.4, precision=0.05)
xs = get_iter_vals(-45, 21, precision=3)  # 160
ys = get_iter_vals(-30, 30, precision=3)
# xs = get_iter_vals(-70, 30, precision=5)  # 256
# ys = get_iter_vals(-50, 50, precision=5)
# xs = get_iter_vals(-56, 24, precision=4)  # 200
# ys = get_iter_vals(-40, 40, precision=4)
# xs = get_iter_vals(-60, 60, precision=3)  # 160
# ys = get_iter_vals(-60, 60, precision=3)
# xs = get_iter_vals(-30, 14, precision=2)  # 108
# ys = get_iter_vals(-20, 20, precision=2)
# xs = get_iter_vals(-15, 14, precision=1)  # 108
# ys = get_iter_vals(-10, 10, precision=1)

# dsize = (256, 256)
# dsize = (200, 200)
dsize = (160, 160)  # 尺寸再小效果就差了


# todo: 先粗范围得到一个初始解，对于处在边界位置的 tx, ty 再扩大搜索范围
# 标签放大，逐类做下内部膨胀，噪点消除和边缘平滑
# 针对 xs/ys 边缘，分成 左边界 / 右边界 分别扩展处理

# @nb.jit(nopython=True)
def bf_match(sat_msk, lbl_msk, txs=xs, tys=ys, last_aff=None):
    """暴力匹配 sat/lbl road msk
    两层优化
    1. 深层 ty 及时跳出，减少无效迭代，最内层的减少能影响所有外层 for 执行，速度 3:45 -> 40s
        有的情况无法顾及，[失败]
    2. dsize 不影响配准 scale (只有 crop pad 影响)，但影响 for 最内层 cv2.warpAffine, 256 -> 160, 速度 40s -> 20s
        暴力法 如何判断什么时候取到了最优?
            设置很长时间不更新，返回也不行
    """
    best_aff = {
        'ratio': 0,
        'theta': 0,
        'scale': 0,
        'tx': 0,
        'ty': 0,
        'M': None,
        # 'road_msk': None
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
                            # 'road_msk': aff_msk
                        }
    return best_aff


def plt_cmp_msk(sat_img, sat_road, aff_road, aff_lbl, title):
    f, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs.flat[0].imshow(sat_img)
    axs.flat[0].set_title(title)
    axs.flat[1].imshow(sat_road, cmap='gray')
    axs.flat[1].set_title('sat_road')
    axs.flat[2].imshow(color_code_target(aff_lbl, label_colors))
    axs.flat[2].set_title('aff_label')
    axs.flat[3].imshow(aff_road, cmap='gray')
    axs.flat[3].set_title('aff_road')
    plt.show()


def plt_two_msk(sat_road_msk, lbl_road_msk):
    f, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs.flat[0].imshow(sat_road_msk, cmap='gray')
    axs.flat[0].set_title('sat_road_msk')
    axs.flat[1].imshow(lbl_road_msk, cmap='gray')
    axs.flat[1].set_title('lbl_road_msk')
    plt.show()


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


def rematch_corner_cases():
    """
    针对匹配处于边界范围的结果，进行再匹配
    """
    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')

    affine_stats = load_pickle(f'{match_dir}/road_affine_stats_num1735.pkl')
    newaff_stats = affine_stats.copy()

    # 全局 tx/ty 边界，首次判断 aff 是否合理
    global_tx_left, global_tx_right = -45, 21
    global_ty_left, global_ty_right = -30, 30

    before_ratios = []
    after_ratios = []

    update_cnt = 0

    for tile, aff in affine_stats.items():

        if global_tx_left < aff['tx'] < global_tx_right and global_ty_left < aff['ty'] < global_ty_right:  # 合法即跳过
            continue

        print()
        print(tile)

        # 添加 old ratio
        before_ratios.append(aff['ratio'])

        # 读图
        # sat_road
        sat_road_msk = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        sat_road_msk = sat_road_msk[:, :, -1]  # 取末尾 alpha 通道
        sat_road_msk[sat_road_msk > 0] = 255  # 二值化
        sat_road_msk = cv2.resize(sat_road_msk, dsize, interpolation=cv2.INTER_NEAREST)

        # lbl_road
        tile_pts = tile_mapping[tile]
        tile_msk = crop_mask_from_gmap(tile_pts)
        tile_msk = cv2.resize(tile_msk, dsize, interpolation=cv2.INTER_NEAREST)
        lbl_road_msk = np.zeros_like(tile_msk, dtype='uint8')
        lbl_road_msk[tile_msk == 20] = 255

        # 循环找到新的匹配
        # 赋值，方便内层 while 迭代更新 tx/ty 边界 进行搜索
        tx_left, tx_right = global_tx_left, global_tx_right
        ty_left, ty_right = global_ty_left, global_ty_right
        while True:
            tx, ty = aff['tx'], aff['ty']

            # 重新分配搜索区间
            txs = get_tx_range(tx, tx_left, tx_right)
            tys = get_tx_range(ty, ty_left, ty_right)
            tx_left, tx_right = txs[0], txs[-1]
            ty_left, ty_right = tys[0], tys[-1]

            # match
            last_ratio = aff['ratio']
            aff = bf_match(sat_road_msk, lbl_road_msk, txs, tys, last_aff=aff)

            # 合法区间内找到，并且由于传入 last_ratio 必然发生了更新 无需再扩展区间
            if tx_left < aff['tx'] < ty_right and ty_left < aff['ty'] < ty_right:
                break
            else:  # 不在合法区间
                if aff['ratio'] == last_ratio:  # 如果无更新，说明还是在原来的边界处取到
                    break
                # 如果有更新，如果在新的边界处取到，继续 while 循环

        # todo: aff mask
        after_ratios.append(aff['ratio'])
        if after_ratios[-1] > before_ratios[-1]:  # 发生了更新
            update_cnt += 1
            print('============> update:', update_cnt)
            print('before ratio:', before_ratios[-1])
            print('after ratio:', aff['ratio'])
            pprint(aff)

            # warp label
            aff_lbl_msk = cv2.warpAffine(tile_msk, aff['M'], dsize, flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
            newaff_stats[tile] = aff

            # resize label
            aff_lbl_msk = cv2.resize(aff_lbl_msk, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f'{lbl_dir}/{tile}.png', color_code_target(aff_lbl_msk, label_colors)[:, :, ::-1])
            np.save(f'{msk_dir}/{tile}.npy', aff_lbl_msk)

    np.save(f'{match_dir}/before_ratios.npy', before_ratios)
    np.save(f'{match_dir}/after_ratios.npy', after_ratios)

    dump_pickle(newaff_stats, f'/datasets/rs_segment/AerialCitys/HZ20/matches/road_affine_stats_after.pkl')


def match_sat_lbl_road():
    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')

    affine_stats = {}

    tiles = sorted(tile_mapping.keys())

    upper = 1755

    # idxs = range(0, upper)  # 208 - 40s
    # idxs = range(600, upper) # 209 - 50s
    idxs = range(1200, upper)  # 110 - 30s
    print('total:', len(idxs))

    for idx in idxs:
        tile = tiles[idx]
        # sat img
        # sat_img = cv2.imread(os.path.join(sat_dir, f'{tile}.jpeg'), cv2.IMREAD_UNCHANGED)
        # sat_img = cv2.resize(sat_img, dsize)

        # sat_road
        sat_road_msk = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        if len(sat_road_msk.shape) < 3:  # 13046-3446 直接 (256,256) 值全 = 0，baidu 存在没有 road 标注的图片
            continue

        print(f'{idx}/{upper}', tile)
        sat_road_msk = sat_road_msk[:, :, -1]  # 取末尾 alpha 通道
        sat_road_msk[sat_road_msk > 0] = 255  # 二值化
        sat_road_msk = cv2.resize(sat_road_msk, dsize, interpolation=cv2.INTER_NEAREST)

        # lbl_road
        tile_pts = tile_mapping[tile]
        tile_msk = crop_mask_from_gmap(tile_pts)
        tile_msk = cv2.resize(tile_msk, dsize, interpolation=cv2.INTER_NEAREST)
        lbl_road_msk = np.zeros_like(tile_msk, dtype='uint8')
        lbl_road_msk[tile_msk == 20] = 255

        # plt_two_msk(sat_road_msk, lbl_road_msk)

        # match
        aff = bf_match(sat_road_msk, lbl_road_msk)
        # aff_road_msk = aff.pop('road_msk')
        if aff['M'] is None:
            continue

        # todo: 边界 research
        # 情况判断：有的确实是在边界发现最优的

        aff_lbl_msk = cv2.warpAffine(tile_msk, aff['M'], dsize, flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
        affine_stats[tile] = aff
        pprint(aff)

        aff_lbl_msk = cv2.resize(aff_lbl_msk, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite(f'{save_dir}/{tile}_road.png', aff_road_msk)
        cv2.imwrite(f'{lbl_dir}/{tile}.png', color_code_target(aff_lbl_msk, label_colors)[:, :, ::-1])
        np.save(f'{msk_dir}/{tile}.npy', aff_lbl_msk)
        # plt_cmp_msk(sat_img, sat_road_msk, aff_road_msk, aff_lbl_msk, title=tile)

    dump_pickle(affine_stats, f'/datasets/rs_segment/AerialCitys/HZ20/matches/road_affine_stats_{upper}.pkl')


if __name__ == '__main__':
    # match_sat_lbl_road()
    rematch_corner_cases()
