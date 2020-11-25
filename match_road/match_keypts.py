"""
不能保证每张 road tile 中都包括 道路，可能为空?
"""
from constants import *
from utils.misc import *
from utils import util
import cv2
import matplotlib.pyplot as plt
from hz.GridMap import load_gmap

from algorithms import neural_best_buddies as NBBs
from options import Options
from models import vgg19_model

# nbb
args = [
    '--imageSize', '256',
    '--k_final', '10',  # 最后1层 k 数量; 计算 AM, choose top3 from 5; 或者直接聚成 3 簇
    '--k_per_level', '15',  # 调大，可预防 k_final cluster 数量不够计算 affine matrix; 15-7, 10-4
    '--fast'
]
opt = Options().parse(args)
opt.results_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/match_road/results'

# map
gmap = load_gmap()
map_h, map_w = 10532, 11153
label_names, label_colors = get_label_name_colors('hz/hz20.csv')


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


def crop_mask_from_gmap(tile_pts, pad=0):  # pad 太大会引入一些外部区域，造成匹配干扰?
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


def match_sat_target_road():
    nbbs = get_nbbs()
    transform = util.get_transform(opt.imageSize)

    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    dsize = (256, 256)

    for idx, (tile, tile_pts) in enumerate(tile_mapping.items()):
        print()
        print(idx, tile)
        save_dir = os.path.join(opt.results_dir, tile)

        # sat_road
        road_img = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        road_img = road_img[:, :, -1]  # 取末尾 alpha 通道
        road_img[road_img > 0] = 255  # 二值化

        # target_road
        tile_msk = crop_mask_from_gmap(tile_pts)
        road_msk = np.zeros_like(tile_msk, dtype='uint8')
        road_msk[tile_msk == 20] = 255
        road_msk = cv2.resize(road_msk, dsize)  # bilinear, 0-255 都有了
        road_msk[road_msk > 0] = 255  # 再次二值化

        road_img = cv2.cvtColor(road_img, cv2.COLOR_GRAY2RGB)
        road_msk = cv2.cvtColor(road_msk, cv2.COLOR_GRAY2RGB)

        A = transform(road_img).unsqueeze(0)  # 256
        B = transform(road_msk).unsqueeze(0)

        # 保存中间结果
        nbbs.save_dir = save_dir
        top_k_correspondence = nbbs.run(A, B)  # pts_A, pts_B
        print(top_k_correspondence)

        if len(top_k_correspondence[0]) < 3:  # 不足够
            print('Stop, not enought pts')
            continue


if __name__ == '__main__':
    match_sat_target_road()
