"""
filter_tiles
    tiles_mapping.json: 利用最近邻仿射矩阵，将 tile 映射到的 标注图片区域
    1. 求出外接矩形
    2. 满足 顶点合理 & 标注区域占比 > thre 保存数据
"""
from utils.misc import *
from hz.GridMap import GridMap
import cv2
from constants import *

label_names, label_colors = get_label_name_colors('hz/hz20.csv')
map_h, map_w = 10532, 11153


def load_gmap():
    gmap = GridMap('hz/index.txt')
    gmap.preprocess()  # map bg=0
    return gmap.data


def filter_tiles(thre=0.7):
    """
    tiles_mapping.json: 利用最近邻仿射矩阵，将 tile 映射到的区域
        1. 求出外接矩形
        2. 满足 顶点合理 & 标注区域占比 > thre 保存数据
    """
    gmap = load_gmap()

    res = {}
    cnt = 0

    tile_mapping = load_json(f'{match_dir}/tiles_mapping.json')  # 3840

    for tile, tile_pts in tile_mapping.items():
        pts = np.array(tile_pts)
        x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
        y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())

        # 顶点合理
        if x_min >= 0 and x_max < map_w and y_min >= 0 and y_max < map_h:
            tile_mask = gmap[y_min:y_max + 1, x_min:x_max + 1]

            # 标注区占比 >= thre
            if (tile_mask > 0).sum() / tile_mask.size >= thre:
                res[tile] = tile_pts
                cnt += 1
                print(cnt)

    dump_json(res, f'{match_dir}/tiles_mapping_{thre}_num{cnt}.json')


def pts_hw2xy(pts):  # match 匹配结果为 h,w -> 转成 x,y 计算 affine matrix
    return np.array(pts)[:, ::-1]


matchs = {
    'left3': [[[118, 144], [142, 18], [48, 152]],
              [[88, 136], [108, 48], [56, 152]]],
    'all': [[[118, 144], [142, 18], [48, 152], [104, 34], [86, 96]],
            [[88, 136], [108, 48], [56, 152], [84, 58], [66, 104]]]
}


def crop_mask_from_gmap(gmap, tile_pts, pad=30):
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


def affine_demo():
    gmap = load_gmap()

    match_pts = {  # h,w; 红, 绿, 蓝, 棕, 青
        '13039-3433': [[[118, 144], [142, 18], [48, 152], [104, 34], [86, 96]],  # sat
                       [[88, 136], [108, 48], [56, 152], [84, 58], [66, 104]]],  # lbl
        # '13039-3432': [[[108, 152], [134, 60], [90, 44], [134, 62]],
        #                [[76, 152], [102, 76], [66, 58], [102, 78]]]
    }

    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    dsize = (160, 160)  # 当时取小尺寸，为了加快 nbb 计算

    for tile, tile_matchs in match_pts.items():
        tile_pts = tile_mapping[tile]

        # get mask
        tile_mask = crop_mask_from_gmap(gmap, tile_pts)
        tile_img = cv2.imread(f'{sat_dir}/{tile}.jpeg')[:, :, ::-1]

        # 基于 160*160 得到的对应点
        sat_pts, lbl_pts = tile_matchs
        if len(sat_pts) < 3:  # 不足够计算仿射
            continue

        # from lbl to sat
        # 能从一组 map pts 选择一组能使 全体映射误差最小的三个点
        M = cv2.estimateAffinePartial2D(pts_hw2xy(lbl_pts), pts_hw2xy(sat_pts), confidence=0.95)[0]
        angle = np.rad2deg(np.arctan(M[1][0] / M[1][1]))
        print(M)
        print(angle)
        # M = cv2.estimateAffine2D(pts_hw2xy(lbl_pts), pts_hw2xy(sat_pts))[0]
        # resize ori mask to 160
        tile_mask = cv2.resize(tile_mask, dsize, interpolation=cv2.INTER_NEAREST)
        # 设置仿射变换插值类型为 INTER_NEAREST, borderValue 为 bg 0
        aff_mask = cv2.warpAffine(tile_mask, M, dsize=dsize, flags=cv2.INTER_NEAREST, borderValue=0)

        tile_mask = cv2.resize(tile_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        aff_mask = cv2.resize(aff_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        plt_img_target(tile_img, tile_mask, label_colors, title=tile)
        plt_img_target(tile_img, aff_mask, label_colors, title=tile)


if __name__ == '__main__':
    # filter_tiles(thre=0.9)
    affine_demo()

    pass
