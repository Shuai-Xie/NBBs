"""
不能保证每张 road tile 中都包括 道路，可能为空?
"""
from constants import *
from utils.misc import *
import cv2
import matplotlib.pyplot as plt
from hz.GridMap import load_gmap

gmap = load_gmap()
map_h, map_w = 10532, 11153
label_names, label_colors = get_label_name_colors('hz/hz20.csv')

results_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/match_road/results'
mkdir(results_dir)


def crop_mask_from_gmap(tile_pts, pad):  # pad 太大会引入一些外部区域，造成匹配干扰?
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


sift = cv2.SIFT_create()


def keypts_matching(img1, img2):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.6 * n.distance]
    print(len(good))

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    plt.imshow(img3)
    plt.show()


def match_sat_target_road():
    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    dsize = (256, 256)

    for idx, (tile, tile_pts) in enumerate(tile_mapping.items()):
        print(idx, tile)

        # sat_road
        road_img = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        road_img = road_img[:, :, -1]  # 取末尾 alpha 通道
        road_img[road_img > 0] = 255  # 二值化

        # target_road
        tile_msk = crop_mask_from_gmap(tile_pts, pad=20)
        road_msk = np.zeros_like(tile_msk, dtype='uint8')
        road_msk[tile_msk == 20] = 255
        road_msk = cv2.resize(road_msk, dsize)  # bilinear, 0-255 都有了
        road_msk[road_msk > 0] = 255  # 再次二值化

        cv2.imwrite(f'match_road/examples/{tile}_img.png', road_img)
        cv2.imwrite(f'match_road/examples/{tile}_msk.png', road_msk)

        if idx == 20:
            exit(0)

        # keypts_matching(road_img, road_msk)


if __name__ == '__main__':
    match_sat_target_road()


def vis_sat_target_road():
    tile_mapping = load_json(f'{match_dir}/tiles_mapping_1.0_num1755.json')
    dsize = (256, 256)

    for idx, (tile, tile_pts) in enumerate(tile_mapping.items()):
        print(idx, tile)
        f, axs = plt.subplots(2, 3, figsize=(9, 6))

        # sat - road - binary
        sat_img = cv2.imread(os.path.join(sat_dir, f'{tile}.jpeg'), cv2.IMREAD_UNCHANGED)
        axs.flat[0].imshow(sat_img)
        axs.flat[0].set_title(tile)

        road_img = cv2.imread(os.path.join(baidu_dir, f"{tile.replace('-', '_')}_road.png"), cv2.IMREAD_UNCHANGED)
        axs.flat[1].imshow(road_img)
        axs.flat[1].set_title('sat_road')

        road_img = road_img[:, :, -1]  # 取末尾 alpha 通道
        road_img[road_img > 0] = 255  # 二值化

        axs.flat[2].imshow(road_img, cmap='gray')
        axs.flat[2].set_title('sat_road_binary')

        # target - road - bianry
        tile_msk = crop_mask_from_gmap(tile_pts)
        tile_msk = cv2.resize(tile_msk, dsize, interpolation=cv2.INTER_NEAREST)
        road_msk = np.zeros_like(tile_msk)
        road_msk[tile_msk == 20] = 20

        axs.flat[3].imshow(color_code_target(tile_msk, label_colors))
        axs.flat[3].set_title('target')

        axs.flat[4].imshow(color_code_target(road_msk, label_colors))
        axs.flat[4].set_title('target_road')

        road_msk[road_msk == 20] = 255
        axs.flat[5].imshow(road_msk, cmap='gray')
        axs.flat[5].set_title('target_road_binary')

        plt.savefig(f'match_road/cmp_road/{tile}.png', bbox_inches='tight', pad_inches=0.)

        plt.show()
        plt.close()
