"""
迭代 仿射矩阵; # 认为是相似变换，少2个参数
    旋转
    放缩
    平移
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from pprint import pprint

img_dir = 'match_road/examples'

# examples = [
#     '13039-3431',
#     '13039-3433',
#     '13040-3431',
#     # '13040-3432'
# ]
examples = [p[:-8] for p in os.listdir(img_dir) if p.endswith('_img.png')]


def create_fake_msk():
    eg = '13039-3433'
    msk = cv2.imread(f'{img_dir}/{eg}_msk.png', cv2.IMREAD_UNCHANGED)

    th_deg = 1.05
    th = np.pi * th_deg / 180
    s = 1.2
    tx, ty = -30, -20

    R = s * np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
    t = np.array([[tx],
                  [ty]])
    M = np.hstack((R, t))

    aff_msk = cv2.warpAffine(msk, M, dsize=(256, 256), flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
    cv2.imwrite(f'{img_dir}/{eg}_msk_fake.png', aff_msk)

    plt_cmp_msk(msk, msk, aff_msk, title=eg)


# 匹配上的前景占比
def matched_fg_ratio(base_msk, aff_msk):
    fg_msk = aff_msk > 0
    fg_sum = fg_msk.sum()
    if fg_sum == 0:
        return 0
    else:
        match_fg = fg_msk & (base_msk == aff_msk)
        return match_fg.sum() / fg_sum


def plt_cmp_msk(base_msk, ori_msk, aff_msk, title):
    f, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs.flat[0].imshow(ori_msk, cmap='gray')
    axs.flat[0].set_title('ori')
    axs.flat[1].imshow(base_msk, cmap='gray')
    axs.flat[1].set_title(title)
    axs.flat[2].imshow(aff_msk, cmap='gray')
    axs.flat[2].set_title('affine')
    plt.show()


def get_iter_vals(min_, max_, precision):
    num = int((max_ - min_) / precision + 1)
    return np.linspace(min_, max_, num)


if __name__ == '__main__':
    thetas = get_iter_vals(-1, 2, precision=0.05)
    scales = get_iter_vals(1.2, 1.4, precision=0.05)
    xs = get_iter_vals(-70, -40, precision=5)
    ys = get_iter_vals(-10, 10, precision=5)

    for eg in examples:
        print(eg)
        img = cv2.imread(f'{img_dir}/{eg}_img.png', cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(f'{img_dir}/{eg}_msk.png', cv2.IMREAD_UNCHANGED)

        # 300s, 崩
        t1 = time.time()

        best_aff = {
            'ratio': 0,
            'theta': 0,
            'scale': 0,
            'M': None,
            'tx': 0,
            'ty': 0,
        }
        best_msk = None

        for th_deg in tqdm(thetas):
            th = np.pi * th_deg / 180
            for s in scales:
                for tx in xs:
                    for ty in ys:
                        R = s * np.array([[np.cos(th), -np.sin(th)],
                                          [np.sin(th), np.cos(th)]])
                        t = np.array([[tx],
                                      [ty]])
                        M = np.hstack((R, t))

                        aff_msk = cv2.warpAffine(msk, M, dsize=(256, 256), flags=cv2.INTER_NEAREST, borderValue=0).astype('uint8')
                        ratio = matched_fg_ratio(img, aff_msk)

                        if ratio > best_aff['ratio']:
                            best_aff = {
                                'ratio': ratio,
                                'theta': th_deg,
                                'scale': s,
                                'M': M,
                                'tx': tx,
                                'ty': ty,
                            }
                            best_msk = aff_msk

        t2 = time.time()
        print(t2 - t1)
        pprint(best_aff)

        plt_cmp_msk(base_msk=img, ori_msk=msk, aff_msk=best_msk, title=eg)
