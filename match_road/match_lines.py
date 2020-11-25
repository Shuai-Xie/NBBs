"""
直线检测
不管是否 在原图中存在 直线的交点，只要检测出方向
即可计算 两个坐标系 各自情况下的交点，就能计算仿射矩阵
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

img_dir = 'match_road/examples'

examples = [
    '13039-3431',
    '13039-3433',
    '13040-3431',
    '13040-3432'
]


def vis_line(img):
    # return (n,1,4)
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = cv2.Canny(img, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLinesP(
        img,
        rho=1,  # 线段以像素为单位的距离精度
        theta=np.pi / 180,  # 线段以弧度为单位的角度精度
        threshold=100,  # 线段长度阈值，越大检测出的线越少
        minLineLength=256 * 0.3,  # 线段以像素为单位的最小长度
        maxLineGap=0.5,  # 越小，重复越少
    )
    if lines is None:
        return img
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    lines = lines.squeeze(1)
    print(lines.shape)
    for x1, y1, x2, y2 in lines:
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img


def plt_cmp_msk(sat_msk, lbl_msk, title):
    f, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs.flat[0].imshow(sat_msk, cmap='gray')
    axs.flat[0].set_title(title)
    axs.flat[1].imshow(lbl_msk, cmap='gray')
    axs.flat[1].set_title('label')

    plt.savefig(f'match_road/lines/{title}.png', bbox_inches='tight', pad_inches=0.)
    plt.show()


for eg in examples:
    img = cv2.imread(f'{img_dir}/{eg}_img.png', cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(f'{img_dir}/{eg}_msk.png', cv2.IMREAD_UNCHANGED)

    img = vis_line(img)
    msk = vis_line(msk)

    plt_cmp_msk(sat_msk=img, lbl_msk=msk, title=eg)
