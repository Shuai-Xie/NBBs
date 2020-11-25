# 查看 image / mask / label 文件夹是否对应
import os

root = '/datasets/RS_Dataset/HZ20'

img_dir = '/datasets/RS_Dataset/HZ20/train/image'
mask_dir = '/datasets/RS_Dataset/HZ20/train/mask'
lbl_dir = '/datasets/RS_Dataset/HZ20/train/label'


def get_sorted_filename(file_dir):
    return sorted([p.split('.')[0] for p in os.listdir(file_dir) if p != '@eaDir'])


img_names = get_sorted_filename(img_dir)
lbl_names = get_sorted_filename(lbl_dir)
mask_names = get_sorted_filename(mask_dir)


def check_img_lbl():
    # img 应当删 13093-3434, 结果错删了 13094-3434
    print(set(img_names) - set(lbl_names))  # {'13093-3434'}
    print(set(lbl_names) - set(img_names))  # {'13094-3434'} 删错了


def filter_bad_masks():
    bad_masks = set(mask_names) - set(img_names)
    for msk in bad_masks:
        os.remove(f'{mask_dir}/{msk}.npy')


def consistent_img_lbl_mask():
    union = set(img_names) | set(lbl_names) | set(mask_names)
    inter = set(img_names) & set(lbl_names) & set(mask_names)
    print(union - inter)


def filter_bad_case():
    chose_imgs = get_sorted_filename(img_dir)
    all_imgs = os.listdir(f'{root}/matches/results')

    print(len(all_imgs))
    print(len(chose_imgs))

    bad_case = set(all_imgs) - set(chose_imgs)
    print(len(bad_case))
    print(bad_case)


filter_bad_case()
