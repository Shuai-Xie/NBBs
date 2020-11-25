from constants import *
from utils.misc import *
from pprint import pprint
import shutil
from tqdm import tqdm


def split_affine_attrs():
    # 4 种 dict 合并方法 https://blog.csdn.net/Jerry_1126/article/details/73017270
    affine_stats = load_pickle(f'{match_dir}/road_affine_stats_num1735_after.pkl')

    ratios = []
    thetas = []
    scales = []
    txs = []
    tys = []

    for tile, aff in affine_stats.items():
        ratios.append(aff['ratio'])
        thetas.append(aff['theta'])
        scales.append(aff['scale'])
        txs.append(aff['tx'])
        tys.append(aff['ty'])

    np.save(f'{match_dir}/affine_ratios.npy', ratios)
    np.save(f'{match_dir}/affine_thetas.npy', thetas)
    np.save(f'{match_dir}/affine_scales.npy', scales)
    np.save(f'{match_dir}/affine_txs.npy', txs)
    np.save(f'{match_dir}/affine_tys.npy', tys)


def vis_hist(bins=10):
    affine_ratios = np.load(f'{match_dir}/affine_ratios.npy')
    ys, xs = np.histogram(affine_ratios, bins=bins)

    plt.barh(range(bins), ys, align='center')  # horizontal bar
    plt.xlim([0, 600])
    plt.yticks(range(bins), labels=xs[:-1])
    ax = plt.gca()
    ax.set_xlabel('match_num')
    # https://matplotlib.org/3.3.2/api/text_api.html#matplotlib.text.Text
    ax.set_ylabel('match_ratio', rotation='horizontal')
    ax.yaxis.set_label_coords(-0.2, 0.97)  # 详细设置 label 位置

    for idx, y in enumerate(ys):
        plt.text(y + 5, idx, s=str(y))

    plt.show()


def vis_update_ratio():
    import seaborn as sns
    sns.set()
    sns.set_style('whitegrid')  # darkgrid

    before_ratios = np.load(f'{match_dir}/before_ratios.npy')
    after_ratios = np.load(f'{match_dir}/after_ratios.npy')
    total_num = len(before_ratios)

    update_msk = before_ratios != after_ratios
    before_ratios = before_ratios[update_msk]
    after_ratios = after_ratios[update_msk]

    num = len(before_ratios)  # update num
    mean_update = (after_ratios - before_ratios).sum() / num
    print(f'update: {num}/{total_num}')
    print(f'mean increase: {mean_update}')

    plt.figure(figsize=(12, 3))
    plt.scatter(range(num), before_ratios, s=3, label='before')
    plt.scatter(range(num), after_ratios, s=3, label='after')
    plt.legend()

    plt.show()


def vis_bad_case():
    affine_stats = load_pickle(f'{match_dir}/road_affine_stats_num1735_after.pkl')

    upper = 1.0

    bad_case_dir = f'{match_dir}/bad_case/bad_case_{upper}'
    mkdir(bad_case_dir)

    cnt = 0
    for tile, aff in affine_stats.items():
        if upper - 0.1 < aff['ratio'] <= upper:
            cnt += 1
            print(tile)
            # sat
            shutil.copy(
                src=f'/datasets/rs_segment/AerialCitys/HZ20/train/image_z16/{tile}.jpeg',
                dst=f'{bad_case_dir}/{tile}.jpeg'
            )
            # label
            shutil.copy(
                src=f'/datasets/rs_segment/AerialCitys/HZ20/matches/match_road/label/{tile}.png',
                dst=f'{bad_case_dir}/{tile}.png'
            )
    print(cnt)


total_num = 149 + 211 + 401 + 558 + 316 + 100
print(total_num)
