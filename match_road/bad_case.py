"""
bad case 特征
- 明显场景不对应; 当 sat 中基本不存在大面积道路时容易发生
- label 图片误标注
- 大量简单类，如全是绿色森林 or 水域
- 道路 fg 不一致，只追求 fg 占比最大的结果
"""
import os
from utils.misc import *
from pprint import pprint

bad_img_dir = '/datasets/rs_segment/AerialCitys/HZ20/matches/bad_case'
bad_txt_dir = 'match_road/bad_cases'

res = {}

global_total_num = 0
global_bad_num = 0

for sub_dir in os.listdir(bad_img_dir):
    total_num = sum([1 for p in os.listdir(os.path.join(bad_img_dir, sub_dir)) if p.endswith('.png')])
    bad_num = len(read_txt_as_list(os.path.join(bad_txt_dir, sub_dir + '.txt')))

    res[sub_dir] = {
        'total': total_num,
        'bad': bad_num,
        'keep_ratio': 1 - bad_num / total_num
    }

    global_total_num += total_num
    global_bad_num += bad_num

res['global'] = {
    'total': global_total_num,
    'bad': global_bad_num,
    'keep': global_total_num - global_bad_num,
    'keep_ratio': 1 - global_bad_num / global_total_num
}

pprint(res)
