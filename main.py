import os
import numpy as np
from models import vgg19_model
from algorithms import neural_best_buddies as NBBs
from utils import util
from utils import MLS
import time
from options import Options

opt = Options().parse()
save_dir = os.path.join(opt.results_dir, opt.name)

vgg19 = vgg19_model.define_Vgg19(opt)

nbbs = NBBs.sparse_semantic_correspondence(
    vgg19,  # model
    opt.gpu_ids, opt.tau, opt.border_size, save_dir,
    opt.k_per_level,
    opt.k_final, opt.fast
)

# read tensor img [1,3,224,224]

t1 = time.time()
A = util.read_image(opt.datarootA, opt.imageSize)
B = util.read_image(opt.datarootB, opt.imageSize)
points = nbbs.run(A, B)
t2 = time.time()
print('time:', t2 - t1)
# L1: 186s
# L2: 53s
# L2_hook: 52s
# L2_nolearn: 65s, warp 过程比 model learn 更耗时
# L2_160: 30s

# 读取特征点，完成图像 warp
mls = MLS.MLS(v_class=np.int32)
mls.run_MLS_in_folder(root_folder=save_dir)
