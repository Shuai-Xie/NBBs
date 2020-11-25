import os
import math
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from . import feature_metric as FM
from utils import draw_correspondence as draw
from utils import util
from pprint import pprint
import time


class sparse_semantic_correspondence:
    """稀疏 语义 一致性"""

    def __init__(self, model, gpu_ids, tau, border_size, save_dir, k_per_level, k_final, fast):
        """
        :param model: vgg19, forward 设置 start/end level 得到中间层特征
        :param gpu_ids:
        :param tau: 0.05, 经验值，选择 较优的 NBBs
        :param border_size:
        :param save_dir: 中间层 一致性点 保存 dir
        :param k_per_level: 每层最多搜索的 pts 数量
        :param k_final:     每层最多保留的 pts 数量
        :param fast: True 保存到 level 2 截至, 1/2 feature map
        """
        self.Tensor = torch.cuda.FloatTensor if gpu_ids else torch.Tensor
        self.model = model
        self.tau = tau
        self.border_size = border_size  # 7
        self.save_dir = save_dir
        self.k_per_level = k_per_level
        self.k_final = k_final
        self.L_final = 2 if fast else 1  # fast 场景到 level2

        # define 5 levels
        self.patch_size_list = [[5, 5], [5, 5], [3, 3], [3, 3], [3, 3]]  # 2*radius -1
        self.search_box_radius_list = [3, 3, 2, 2, 2]
        self.draw_radius = [2, 2, 2, 4, 8]
        self.pad_mode = 'reflect'

    def find_mapping(self, A, B, patch_size, initial_mapping, search_box_radius):
        """
        :param A: F_Am_normalized 归一化特征单位向量
        :param B: F_Bm_normalized
        :param patch_size: [3, 3]
        :param initial_mapping: initial_map_a_to_b, identity_map, 坐标位 [i][j]
        :param search_box_radius: 2
        :return:
        """
        A_to_B_map = self.Tensor(1, 2, A.size(2), A.size(3))

        dx, dy = math.floor(patch_size[0] / 2), math.floor(patch_size[1] / 2)
        pad_size = [dy, dy, dx, dx]

        # pad img, 考虑边缘
        A_padded = F.pad(A, pad_size, self.pad_mode)
        B_padded = F.pad(B, pad_size, self.pad_mode)

        # O(H*W * H*W)
        for i in range(A.size(2)):  # H
            for j in range(A.size(3)):  # W
                # patch A
                candidate_patch_A = A_padded[:, :, i:(i + 2 * dx + 1), j:(j + 2 * dy + 1)]
                index = self.find_closest_patch_index(  # todo: 并行化
                    B_padded,
                    candidate_patch_A,
                    initial_mapping[0, :, i, j],  # (i,j)
                    search_box_radius
                )
                # A (i,j) 对应于 B (index)
                A_to_B_map[:, :, i, j] = self.Tensor([index[0] - dx, index[1] - dy])

        return A_to_B_map

    def find_closest_patch_index(self, B, patch_A, inital_pixel, search_box_radius):
        """
        :param B: B_padded feature
        :param patch_A: patch features C*3*3
        :param inital_pixel:  initial_mapping[0, :, i, j] -> 初始 A 的 i,j; 寻找对应 B 上的区域
        :param search_box_radius: 2
        :return:
        """
        dx, dy = math.floor(patch_A.size(2) / 2), math.floor(patch_A.size(3) / 2)
        search_dx = search_dy = search_box_radius  # 搜索 radius, 点的半径 所在区域

        # 初始 A(i,j) 对应的 B 区域
        up_boundary = int(inital_pixel[0] - search_dx) if inital_pixel[0] - search_dx > 0 else 0
        down_boundary = int(inital_pixel[0] + 2 * dx + search_dx + 1) if inital_pixel[0] + 2 * dx + search_dx + 1 < B.size(2) else B.size(2)
        left_boundary = int(inital_pixel[1] - search_dy) if inital_pixel[1] - search_dy > 0 else 0
        right_boundary = int(inital_pixel[1] + 2 * dy + search_dy + 1) if inital_pixel[1] + 2 * dy + search_dy + 1 < B.size(3) else B.size(3)

        # pad B 1*512*3*3 -> 1*512*5*5; pad 后 对每个位置 做 conv
        search_box_B = B[:, :, up_boundary:down_boundary, left_boundary:right_boundary]

        # patch_A 1*512*3*3 作为卷积核 weight
        # 将 search_box_B 从 512 -> 1 维
        result_B = F.conv2d(search_box_B, patch_A.contiguous()).data
        # print(patch_A.shape)  # [1, 512, 3, 3]
        # print(search_box_B.shape)  # [1, 512, 5, 5]
        # print(result_B.shape)  # [1, 1, 3, 3]

        # cos distance
        distance = result_B
        # todo: 可能存在多个 max，返回一个即可
        _, _, max_i, max_j = torch.where(distance == distance.max())
        if len(max_i) > 1:
            max_i, max_j = max_i[0], max_j[0]
        # +dx, +dy 转换到 B patch 坐标系
        # +up_boundary, +left_boundary 转换到 B 全图坐标系
        closest_patch_index = [max_i + dx + up_boundary, max_j + dy + left_boundary]

        return closest_patch_index

    def warp(self, A_size, B, patch_size, mapping_a_to_b):  # mapping_a_to_b
        """
        根据 mapping_a_to_b 将 A 对应 B 的区域 映射到 A 自身的位置
        """
        assert (B.size() == A_size)
        [dx, dy] = [math.floor(patch_size[0] / 2), math.floor(patch_size[1] / 2)]
        B_padded = F.pad(B, [dy, dy, dx, dx], self.pad_mode)

        # 保存从 B 映射到的 A 中每个 patch 对应的 B feature
        warped_A = self.Tensor(B_padded.size()).fill_(0.0)

        counter = self.Tensor(B_padded.size()).fill_(0.0)
        patch_cnt = self.Tensor(B_padded.size(0), B_padded.size(1), patch_size[0], patch_size[1]).fill_(1.0)

        for i in range(A_size[2]):
            for j in range(A_size[3]):
                # A region 对应的 B region 添加到 A 中 (i,j) 位置
                ab_i, ab_j = map(int, mapping_a_to_b[0, :, i, j])
                # 因为 stride=1, 而 patch_size>1, patch 内部分 pixel 多次加 feature，所以用 counter 平均特征
                warped_A[:, :, i:(i + 2 * dx + 1), j:(j + 2 * dy + 1)] += \
                    B_padded[:, :, ab_i:ab_i + 2 * dx + 1, ab_j:ab_j + 2 * dy + 1]
                counter[:, :, i:(i + 2 * dx + 1), j:(j + 2 * dy + 1)] += patch_cnt

        warped_A = warped_A[:, :, dx:(warped_A.size(2) - dx), dy:(warped_A.size(3) - dy)] / \
                   counter[:, :, dx:(warped_A.size(2) - dx), dy:(warped_A.size(3) - dy)]
        return warped_A

    def mapping_to_image_size(self, mapping, level, original_image_size):
        if level == 1:
            return mapping
        else:
            identity_map_L = self.identity_map(mapping.size())
            identity_map_original = self.identity_map(original_image_size)
            factor = int(math.pow(2, level - 1))
            return identity_map_original + self.upsample_mapping(mapping - identity_map_L, factor=factor)

    def upsample_mapping(self, mapping, factor=2):
        upsampler = torch.nn.Upsample(scale_factor=factor, mode='nearest')
        return upsampler(factor * mapping).data

    def normalize_0_to_1(self, F):
        assert (F.dim() == 4)
        max_val = F.max()
        min_val = F.min()
        if max_val != min_val:
            F_normalized = (F - min_val) / (max_val - min_val)
        else:
            F_normalized = self.Tensor(F.size()).fill_(0)

        return F_normalized

    def identity_map(self, size):
        """
        C=2，存储 grid i,j 坐标，element-wise 组合可以得到 坐标位置
        :param size: feature tensor size, 1,512,14,14
        :return:
        """
        idnty_map = self.Tensor(size[0], 2, size[2], size[3])
        # size[0] = 1
        idnty_map[0, 0, :, :].copy_(
            torch.arange(0, size[2]).repeat(size[3], 1).transpose(0, 1)  # arange(size[2]), i; (W,H)->(H,W)
        )
        idnty_map[0, 1, :, :].copy_(
            torch.arange(0, size[3]).repeat(size[2], 1)  # arange(size[3]), j
        )
        return idnty_map

    def find_neural_best_buddies(self, correspondence, F_A, F_B, F_Am, F_Bm, patch_size,
                                 initial_map_a_to_b, initial_map_b_to_a, search_box_radius,
                                 deepest_level=False):
        """
        :param correspondence: 对应点 来自 上层
        :param F_A: model features
        :param F_B:
        :param F_Am: clone features
        :param F_Bm:
        :param patch_size: 对应 level 的 region patch 大小
        :param initial_map_a_to_b: 初始 i,j 坐标位, find_mapping 会更新得到 a_to_b, b_to_a
        :param initial_map_b_to_a:
        :param search_box_radius:
        :param deepest_level: 深层，控制
        :return:
            refined_correspondence
            a_to_b
            b_to_a
        """
        # 归一化: feature vector -> 单位向量
        # 方便后面计算 feature vector 间 cos similarity
        F_Am_normalized = FM.normalize_per_pix(F_Am)
        F_Bm_normalized = FM.normalize_per_pix(F_Bm)

        # NN, d_max
        t1 = time.time()
        a_to_b = self.find_mapping(F_Am_normalized, F_Bm_normalized, patch_size, initial_map_a_to_b, search_box_radius)
        b_to_a = self.find_mapping(F_Bm_normalized, F_Am_normalized, patch_size, initial_map_b_to_a, search_box_radius)
        print('map time:', time.time() - t1)

        if deepest_level:
            # mutual NN, correspondence
            # return [[A_pts], [B_pts]] NBBs
            refined_correspondence = self.find_best_buddies(a_to_b, b_to_a)
            # cal rank activations
            # return [[A_pts], [B_pts], [rank_activations]]
            refined_correspondence = self.calculate_activations(refined_correspondence, F_A, F_B)
        else:
            refined_correspondence = correspondence
            for i in range(len(correspondence[0]) - 1, -1, -1):
                top_left_1, bottom_right_1 = self.extract_receptive_field(correspondence[0][i][0], correspondence[0][i][1], search_box_radius,
                                                                          [a_to_b.size(2), a_to_b.size(3)])
                top_left_2, bottom_right_2 = self.extract_receptive_field(correspondence[1][i][0], correspondence[1][i][1], search_box_radius,
                                                                          [a_to_b.size(2), a_to_b.size(3)])
                # 针对每个 patch, find_best_buddies
                refined_correspondence_i = self.find_best_buddies(a_to_b, b_to_a, top_left_1, bottom_right_1, top_left_2, bottom_right_2)
                refined_correspondence_i = self.calculate_activations(refined_correspondence_i, F_A, F_B)
                refined_correspondence = self.replace_refined_correspondence(refined_correspondence, refined_correspondence_i, i)

        return refined_correspondence, a_to_b, b_to_a

    def find_best_buddies(self, a_to_b, b_to_a,
                          top_left_1=[0, 0],
                          bottom_right_1=[float('inf'), float('inf')],
                          top_left_2=[0, 0],
                          bottom_right_2=[float('inf'), float('inf')]):
        """
        :param a_to_b: A/B patch 内 i,j 映射
        :param b_to_a:
        :param top_left_1: # 后面四组坐标位; 中间层特征要使用; 表示1个 patch 范围
        :param bottom_right_1:
        :param top_left_2:
        :param bottom_right_2:
        :return:
        """
        assert (a_to_b.size() == b_to_a.size())
        correspondence = [[], []]  # A pts, B pts
        # number_of_cycle_consistencies = 0

        for i in range(top_left_1[0], min(bottom_right_1[0], a_to_b.size(2))):
            for j in range(top_left_1[1], min(bottom_right_1[1], a_to_b.size(3))):
                # 通过 A(i,j) -> a_to_b(i,j) A 对应于 B 的 i,j -> b_to_a(map)
                ab_i, ab_j = map(int, a_to_b[0, :, i, j])  # A->B
                aba_i, aba_j = b_to_a[0, :, ab_i, ab_j]  # A->B->A

                d = FM.spatial_distance(  # 判断二点 是否循环对应
                    point_A=self.Tensor([i, j]),
                    point_B=self.Tensor([aba_i, aba_j])
                )
                if d == 0:  # 满足循环对应, long 型 tensor 不能直接和 float 比较
                    if top_left_2[0] <= ab_i < bottom_right_2[0] and top_left_2[1] <= ab_j < bottom_right_2[1]:
                        correspondence[0].append([i, j])  # A 中 i,j 对应
                        correspondence[1].append([ab_i, ab_j])  # B 中 ab_i,ab_j
                        # number_of_cycle_consistencies += 1

        # 每个 patch 内输出一组
        # print('number_of_cycle_consistencies:', number_of_cycle_consistencies)
        return correspondence

    def extract_receptive_field(self, x, y, radius, width):
        center = [2 * x, 2 * y]
        top_left = [max(center[0] - radius, 0), max(center[1] - radius, 0)]
        bottom_right = [min(center[0] + radius + 1, width[0]), min(center[1] + radius + 1, width[1])]
        return [top_left, bottom_right]

    def replace_refined_correspondence(self, correspondence, refined_correspondence_i, index):
        new_correspondence = correspondence
        activation = correspondence[2][index]
        new_correspondence[0].pop(index)
        new_correspondence[1].pop(index)
        new_correspondence[2].pop(index)

        for j in range(len(refined_correspondence_i[0])):
            new_correspondence[0].append(refined_correspondence_i[0][j])
            new_correspondence[1].append(refined_correspondence_i[1][j])
            new_correspondence[2].append(activation + refined_correspondence_i[2][j])

        return new_correspondence

    def calculate_activations(self, correspondence, F_A, F_B):
        """
        :param correspondence: [[A_pts], [B_pts]]
        :param F_A: feature map
        :param F_B:
        :return: response_correspondence
            [[A_pts], [B_pts]] + [rank_activations]
        """
        # normalized featrue map
        # FM.response: l2 and min_max norm F_A, F_B -> [0, 1]
        self.H_A = FM.FA_to_HA_norm(F_A)
        self.H_B = FM.FA_to_HA_norm(F_B)

        response_correspondence = correspondence
        response_correspondence.append([])
        for i in range(len(correspondence[0])):
            # pts and activations
            pt_A, pt_B = correspondence[0][i], correspondence[1][i]
            response_A_i = self.H_A[0, 0, pt_A[0], pt_A[1]]
            response_B_i = self.H_B[0, 0, pt_B[0], pt_B[1]]
            # rank, add for each layer
            correspondence_avg_response_i = (response_A_i + response_B_i) * 0.5
            response_correspondence[2].append(correspondence_avg_response_i)
        return response_correspondence

    def limit_correspondence_number_per_level(self, correspondence, F_A, F_B, tau, top=5):
        correspondence_avg_response = self.Tensor(len(correspondence[0])).fill_(0)
        for i in range(len(correspondence[0])):
            correspondence_avg_response[i] = correspondence[2][i]

        top_response_correspondence = [[], [], []]
        if len(correspondence[0]) > 0:
            [sorted_correspondence, ind] = correspondence_avg_response.sort(dim=0, descending=True)
            for i in range(min(top, len(correspondence[0]))):
                top_response_correspondence[0].append(correspondence[0][ind[i]])
                top_response_correspondence[1].append(correspondence[1][ind[i]])
                top_response_correspondence[2].append(sorted_correspondence[i])

        return top_response_correspondence

    def threshold_response_correspondence(self, correspondence, H_A, H_B, th):
        """
        :param correspondence: [[A_pts], [B_pts], [rank_activation]]
        :param H_A: normalized activation map
        :param H_B:
        :param th: tau=0.05 response threshold
        :return:
        """
        # mask
        M_A = H_A.ge(th)
        M_B = H_B.ge(th)
        # print('A mask:', M_A.sum())  # > thre 较多，但是二者位置能对应的的不多
        # print('B mask:', M_B.sum())

        high_correspondence = [[], [], []]

        # 遍历 对应点
        for i in range(len(correspondence[0])):
            M_A_i = M_A[0, 0, correspondence[0][i][0], correspondence[0][i][1]]
            M_B_i = M_B[0, 0, correspondence[1][i][0], correspondence[1][i][1]]
            # 只判断 第0维 feature 是否 > 0.05?
            if M_A_i and M_B_i:
                high_correspondence[0].append(correspondence[0][i])
                high_correspondence[1].append(correspondence[1][i])
                high_correspondence[2].append(correspondence[2][i])

        return high_correspondence

    def make_correspondence_unique(self, correspondence):
        unique_correspondence = correspondence
        for i in range(len(unique_correspondence[0]) - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                if self.is_same_match(unique_correspondence[0][i], unique_correspondence[0][j]):
                    unique_correspondence[0].pop(i)
                    unique_correspondence[1].pop(i)
                    unique_correspondence[2].pop(i)
                    break

        return unique_correspondence

    def remove_border_correspondence(self, correspondence, border_width, image_width):
        filtered_correspondence = correspondence
        for i in range(len(filtered_correspondence[0]) - 1, -1, -1):
            x_1 = filtered_correspondence[0][i][0]
            y_1 = filtered_correspondence[0][i][1]
            x_2 = filtered_correspondence[1][i][0]
            y_2 = filtered_correspondence[1][i][1]
            if x_1 < border_width or x_1 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif x_2 < border_width or x_2 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif y_1 < border_width or y_1 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)
            elif y_2 < border_width or y_2 > image_width - border_width:
                filtered_correspondence[0].pop(i)
                filtered_correspondence[1].pop(i)
                filtered_correspondence[2].pop(i)

        return filtered_correspondence

    def is_same_match(self, corr_1, corr_2):
        if corr_1[0] == corr_2[0] and corr_1[1] == corr_2[1]:
            return True

    def scale_correspondence(self, correspondence, level):
        scaled_correspondence = [[], [], []]
        scale_factor = int(math.pow(2, level - 1))
        for i in range(len(correspondence[0])):
            scaled_correspondence[0].append([scale_factor * correspondence[0][i][0], scale_factor * correspondence[0][i][1]])
            scaled_correspondence[1].append([scale_factor * correspondence[1][i][0], scale_factor * correspondence[1][i][1]])
            scaled_correspondence[2].append(correspondence[2][i])

        return scaled_correspondence

    def save_correspondence_as_txt(self, correspondence, name=''):
        self.save_points_as_txt(correspondence[0], 'correspondence_A' + name)
        self.save_points_as_txt(correspondence[1], 'correspondence_Bt' + name)

    def save_points_as_txt(self, points, name):
        file_name = os.path.join(self.save_dir, name + '.txt')
        with open(file_name, 'wt') as opt_file:
            for i in range(len(points)):
                opt_file.write('%i, %i\n' % (points[i][0], points[i][1]))

    def top_k_in_clusters(self, correspondence, k):
        """
        :param correspondence: [[A_pts], [B_pts], [rank_activation >= tau]]
        :param k:
        :return: top_cluster_correspondence: [[A_pts], [B_pts], [rank_activation >= tau]]
        """
        if k > len(correspondence[0]):
            return correspondence

        correspondence_R_4 = []
        for i in range(len(correspondence[0])):
            correspondence_R_4.append([  # 聚类特征向量 dim=4, 由2组对应的空间坐标组成
                correspondence[0][i][0],
                correspondence[0][i][1],
                correspondence[1][i][0],
                correspondence[1][i][1]
            ])

        top_cluster_correspondence = [[], [], []]
        # print("Calculating K-means...")
        kmeans = KMeans(n_clusters=k, random_state=0).fit(correspondence_R_4)
        labels = kmeans.labels_

        # 取每个簇 i 下的极值 idx
        cluster = {
            i: {'max_activation': 0, 'max_activation_idx': -1}
            for i in range(k)
        }

        for j in range(len(correspondence[0])):
            if correspondence[2][j] > cluster[labels[j]]['max_activation']:
                cluster[labels[j]]['max_activation'] = correspondence[2][j]
                cluster[labels[j]]['max_activation_idx'] = j

        # pprint(cluster)

        for i in range(k):
            max_activation_idx = cluster[i]['max_activation_idx']
            top_cluster_correspondence[0].append(correspondence[0][max_activation_idx])
            top_cluster_correspondence[1].append(correspondence[1][max_activation_idx])
            top_cluster_correspondence[2].append(correspondence[2][max_activation_idx])

        return top_cluster_correspondence

    def caculate_mid_correspondence(self, correspondence):
        mid_correspondence = []
        for i in range(len(correspondence[0])):
            x_m = math.floor((correspondence[0][i][0] + correspondence[1][i][0]) / 2)
            y_m = math.floor((correspondence[0][i][1] + correspondence[1][i][1]) / 2)
            mid_correspondence.append([x_m, y_m])

        return mid_correspondence

    def transfer_style_local(self, F_A, F_B, patch_size, image_width, mapping_a_to_b, mapping_b_to_a, L):
        """
        F_A, F_B 深层特征图
        mapping_a_to_b, mapping_b_to_a 从 F_B, F_A 得到映射特征图
           FL_1A, FL_1B 浅层特征图
         + RL_1B, RL_1A 映射特征图; 从 wraped L 层 学到 L-1 层
           取均值，得到 common local feature
        """

        # L 层的 mapping, upsample 到 L-1 层
        initial_map_a_to_b = self.upsample_mapping(mapping_a_to_b)  # nearest 上采样 mapping, 在对应区域 coarse-to-fine
        initial_map_b_to_a = self.upsample_mapping(mapping_b_to_a)

        FL_1A = self.features_A[L - 2]  # -1 上一层, -1 idx 从 0
        FL_1B = self.features_B[L - 2]

        t1 = time.time()

        # B->A
        F_B_warped = self.warp(F_A.size(), F_B, patch_size, mapping_a_to_b)  # B 特征 wrap 到 A(i,j) 位置
        F_A_warped = self.warp(F_B.size(), F_A, patch_size, mapping_b_to_a)  # A 特征 wrap 到 B(i,j) 位置

        # Note: freeze 中间 layer [L-1,L]；
        # 更新对象：和 L-1 层 feature 同 size 的 feature; 使模型输出 和 warped feature 相似
        RL_1B = self.model.deconv(F_B_warped, image_width, L, L - 1)
        RL_1A = self.model.deconv(F_A_warped, image_width, L, L - 1)
        print('warp time:', time.time() - t1)
        # 1.0, 2.19, 2.92, 2.70  # 模型学得过程竟然还更快一些

        # RL_1B = self.warp(FL_1A.size(), FL_1B, patch_size, initial_map_a_to_b)
        # RL_1A = self.warp(FL_1B.size(), FL_1A, patch_size, initial_map_a_to_b)
        # print('warp time:', time.time() - t1)
        # 0.22, 0.88, 3.53, 14

        # 浅层 feature + 深层 feature 映射坐标位
        FL_1Am = (FL_1A + RL_1B) * 0.5  # unnormalized
        FL_1Bm = (FL_1B + RL_1A) * 0.5

        return [FL_1A, FL_1B, FL_1Am, FL_1Bm, initial_map_a_to_b, initial_map_b_to_a]

    def finalize_correspondence(self, correspondence, image_width, L):
        print("Drawing correspondence...")
        unique_correspondence = self.make_correspondence_unique(correspondence)
        scaled_correspondence = self.scale_correspondence(unique_correspondence, L)
        # draw.draw_correspondence(self.A, self.B, scaled_correspondence, self.draw_radius[L - 1], self.save_dir, L)
        scaled_correspondence = self.remove_border_correspondence(scaled_correspondence, self.border_size, image_width)
        print("No. of correspondence: ", len(scaled_correspondence[0]))
        return scaled_correspondence

    def run(self, A, B):
        assert (A.size() == B.size())
        image_width = A.size(3)
        util.mkdir(self.save_dir)

        print("Saving original images...")
        util.save_final_image(A, 'original_A', self.save_dir)
        util.save_final_image(B, 'original_B', self.save_dir)

        self.A = self.Tensor(A.size()).copy_(A)
        self.B = self.Tensor(B.size()).copy_(B)

        print("Starting algorithm...")
        # coarse-to-fine
        L_start = 5

        # todo: 直接获取所有需要使用的中间特征
        self.model.set_input(self.A)
        self.features_A = self.model.get_all_layer_output(L_start)
        self.model.set_input(self.B)
        self.features_B = self.model.get_all_layer_output(L_start)

        F_A = self.features_A[L_start - 1]
        F_B = self.features_B[L_start - 1]
        F_Am, F_Bm = F_A.clone(), F_B.clone()

        # 1.初始坐标位 a_to_b, b_to_a
        initial_map_a_to_b = self.identity_map(F_B.size())  # 1,512,14,14
        initial_map_b_to_a = initial_map_a_to_b.clone()

        # L: deepest to shallowest
        for L in range(L_start, self.L_final - 1, -1):  # 5, self.L_final = 2 if fast else 1
            patch_size = self.patch_size_list[L - 1]  # [3,3]
            search_box_radius = self.search_box_radius_list[L - 1]  # 2, (path+1)/2
            draw_radius = self.draw_radius[L - 1]  # 8

            if L == L_start:
                deepest_level = True  # 最深层
                correspondence = []  # begin
            else:
                deepest_level = False
                # correspondence: 中间层，基于上轮(+1层) correspondence 寻找 search region

            print("Finding best-buddies for the " + str(L) + "-th level")

            # 2.对应后坐标位 和 NBBs
            # correspondence: [[A_pts], [B_pts], [rank_activation]]
            # mapping_a_to_b, mapping_b_to_a; 坐标位
            correspondence, mapping_a_to_b, mapping_b_to_a = self.find_neural_best_buddies(
                correspondence, F_A, F_B, F_Am, F_Bm, patch_size,
                initial_map_a_to_b, initial_map_b_to_a,
                search_box_radius, deepest_level
            )

            # 3.tau 过滤 H(p)
            # correspondence: [[A_pts], [B_pts], [rank_activation >= tau]]
            correspondence = self.threshold_response_correspondence(correspondence,
                                                                    self.H_A, self.H_B,  # l2 and min_max norm feature
                                                                    self.tau)  # 可以不用手工设置，保留一些统计结果?
            # print(f'thre >= {self.tau}:', len(correspondence[0]))

            # 设置 k_per_level, 由深到浅的每一层 都执行 k-means 选择 NBBs; 减少计算量
            if self.k_per_level < float('inf'):
                correspondence = self.top_k_in_clusters(correspondence, int(self.k_per_level))
                # print(f'cluster k({self.k_per_level}):', len(correspondence[0]))

            # scale_correspondence 坐标位，在原图画出对应
            if L > self.L_final:
                # print("Drawing correspondence...")
                scaled_correspondence = self.scale_correspondence(correspondence, L)
                # draw.draw_correspondence(self.A, self.B, scaled_correspondence, draw_radius, self.save_dir, L)

                # 获取 L-1 层特征
                F_A, F_B, F_Am, F_Bm, initial_map_a_to_b, initial_map_b_to_a = self.transfer_style_local(
                    F_A, F_B, patch_size, image_width, mapping_a_to_b, mapping_b_to_a, L
                )

        filtered_correspondence = self.finalize_correspondence(correspondence, image_width, self.L_final)
        top_k_correspondence = self.top_k_in_clusters(filtered_correspondence, self.k_final)

        # draw and save
        # final L=1
        draw_radius = self.draw_radius[self.L_final - 1]
        # draw.draw_correspondence(self.A, self.B, filtered_correspondence, draw_radius, self.save_dir, self.L_final)
        # self.save_correspondence_as_txt(filtered_correspondence)

        # final L=1, top k
        draw.draw_correspondence(self.A, self.B, top_k_correspondence, draw_radius, self.save_dir, self.L_final, self.k_final)
        self.save_correspondence_as_txt(top_k_correspondence, name='_top_' + str(self.k_final))

        # return scaled_correspondence
        return top_k_correspondence
