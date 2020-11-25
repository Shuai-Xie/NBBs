import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import os

from pkg_resources import parse_version


def define_Vgg19(opt):
    use_gpu = len(opt.gpu_ids) > 0

    # 直接采用 pretrained 模型
    vgg19net = vgg19(models.vgg19(pretrained=True), opt)

    if use_gpu:
        assert (torch.cuda.is_available())
        vgg19net.cuda(opt.gpu_ids[0])  # 只用第1块卡

    return vgg19net


class vgg19(nn.Module):
    def __init__(self, basic_model, opt):
        """
        :param basic_model: 使用 torchvision pretrain vgg19
        :param opt:
        """
        super(vgg19, self).__init__()

        # model
        # 通过 Sequential layer 数字截取
        self.layer_1 = self.make_layers(basic_model, 0, 2)
        self.layer_2 = self.make_layers(basic_model, 2, 7)
        self.layer_3 = self.make_layers(basic_model, 7, 12)
        self.layer_4 = self.make_layers(basic_model, 12, 21)
        self.layer_5 = self.make_layers(basic_model, 21, 30)

        # 5 层 layer 特征
        self.layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4, self.layer_5]
        # channels for each level
        self.channels = [3, 64, 128, 256, 512, 512]

        # opt
        image_height = image_width = opt.imageSize  # 224
        self.Tensor = torch.cuda.FloatTensor if opt.gpu_ids else torch.Tensor
        self.input = self.Tensor(opt.batchSize, opt.input_nc, image_height, image_width)  # (1,3,224,224)
        self.convergence_threshold = opt.convergence_threshold  # 0.001
        self.old_lr = opt.lr  # 0.05
        self.beta = opt.beta1  # 0.5

    def make_layers(self, basic_model, start_layer, end_layer):
        """
        从 basic_model 按照 [start_layer, end_layer) 截取得到对应的 layer, 即子模型
        end_layer 所在位置 对应原始网络 每个 layer 首层 conv 的输出，即获取特定位置的 feature map
        """
        layer = []
        features = next(basic_model.children())  # 获取到 VGG.features 即 FC 层之前
        original_layer_number = 0
        for module in features.children():
            if start_layer <= original_layer_number < end_layer:
                layer += [module]
            original_layer_number += 1
        # print(layer)
        return nn.Sequential(*layer)

    def make_classifier_layer(self, old_classifier, dropout_layers):
        classifier_layer = []
        features = next(old_classifier.children())
        for name, module in old_classifier.named_children():
            if int(name) not in dropout_layers:
                classifier_layer += [module]
        return nn.Sequential(*classifier_layer)

    def set_input(self, input_A):
        self.input = input_A

    @torch.no_grad()
    def get_all_layer_output(self, level=5):
        features = []
        layer_i_input = self.input
        for i in range(level):  # 1-5
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layer_i_input = layer_i_output
            features.append(layer_i_output.data)
        return features

    def forward(self, level=5, start_level=0):
        """
        通过设置 level 来得到不同中间层输出，其实可以使用 hook 一次性拿到所有想要层的特征
        :param level: 结束 level
        :param start_level: 起始 level
        :return:
        """
        assert (level >= start_level)
        layer_i_output = layer_i_input = self.input

        # 可能存在 每个 level 输出，浅层需要重复推理
        for i in range(start_level, level):
            layer_i = self.layers[i]
            layer_i_output = layer_i(layer_i_input)
            layer_i_input = layer_i_output

        return layer_i_output

    def deconv(self, features, original_image_width, src_level, dst_level):
        """
        features: warped feature, 作为 src feature，learn L-1 feature
        original_image_width: 224
        src_level: L
        dst_level: L-1
        print_errors:
        """
        # dst feature
        # 计算 L-1 层特征 size: B,C,H,W
        dst_feature_size = self.get_layer_size(dst_level, batch_size=features.size(0), width=original_image_width)
        deconvolved_feature = self.Tensor(dst_feature_size)
        deconvolved_feature.data.fill_(0)
        deconvolved_feature.requires_grad_()  # requires_grad=True

        optimizer = torch.optim.Adam([{'params': deconvolved_feature}],
                                     lr=self.old_lr, betas=(self.beta, 0.999))
        # src feature
        src_feature_size = self.get_layer_size(src_level, batch_size=features.size(0), width=original_image_width)
        src_feature = self.Tensor(src_feature_size)  # requires_grad=False
        src_feature.data.copy_(features)  # copy warped feature

        criterion = nn.MSELoss(reduction='mean')

        i = 0
        self.reset_last_losses()
        while self.convergence_criterion() > self.convergence_threshold:
            optimizer.zero_grad()
            # init L-1 feature 作为 输入
            self.set_input(deconvolved_feature)
            # 使用 [L-1,L] 这一段 layer 作为模型; 没有优化 layer 参数，而是优化 deconvolved_feature
            deconvolved_feature_forward = self.forward(level=src_level, start_level=dst_level)
            # loss
            loss_perceptual = criterion(deconvolved_feature_forward, src_feature)
            loss_perceptual.backward()
            loss = loss_perceptual.item()
            self.update_last_losses(loss)
            # if i % 5 == 0:
            #     print(self.last_losses)
            optimizer.step()
            i += 1

        return deconvolved_feature

    def reset_last_losses(self):
        self.last_losses = np.array([0, 100, 200, 300, 400, 500])

    def update_last_losses(self, loss):
        self.last_losses = np.delete(self.last_losses, 0)  # 删除队头
        self.last_losses = np.append(self.last_losses, loss)  # 加入新的

    def convergence_criterion(self):  # 计算 loss 更新均值
        convergence_criterion = np.average(np.abs(np.diff(self.last_losses)))
        return convergence_criterion

    def get_layer_size(self, level, batch_size=1, width=224):
        if level == 0:
            width_layer = width
        else:
            width_layer = int(width / (2 ** int(level - 1)))  # width 下采样
        return torch.Size([batch_size, self.channels[int(level)], width_layer, width_layer])
