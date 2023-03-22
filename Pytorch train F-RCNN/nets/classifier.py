import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   after ROIPooling results run a regression task
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   after ROIPooling results run a classification task
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   initilize network weights
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)
        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   use ROI bbox to get common use feature map
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   use classifier network to get feature map
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   f7 shape [300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   after ROIPooling results run regression
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        #-----------------------------------#
        #   after ROIPooling results run classification
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)
        #-----------------------------------#
        #   intilize weights
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)
        
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   use the ROI bbox get feature maps for common use
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   use classifier network to extract the featuret
        #-----------------------------------#
        fc7 = self.classifier(pool)
        #--------------------------------------------------------------#
        #   f7 shape [300, 2048]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)
        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
