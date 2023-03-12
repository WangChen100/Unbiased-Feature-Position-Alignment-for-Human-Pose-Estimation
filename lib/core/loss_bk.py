# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
# import numpy as np
# from sklearn.mixture import GaussianMixture as GMM
# from core.inference import get_max_preds

class JointsOffsetLoss(nn.Module):
    def __init__(self, use_target_weight, offset_weight, smooth_l1):
        super(JointsOffsetLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.offset_weight = offset_weight
        self.criterion = nn.MSELoss(reduction='mean')
        self.criterion_offset = nn.SmoothL1Loss(reduction='mean') if smooth_l1 else nn.L1Loss(reduction='mean')

    def forward(self, output, hm_hps, target, target_offset, target_weight):
        """
        calculate loss
        :param output: [batch, joints, height, width]
        :param hm_hps: [batch, 2*joints, height, width]
        :param target: [batch, joints, height, width]
        :param target_offset: [batch, 2*joints, height, width]
        :param mask_01: [batch, joints, height, width]
        :param mask_g: [batch, joints, height, width]
        :param target_weight: [batch, joints, 1]
        :return: loss=joint_loss+weight*offset_loss
        """
        batch_size, num_joints, _, _ = output.shape

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, dim=1)
        offsets_pred = hm_hps.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)
        offsets_gt = target_offset.reshape((batch_size, 2*num_joints, -1)).split(2, dim=1)

        del batch_size, _

        joint_l2_loss, offset_loss = 0.0, 0.0

        for idx in range(num_joints):
            offset_pred = offsets_pred[idx] * heatmaps_gt[idx]  # [batch_size, 2, h*w]
            offset_gt = offsets_gt[idx] * heatmaps_gt[idx]      # [batch_size, 2, h*w]
            heatmap_pred = heatmaps_pred[idx].squeeze()     # [batch_size, h*w]
            heatmap_gt = heatmaps_gt[idx].squeeze()         # [batch_size, h*w]

            if self.use_target_weight:
                joint_l2_loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                offset_loss += self.criterion_offset(
                    offset_pred.mul(target_weight[:, idx, None]),
                    offset_gt.mul(target_weight[:, idx, None])
                )  # target_weight[:, idx].unsqueeze(2)
            else:
                joint_l2_loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
                offset_loss += self.criterion_offset(offset_pred, offset_gt)

        loss = joint_l2_loss + self.offset_weight * offset_loss

        return loss / num_joints, offset_loss / num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
