# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Xingyi Zhou (zhouxy2017@gmail.com)
# Modified by Chen Wang (wangchen199179@gmail.com)
# ------------------------------------------------------------------------------

import numpy as np

def draw_dense_reg(hm_size, joint, sigma, locref_stdev, kpd):

    offset_map = np.empty([2, hm_size[1], hm_size[0]], dtype=np.float32)  # [2, height, width]

    y, x = np.ogrid[0:hm_size[1], 0:hm_size[0]]
    mat_x, mat_y = np.meshgrid(x, y)  # standard cooridate field

    offset_map[0] = joint[0] - mat_x  # x-axis offset field
    offset_map[1] = joint[1] - mat_y  # y-axis offset field

    h = np.sum(offset_map*offset_map, axis=0)  # distance**2, [height, width]
    heatmap = np.exp(- h / (2 * sigma ** 2))
    heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0  # gaussian heatmap

    offset_map /= locref_stdev  # rescale offset map
    mask01 = np.where(h <= kpd**2, 1, 0)  # 0-1 mask
    # offset_map *= mask01[None, ...]
        
    return heatmap, offset_map, mask01


if __name__ == '__main__':
    heatmap_size=[24, 32]
    joints=np.array([10.6, 5.4])
    sigma=3
    mask_sigma=1
    hm, om, mask, hm_ce = draw_dense_reg(heatmap_size, joints, sigma, 1.5, mask_sigma, 4)
