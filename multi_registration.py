import torch.nn as nn
import torch
import numpy as np
from torch.distributions.normal import Normal
from model.unet_bb_convlstm import Unet_b_clstm
from model.STN import SpatialTransformer, ResizeTransform, VecInt


class MC(nn.Module):
    def __init__(self, vol_size, inshape, nb_features=None, nb_levels=None,
                 feat_mult=1, int_steps=7, int_downsize=2, bidir=False, use_probs=False):
        super(MC, self).__init__()
        self.training = True
        self.unet_clstm = Unet_b_clstm(nb_features=nb_features, nb_levels=nb_levels, feat_mult=feat_mult)

        self.flow = nn.Conv3d(self.unet_clstm.dec_nf[-1], 3,  kernel_size=3, padding=1)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        resize = int_steps > 0 and int_downsize > 1
        self.resize = ResizeTransform(int_downsize) if resize else None
        self.fullsize = ResizeTransform(1 / int_downsize) if resize else None

        self.bidir = bidir

        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None
        self.STN = SpatialTransformer(vol_size)

    def forward(self, source, target, registration=False):
        x = torch.cat([source, target], dim=2)
        x = self.unet_clstm(x)
        y_source_list=[]
        flow_list=[]
        y_target_list=[]
        preint_flow_list=[]
        for i in range(len(x)):
            flow_field = self.flow(x[i])  #按照时间顺序得到序列flow
            # resize flow for integration
            pos_flow = flow_field
            # if self.resize:
            #     pos_flow = self.resize(pos_flow)
            preint_flow = pos_flow
            neg_flow = -pos_flow if self.bidir else None
            # if self.integrate:
            #     pos_flow = self.integrate(pos_flow)
            #     neg_flow = self.integrate(neg_flow) if self.bidir else None
            #     if self.fullsize:
            #         pos_flow = self.fullsize(pos_flow)
            #         neg_flow = self.fullsize(neg_flow) if self.bidir else None

            self.source_t=np.squeeze(source[:, i: i+1, ...], axis=1)
            y_source = self.STN(self.source_t, pos_flow)
            y_target = self.STN(target, neg_flow) if self.bidir else None
            y_source_list.append(y_source), flow_list.append(pos_flow), y_target_list.append(y_target), preint_flow_list.append(preint_flow)

        if not registration:
            return (y_source_list, y_target_list, preint_flow_list) if self.bidir else (y_source_list, preint_flow_list)
        else:
            return y_source_list, flow_list


