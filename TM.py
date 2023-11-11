import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.init as init

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dp_min, dp_max = -0.00, 0.3
dt_min, dt_max = -0.0000, 0.005
fp_min, fp_max = -0.00, 0.7
img_w = 192
img_h = 256

b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000]).astype(np.float32)
b_fit = np.expand_dims(b_values, -1)
b_fit = torch.from_numpy(b_fit).to(device)

class TMnet(nn.Module):
    def __init__(self):
        super(TMnet, self).__init__()
        self.d1 = DownsampleLayer(9, 64)  # 9-64
        self.d2 = DownsampleLayer(64, 128)  # 64-128
        self.d3 = DownsampleLayer(128, 256)  # 128-256
        self.d4 = DownsampleLayer(256, 512)  # 256-512

        self.u1 = UpSampleLayer(512, 512)  # 512-1024-512
        self.u2 = UpSampleLayer(1024, 256)  # 1024-512-256
        self.u3 = UpSampleLayer(512, 128)  # 512-256-128
        self.u4 = UpSampleLayer(256, 64)  # 256-128-64

        self.o = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        max_mat = torch.max(inputs.reshape((inputs.size(0), -1)), dim=1)[0].reshape((inputs.size(0), 1, 1, 1))
        # inputs_norm = torch.clamp(inputs / max_mat, 0, 1)
        d_1, d1 = self.d1(inputs)
        d_2, d2 = self.d2(d1)
        d_3, d3 = self.d3(d2)
        d_4, d4 = self.d4(d3)

        u1 = self.u1(d4, d_4)
        u2 = self.u2(u1, d_3)
        u3 = self.u3(u2, d_2)
        u4 = self.u4(u3, d_1)

        # mask = (inputs[:, :1] > 0).float()
        # params = torch.clamp(torch.abs(self.o(u4)), 0, 1)  # * mask
        out_params = self.sigmoid(self.o(u4)[:, :3])  #
        dp = out_params[:, 0:1] * (dp_max - dp_min) + dp_min
        dt = out_params[:, 1:2] * (dt_max - dt_min) + dt_min
        fp = out_params[:, 2:3] * (fp_max - fp_min) + fp_min
        params = torch.cat((dp, dt, fp), dim=1)

        out_rec = self.ivim_matmul(params) * inputs[:, :1]

        return out_rec, params

    def ivim_matmul(self, params):
        flat = params.view(params.size(0), 3, params.size(2) * params.size(3))
        dp = flat[:, 0].unsqueeze(1)
        dt = flat[:, 1].unsqueeze(1)
        fp = flat[:, 2].unsqueeze(1)
        b_fit_ = b_fit.unsqueeze(0).repeat(params.size(0), 1, 1)
        outputs = fp * torch.exp(-torch.bmm(b_fit_, dp)) + (1 - fp) * torch.exp(-torch.bmm(b_fit_, dt))
        outputs = outputs.view(params.size(0), b_values.shape[0], img_w, img_h)
        # print(outputs.shape)
        return outputs

class DownsampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.Conv_BN_ReLU_2(x)
        out_2 = self.downsample(out)
        return out, out_2


class UpSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out):
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out, out), dim=1)
        return cat_out

def order(ivim):
    Dp_, Dt_, Fp_ = ivim[..., 0:1], ivim[..., 1:2], ivim[..., 2:]
    if np.mean(Dp_) < np.mean(Dt_):
        print('swap')
        temp = copy.deepcopy(Dp_)
        Dp_ = copy.deepcopy(Dt_)
        Dt_ = temp
        Fp_ = 1-Fp_
    return np.concatenate((Dp_, Dt_, Fp_), -1)




