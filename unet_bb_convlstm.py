import torch.nn as nn
import torch
import numpy as np
from model.convlstm3d import ConvLSTM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Unet_b_clstm(nn.Module):

    def __init__(self, nb_features=None, nb_levels=None, feat_mult=1):
        super(Unet_b_clstm, self).__init__()

        if nb_features is None:
            nb_features = default_unet_features()
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features
        self.maxpool=nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        prev_nf = 2
        self.down = nn.ModuleList()
        for nf in self.enc_nf:
            self.down.append(ConvBlock(prev_nf, nf, stride=2))
            prev_nf = nf
        enc_history = list(reversed(self.enc_nf))  #32,32,32,16

        self.up = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[2:len(self.enc_nf) + 2 ]):
            channels = prev_nf * 2 if i > 0 else prev_nf
            self.up.append(ConvBlock(channels, nf, stride=1))
            prev_nf = nf

        prev_nf += 16
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf) + 2:]:
            self.extras.append(ConvBlock(prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        self.x_enc_pop = []
        self.x_enc_last = []
        for i in range(x.shape[1]):
            self.x_enc_layers = []
            x_enc = [np.squeeze(x[:, i:i + 1, ...], axis=1)]
            for layer in self.down:
                self.x_enc_layers.append(layer(x_enc[-1]))
                x_enc[-1]=self.maxpool(self.x_enc_layers[-1])
            self.x_enc_pop.append(self.x_enc_layers)
            self.x_last = (x_enc[-1]).unsqueeze(1)
            self.x_enc_last.append(self.x_last)
            if i == 0:
                self.x_enc_la=self.x_last
            else:
                self.x_enc_la=torch.cat([self.x_enc_la, self.x_last], dim=1)

        clstm_input=self.x_enc_la
        clstm_out = ConvLSTM(input_dim=clstm_input.shape[2], hidden_dim=32, kernel_size=[(3, 3, 3), (3, 3, 3)],
                     num_layers=2, batch_first=True, bias=False, return_all_layers=True)(clstm_input)
        dec_input=clstm_out[0][0]
        x_dec_last = []
        self.x_dec_pop=list(reversed(self.x_enc_pop))
        for i in range(dec_input.shape[1]):
            x_dec_list = [np.squeeze(dec_input[:, i: i+1, ...], axis=1)]
            self.x_dec=x_dec_list[-1]
            for layer in self.up:
                self.x_dec = layer(self.x_dec)
                self.x_dec = self.upsample(self.x_dec)
                self.x_dec = torch.cat([self.x_dec, self.x_dec_pop[i].pop()], dim=1)

            for layer in self.extras:
                self.x_dec = layer(self.x_dec)
            x_dec_last.append(self.x_dec)

        return x_dec_last

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

def get_len(x_shape):
    length=1
    for dim in x_shape:
        length *= dim

    return length
