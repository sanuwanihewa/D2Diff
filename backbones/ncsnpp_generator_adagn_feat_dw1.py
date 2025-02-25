# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
''' Codes adapted from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py
'''

from . import utils, layers, layerspp, dense_layer
import torch.nn as nn
import functools
import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp_Adagn
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp_Adagn
ResnetBlock_Feat = layerspp.ConvFeatBlock
ResnetBlock_Adapt_Feat = layerspp.ConvBlock
ResnetBlock_Feat_GAP = layerspp.ConvBlock_GAP
ResnetBlockBigGAN_one = layerspp.ResnetBlockBigGANpp_Adagn_one
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
default_initializer = layers.default_init
dense = dense_layer.dense
from backbones.gcn_lib import *
from backbones.ScConv import *

class ViGBlock(nn.Module):
    def __init__(self, in_features, HW):
        super(ViGBlock, self).__init__()
        self.k = 9  # neighbor num
        self.conv = 'mr'  # graph conv layer {edge, mr}
        self.act = nn.GELU()  # activation function
        self.norm = 'instance'  # normalization type
        self.bias = True  # bias for convolution layers
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated KNN or not
        self.epsilon = 0.2  # stochastic epsilon for graph convolution
        self.use_stochastic = False  # use stochastic GCN
        self.drop_path = 0.0  # drop path rate
        self.HW = HW  # input image height and width

        # Define a simple MLP to process the time embedding (temb)
        self.temb_dense = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features)
        )

        # Define a simple MLP to process the latent embedding (zemb)
        self.zemb_dense = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, in_features)
        )

        # Define the main convolution block
        vig_blocks = [
            Grapher(in_channels=in_features, kernel_size=7, dilation=1, conv=self.conv, act='gelu',
                    norm='instance', bias=self.bias, stochastic=self.use_stochastic, epsilon=self.epsilon,
                    r=1, n=self.HW, drop_path=self.drop_path, relative_pos=True),
            ScConv(in_features),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            Grapher(in_features, 7, 1, self.conv, 'gelu', 'instance',
                    self.bias, self.use_stochastic, self.epsilon, r=1, n=self.HW, drop_path=self.drop_path,
                    relative_pos=True),
            ScConv(in_features),
            nn.InstanceNorm2d(in_features)
        ]
        self.conv_block = nn.Sequential(*vig_blocks)

    def forward(self, x, temb=None, zemb=None):
        # Process latent embedding (zemb) and add it as a condition
        if zemb is not None:
            zemb = self.zemb_dense(zemb)
            zemb = zemb[:, :, None, None]  # Reshape for broadcasting
            x = x + zemb

        # Pass through the main convolution block
        h = self.conv_block(x)

        # Process time embedding (temb) and add it as a condition
        if temb is not None:
            temb = self.temb_dense(temb)
            temb = temb[:, :, None, None]  # Reshape for broadcasting
            h += temb

        return x + h

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def initDCTKernel(N):
    kernel = np.zeros((N, N, N * N))
    cnum = 0
    for i in range(N):
        for j in range(N):
            ivec = np.linspace(0.5 * math.pi / N * i, (N - 0.5) * math.pi / N * i, num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0.5 * math.pi / N * j, (N - 0.5) * math.pi / N * j, num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            if i == 0 and j == 0:
                slice = slice / N
            elif i * j == 0:
                slice = slice * np.sqrt(2) / N
            else:
                slice = slice * 2.0 / N

            kernel[:, :, cnum] = slice
            cnum = cnum + 1
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (3, 0, 1, 2))
    return kernel


################################## Generate the 2D-iDCT Kernels of size NxN ##################################
def initIDCTKernel(N):
    kernel = np.zeros((N, N, N * N))
    for i_ in range(N):
        i = N - i_ - 1
        for j_ in range(N):
            j = N - j_ - 1
            ivec = np.linspace(0, (i + 0.5) * math.pi / N * (N - 1), num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0, (j + 0.5) * math.pi / N * (N - 1), num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            ic = np.sqrt(2.0 / N) * np.ones(N)
            ic[0] = np.sqrt(1.0 / N)
            jc = np.sqrt(2.0 / N) * np.ones(N)
            jc[0] = np.sqrt(1.0 / N)
            cmatrix = np.outer(ic, jc)

            slice = slice * cmatrix
            slice = slice.reshape((1, N * N))
            slice = slice[np.newaxis, :]
            kernel[i_, j_, :] = slice / (N * N)
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (0, 3, 1, 2))
    return kernel

class AdaptiveCombiner(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveCombiner, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 3, kernel_size=1),  # One score per tensor
            nn.Softmax(dim=1)
        )
        self.final_conv = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)

    def forward(self, f1, f2, f3):
        combined = torch.cat([f1, f2, f3], dim=1)  # Concatenate along channels
        attention_scores = self.attention(combined)
        combined_f= (
            attention_scores[:, 0:1] * f1 + 
            attention_scores[:, 1:2] * f2 + 
            attention_scores[:, 2:3] * f3
        )
        return self.final_conv(combined_f)



@utils.register_model(name='ncsnpp_freq')
class NCSNpp_Freq(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
       
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            # assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

          

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
            ConvBlock = functools.partial(ResnetBlock_Feat,
                                          act=act, in_ch=config.num_channels
                                          )

        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        self.in_kernel_3x3 = nn.Parameter(torch.Tensor(initDCTKernel(3)).contiguous())
        self.in_kernel_5x5 = nn.Parameter(torch.Tensor(initDCTKernel(5)).contiguous())
        self.in_kernel_7x7 = nn.Parameter(torch.Tensor(initDCTKernel(7)).contiguous())

       

       
        modules.append(ConvBlock(in_ch=1, out_ch=nf))
      
       

       

        hs_c = [nf * 4]

        in_ch = nf * 4
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
               
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                  
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                   
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))
                  

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                   
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
    
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

        self.combiner_3_3 = AdaptiveCombiner(in_channels=9)
        self.combiner_5_5 = AdaptiveCombiner(in_channels=25)
        self.combiner_7_7 = AdaptiveCombiner(in_channels=49)

       

    def forward(self, x, cond1, cond2, cond3, time_cond, z):
    
        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        h_x_3_c1 = F.conv2d(input=cond1, weight=self.in_kernel_3x3, padding=1)
        h_x_5_c1 = F.conv2d(input=cond1, weight=self.in_kernel_5x5, padding=2)
        h_x_7_c1 = F.conv2d(input=cond1, weight=self.in_kernel_7x7, padding=3)

        h_x_3_c2 = F.conv2d(input=cond2, weight=self.in_kernel_3x3, padding=1)
        h_x_5_c2 = F.conv2d(input=cond2, weight=self.in_kernel_5x5, padding=2)
        h_x_7_c2 = F.conv2d(input=cond2, weight=self.in_kernel_7x7, padding=3)

        h_x_3_c3 = F.conv2d(input=cond3, weight=self.in_kernel_3x3, padding=1)
        h_x_5_c3 = F.conv2d(input=cond3, weight=self.in_kernel_5x5, padding=2)
        h_x_7_c3 = F.conv2d(input=cond3, weight=self.in_kernel_7x7, padding=3)

        x_feat = modules[m_idx](x)
        m_idx += 1

        #combiner ablation
        # c_feat1 = modules[m_idx](torch.cat((h_x_3_c1, h_x_5_c1, h_x_7_c1), axis=1))
        # m_idx += 1
        # c_feat2 = modules[m_idx](torch.cat((h_x_3_c2, h_x_5_c2, h_x_7_c2), axis=1))
        # m_idx += 1
        # c_feat3 = modules[m_idx](torch.cat((h_x_3_c3, h_x_5_c3, h_x_7_c3), axis=1))
        # m_idx += 1

        c_feat1 = self.combiner_3_3(h_x_3_c1,h_x_3_c2,h_x_3_c3)
        c_feat2 = self.combiner_5_5(h_x_5_c1,h_x_5_c2,h_x_5_c3)
        c_feat3 = self.combiner_7_7(h_x_7_c1,h_x_7_c2,h_x_7_c3)

        # attention_map_to_plot1 = h_x_3_c1[0, 0, :, :] 
        # attention_map_to_plot1 = to_range_0_1(attention_map_to_plot1);
        # attention_map_to_plot1 = attention_map_to_plot1/attention_map_to_plot1.max()
        # plt.imshow(attention_map_to_plot1.detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Attention Map')
        # plt.axis('off')
        # plt.savefig('/data/shew0029/MedSyn/VQGAN_3D/SynDiff-multi/SynDiff-main/results/exp_dw/attn_maps/attention_map1.png', bbox_inches='tight', pad_inches=0)

      
        
        # plt.close()

      
       

        # hs = [torch.cat((x_feat, c_feat1,c_feat2,c_feat3), axis=1)]
        hs = [torch.cat((x_feat, c_feat1, c_feat2, c_feat3), axis=1)]
      

        # hs = [modules[m_idx](x)]
        # m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        
        # h_3=self.conv_k3(h)
        # h_5=self.conv_k5(h)
        # h_7=self.conv_k7(h)
      
        # h1 = F.conv2d(input=h_3, weight=self.out_kernel_3x3, padding=1)
        # h2 = F.conv2d(input=h_5, weight=self.out_kernel_5x5, padding=2)
        # h3 = F.conv2d(input=h_7, weight=self.out_kernel_7x7, padding=3)
        # h = self.conv_final(torch.cat((h1, h2, h3), axis=1))

        if not self.not_use_tanh:
            return torch.tanh(h)
        return h
       

@utils.register_model(name='ncsnpp_manifold')
class NCSNpp_Gnn(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)
        self.HW = 256 // 4 * 256 // 4

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            # assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
            ConvBlock = functools.partial(ResnetBlock_Feat,
                                          act=act, in_ch=config.num_channels
                                          )

        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))

        hs_c = [nf * 4]

        in_ch = nf * 4
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        # modules.append(AttnBlock(channels=in_ch))
      

        modules.append(ViGBlock(256, self.HW))

        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def forward(self, x, cond1, cond2, cond3, time_cond, z):
      
        # timestep/noise_level embedding; only for continuous training

        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        x_feat = modules[m_idx](x)
        m_idx += 1

        cond1_feat = modules[m_idx](cond1)
        m_idx += 1

        cond2_feat = modules[m_idx](cond2)
        m_idx += 1

        cond3_feat = modules[m_idx](cond3)
        m_idx += 1

        hs = [torch.cat((x_feat, cond1_feat, cond2_feat, cond3_feat), axis=1)]


      
      

        # hs = [modules[m_idx](x)]
        # m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
       
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        if not self.not_use_tanh:

            return torch.tanh(h)
        else:
            return h



@utils.register_model(name='ncsnpp_spatial')
class NCSNpp_Spatial(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.not_use_tanh = config.not_use_tanh
        self.act = act = nn.SiLU()
        self.z_emb_dim = z_emb_dim = config.z_emb_dim

        self.nf = nf = config.num_channels_dae
        ch_mult = config.ch_mult
        self.num_res_blocks = num_res_blocks = config.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.attn_resolutions
        dropout = config.dropout
        resamp_with_conv = config.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.conditional  # noise-conditional
        fir = config.fir
        fir_kernel = config.fir_kernel
        self.skip_rescale = skip_rescale = config.skip_rescale
        self.resblock_type = resblock_type = config.resblock_type.lower()
        self.progressive = progressive = config.progressive.lower()
        self.progressive_input = progressive_input = config.progressive_input.lower()
        self.embedding_type = embedding_type = config.embedding_type.lower()
        init_scale = 0.
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = config.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            # assert config.training.continuous, "Fourier features are only used for continuous training."

            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=config.fourier_scale
            ))
            embed_dim = 2 * nf

        elif embedding_type == 'positional':
            embed_dim = nf

        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
                                     with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample,
                                                 fir=fir, fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample,
                                       with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                                                   fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)
            ConvBlock = functools.partial(ResnetBlock_Feat,
                                          act=act, in_ch=config.num_channels
                                          )

        elif resblock_type == 'biggan_oneadagn':
            ResnetBlock = functools.partial(ResnetBlockBigGAN_one,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            temb_dim=nf * 4,
                                            zemb_dim=z_emb_dim)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = config.num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))
        modules.append(ConvBlock(in_ch=channels, out_ch=nf))

        hs_c = [nf * 4]

        in_ch = nf * 4
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                    num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

        mapping_layers = [PixelNorm(),
                          dense(config.nz, z_emb_dim),
                          self.act, ]
        for _ in range(config.n_mlp):
            mapping_layers.append(dense(z_emb_dim, z_emb_dim))
            mapping_layers.append(self.act)
        self.z_transform = nn.Sequential(*mapping_layers)

    def forward(self, x, cond1, cond2, cond3, time_cond, z):
        # to_range_0_1 = lambda x: (x + 1.) / 2.
        # timestep/noise_level embedding; only for continuous training

        zemb = self.z_transform(z)
        modules = self.all_modules
        m_idx = 0

        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond

            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.config.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        x_feat = modules[m_idx](x)
        m_idx += 1

        cond1_feat = modules[m_idx](cond1)
        m_idx += 1

        cond2_feat = modules[m_idx](cond2)
        m_idx += 1

        cond3_feat = modules[m_idx](cond3)
        m_idx += 1

        hs = [torch.cat((x_feat, cond1_feat, cond2_feat, cond3_feat), axis=1)]

        
        # attention_map_to_plot1 = cond1_feat[0, 0, :, :] 
        # attention_map_to_plot1 = to_range_0_1(attention_map_to_plot1);
        # attention_map_to_plot1 = attention_map_to_plot1/attention_map_to_plot1.max()
        # plt.imshow(attention_map_to_plot1.detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.title('Attention Map')
        # plt.axis('off')
        # plt.savefig('/data/shew0029/MedSyn/VQGAN_3D/SynDiff-multi/SynDiff-main/results/exp_dw/attn_maps/attention_map2.png', bbox_inches='tight', pad_inches=0)
      

        # hs = [modules[m_idx](x)]
        # m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb, zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb, zemb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb, zemb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb, zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb, zemb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        if not self.not_use_tanh:

            return torch.tanh(h)
        else:
            return h
