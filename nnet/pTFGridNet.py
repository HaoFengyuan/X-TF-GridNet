import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from libs.utils import get_layer


class pTFGridNet(nn.Module):
    def __init__(self,
                 n_fft=256,
                 n_layers=6,
                 lstm_hidden_units=256,
                 attn_n_head=4,
                 attn_approx_qk_dim=516,
                 emb_dim=32,
                 emb_ks=8,
                 emb_hs=1,
                 num_spks=101,
                 activation="prelu",
                 eps=1.0e-5):
        super().__init__()
        self.n_layers = n_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        # Speech Encoder
        t_ksize, f_ksize = 3, 3
        ks, padding = (t_ksize, f_ksize), (t_ksize // 2, f_ksize // 2)
        self.conv = nn.Sequential(nn.Conv2d(2, emb_dim, ks, padding=padding),
                                  nn.GroupNorm(1, emb_dim, eps=eps))

        # Speaker Encoder
        self.aux_encoder = AuxEncoder(emb_dim, num_spks)

        # Speaker Extractor
        self.fusion_blocks = nn.ModuleList([FusionModule(emb_dim) for _ in range(n_layers)])
        self.separate_blocks = nn.ModuleList([GridNetBlock(emb_dim,
                                                           emb_ks,
                                                           emb_hs,
                                                           n_freqs,
                                                           lstm_hidden_units,
                                                           n_head=attn_n_head,
                                                           approx_qk_dim=attn_approx_qk_dim,
                                                           activation=activation,
                                                           eps=eps) for _ in range(n_layers)])

        # Speech Decoder
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=padding)

    def forward(self,
                mix: torch.Tensor,
                aux: torch.Tensor,
                aux_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.
            Args:
                mix (torch.Tensor): batched audio tensor with N samples [B, 2, T, F]
                aux (torch.Tensor): batched audio tensor with N samples [B, T, F]
                aux_lengths (torch.Tensor): aux input lengths [B]
            Returns:
                enhanced (torch.Tensor): [B, 2, T, F] audio tensors with N samples.
        """
        # Speech Encoder
        esti = self.conv(mix)  # [B, -1, T, F]
        aux = self.conv(aux)  # [B, -1, T, F]

        # Speaker Encoder
        aux, speak_pred = self.aux_encoder(aux, aux_lengths)

        # Speaker Extractor
        for i in range(self.n_layers):
            esti = self.fusion_blocks[i](aux, esti)
            esti = self.separate_blocks[i](esti)  # [B, -1, T, F]

        # Speech Decoder
        esti = self.deconv(esti)  # [B, 2, T, F]

        return esti, speak_pred


class GridNetBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 n_freqs,
                 hidden_channels,
                 n_head=4,
                 approx_qk_dim=512,
                 activation='prelu',
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        # Intra-Frame Full-Band Module
        in_channels = emb_dim * emb_ks

        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Sub-Band Temporal Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Cross-Frame Self-Attention Module
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(f"attn_conv_Q_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_K_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_V_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, emb_dim // n_head, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps)))
        self.add_module("attn_concat_proj",
                        nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size=1),
                                      get_layer(activation)(),
                                      LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)))

    def __getitem__(self, item):
        return getattr(self, item)

    def forward(self, x):
        B, C, old_T, old_F = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        F = math.ceil((old_F - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = nn.functional.pad(x, (0, F - old_F, 0, T - old_T))

        # Intra-Frame Full-Band Module
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, F]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, F)  # [BT, C, F]
        intra_rnn = nn.functional.unfold(intra_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, F]
        intra_rnn = intra_rnn.view([B, T, C, F])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, F]
        intra_rnn = intra_rnn + input_  # [B, C, T, F]

        # Sub-Band Temporal Module
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * F, C, T)  # [BF, C, T]
        inter_rnn = nn.functional.unfold(inter_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
        inter_rnn = self.inter_rnn(inter_rnn)[0]  # [BF, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
        inter_rnn = inter_rnn.view([B, F, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, F]
        inter_rnn = inter_rnn + input_  # [B, C, T, F]

        # Cross-Frame Self-Attention Module
        inter_rnn = inter_rnn[..., :old_T, :old_F]
        batch = inter_rnn

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self[f"attn_conv_Q_{ii}"](batch))  # [B, C, T, F]
            all_K.append(self[f"attn_conv_K_{ii}"](batch))  # [B, C, T, F]
            all_V.append(self[f"attn_conv_V_{ii}"](batch))  # [B, C, T, F/H]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, F]
        K = torch.cat(all_K, dim=0)  # [B', C, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C, T, F/H]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*F]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*F]
        V = V.transpose(1, 2)  # [B', T, C, F/H]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/H]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / emb_dim ** 0.5  # [B', T, T]
        attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/H]

        V = V.reshape(old_shape)  # [B', T, C, F/H]
        V = V.transpose(1, 2)  # [B', C, T, F/H]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [H, B, C, T, F/H]
        batch = batch.transpose(0, 1)  # [B, H, C, T, F/H])
        batch = batch.contiguous().view([B, self.n_head * emb_dim, old_T, -1])  # [B, C, T, F]
        batch = self["attn_concat_proj"](batch)  # [B, C, T, F]
        out = batch + inter_rnn

        return out


class AuxEncoder(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_spks):
        super(AuxEncoder, self).__init__()
        k1, k2 = (1, 3), (1, 3)
        self.d_feat = emb_dim

        self.aux_enc = nn.ModuleList([EnUnetModule(emb_dim, emb_dim, (1, 5), k2, scale=4),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=3),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=2),
                                      EnUnetModule(emb_dim, emb_dim, k1, k2, scale=1)])
        self.out_conv = nn.Linear(emb_dim, emb_dim)
        self.speaker = nn.Linear(emb_dim, num_spks)

    def forward(self,
                auxs: torch.Tensor,
                aux_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aux_lengths = (((aux_lengths // 3) // 3) // 3) // 3

        for i in range(len(self.aux_enc)):
            auxs = self.aux_enc[i](auxs)  # [B, C, T, F]

        auxs = torch.stack([torch.mean(
            aux[:, :aux_length, :], dim=(1, 2)) for aux, aux_length in zip(auxs, aux_lengths)], dim=0)  # [B, C]
        auxs = self.out_conv(auxs)

        return auxs, self.speaker(auxs)


class FusionModule(nn.Module):
    def __init__(self,
                 emb_dim,
                 nhead=4,
                 dropout=0.1):
        super(FusionModule, self).__init__()
        self.nhead = nhead
        self.dropout = dropout
        param_size = [1, 1, emb_dim]

        self.attn = nn.MultiheadAttention(emb_dim,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)
        self.fusion = nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=1)
        self.alpha = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.zeros_(self.alpha)

    def forward(self,
                aux: torch.Tensor,
                esti: torch.Tensor) -> torch.Tensor:
        aux = aux.unsqueeze(1)  # [B, 1, C]
        flatten_esti = esti.flatten(start_dim=2).transpose(1, 2)  # [B, T*F, C]
        aux_adapt = self.attn(aux, flatten_esti, flatten_esti, need_weights=False)[0]
        aux = aux + self.alpha * aux_adapt  # [B, 1, C]

        aux = aux.unsqueeze(-1).transpose(1, 2).expand_as(esti)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]

        return esti


class EnUnetModule(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 scale: int):
        super(EnUnetModule, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.cin = cin
        self.cout = cout
        self.scale = scale

        self.in_conv = nn.Sequential(GateConv2d(cin, cout, k1, (1, 2)),
                                     nn.BatchNorm2d(cout),
                                     nn.PReLU(cout))
        self.encoder = nn.ModuleList([Conv2dUnit(k2, cout) for _ in range(scale)])
        self.decoder = nn.ModuleList([Deconv2dUnit(k2, cout, 1)])
        for i in range(1, scale):
            self.decoder.append(Deconv2dUnit(k2, cout, 2))
        self.out_pool = nn.AvgPool2d((3, 1))

    def forward(self, x: torch.Tensor):
        x_resi = self.in_conv(x)
        x = x_resi
        x_list = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x_list.append(x)

        x = self.decoder[0](x)
        for i in range(1, len(self.decoder)):
            x = self.decoder[i](torch.cat([x, x_list[-(i + 1)]], dim=1))
        x_resi = x_resi + x

        return self.out_pool(x_resi)


class GateConv2d(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k: tuple,
                 s: tuple):
        super(GateConv2d, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s

        self.conv = nn.Sequential(nn.ConstantPad2d((0, 0, k[0] - 1, 0), value=0.),
                                  nn.Conv2d(in_channels=cin,
                                            out_channels=cout * 2,
                                            kernel_size=k,
                                            stride=s))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)

        return outputs * gate.sigmoid()


class Conv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int):
        super(Conv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.conv = nn.Sequential(nn.Conv2d(c, c, k, (1, 2)),
                                  nn.BatchNorm2d(c),
                                  nn.PReLU(c))

    def forward(self, x):
        return self.conv(x)


class Deconv2dUnit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 expend_scale: int):
        super(Deconv2dUnit, self).__init__()
        self.k = k
        self.c = c
        self.expend_scale = expend_scale
        self.deconv = nn.Sequential(nn.ConvTranspose2d(c * expend_scale, c, k, (1, 2)),
                                    nn.BatchNorm2d(c),
                                    nn.PReLU(c))

    def forward(self, x):
        return self.deconv(x)


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        self.eps = eps

        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,F]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        self.eps = eps

        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,1]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat


if __name__ == "__main__":
    import toml

    configs = toml.load('/data/haofengyuan/Speech_extraction/Github/X-TF-GridNet/configs/train_config.toml')
    gpuids = tuple(configs['gpu']['gpu_ids'])
    device = torch.device("cuda:{}".format(gpuids[0]))

    net = pTFGridNet(n_fft=configs['signal']['fft_num'],
                     n_layers=configs['net']['n_layers'],
                     lstm_hidden_units=configs['net']['lstm_hidden_units'],
                     attn_n_head=configs['net']['attn_n_head'],
                     attn_approx_qk_dim=configs['net']['attn_approx_qk_dim'],
                     emb_dim=configs['net']['emb_dim'],
                     emb_ks=configs['net']['emb_ks'],
                     emb_hs=configs['net']['emb_hs'],
                     num_spks=configs['path']['num_spks'],
                     activation=configs['net']['activation'],
                     eps=configs['net']['eps']).to(device)

    # num_params = sum([param.nelement() for param in net.parameters()]) / 10.0 ** 6
    # print(num_params)

    # aux = torch.ones((2, 2, 501, 129), device=device)
    # inpt = torch.ones((2, 2, 501, 129), device=device)
    # aux_len = torch.tensor([501, 401], dtype=torch.int, device=device)
    # output, spk_pred = net(inpt, aux, aux_len)
    # print(output.shape)

    from ptflops import get_model_complexity_info
    
    
    def input_constructor(input_shape):
        inputs = {'mix': torch.ones((1, *input_shape), device=device),
                  'aux': torch.ones((1, *input_shape), device=device),
                  'aux_lengths': torch.tensor([input_shape[1]], dtype=torch.int, device=device)}

        return inputs
    
    macs, params = get_model_complexity_info(net, (2, 501, 129),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             input_constructor=input_constructor,
                                             verbose=True,
                                             output_precision=4)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
