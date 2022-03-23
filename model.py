import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
        
class Conv3d_rmap(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, padding, stride, bias=True):
        super(Conv3d_rmap, self).__init__()
        self.pad_u = padding[2]
        self.pad_v = padding[1]
        self.pad_r = padding[0]
        self.conv = nn.Conv3d(in_ch, out_ch, k_size, padding=(0,0,0), stride=stride, bias=bias)

    def forward(self, x):
        x_u = F.pad(x, (self.pad_u,self.pad_u,0,0,0,0), mode='circular')
        x_v = F.pad(x_u, (0,0,self.pad_v,self.pad_v,self.pad_r,self.pad_r), mode='replicate')
        y = self.conv(x_v)
        return y

class PolarConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0):
        super(PolarConvBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv3d_rmap(in_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            Conv3d_rmap(out_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
        )
        self.actlayer = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            Conv3d_rmap(out_ch, out_ch, (2,3,3), padding=(0,1,1), stride=(2,1,1)),
        )
        self.identity = nn.Sequential(
            Conv3d_rmap(in_ch, out_ch, (1,1,1), padding=(0,0,0), stride=(1,1,1), bias=False),
            nn.BatchNorm3d(out_ch),
        )
        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            y = self.downsample(self.dropout(self.actlayer(self.conv(x) + self.identity(x))))
        else:
            y = self.downsample(self.actlayer(self.conv(x) + self.identity(x)))
        return y

class PolarLastBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=0):
        super(PolarLastBlock, self).__init__()
        self.conv = nn.Sequential(
            Conv3d_rmap(in_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            Conv3d_rmap(out_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
        )
        self.actlayer = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1)),
        )
        self.identity = nn.Sequential(
            Conv3d_rmap(in_ch, out_ch, (1,1,1), padding=(0,0,0), stride=(1,1,1), bias=False),
            nn.BatchNorm3d(out_ch),
        )
        if dropout_rate > 0:
            self.dropout = nn.Dropout3d(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        if self.dropout is not None:
            y = self.downsample(self.dropout(self.actlayer(self.conv(x) + self.identity(x))))
        else:
            y = self.downsample(self.actlayer(self.conv(x) + self.identity(x)))
        return y

class LocConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LocConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, (3,3,3), padding=(1,1,1), stride=(1,1,1)),
            nn.BatchNorm3d(out_ch),
        )
        self.actlayer = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
        )
        self.identity = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )

    def forward(self, x):
        y = self.downsample(self.actlayer(self.conv(x) + self.identity(x)))
        return y

class PolarLayer(nn.Module):
    def __init__(self, polar_size, r_spacing, im_size, im_spacing):
        super(PolarLayer, self).__init__()
        self.polar_size = polar_size
        u = torch.linspace(0, 360, steps=self.polar_size[0]) * np.pi / 180
        v = torch.linspace(-90, 90, steps=self.polar_size[1]) * np.pi / 180
        r = torch.linspace(0, (self.polar_size[2]*r_spacing)/(im_size*im_spacing*0.5), steps=self.polar_size[2])
        grid_r, grid_v, grid_u = torch.meshgrid(r, v, u)
        self.grid_x = grid_r * torch.cos(grid_v) * torch.sin(grid_u)
        self.grid_y = grid_r * torch.cos(grid_v) * torch.cos(grid_u)
        self.grid_z = grid_r * torch.sin(grid_v)
        self.grid_x = self.grid_x.unsqueeze(0).unsqueeze(0)
        self.grid_y = self.grid_y.unsqueeze(0).unsqueeze(0)
        self.grid_z = self.grid_z.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input, polar_cp):
        output_list = []
        for i in range(input.shape[0]):
            sample_grid = torch.cat([self.grid_x.cuda() + polar_cp[i, 0], self.grid_y.cuda() + polar_cp[i, 1], self.grid_z.cuda() + polar_cp[i, 2]], dim=1)
            sample_grid = sample_grid.permute(0,2,3,4,1).contiguous()
            resampled_input = F.grid_sample(input[i:i+1], sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            output_list.append(resampled_input)
        output = torch.cat(output_list, dim=0)
        return output

class PTN(nn.Module):
    def __init__(self, in_ch, base_ch, polar_size, r_spacing, im_size, im_spacing):
        super(PTN, self).__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.polar_size_im = polar_size
        self.polar_size_f0 = [polar_size[0], polar_size[1], polar_size[2] // 2]

        # localization sub-network
        self.loc_conv1 = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, (7,7,7), padding=(3,3,3), stride=(2,2,2)),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((3, 3, 3), padding=(1,1,1), stride=(2, 2, 2)),
        )
        self.loc_conv2 = LocConvBlock(base_ch, base_ch)
        self.loc_conv3 = LocConvBlock(base_ch, base_ch*2)
        self.loc_conv4 = LocConvBlock(base_ch*2, base_ch*4)
        self.loc_conv5 = LocConvBlock(base_ch*4, base_ch*8)
        self.loc_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.loc_fc = nn.Linear(base_ch*8, 3)
        self.loc_actlayer = nn.Tanh()

        # polar transform layer
        self.polar_layer_im = PolarLayer(self.polar_size_im, r_spacing, im_size, im_spacing)
        self.polar_layer_f0 = PolarLayer(self.polar_size_f0, r_spacing * 2, im_size, im_spacing)

        # surface regression sub-network
        self.reg_conv1 = PolarConvBlock(in_ch, base_ch, dropout_rate=0)
        self.reg_conv2 = PolarConvBlock(base_ch*2, base_ch*2, dropout_rate=0)
        self.reg_conv3 = PolarConvBlock(base_ch*2, base_ch*4, dropout_rate=0)
        self.reg_conv4 = PolarConvBlock(base_ch*4, base_ch*8, dropout_rate=0)
        self.reg_conv5 = PolarLastBlock(base_ch*8, base_ch*8, dropout_rate=0)
        self.reg_out = nn.Sequential( 
            nn.Conv3d(base_ch*8, base_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, 1, 1),
            nn.Sigmoid()
        )

    ## Input
    # im: input image (stored in Cartesian grid (x,y,z))
    # gt_cp: ground-truth centroid point used as the orgin point from polar transformation
    # gt_mask: ground-truth segmentation mask of the input image (stored in Cartesian grid (x,y,z))
    ## Output
    # pred_srm: predicted surface radius map (SRM)
    # pred_cp: predicted prostate centroid point
    # gt_srm: ground-truth SRM transformed using the input gt_cp and gt_mask
    def forward(self, im, gt_cp=None, gt_mask=None):
        f0 = self.loc_conv1(im)
        f1 = self.loc_conv2(f0)
        f1 = self.loc_conv3(f1)
        f1 = self.loc_conv4(f1)
        f1 = self.loc_conv5(f1)
        f1 = self.loc_pool(f1)
        f1 = f1.reshape(f1.shape[0], -1)
        f1 = self.loc_fc(f1)
        pred_cp = self.loc_actlayer(f1)

        if gt_cp is not None:
            cp = gt_cp
        else:
            cp = pred_cp
        im_polar = self.polar_layer_im(im, cp)
        f0_polar = self.polar_layer_f0(f0, cp)
        if gt_mask is not None:
            gt_srm = (torch.sum(self.polar_layer_im(gt_mask, cp), dim=2, keepdim=True) + 0.5) / self.polar_size_im[2]

        reg_f0 = self.reg_conv1(im_polar)
        reg_f1 = self.reg_conv2(torch.cat([reg_f0, f0_polar], dim=1))
        reg_f1 = self.reg_conv3(reg_f1)
        reg_f1 = self.reg_conv4(reg_f1)
        reg_f1 = self.reg_conv5(reg_f1)

        pred_srm = self.reg_out(reg_f1)

        if gt_mask is not None:
            return pred_srm, pred_cp, gt_srm
        else:
            return pred_srm, pred_cp

    def description(self):
        return 'Polar transform network with {0:d}-ch input for prostate segmentation (base channel = {1:d})'.format(self.in_ch, self.base_ch)