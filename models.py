import torch
from torch import nn
import torch.nn.functional as F

class SubfieldSplitSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat((F.softmax(x[:, :2], dim=1), F.softmax(x[:, 2:], dim=1)), dim=1)

class CBAM(nn.Module):
    def __init__(self, in_ch, r):
        super().__init__()
        self.attention_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)

        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(in_ch // r),
            nn.Linear(in_ch // r, in_ch)
        )

    def cam(self, x):
        N, C, H, W, D = x.shape

        max_ch = F.max_pool3d(x, kernel_size=(H, W, D)).view(N, C)
        avg_ch = torch.mean(x, dim=(2, 3, 4)).view(N, C)

        att_max = self.mlp(max_ch)
        att_avg = self.mlp(avg_ch)

        att = torch.sigmoid(att_max + att_avg)
        att = att.view(N, C, 1, 1, 1)

        return att * x

    def sam(self, x):
        max_ch = torch.max(x, dim=1, keepdim=True).values
        avg_ch = torch.mean(x, dim=1, keepdim=True)

        att_map = torch.cat([max_ch, avg_ch], dim=1)

        att_map = self.attention_conv(att_map)
        att_map = torch.sigmoid(att_map)

        return att_map * x

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x

class NestedResUNet6(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_ch, out_ch, residual=False, dropout_p=0.0,
                     cbam=False, cbam_r=16):
            super().__init__()
            self.residual = residual
            self.out_ch = out_ch

            conv_class = nn.Conv3d
            conv_params = dict(kernel_size=3, padding=1)

            if self.residual:
                self.res_conv = conv_class(in_ch, out_ch, **conv_params)

            self.conv1 = conv_class(in_ch, out_ch, bias=False, **conv_params)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.activation1 = nn.ReLU(inplace=True)
            self.conv2 = conv_class(out_ch, out_ch, bias=False, **conv_params)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.activation2 = nn.ReLU(inplace=True)

            self.cbam = None
            if cbam:
                self.cbam = CBAM(out_ch, r=cbam_r)

            self.dropout = None
            if dropout_p != 0.0:
                self.dropout = nn.Dropout3d(p=dropout_p)

        def forward(self, x):
            x_in = x

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.activation1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.activation2(x)

            if self.cbam is not None:
                x = self.cbam(x)

            if self.residual:
                x = self.res_conv(x_in) + x

            if self.dropout is not None:
                x = self.dropout(x)

            return x

    def __init__(self, input_channels, output_channels, filters, dropout_p=0.0, subfields=False, cbam=False):
        super().__init__()

        self.dropout = None
        if dropout_p != 0.0:
            self.dropout = nn.Dropout3d(p=dropout_p)

        self.down = nn.AvgPool3d(kernel_size=2, stride=2, count_include_pad=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        block_params = dict(dropout_p=dropout_p, cbam=cbam)

        self.conv0_0 = self.Block(input_channels, filters, **block_params, residual=True)
        self.conv1_0 = self.Block(filters, filters, **block_params)
        self.conv0_1 = self.Block(filters*2, filters, **block_params, residual=True)

        self.conv2_0 = self.Block(filters, filters, **block_params)
        self.conv1_1 = self.Block(filters*3, filters, **block_params)
        self.conv0_2 = self.Block(filters*2, filters, **block_params, residual=True)

        self.conv3_0 = self.Block(filters, filters, **block_params)
        self.conv2_1 = self.Block(filters*3, filters, **block_params)
        self.conv1_2 = self.Block(filters*3, filters, **block_params)
        self.conv0_3 = self.Block(filters*2, filters, **block_params, residual=True)

        self.out_conv = nn.Conv3d(filters, output_channels, kernel_size=3, padding=1)

        self.subfields = subfields
        if subfields:
            assert output_channels == 5
            self.hypothesis = SubfieldSplitSoftmax()
        else:
            self.hypothesis = nn.Softmax(dim=1)

    def forward(self, x, id=None):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.down(x0_0))
        x0_1 = self.conv0_1(torch.cat((x0_0, self.up(x1_0)), 1))

        x2_0 = self.conv2_0(self.down(x1_0))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.up(x2_0), self.down(x0_1)), 1))
        x0_2 = self.conv0_2(torch.cat((x0_1, self.up(x1_1)), 1))

        x3_0 = self.conv3_0(self.down(x2_0))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.up(x3_0), self.down(x1_1)), 1))
        x1_2 = self.conv1_2(torch.cat((x1_1, self.up(x2_1), self.down(x0_2)), 1))
        x0_3 = self.conv0_3(torch.cat((x0_2, self.up(x1_2)), 1))

        x_out = self.out_conv(x0_3)
        x_out = self.hypothesis(x_out)
        return x_out