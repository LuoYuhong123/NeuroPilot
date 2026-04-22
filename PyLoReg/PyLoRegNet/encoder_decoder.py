import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act="relu", norm="bn", padding_mode="reflect"):
        super().__init__()

        def get_norm(c):
            if norm == "bn":
                return nn.BatchNorm2d(c)
            elif norm == "in":
                return nn.InstanceNorm2d(c, affine=True)
            else:
                return nn.Identity()

        def get_act():
            if act == "relu":
                return nn.ReLU(inplace=True)
            elif act == "leaky_relu":
                return nn.LeakyReLU(0.1, inplace=True)
            else:
                return nn.GELU()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            get_norm(out_ch),
            get_act(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode=padding_mode),
            get_norm(out_ch),
            get_act(),
        )

    def forward(self, x):
        return self.block(x)



class Encoder(nn.Module):
    """
    Encoder: 多层下采样 + DoubleConv
    down_factors: [2,2,2] 表示连续三次下采样，每次 /2
    channels: [64, 128, 256, 512]
    """
    def __init__(
        self,
        in_ch=1,
        out_ch=128,
        down_num=4,
        act="relu",
        norm="in",
        padding_mode="reflect",
    ):
        super().__init__()
        # 每一层下采样倍率（这里还是全部用 2）
        down_factors = [2] * down_num
        channels = [out_ch // 2] + [out_ch] * down_num
        assert len(channels) == len(down_factors) + 1

        # stem 卷积（保持尺寸不变）
        self.stem = nn.Conv2d(
            in_ch, channels[0],
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode
        )

        # 下采样改成 AvgPool2d + DoubleConv
        pools = []
        blocks = []
        for i, s in enumerate(down_factors):
            # 平均池化做下采样，kernel_size=stride=s
            pools.append(nn.AvgPool2d(kernel_size=s, stride=s))

            # 下采样之后做特征提取
            blocks.append(
                DoubleConv(
                    channels[i],
                    channels[i + 1],
                    act=act,
                    norm=norm,
                    padding_mode=padding_mode,
                )
            )

        self.pools = nn.ModuleList(pools)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.stem(x)
        f_list = [x]
        # 每层：先下采样，再 DoubleConv
        for pool, block in zip(self.pools, self.blocks):
            x = pool(x)      # AvgPool2d 下采样
            x = block(x)     # 提取特征
            f_list.append(x)
        return f_list



class Decoder(nn.Module):
    """
    Decoder: 多层上采样（无 skip）+ DoubleConv
    up_factors:   [2,2,2] 表示连续三次上采样，每次 ×2
    channels:     [512, 256, 128, 64]
    """
    def __init__(
        self,
        out_ch=1,
        in_ch=128,
        # channels=[512, 256, 128, 64],   [2, 2, 2]
        # up_factors=[2, 2],
        up_num = 4,
        act="relu",
        norm="",
        padding_mode="reflect",
        up_mode="interp",
    ):
        super().__init__()
        up_factors = [2] * up_num
        channels= [in_ch] * up_num+[ in_ch//2]
        # print('channels ---> ',channels, up_factors)
        assert len(channels) == len(up_factors) + 1

        self.up_mode = up_mode

        blocks = []
        for i, s in enumerate(up_factors):
            in_c = channels[i]
            out_c = channels[i+1]

            if up_mode == "interp":
                up = nn.Sequential(
                    nn.Upsample(scale_factor=s, mode="bilinear", align_corners=True),
                    nn.Conv2d(in_c, out_c, kernel_size=1),
                )
            else:  # deconv
                up = nn.ConvTranspose2d(in_c, out_c, kernel_size=s, stride=s)

            block = nn.Sequential(
                up,
                DoubleConv(out_c, out_c, act=act, norm=norm, padding_mode=padding_mode)
            )
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(channels[-1], out_ch, kernel_size=1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x



class AutoEncoder(nn.Module):
    def __init__(self, 
                 in_ch=1,
                    layer_num=4,
                    channel_num=128):
        super().__init__()
        self.encoder = Encoder( in_ch = in_ch,
                                out_ch = channel_num,
                                down_num = layer_num,  )
        self.decoder = Decoder( out_ch = in_ch,
                                in_ch = channel_num,
                                up_num = layer_num,  )

    def forward(self, x):
        encoder_f_list = self.encoder(x)
        encoder_f = encoder_f_list[-1]
        output = self.decoder(encoder_f)
        return output, encoder_f_list



if __name__ == "__main__":
    net = AutoEncoder()
    x = torch.randn(1, 3, 256, 256)
    y = net(x)
    print(x.shape, y.shape)
