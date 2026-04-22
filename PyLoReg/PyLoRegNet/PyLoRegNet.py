import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder          # 目前没用到，可以保留以兼容别处
from .transformer import FeatureTransformer, FeatureFlowAttention  # 暂时不用
from .matching import global_correlation_softmax, local_correlation_softmax  # 不再在本文件中调用
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position
from .encoder_decoder import Encoder, Decoder
# from PyLoReg.PyLoRegNet.encoder_decoder import Encoder, Decoder

def _load_into_module(module, ckpt_path, name=""):
    if not (os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0):
        print(f"[load {name}] checkpoint not found or empty: {ckpt_path}")
        return

    print(f"[load {name}] loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 兼容不同的保存格式
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # 去掉 'module.' 前缀（如果是 DDP 保存的）
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }

    # 如果外面又包了一层 DDP，这里取 .module
    target = module.module if hasattr(module, "module") else module

    missing, unexpected = target.load_state_dict(state, strict=False)
    print(f"[load {name}] missing   : {len(missing)}")
    print(f"[load {name}] unexpected: {len(unexpected)}")


# ==================== PWC-style Cost Volume ====================

def build_cost_volume_slow(f0, f1, radius=4, normalize=True):
    """
    f0, f1: (B, C, H, W)
    radius: 搜索半径 r，窗口大小为 (2r+1) × (2r+1)
    返回:
        cost_volume: (B, (2r+1)^2, H, W)
    """
    B, C, H, W = f0.shape

    if normalize:
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)

    pad = radius
    # 在四周 pad 一圈，后面直接用平移索引
    f1_padded = F.pad(f1, (pad, pad, pad, pad), mode="replicate")

    cost_list = []

    # dy, dx 遍历 [-r, r]
    for dy in range(-radius, radius + 1):
        y0 = pad + dy
        y1 = y0 + H
        for dx in range(-radius, radius + 1):
            x0 = pad + dx
            x1 = x0 + W
            # 平移后的 f1
            f1_shift = f1_padded[:, :, y0:y1, x0:x1]  # (B, C, H, W)
            # 点乘相似度
            cost = (f0 * f1_shift).sum(1, keepdim=True)  # (B,1,H,W)
            cost_list.append(cost)

    cost_volume = torch.cat(cost_list, dim=1)  # (B, D, H, W), D = (2r+1)^2
    cost_volume = cost_volume / (C ** 0.5)
    return cost_volume



def build_cost_volume(f0, f1, radius=4, normalize=True):
    """
    f0, f1: (B, C, H, W)
    return: (B, (2r+1)^2, H, W)
    """
    B, C, H, W = f0.shape
    D = (2 * radius + 1)**2

    # normalize
    if normalize:
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)

    # padded f1 for all shifted windows
    f1_padded = F.pad(f1, (radius, radius, radius, radius), mode='replicate')

    # unfold f1 to get all (2r+1)^2 neighbor patches
    # output shape = (B, C*(2r+1)^2, H*W)
    f1_unfold = F.unfold(
        f1_padded,
        kernel_size=2*radius+1,
        padding=0,
        stride=1
    )

    # reshape to (B, C, D, H*W)
    f1_unfold = f1_unfold.view(B, C, D, H * W)

    # flatten f0 to (B, C, 1, H*W)
    f0_flat = f0.view(B, C, 1, H * W)

    # cost = 点乘
    cost = (f0_flat * f1_unfold).sum(1) / (C ** 0.5)
    # shape: (B, D, H*W)

    # reshape back to (B, D, H, W)
    cost = cost.view(B, D, H, W)

    return cost



class CostVolumeRefineNet(nn.Module):
    """
    小 CNN，把 cost volume → flow (B,2,H,W)
    可以按需换成更大的 UNet
    """
    def __init__(self, in_ch, mid_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, 2, 3, padding=1),  # 输出光流 (B,2,H,W)
        )

    def forward(self, x):
        return self.net(x)


# ==================== 通用上采样模块 ====================
class DeconvUpsampler(nn.Module):
    """
    使用 ConvTranspose2d 的上采样模块：
        - 支持 2×, 4×, 8×, 16× ...（2 的幂），由 max_upsample 控制上限
        - forward 时通过 upsample_factor 选择用前几层
    """
    def __init__(self, in_ch=2, base_ch=32, max_upsample=16):
        super().__init__()
        assert max_upsample >= 2, "DeconvUpsampler: max_upsample 必须 >= 2"
        assert (max_upsample & (max_upsample - 1)) == 0, \
            f"DeconvUpsampler: max_upsample={max_upsample} 不是 2 的幂"

        self.max_upsample = max_upsample
        num_layers = int(math.log2(max_upsample))  # 例如 16 -> 4 层
        act = nn.LeakyReLU(0.1, inplace=True)

        blocks = []

        # 第一层：in_ch -> base_ch，stride=2
        blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1),
            act
        ))
        # 中间层：base_ch -> base_ch，stride=2
        for _ in range(num_layers - 1):
            blocks.append(nn.Sequential(
                nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1),
                act
            ))

        self.blocks = nn.ModuleList(blocks)
        # 最后统一 refine 到 2 通道
        self.out_conv = nn.Conv2d(base_ch, 2, kernel_size=3, padding=1)

    def forward(self, flow, upsample_factor=2):
        """
        flow: [B, 2, H, W]
        upsample_factor: 2,4,8,16,... <= max_upsample
        """
        assert upsample_factor >= 2, "DeconvUpsampler: upsample_factor 必须 >= 2"
        assert (upsample_factor & (upsample_factor - 1)) == 0, \
            f"DeconvUpsampler: upsample_factor={upsample_factor} 不是 2 的幂"
        assert upsample_factor <= self.max_upsample, \
            f"DeconvUpsampler: upsample_factor={upsample_factor} > max_upsample={self.max_upsample}"

        num_used = int(math.log2(upsample_factor))
        x = flow
        for i in range(num_used):
            x = self.blocks[i](x)

        x = self.out_conv(x)
        return x



class PixelShuffleUpsampler(nn.Module):
    """
    使用 PixelShuffle 的上采样模块：
        - 支持 2×, 4×, 8× ...（2 的幂），由 max_upsample 控制上限
        - 每一层都是：Conv -> LeakyReLU -> Conv -> PixelShuffle(2)，输出 2 通道 flow
        - 最后统一一个 3x3 Conv refine
    """
    def __init__(self, in_ch=2, base_ch=32, max_upsample=4):
        super().__init__()
        assert max_upsample >= 2, "PixelShuffleUpsampler: max_upsample 必须 >= 2"
        assert (max_upsample & (max_upsample - 1)) == 0, \
            f"PixelShuffleUpsampler: max_upsample={max_upsample} 不是 2 的幂"

        self.max_upsample = max_upsample
        num_layers = int(math.log2(max_upsample))  # 例如 4 -> 2 层
        act = nn.LeakyReLU(0.1, inplace=True)

        blocks = []
        in_channels = in_ch

        for _ in range(num_layers):
            blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1),
                act,
                nn.Conv2d(base_ch, 2 * 2 * 2, kernel_size=3, padding=1),
                nn.PixelShuffle(2)   # 输出 2 通道 flow
            ))
            in_channels = 2  # 下一层输入仍然是 2 通道 flow

        self.blocks = nn.ModuleList(blocks)
        # 最终再来一个小 conv 做 refine
        self.refine = nn.Conv2d(2, 2, kernel_size=3, padding=1)

    def forward(self, flow, upsample_factor=2):
        """
        flow: [B, 2, H, W]
        upsample_factor: 2,4,... <= max_upsample
        """
        assert upsample_factor >= 2, "PixelShuffleUpsampler: upsample_factor 必须 >= 2"
        assert (upsample_factor & (upsample_factor - 1)) == 0, \
            f"PixelShuffleUpsampler: upsample_factor={upsample_factor} 不是 2 的幂"
        assert upsample_factor <= self.max_upsample, \
            f"PixelShuffleUpsampler: upsample_factor={upsample_factor} > max_upsample={self.max_upsample}"

        num_used = int(math.log2(upsample_factor))

        x = flow
        for i in range(num_used):
            x = self.blocks[i](x)

        x = self.refine(x)
        return x


# ==================== GMFlow 主体（PWC-style Cost Volume + CNN Refinement） ====================

class PyLoRegNet(nn.Module):
    def __init__(self,
                 use_feature_num=4,
                 feature_channels=128,
                 **kwargs,
                 ):
        super(PyLoRegNet, self).__init__()
        self.feature_channels = feature_channels
        self.use_feature_num = use_feature_num

        self.layer_num = 4  # Encoder/Decoder 层数

        # Encoder / Decoder（你自定义的 UNet 编码器）
        self.backbone_encoder = Encoder(
            in_ch=1,
            out_ch=feature_channels,
            down_num=self.layer_num
        )
        self.backbone_decoder = Decoder(
            out_ch=1,
            in_ch=feature_channels,
            up_num=self.layer_num
        )

        self.deconv_upsampler = DeconvUpsampler(
            in_ch=2,
            base_ch=32,
            max_upsample=16,
        )
        self.pixelshuffle_upsampler = PixelShuffleUpsampler(
            in_ch=2,
            base_ch=32,
            max_upsample=4,
        )

        # PWC-style cost volume + CNN refine
        self.cost_radius = 4
        cv_in_ch = (2 * self.cost_radius + 1) ** 2
        self.cost_net = CostVolumeRefineNet(
            in_ch=cv_in_ch,
            mid_ch=feature_channels,  # 你可以改成 64/96 等
        )

    def extract_feature(self, img):
        # encoder 返回：从高分辨率 -> 低分辨率的列表
        features = self.backbone_encoder(img)   # list: [H,W], [H/2,W/2], ...
        # 翻转为 低分辨率 -> 高分辨率
        features = features[::-1]
        return features

    def upsample_flow(self, flow, mode='bilinear', upsample_factor=2):
        """
        统一上采样接口：
            mode: 'bilinear' | 'pixelshuffle' | 'deconv'
        """
        if mode == 'bilinear':
            up_flow = F.interpolate(
                flow,
                scale_factor=upsample_factor,
                mode='bilinear',
                align_corners=True
            ) * upsample_factor

        elif mode == 'pixelshuffle':
            up_flow = self.pixelshuffle_upsampler(flow, upsample_factor=upsample_factor)
            # 光流语义：分辨率放大 K 倍，位移也要乘 K
            up_flow = up_flow * upsample_factor

        elif mode == 'deconv':
            up_flow = self.deconv_upsampler(flow, upsample_factor=upsample_factor)
            up_flow = up_flow * upsample_factor

        else:
            raise ValueError(f"Unknown upsample mode: {mode}")

        return up_flow

    def forward(self, GT, img1,
                use_feature_num = 3,
                **kwargs,
                ):
        """
        GT:    img0 (B,1,H,W)
        img1:  img1 (B,1,H,W)
        现在匹配模块是：PWC-style cost volume + CNN refine，不再用 transformer/global corr
        """
        img0 = GT
        results_dict = {}
        flow_preds = []
        flow_per_preds = []

        # 如果你要对输入做归一化，可以恢复这句（注意单通道版本）
        # img0, img1 = normalize_img(img0, img1)

        # resolution: low -> high
        feature0_list = self.extract_feature(img0)
        feature1_list = self.extract_feature(img1)

        # 用最低分辨率 feature 重建一张图（可选，用于可视化/辅助 loss）
        recon_img0 = self.backbone_decoder(feature0_list[0])
        recon_img1 = self.backbone_decoder(feature1_list[0])

        if 0:
            for i, f in enumerate(feature0_list):
                try:
                    print(f"[{i}] shape =", f.shape)
                except:
                    # 万一某个不是 tensor / ndarray
                    print(f"[{i}] type =", type(f))

        # 丢掉最高分辨率那个 feature（保留多尺度的 3 个）
        # feature0_list = feature0_list[:-1]
        # feature1_list = feature1_list[:-1]

        flow_all = None
        feature_num = self.layer_num
           # 用前三个尺度做多尺度 refinement

        for scale_idx in range(use_feature_num):
            now_upsample_factor = 2 ** (feature_num - scale_idx)  # 最后一轮循环 = 2**(4-2)=4
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            # 当前尺度：如果已有 coarse flow，则先 warp feature1
            if flow_all is not None:
                flow_for_warp = flow_all
                if flow_for_warp.shape[-2:] != feature1.shape[-2:]:
                    H1, W1 = feature1.shape[-2:]
                    H0, W0 = flow_for_warp.shape[-2:]
                    scale_y = H1 / H0
                    scale_x = W1 / W0

                    flow_for_warp = F.interpolate(
                        flow_for_warp, size=(H1, W1),
                        mode='bilinear', align_corners=True
                    )
                    scale = torch.tensor([scale_x, scale_y],
                                         device=flow_for_warp.device,
                                         dtype=flow_for_warp.dtype).view(1, 2, 1, 1)
                    flow_for_warp = flow_for_warp * scale

                feature1_used = flow_warp(feature1, flow_for_warp)
            else:
                feature1_used = feature1

            # -------- PWC-style: cost volume + CNN refine ----------
            cost = build_cost_volume(
                feature0,           # (B,C,H,W)
                feature1_used,      # (B,C,H,W)
                radius=self.cost_radius,
                normalize=True,
            )   # (B, (2r+1)^2, H, W)

            # 小 CNN 从 cost volume 预测当前尺度的 Δflow
            flow_pred = self.cost_net(cost)   # (B,2,H,W)
            # print('flow_pred ---> ',flow_pred.shape, img1.shape)
            if flow_pred.shape[-2:] == img1.shape[-2:]:
                flow_pred_upsample = flow_pred
            else:
                flow_pred_upsample = self.upsample_flow(flow_pred, mode='deconv', 
                                            upsample_factor=now_upsample_factor)
            flow_per_preds.append(flow_pred_upsample)
            # 多尺度累积：上采样 + 残差加
            if flow_all is not None:
                if flow_all.shape != flow_pred.shape:
                    # 尺度间固定 2× 上采样
                    flow_all = self.upsample_flow(flow_all, mode='deconv', upsample_factor=2)
                flow_all = flow_all + flow_pred
            else:
                flow_all = flow_pred

            flow_preds.append(flow_all)

        # 最后一次：上采样到原分辨率（这里 upsample_factor 已经是最后一轮循环里的值，例如 4）
        if now_upsample_factor>1:
            flow_all = self.upsample_flow(flow_all, mode='deconv', upsample_factor=now_upsample_factor)
        flow_preds.append(flow_all)
        
        if 0:
            for i, f in enumerate(flow_preds):
                print(f"[DEBUG] flow_preds[{i}] shape = {tuple(f.shape)}")
                
        results_dict.update({
            'flow_per_preds':flow_per_preds,
            'flow_preds': flow_preds,
            'recon_img0': recon_img0,
            'recon_img1': recon_img1,
            'feature0_list': feature0_list,
            'feature1_list': feature1_list,
        })

        return results_dict
