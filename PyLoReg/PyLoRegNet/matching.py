import torch
import torch.nn.functional as F

from .geometry import coords_grid, generate_window_grid, normalize_coords


def global_correlation_softmax(feature0, feature1,
                               pred_bidir_flow=False,
                               ):
    # global correlation
    b, c, h, w = feature0.shape
    feature0 = feature0.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, C]
    feature1 = feature1.view(b, c, -1)  # [B, C, H*W]

    correlation = torch.matmul(feature0, feature1).view(b, h, w, h, w) / (c ** 0.5)  # [B, H, W, H, W]

    # flow from softmax
    init_grid = coords_grid(b, h, w).to(correlation.device)  # [B, 2, H, W]
    grid = init_grid.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    correlation = correlation.view(b, h * w, h * w)  # [B, H*W, H*W]

    if pred_bidir_flow:
        correlation = torch.cat((correlation, correlation.permute(0, 2, 1)), dim=0)  # [2*B, H*W, H*W]
        init_grid = init_grid.repeat(2, 1, 1, 1)  # [2*B, 2, H, W]
        grid = grid.repeat(2, 1, 1)  # [2*B, H*W, 2]
        b = b * 2

    prob = F.softmax(correlation, dim=-1)  # [B, H*W, H*W]

    correspondence = torch.matmul(prob, grid).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    # when predicting bidirectional flow, flow is the concatenation of forward flow and backward flow
    flow = correspondence - init_grid

    return flow, prob




# --- 1. 原始代码所需的辅助函数 ---
# --- 3. 新的辅助函数：用于拼接 ---
def stitch_flow(local_flow_patches, b, l, h, w, p, s, pad):
    """
    将 patch 化的 flow 拼接回完整的 flow 场。
    local_flow_patches: [B*L, 2, P, P]
    b, l, h, w, p, s, pad: batch, num_patches, height, width, patch_size, stride, padding
    """
    # 3.1. 将 flow patches 重塑为 F.fold 可以理解的格式
    # [B*L, 2, P, P] -> [B, L, 2*P*P] -> [B, 2*P*P, L]
    local_flow_unfolded = local_flow_patches.view(b, l, 2 * p * p).permute(0, 2, 1)
    # 3.2. 使用 F.fold 拼接
    # F.fold 会自动将重叠区域的值 *相加*
    output_flow = F.fold(local_flow_unfolded, (h, w), kernel_size=p, stride=s, padding=pad)
    # 3.3. 创建一个归一化掩码 (Normalization Mask)
    # 这是最关键的一步：处理重叠区域的平均
    # 我们创建一个ones张量, 然后用同样的unfold/fold流程处理它
    # 这样 fold 之后，每个像素的值就是它被重叠的次数
    ones = torch.ones(b, 1, h, w).to(local_flow_patches.device)
    # [B, 1, H, W] -> [B, 1*P*P, L]
    ones_patches = F.unfold(ones, kernel_size=p, stride=s, padding=pad)
    # [B, 1*P*P, L] -> [B, 1, H, W]
    norm_mask = F.fold(ones_patches, (h, w), kernel_size=p, stride=s, padding=pad)
    # 3.4. 归一化 Flow
    # 将相加的 flow 除以重叠次数，得到平均 flow
    final_flow = output_flow / norm_mask.clamp(min=1e-6)
    return final_flow


# --- 4. 你要求的新函数 ---
def patched_correlation_softmax(feature0, feature1,
                                patch_size=32,
                                pred_bidir_flow=False,
                                ):
    """
    使用 patch 和 50% 重叠来计算相关性和 flow，以优化内存和速度。
    """
    b, c, h, w = feature0.shape
    p = patch_size      # Patch size (P)
    s = p // 2          # Stride (S) (50% overlap)
    pad = p // 2        # Padding (P/2)

    # 1. 将 feature0 和 feature1 "拆分" (unfold) 成 patch
    # F.unfold 会创建 [B, C*P*P, L] 形状的张量, L 是 patch 的数量
    patches0_unfolded = F.unfold(feature0, kernel_size=p, stride=s, padding=pad)
    patches1_unfolded = F.unfold(feature1, kernel_size=p, stride=s, padding=pad)
    # L =  patch 的数量
    l = patches0_unfolded.shape[-1]
    # 2. 将 batch 维度和 patch 维度合并
    # [B, C*P*P, L] -> [B, L, C*P*P] -> [B*L, C, P, P]
    patches0 = patches0_unfolded.permute(0, 2, 1).view(b * l, c, p, p)
    patches1 = patches1_unfolded.permute(0, 2, 1).view(b * l, c, p, p)
    # 3. 在这个 "大" batch 的 patches 上计算 "局部" flow
    # 我们复用你原来的函数，但现在它是在 [B*L] 个小 patch 上并行计算
    # 注意：pred_bidir_flow 的处理需要在 stitch 之前和之后分开做
    local_flow, _ = global_correlation_softmax(patches0, patches1,
                                             pred_bidir_flow=pred_bidir_flow)
    # 4. 拼接 (Stitch)
    if pred_bidir_flow:
        # 如果是双向流，local_flow 形状为 [2*B*L, 2, P, P]
        # 我们需要先把它拆开，分别拼接
        flow_fwd, flow_bwd = torch.chunk(local_flow, 2, dim=0)
        final_flow_fwd = stitch_flow(flow_fwd, b, l, h, w, p, s, pad)
        final_flow_bwd = stitch_flow(flow_bwd, b, l, h, w, p, s, pad)
        # 最终在 batch 维上合并
        final_flow = torch.cat((final_flow_fwd, final_flow_bwd), dim=0) # [2*B, 2, H, W]
    else:
        # 如果是单向流，local_flow 形状为 [B*L, 2, P, P]
        final_flow = stitch_flow(local_flow, b, l, h, w, p, s, pad)
    # 注意：我们无法返回一个有意义的 'prob' 矩阵
    # 因为 'prob' 是局部的 [B*L, P*P, P*P]，拼接它没有意义且内存开销更大
    return final_flow, None





def local_correlation_softmax(feature0, feature1, local_radius,
                              padding_mode='zeros'):
    """
    局部窗口版 softmax 匹配（不使用 grid_sample，显存更友好）

    feature0, feature1: [B, C, H, W]
    local_radius: R，搜索窗口半径（像素）
    返回:
      flow:       [B, 2, H, W]
      match_prob: [B, H*W, K]，K = (2R+1)^2
    """
    b, c, h, w = feature0.shape
    device = feature0.device
    HW = h * w

    # 原始像素坐标网格 [B, 2, H, W]，coords[0]=x, coords[1]=y （假设与你原来的 coords_grid 一致）
    coords_init = coords_grid(b, h, w).to(device)       # [B, 2, H, W]
    coords_flat = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, HW, 2]

    # 展平成 [B, HW, C] 以便后面做矩阵乘
    feature0_flat = feature0.permute(0, 2, 3, 1).reshape(b, HW, c)  # [B, HW, C]

    # 构造局部位移列表 offsets: K x 2，格式为 (dx, dy)
    rs = local_radius
    offsets = []
    for dy in range(-rs, rs + 1):
        for dx in range(-rs, rs + 1):
            offsets.append((dx, dy))
    offsets = torch.tensor(offsets, dtype=torch.float32, device=device)  # [K, 2]
    K = offsets.shape[0]

    # 预分配相关性 & 概率
    corr = feature0.new_full((b, HW, K), -1e9)   # [B, HW, K]
    # 用于期望坐标的偏移 (dx, dy)
    # offsets: [K, 2]

    # 先算出每个像素的 (y, x) 坐标，用于生成 mask
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )  # [H, W]
    yy = yy.reshape(1, HW)  # [1, HW]
    xx = xx.reshape(1, HW)  # [1, HW]

    # 对 K 个偏移逐个计算相关性（不存 window_feature，显存小很多）
    for k, (dx, dy) in enumerate(offsets):
        dx = int(dx.item())
        dy = int(dy.item())

        # 计算在该偏移下，哪些像素是有效的（源点和目标点都在图像内）
        x2 = xx + dx
        y2 = yy + dy
        valid = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)   # [1, HW]
        valid = valid.expand(b, HW)                           # [B, HW]

        if not valid.any():
            # 完全没有有效点，就保持 corr[...,k] = -1e9 让 softmax 忽略
            continue

        # 有效点索引（扁平 idx）
        # dst 像素索引：0..HW-1
        # src 像素索引：y2*w + x2
        dst_idx = torch.arange(HW, device=device).unsqueeze(0).expand(b, HW)  # [B, HW]
        src_idx = (y2 * w + x2).long()                                        # [1, HW] → broadcast

        # 只保留有效位置
        dst_idx_valid = dst_idx[valid]    # [N_valid_total]
        src_idx_valid = src_idx.expand(b, HW)[valid]  # [N_valid_total]

        # 把 [B, HW, C] 展平到 [B*HW, C]，方便用索引
        feat0_all = feature0_flat.reshape(b * HW, c)   # [B*HW, C]
        feat1_flat = feature1.permute(0, 2, 3, 1).reshape(b * HW, c)  # [B*HW, C]

        # 对应位置点乘
        corr_vals = (feat0_all[dst_idx_valid] * feat1_flat[src_idx_valid]).sum(dim=-1)  # [N_valid_total]

        # 写回 corr 的第 k 通道
        corr_k = corr[:, :, k].reshape(-1)               # [B*HW]
        corr_k[valid.view(-1)] = corr_vals / (c ** 0.5)  # 只覆盖有效位置
        corr[:, :, k] = corr_k.view(b, HW)

    # softmax 前 corr 无效位置仍然是 -1e9，相当于 mask 掉
    prob = F.softmax(corr, dim=-1)   # [B, HW, K]

    # 根据概率和 offsets 求期望位移 Δx, Δy
    # prob: [B, HW, K], offsets: [K, 2]
    # → disp: [B, HW, 2]
    disp = torch.matmul(prob, offsets)   # [B, HW, 2]

    # correspondence = coords_init + disp
    correspondence = coords_flat + disp  # [B, HW, 2]
    correspondence = correspondence.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init  # [B, 2, H, W]，与 global 形式一致

    return flow, prob  # prob: [B, H*W, K]





def local_correlation_softmax_v3(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              chunk_size=1024,  # 新增：分块大小（按 H*W 维度分）
                              ):
    """
    feature0, feature1: [B, C, H, W]
    local_radius: R
    返回:
      flow:       [B, 2, H, W]
      match_prob: [B, H*W, (2R+1)^2]
    使用分块在 H*W 维度上循环，显著降低 grid_sample 的峰值显存占用。
    """
    device = feature0.device
    b, c, h, w = feature0.size()
    HW = h * w

    # [B, 2, H, W] 和 [B, H*W, 2]
    coords_init = coords_grid(b, h, w).to(device)          # 原始坐标
    coords_flat = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, HW, 2]

    # 局部窗口大小
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    K = local_h * local_w   # (2R+1)^2

    # 预生成 [-R,R]×[-R,R] 的偏移 grid: [K, 2] → [1,1,K,2] 方便广播
    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w,
                                       device=device)        # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(1, 1, K, 2)            # [1,1,K,2]

    # 预分配结果
    # 概率： [B, HW, K]
    match_prob = feature0.new_zeros(b, HW, K)
    # 期望坐标： [B, HW, 2]
    correspondence_flat = feature0.new_zeros(b, HW, 2)

    # 为了更方便 chunk，预先把 feature0 展平成 [B, HW, C]
    feature0_flat = feature0.permute(0, 2, 3, 1).reshape(b, HW, c)  # [B, HW, C]

    # 确保 feature1 类型和连续性适合 grid_sample
    feature1 = feature1.to(dtype=torch.float32).contiguous()

    # 分块循环：一次处理 chunk_size 个像素
    for start in range(0, HW, chunk_size):
        end = min(HW, start + chunk_size)
        hw_chunk = end - start

        # 当前 chunk 的坐标: [B, hw_chunk, 2]
        coords_chunk = coords_flat[:, start:end, :]  # [B, hw, 2]

        # 构造局部 sample coords： [B, hw_chunk, K, 2]
        sample_coords = coords_chunk.unsqueeze(2) + window_grid  # 广播: (B,hw,1,2)+(1,1,K,2)
        # 记录 softmax 用的“真实坐标”（未归一化）
        sample_coords_softmax = sample_coords  # [B, hw, K, 2]

        # 有效性 mask: [B, hw, K]
        x = sample_coords[..., 0]
        y = sample_coords[..., 1]
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

        # 归一化到 [-1,1]，供 grid_sample 使用
        sample_coords_norm = normalize_coords(sample_coords, h, w)  # [B, hw, K, 2]
        sample_coords_norm = sample_coords_norm.to(dtype=torch.float32).contiguous()

        # grid_sample: input [B, C, H, W], grid [B, H_out, W_out, 2]
        # 这里 H_out=hw_chunk, W_out=K
        window_feature = F.grid_sample(
            feature1, sample_coords_norm,
            mode='bilinear',
            padding_mode=padding_mode,
            align_corners=True
        )  # [B, C, hw, K]
        window_feature = window_feature.permute(0, 2, 1, 3)  # [B, hw, C, K]

        # 当前 chunk 的 feature0: [B, hw, 1, C]
        f0_chunk = feature0_flat[:, start:end, :].unsqueeze(2)  # [B, hw, 1, C]

        # 计算相关性: [B, hw, K]
        corr_chunk = torch.matmul(f0_chunk, window_feature).squeeze(2) / (c ** 0.5)

        # mask 无效位置
        corr_chunk[~valid] = -1e9

        # softmax 得到匹配概率: [B, hw, K]
        prob_chunk = F.softmax(corr_chunk, dim=-1)

        # 保存概率
        match_prob[:, start:end, :] = prob_chunk

        # 计算期望坐标: [B, hw, 2]
        correspondence_chunk = torch.matmul(
            prob_chunk.unsqueeze(2),           # [B, hw, 1, K]
            sample_coords_softmax              # [B, hw, K, 2]
        ).squeeze(2)                            # [B, hw, 2]

        correspondence_flat[:, start:end, :] = correspondence_chunk

    # reshape 回 [B, 2, H, W]
    correspondence = correspondence_flat.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init           # [B, 2, H, W]
    return flow, match_prob                       # match_prob: [B, H*W, K]




def local_correlation_softmax_v2(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)    # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w,
                                       device=feature0.device)            # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)          # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid                   # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]

    # --- 关键修复：确保输入 grid_sample 的两个张量都是 float32 且 contiguous ---
    feature1 = feature1.to(dtype=torch.float32).contiguous()
    sample_coords_norm = sample_coords_norm.to(dtype=torch.float32).contiguous()

    # grid_sample: input [B, C, H_in, W_in], grid [B, H_out, W_out, 2]
    # 这里 H_out = H*W, W_out = (2R+1)^2
    print('local ---> ', feature1.shape, sample_coords_norm.shape)
    window_feature = F.grid_sample(
        feature1, sample_coords_norm,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True
    ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]

    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax) \
        .squeeze(-2).view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob






def local_correlation_softmax_old(feature0, feature1, local_radius,
                              padding_mode='zeros',
                              ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1)^2, 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    print('local ---> ', feature1.shape, sample_coords_norm.shape)
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    # mask invalid locations
    corr[~valid] = -1e9

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)^2]

    correspondence = torch.matmul(prob.unsqueeze(-2), sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

    flow = correspondence - coords_init
    match_prob = prob

    return flow, match_prob
