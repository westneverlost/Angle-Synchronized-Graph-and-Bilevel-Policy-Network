import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 工具函数
# =========================

def angle_to_vec(theta: torch.Tensor):
    """θ -> (cosθ, sinθ); theta:[N]"""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

def vec_to_angle(v: torch.Tensor):
    """(cosθ, sinθ)->θ; v:[N,2]"""
    return torch.atan2(v[..., 1], v[..., 0])

def apply_deltas(boxes, d):  # boxes:[N,5], d:[N,6] -> (Δx,Δy,Δw,Δh,Δcos,Δsin)
    x,y,w,h,theta = boxes.unbind(-1)
    dx,dy,dw,dh,dcos,dsin = d.unbind(-1)
    # 尺度更新（log-space 或直接残差均可，这里用相对残差）
    w_ = w * (1.0 + torch.tanh(dw))
    h_ = h * (1.0 + torch.tanh(dh))
    x_ = x + dx
    y_ = y + dy
    # 角度用向量加法避免折返
    v = angle_to_vec(theta) + torch.stack([dcos, dsin], dim=-1)
    th_ = vec_to_angle(v)
    return torch.stack([x_, y_, w_.clamp_min(1e-3), h_.clamp_min(1e-3), th_], dim=-1)

def knn_graph(centers: torch.Tensor, k: int):
    """
    简单的 GPU kNN（O(N^2) 适合 N<=400），centers:[N,2] -> idx:[N,k]
    如需更快可替换为 faiss/torch_cluster。
    """
    with torch.no_grad():
        dist = torch.cdist(centers, centers, p=2)
        idx = dist.topk(k=k+1, largest=False).indices[:, 1:]  # 去掉自己
    return idx  # [N,k]

def edge_features(boxes_i, boxes_j):
    """
    构造旋转感知的边特征 e_ij：
    [距, 尺度比, cosΔθ, sinΔθ]
    """
    ci = boxes_i[..., :2]
    cj = boxes_j[..., :2]
    wi = boxes_i[..., 2]
    hi = boxes_i[..., 3]
    wj = boxes_j[..., 2]
    hj = boxes_j[..., 3]
    thetai = boxes_i[..., 4]
    thetaj = boxes_j[..., 4]
    dist = torch.norm(ci - cj, dim=-1, keepdim=True)  # [N,k,1]
    scale = torch.log((torch.minimum(wi, hi) / torch.minimum(wj, hj)).clamp_min(1e-6)).unsqueeze(-1)
    dtheta = thetai - thetaj
    cosd = torch.cos(dtheta).unsqueeze(-1)
    sind = torch.sin(dtheta).unsqueeze(-1)
    return torch.cat([dist, scale, cosd, sind], dim=-1)  # [N,k,4]

# =========================
# 等变对齐 + 消息传递
# =========================

class SE2Align(nn.Module):
    """对邻居特征做相对角度对齐：在 (cosθ,sinθ) 子空间旋转；其余通道不变。"""
    def __init__(self, feat_dim: int):
        super().__init__()
        assert feat_dim >= 2, "roi_feat 前两个通道预留为角编码/显式取向槽更稳"
        self.feat_dim = feat_dim

    def forward(self, nbr_feat, dtheta):
        """
        nbr_feat: [N,k,C], dtheta:[N,k]
        旋转前两个通道；其余通道保持
        """
        c = torch.cos(dtheta)[..., None]
        s = torch.sin(dtheta)[..., None]
        xy = nbr_feat[..., :2]
        x = xy[..., 0:1]; y = xy[..., 1:2]
        x2 =  c * x - s * y
        y2 =  s * x + c * y
        head = torch.cat([x2, y2], dim=-1)
        tail = nbr_feat[..., 2:]
        return torch.cat([head, tail], dim=-1)

class GraphMessageLayer(nn.Module):
    def __init__(self, in_dim, edge_dim=4, hid=256):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hid//2), nn.ReLU(inplace=True),
            nn.Linear(hid//2, hid//2), nn.ReLU(inplace=True))
        self.msg_mlp  = nn.Sequential(
            nn.Linear(in_dim + hid//2, hid), nn.ReLU(inplace=True),
            nn.Linear(hid, in_dim))
        self.attn = nn.Linear(in_dim*2 + hid//2, 1)
        self.align = SE2Align(in_dim)

    def forward(self, x, nbr_idx, boxes):
        """
        x: [N,C], nbr_idx:[N,k], boxes:[N,5]
        """
        N, C = x.shape
        k = nbr_idx.shape[1]
        nbr = x[nbr_idx]                     # [N,k,C]
        # 相对角对齐
        dtheta = (boxes[..., 4:5] - boxes[nbr_idx][..., 4:5]).squeeze(-1)  # [N,k]
        nbr_aligned = self.align(nbr, dtheta)
        # 边特征
        e = edge_features(boxes.unsqueeze(1).expand(-1,k,-1),
                          boxes[nbr_idx])
        e_emb = self.edge_mlp(e)             # [N,k,h/2]
        # 注意力
        q = x.unsqueeze(1).expand(-1,k,-1)
        attn_in = torch.cat([q, nbr_aligned, e_emb], dim=-1)
        alpha = F.softmax(self.attn(attn_in).squeeze(-1), dim=1)  # [N,k]
        # 消息
        msg_in = torch.cat([nbr_aligned, e_emb], dim=-1)          # [N,k,C+h/2]
        msg = self.msg_mlp(msg_in)                                # [N,k,C]
        msg = (alpha.unsqueeze(-1) * msg).sum(dim=1)              # [N,C]
        return x + msg  # 残差

class SE2ASGHead(nn.Module):
    """
    放在 decode 与 NMS 之间：
    - 选 TopN 候选 → 提供 roi_feats:[N,C] 与 boxes:[N,5]
    - 返回 refined boxes 与（可选）score/class 校正
    """
    def __init__(self, in_dim=256, depth=3, edge_dim=4):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphMessageLayer(in_dim, edge_dim=edge_dim, hid=256)
            for _ in range(depth)
        ])
        self.delta_pred = nn.Linear(in_dim, 6)   # Δx,Δy,Δw,Δh,Δcos,Δsin
        self.score_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1))                   # 置信度微调（可与原 score 融合）

    @torch.no_grad()
    def build_graph(self, boxes, k=8):
        centers = boxes[..., :2]   # [N,2]
        return knn_graph(centers, k)  # [N,k]

    def forward(self, roi_feats, boxes, scores=None, nbr_idx=None, k=8):
        """
        roi_feats:[N,C], boxes:[N,5], scores:[N] or None
        """
        x = roi_feats
        if nbr_idx is None:
            nbr_idx = self.build_graph(boxes, k=k)
        for layer in self.layers:
            x = layer(x, nbr_idx, boxes)
        delta = self.delta_pred(x)
        boxes_refined = apply_deltas(boxes, delta)
        score_delta = self.score_head(x).squeeze(-1) if scores is not None else None
        return boxes_refined, score_delta, dict(feat=x, nbr_idx=nbr_idx)

# =========================
# 损失（示例）
# =========================

def angle_sync_loss(boxes_refined, nbr_idx, weight=1.0, prior_delta=None):
    """
    角度同步：∑_ij || u(θ_i) - R(Δ̂θ_ij) u(θ_j) ||^2
    - prior_delta: [N,k]，可由局部主方向/霍夫先验提供；缺省为 0
    """
    theta = boxes_refined[..., 4]
    N,k = nbr_idx.shape
    th_i = theta.unsqueeze(1).expand(-1,k)
    th_j = theta[nbr_idx]
    if prior_delta is None:
        d_hat = torch.zeros_like(th_i)
    else:
        d_hat = prior_delta
    ui = angle_to_vec(th_i.reshape(-1)).reshape(N,k,2)
    # R(d_hat) u(θ_j)
    c = torch.cos(d_hat); s = torch.sin(d_hat)
    uj = angle_to_vec(th_j.reshape(-1)).reshape(N,k,2)
    x = c*uj[...,0] - s*uj[...,1]
    y = s*uj[...,0] + c*uj[...,1]
    uj_rot = torch.stack([x,y], dim=-1)
    loss = (ui - uj_rot).pow(2).sum(dim=-1).mean()
    return weight * loss

def smooth_l1_boxes(pred, target, beta=1.0):
    return F.smooth_l1_loss(pred, target, beta=beta, reduction='mean')

# 旋转 IoU/GIoU 的实现可调用 mmrotate 的算子；这里占位：
def rotated_iou_loss(pred_boxes, gt_boxes):
    # TODO: 替换为 mmrotate.ops.box_iou_rotated / custom CUDA
    return smooth_l1_boxes(pred_boxes, gt_boxes, beta=1.0)
