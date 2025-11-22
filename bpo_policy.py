import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =============== 动作空间离散化 ===============
# 为工程稳健，动作设成离散表；可按需扩展
R_AUG_CHOICES = [
    "uniform_0_180",      # U[0,180)
    "tri_0_pm30",         # 0/±30°混合
    "bimodal_pm15_30",    # ±15/±30°加权
]
DNMS_CHOICES = [         # 小/中/大 目标 IoU 阈值组
    (0.35,0.45,0.55),
    (0.40,0.50,0.55),
    (0.45,0.55,0.60),
]
AMTS_CHOICES = [         # (tile, stride, overlap, rot_slice)
    (768,  256, 0.33, True),
    (1024, 384, 0.25, True),
    (1024, 384, 0.25, False),
    (1280, 512, 0.20, True),
]
DPA_CHOICES = [          # (pos_rIoU, neg_rIoU)
    (0.5, 0.3),
    (0.6, 0.3),
    (0.6, 0.4),
]

# =============== 策略网络（多头离散） ===============
class MultiHeadPolicy(nn.Module):
    def __init__(self, state_dim=32, hid=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hid), nn.ReLU(True),
            nn.Linear(hid, hid), nn.ReLU(True))
        self.pi_aug  = nn.Linear(hid, len(R_AUG_CHOICES))
        self.pi_dnms = nn.Linear(hid, len(DNMS_CHOICES))
        self.pi_amts = nn.Linear(hid, len(AMTS_CHOICES))
        self.pi_dpa  = nn.Linear(hid, len(DPA_CHOICES))
        self.v_head  = nn.Linear(hid, 1)

    def forward(self, s):
        h = self.backbone(s)
        logits = {
            "aug": self.pi_aug(h),
            "dnms": self.pi_dnms(h),
            "amts": self.pi_amts(h),
            "dpa": self.pi_dpa(h)
        }
        v = self.v_head(h).squeeze(-1)
        return logits, v

    def sample(self, s):
        logits, v = self(s)
        actions = {}
        logps   = {}
        ent     = 0.0
        for k, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            actions[k] = a
            logps[k] = dist.log_prob(a)
            ent += dist.entropy().mean()
        return actions, logps, v, ent

    def log_prob(self, s, actions):
        logits, v = self(s)
        logps = {}
        ent = 0.0
        for k, lg in logits.items():
            dist = torch.distributions.Categorical(logits=lg)
            logps[k] = dist.log_prob(actions[k])
            ent += dist.entropy().mean()
        return logps, v, ent

# =============== 环境封装（需要你实现的 4 个 hook） ===============
@dataclass
class EvalMetrics:
    ap_proxy: float
    recall_small: float
    latency_ms: float

class PolicyEnv:
    """
    你需要实现下面四个 hook，使其能在训练/推理流程中即时生效：
      - set_aug(policy_choice)
      - set_dnms(th_tuple)
      - set_amts(t_s_o_rot)
      - set_dpa(pos_neg_tuple)
    另需实现 quick_eval()：在小验证子集上返回代理指标（ap_proxy/recall_small/latency）
    """
    def __init__(self, set_aug, set_dnms, set_amts, set_dpa, quick_eval, budget_ms=12.5, beta_small=0.2):
        self.set_aug = set_aug
        self.set_dnms = set_dnms
        self.set_amts = set_amts
        self.set_dpa  = set_dpa
        self.quick_eval = quick_eval
        self.budget_ms = budget_ms
        self.beta_small = beta_small
        self.lam = 0.0  # 对偶变量

    @torch.no_grad()
    def step(self, actions):
        # 应用动作
        self.set_aug(R_AUG_CHOICES[actions["aug"].item()])
        self.set_dnms(DNMS_CHOICES[actions["dnms"].item()])
        self.set_amts(AMTS_CHOICES[actions["amts"].item()])
        self.set_dpa(DPA_CHOICES[actions["dpa"].item()])
        # 评估
        m = self.quick_eval()  # -> EvalMetrics
        # 奖励：ΔAP_proxy + β * Recall_small - λ * 超预算惩罚
        # 注意：这里默认为“相对上一步”的增量由 quick_eval 内部维护；也可直接用绝对值
        penalty = max(0.0, m.latency_ms - self.budget_ms)
        reward = m.ap_proxy + self.beta_small * m.recall_small - self.lam * penalty
        # 更新对偶变量（投影到非负）
        self.lam = max(0.0, self.lam + 0.01 * penalty)
        # 状态可由 quick_eval 返回的统计拼装；这里给出示例
        state = torch.tensor([
            m.ap_proxy, m.recall_small, m.latency_ms,
            float(actions["aug"]), float(actions["dnms"]),
            float(actions["amts"]), float(actions["dpa"])
        ], dtype=torch.float32)
        return state, reward

# =============== 简易 PPO ===============
class PPO:
    def __init__(self, policy: MultiHeadPolicy, lr=3e-4, clip=0.2, vf_coef=0.5, ent_coef=1e-3):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def update(self, batch):
        s = batch["s"]            # [B, S]
        a = batch["a"]            # dict of tensors[B]
        old_logp = batch["logp"]  # dict of tensors[B]
        ret = batch["ret"]        # [B]
        adv = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std()+1e-6)

        logp, v, ent = self.policy.log_prob(s, a)
        pi_loss = 0.0
        for k in logp.keys():
            ratio = torch.exp(logp[k] - old_logp[k])
            l1 = ratio * adv
            l2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            pi_loss += -torch.min(l1, l2).mean()
        v_loss = F.mse_loss(v, ret)
        loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.opt.step()
        return dict(loss=loss.item(), pi=pi_loss.item(), v=v_loss.item(), ent=ent.item())

# =============== 训练样例（外层） ===============
def optimize_policies(env: PolicyEnv, state_dim=32, epochs=10, steps_per_epoch=128, gamma=0.99, lam_gae=0.95):
    policy = MultiHeadPolicy(state_dim=7)  # 采用简单状态 7 维；若你扩充统计量可改
    ppo = PPO(policy)

    for ep in range(epochs):
        buf = []
        s = torch.zeros(7)
        for t in range(steps_per_epoch):
            a, logps, v, _ = policy.sample(s.unsqueeze(0))
            s2, r = env.step(a)
            buf.append((s, a, logps, r, v))
            s = s2
        # GAE & returns
        rets, advs = [], []
        g = 0.0; gae = 0.0
        with torch.no_grad():
            for (_, _, _, r, v) in reversed(buf):
                g = r + gamma * g
                delta = r + gamma * 0.0 - v.item()  # 无 bootstrap，简化
                gae = delta + gamma * lam_gae * gae
                rets.append(g); advs.append(gae)
        rets = torch.tensor(list(reversed(rets)), dtype=torch.float32)
        advs = torch.tensor(list(reversed(advs)), dtype=torch.float32)
        batch = {
            "s": torch.stack([x[0] for x in buf], dim=0),
            "a": {k: torch.stack([x[1][k] for x in buf], dim=0) for k in buf[0][1].keys()},
            "logp": {k: torch.stack([x[2][k] for x in buf], dim=0) for k in buf[0][2].keys()},
            "ret": rets,
            "adv": advs
        }
        stats = ppo.update(batch)
        print(f"[PPO] epoch {ep+1}/{epochs}  loss={stats['loss']:.4f}  pi={stats['pi']:.4f}  v={stats['v']:.4f}  ent={stats['ent']:.4f}")

    return policy
