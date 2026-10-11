#!/usr/bin/env python
"""
03_activation_comparison.py
============================
Extension of the AT-PINN paper reproduction:
Compares four activation functions in the BatchedMLP pricer —
  • tanh   (original paper baseline)
  • GELU   (smooth Gaussian-gating)
  • SiLU   (Swish — self-gated, popular in modern deep nets)
  • Sine   (SIREN — periodic, theoretically suited to PDE solutions)

For each activation the script:
  1. Trains European-put and American-put PINNs (quick mode by default)
  2. Saves model weights  ->  results/act_<name>_<instrument>.pt
  3. Records MAE / RMSE / EV / training-time
  4. Plots price curves, error curves, and a summary bar-chart

Run:
    python 03_activation_comparison.py

Set ACT_MODE = "medium" or "paper" for longer / paper-accurate runs.
"""

# --- 0. imports & config ----------------------------------------------------
import os, math, time, json, random, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; remove if you want interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from scipy.special import ndtr
from scipy.linalg import solve_banded

try:
    from IPython.display import display
except Exception:
    display = print

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| device:", DEVICE)
os.makedirs("results", exist_ok=True)

# -- budget (same presets as the base script) ---------------------------------
ACT_MODE = "quick"    # "quick" | "medium" | "paper"
SEED     = 42

_PRESETS = {
    "quick":  dict(E1=3000,  E2=500,  M=6,  ),
    "medium": dict(E1=12000, E2=2000, M=15, ),
    "paper":  dict(E1=50000, E2=5000, M=30, ),
}
CFG = dict(
    **_PRESETS[ACT_MODE],
    hidden=(50, 50, 50, 50), lr1=1e-3, lr2=3e-4, lr_final_ratio=0.01,
    N_INT=150, N_TERM=128, N_BND_EACH=64,
    S_MIN_FRAC=0.25, S_MAX_FRAC=2.0,
    KINK_STD=0.15, T_NEAR_MATURITY=True,
    lam_r=1.0, lam_i=1.0, lam_b=1.0,
    lam_anc=1.0,
    INIT_NOISE_STD=0.0, ANCHOR_NOISE_STD=0.0,
    OUTPUT_MAP="bounded",
    K=45.0, SIGMA=0.20, R=0.05, T=0.5,
    EVAL_S_LO=0.5, EVAL_S_HI=1.5,
    REL_ERR_MIN_PRICE=1.00,
)
print("ACT_MODE =", ACT_MODE, {k: CFG[k] for k in ("E1", "E2", "M")})

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# --- 1. activation functions ------------------------------------------------
def _sine_act(x):
    """SIREN first-layer scale is handled by Glorot-equivalent init below."""
    return torch.sin(x)

ACTIVATIONS = {
    "tanh":  torch.tanh,
    "gelu":  torch.nn.functional.gelu,
    "silu":  torch.nn.functional.silu,    # = Swish
    "sine":  _sine_act,                   # SIREN
}

# --- 2. core utilities (copied from base script verbatim) -------------------
def metrics(y, yhat, rel_min=None):
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    rel_min = CFG["REL_ERR_MIN_PRICE"] if rel_min is None else rel_min
    err = y - yhat
    mask = y >= rel_min
    return dict(
        MAE    = float(np.mean(np.abs(err))),
        RMSE   = float(np.sqrt(np.mean(err ** 2))),
        EV     = float(1.0 - np.var(err, ddof=1) / np.var(y, ddof=1)),
        RelErr_pct = float(100 * np.mean(np.abs(err[mask]) / np.abs(y[mask]))) if mask.any() else float("nan"),
        MaxAbsErr  = float(np.max(np.abs(err))),
    )

def bs_price(S, K, tau, r, sigma, kind):
    S = np.asarray(S, float)
    if tau <= 0:
        return np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if kind == "call":
        return S * ndtr(d1) - K * np.exp(-r * tau) * ndtr(d2)
    return K * np.exp(-r * tau) * ndtr(-d2) - S * ndtr(-d1)

class FDSolver:
    def __init__(self, K, sigma, r, T, kind="put", american=False, N=2001, M=2000, lo=0.05, hi=5.0):
        self.K, self.sigma, self.r, self.T, self.kind, self.american = K, sigma, r, T, kind, american
        x = np.linspace(np.log(lo * K), np.log(hi * K), N)
        self.x, self.S = x, np.exp(x)
        dx = x[1] - x[0]
        payoff = np.maximum(self.S - K, 0.0) if kind == "call" else np.maximum(K - self.S, 0.0)
        a = 0.5 * sigma ** 2 / dx ** 2
        b = (r - 0.5 * sigma ** 2) / (2 * dx)
        lower, diag, upper = (a - b) * np.ones(N), (-2 * a - r) * np.ones(N), (a + b) * np.ones(N)

        def step(Vold, dt, tau_new):
            ab = np.zeros((3, N))
            ab[0, 1:]  = -dt * upper[:-1]
            ab[1, :]   =  1.0 - dt * diag
            ab[2, :-1] = -dt * lower[1:]
            ab[1, 0] = ab[1, -1] = 1.0; ab[0, 1] = 0.0; ab[2, -2] = 0.0
            disc = K * np.exp(-r * tau_new)
            if kind == "put":
                vlo = (K - self.S[0]) if american else disc - self.S[0]; vhi = 0.0
            else:
                vlo = 0.0; vhi = self.S[-1] - disc
            rhs = Vold.copy(); rhs[0], rhs[-1] = vlo, vhi
            if not american:
                return solve_banded((1, 1), ab, rhs)
            V = Vold.copy(); rho = 1e8
            for _ in range(30):
                active = V < payoff
                ab2 = ab.copy(); ab2[1, 1:-1] += rho * active[1:-1]
                rhs2 = rhs.copy(); rhs2[1:-1] += rho * active[1:-1] * payoff[1:-1]
                Vn = solve_banded((1, 1), ab2, rhs2)
                if np.array_equal(Vn < payoff, active): V = Vn; break
                V = Vn
            return V

        dt = T / M
        steps = [dt / 2] * 4 + [dt] * (M - 2)
        V, tau = payoff.copy(), 0.0
        self.taus, self.Vs = [0.0], [V.copy()]
        for h in steps:
            tau += h; V = step(V, h, tau)
            self.taus.append(tau); self.Vs.append(V.copy())
        self.taus, self.Vs = np.array(self.taus), np.array(self.Vs)

    def price(self, S, tau):
        n = int(np.argmin(np.abs(self.taus - tau)))
        return np.interp(np.asarray(S, float), self.S, self.Vs[n])

@dataclass
class Spec:
    kind: str; K: float; sigma: float; r: float; T: float
    Smin: float = None; Smax: float = None
    def __post_init__(self):
        self.Smin = CFG["S_MIN_FRAC"] * self.K if self.Smin is None else self.Smin
        self.Smax = CFG["S_MAX_FRAC"] * self.K if self.Smax is None else self.Smax
    @property
    def is_call(self): return self.kind.endswith("call")
    @property
    def american(self): return self.kind.startswith("american")

def payoff_t(spec, S):
    return torch.relu(S - spec.K) if spec.is_call else torch.relu(spec.K - S)

def bounds_t(spec, S, t):
    tau = spec.T - t; disc = spec.K * torch.exp(-spec.r * tau)
    if spec.kind == "european_call": return torch.relu(S - disc), S
    if spec.kind == "european_put":  return torch.relu(disc - S), disc
    return torch.relu(spec.K - S), torch.full_like(S, spec.K)

# --- 3. parametric BatchedMLP with swappable activation ---------------------
class BatchedMLP(nn.Module):
    """M independent networks evaluated in one batched pass.
    activation_fn: callable applied to every hidden layer (NOT the output).
    For SIREN the first layer uses a larger init scale (omega_0 = 30).
    """
    def __init__(self, M, sizes, seed=0, activation_fn=torch.tanh, is_siren=False):
        super().__init__()
        self.act = activation_fn
        g = torch.Generator().manual_seed(seed)
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for idx, (a, b_) in enumerate(zip(sizes[:-1], sizes[1:])):
            lim = math.sqrt(6.0 / (a + b_))
            W_init = (torch.rand(M, a, b_, generator=g) * 2 - 1) * lim
            if is_siren and idx == 0:
                # SIREN first-layer: uniform in [-1/fan_in, 1/fan_in] scaled by omega_0
                omega_0 = 30.0
                W_init = (torch.rand(M, a, b_, generator=g) * 2 - 1) / a * omega_0
            self.W.append(nn.Parameter(W_init))
            self.b.append(nn.Parameter(torch.zeros(M, 1, b_)))

    def forward(self, h):           # h: [M, N, in]
        for W, b_ in zip(self.W[:-1], self.b[:-1]):
            h = self.act(torch.baddbmm(b_, h, W))
        return torch.baddbmm(self.b[-1], h, self.W[-1])

    @property
    def n_params_per_model(self):
        return sum(p[0].numel() for p in list(self.W) + list(self.b))


class Pricer(nn.Module):
    def __init__(self, spec, M=1, seed=0, activation_fn=torch.tanh, is_siren=False):
        super().__init__()
        self.spec = spec
        self.net = BatchedMLP(M, (2,) + tuple(CFG["hidden"]) + (1,),
                              seed, activation_fn=activation_fn, is_siren=is_siren)
        self.xc = 0.5 * (math.log(spec.Smin) + math.log(spec.Smax))
        self.xh = 0.5 * (math.log(spec.Smax) - math.log(spec.Smin))

    def raw(self, x, t):
        inp = torch.cat([(x - self.xc) / self.xh, 2.0 * t / self.spec.T - 1.0], dim=-1)
        return self.net(inp)

    def forward(self, x, t):
        sp = self.spec; z = self.raw(x, t)
        if CFG["OUTPUT_MAP"] == "raw":
            return sp.K * z
        S = torch.exp(x); _, U = bounds_t(sp, S, t)
        return U * torch.sigmoid(z)

    def project(self, V, x, t):
        S = torch.exp(x); L, U = bounds_t(self.spec, S, t)
        return torch.maximum(torch.minimum(V, U), L)

# --- 4. sampling & PDE loss (unchanged from base) ---------------------------
def sample_points(spec, seed):
    g = torch.Generator().manual_seed(seed)
    xmin, xmax, xK, T = math.log(spec.Smin), math.log(spec.Smax), math.log(spec.K), spec.T
    def sx(n):
        nu = n // 2
        xu = xmin + (xmax - xmin) * torch.rand(nu, 1, generator=g)
        xk = (xK + CFG["KINK_STD"] * torch.randn(n - nu, 1, generator=g)).clamp(xmin, xmax)
        return torch.cat([xu, xk])
    def st(n):
        if not CFG["T_NEAR_MATURITY"]: return T * torch.rand(n, 1, generator=g)
        nu = n // 2
        return torch.cat([T * torch.rand(nu, 1, generator=g),
                          T * torch.sqrt(torch.rand(n - nu, 1, generator=g))])
    return dict(
        int =(sx(CFG["N_INT"]),      st(CFG["N_INT"])),
        term=(sx(CFG["N_TERM"]),     torch.full((CFG["N_TERM"], 1), T)),
        low =(torch.full((CFG["N_BND_EACH"], 1), xmin), T * torch.rand(CFG["N_BND_EACH"], 1, generator=g)),
        high=(torch.full((CFG["N_BND_EACH"], 1), xmax), T * torch.rand(CFG["N_BND_EACH"], 1, generator=g)),
    )

def stack_points(spec, seeds):
    sets = [sample_points(spec, s) for s in seeds]
    return {k: tuple(torch.stack([s[k][i] for s in sets]).to(DEVICE) for i in range(2)) for k in sets[0]}

def pde_residual(model, x, t):
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    V = model(x, t)
    Vt, Vx = torch.autograd.grad(V.sum(), (t, x), create_graph=True)
    Vxx = torch.autograd.grad(Vx.sum(), x, create_graph=True)[0]
    sp = model.spec
    R = Vt + 0.5 * sp.sigma ** 2 * Vxx + (sp.r - 0.5 * sp.sigma ** 2) * Vx - sp.r * V
    return R, V

def pinn_loss(model, pts):
    sp = model.spec; K = sp.K
    xi, ti = pts["int"]; R, V = pde_residual(model, xi, ti); R = R / K
    Si = torch.exp(xi)
    if sp.american:
        pay = payoff_t(sp, Si)
        cont = (V.detach() > pay + 1e-3 * K).float()
        Lr = (cont * R ** 2 + (1 - cont) * torch.relu(R) ** 2).mean(dim=(1, 2))
        Lr = Lr + (torch.relu(pay - V) / K).pow(2).mean(dim=(1, 2))
    else:
        Lr = (R ** 2).mean(dim=(1, 2))
    xt, tt = pts["term"]
    Li = ((model(xt, tt) - payoff_t(sp, torch.exp(xt))) / K).pow(2).mean(dim=(1, 2))
    xl, tl = pts["low"]; xh, th = pts["high"]
    Sl, Sh = torch.exp(xl), torch.exp(xh)
    disc_l = sp.K * torch.exp(-sp.r * (sp.T - tl))
    disc_h = sp.K * torch.exp(-sp.r * (sp.T - th))
    if sp.kind == "european_call":  hl, hh = torch.zeros_like(Sl), Sh - disc_h
    elif sp.kind == "european_put": hl, hh = disc_l - Sl, torch.zeros_like(Sh)
    else:                           hl, hh = sp.K - Sl, torch.zeros_like(Sh)
    Lb = 0.5 * (((model(xl, tl) - hl) / K).pow(2).mean(dim=(1, 2)) +
                ((model(xh, th) - hh) / K).pow(2).mean(dim=(1, 2)))
    total = CFG["lam_r"] * Lr + CFG["lam_i"] * Li + CFG["lam_b"] * Lb
    return total, (Lr.detach(), Li.detach(), Lb.detach())

def _flat_params(model):
    return list(model.net.W) + list(model.net.b)

@torch.no_grad()
def predict(model, S, t, project=True):
    M = model.net.W[0].shape[0]
    x  = torch.log(torch.tensor(np.asarray(S), dtype=torch.float32, device=DEVICE)).view(1, -1, 1).expand(M, -1, -1)
    tt = torch.full_like(x, float(t))
    V  = model(x, tt)
    if project: V = model.project(V, x, tt)
    return V.squeeze(-1).cpu().numpy()

def ensemble_stats(pred):
    return pred.mean(0), pred.std(0, ddof=1)

# --- 5. training (with per-activation LR warm-up for SIREN) -----------------
def train(model, pts_fn, epochs, lr, anchor=None, tag="", lam_anc=None, is_siren=False):
    lam_anc = CFG["lam_anc"] if lam_anc is None else lam_anc
    # SIREN benefits from a lower initial LR to avoid early divergence
    actual_lr = lr * 0.3 if is_siren else lr
    params  = _flat_params(model)
    opt     = torch.optim.Adam(params, lr=actual_lr)
    gamma   = CFG["lr_final_ratio"] ** (1.0 / epochs)
    sched   = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    pts     = pts_fn()
    Ntheta  = model.net.n_params_per_model
    log_every = max(1, epochs // 8)
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        tot, (Lr, Li, Lb) = pinn_loss(model, pts)
        if anchor is not None:
            pen = sum(((p - a) ** 2).sum(dim=tuple(range(1, p.dim()))) for p, a in zip(params, anchor))
            tot_all = tot + lam_anc / Ntheta * pen
        else:
            pen, tot_all = torch.zeros_like(tot), tot
        tot_all.sum().backward()
        # gradient clipping helps SIREN (large initial gradients)
        if is_siren:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step(); sched.step()
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            rec = (ep, tot.mean().item(), Lr.mean().item(), Li.mean().item(),
                   Lb.mean().item(), pen.mean().item())
            hist.append(rec)
            print(f"{tag} ep {ep:6d}/{epochs} | L {rec[1]:.3e} "
                  f"(res {rec[2]:.2e} term {rec[3]:.2e} bnd {rec[4]:.2e}) | {time.time()-t0:6.0f}s")
    return pd.DataFrame(hist, columns=["epoch","loss","res","term","bnd","anchor_pen"])

def run_at_pinn_act(spec, act_name, act_fn, is_siren=False, seed=SEED, tag=""):
    """Full AT-PINN Algorithm 1 for a given activation function."""
    M, E1, E2 = CFG["M"], CFG["E1"], CFG["E2"]
    # -- Stage 1 --------------------------------------------------------------
    t0 = time.time()
    m1 = Pricer(spec, M=1, seed=seed, activation_fn=act_fn, is_siren=is_siren).to(DEVICE)
    h1 = train(m1, lambda: stack_points(spec, [seed]), E1, CFG["lr1"],
               tag=f"[{tag}/{act_name} S1]", is_siren=is_siren)
    theta1 = [p.detach().clone() for p in _flat_params(m1)]
    t1 = time.time() - t0

    # -- Stage 2 --------------------------------------------------------------
    t0 = time.time()
    ens = Pricer(spec, M=M, seed=seed+1, activation_fn=act_fn, is_siren=is_siren).to(DEVICE)
    g = torch.Generator().manual_seed(seed + 123)
    with torch.no_grad():
        for p, th in zip(_flat_params(ens), theta1):
            p.copy_(th.expand_as(p))
    anchor = [th.expand(M, *th.shape[1:]).clone() for th in theta1]
    member_seeds = [seed * 1000 + 17 * (m_ + 1) for m_ in range(M)]
    h2 = train(ens, lambda: stack_points(spec, member_seeds), E2, CFG["lr2"],
               anchor=anchor, tag=f"[{tag}/{act_name} S2]", is_siren=is_siren)
    t2 = time.time() - t0

    # -- Save weights ----------------------------------------------------------
    tag_clean = tag.lower().replace(" ", "_")
    pt_path_ens = f"results/act_{act_name}_{tag_clean}_ensemble.pt"
    pt_path_m1  = f"results/act_{act_name}_{tag_clean}_stage1.pt"
    torch.save([m.state_dict() for m in [ens]], pt_path_ens)
    torch.save(m1.state_dict(), pt_path_m1)
    print(f"  -> saved: {pt_path_ens}  |  {pt_path_m1}")

    return m1, ens, h1, h2, dict(stage1_s=t1, total_s=t1 + t2)

# --- 6. ground truth references ---------------------------------------------
_FD_CACHE = {}
def truth_fn(spec):
    if spec.kind != "american_put":
        return lambda S, t: bs_price(S, spec.K, spec.T - t, spec.r, spec.sigma,
                                     "call" if spec.is_call else "put")
    key = (spec.K, spec.sigma, spec.r, spec.T)
    if key not in _FD_CACHE:
        _FD_CACHE[key] = FDSolver(spec.K, spec.sigma, spec.r, spec.T, "put", american=True)
    fd = _FD_CACHE[key]
    return lambda S, t: fd.price(S, spec.T - t)

# --- 7. main experiment loop -------------------------------------------------
INSTRUMENTS = {
    "european_put": (
        Spec("european_put", CFG["K"], CFG["SIGMA"], CFG["R"], CFG["T"]),
        np.linspace(30.0, 60.0, 121),
    ),
    "american_put": (
        Spec("american_put", CFG["K"], CFG["SIGMA"], CFG["R"], CFG["T"]),
        np.linspace(38.0, 65.0, 109),
    ),
}

ALL_RESULTS = {}   # act_name -> instrument -> dict

for act_name, act_fn in ACTIVATIONS.items():
    is_siren = (act_name == "sine")
    print("=" * 70)
    print(f"  ACTIVATION: {act_name.upper()}")
    print("=" * 70)
    ALL_RESULTS[act_name] = {}

    for inst_name, (spec, S_grid) in INSTRUMENTS.items():
        print(f"\n  -- {inst_name} --")
        m1, ens, h1, h2, tm = run_at_pinn_act(
            spec, act_name, act_fn, is_siren=is_siren, seed=SEED, tag=inst_name[:3]
        )
        truth = truth_fn(spec)(S_grid, 0.0)
        pred  = predict(ens, S_grid, 0.0)
        mu, sd = ensemble_stats(pred)
        m = metrics(truth, mu)
        ALL_RESULTS[act_name][inst_name] = dict(
            m1=m1, ens=ens, h1=h1, h2=h2,
            S_grid=S_grid, truth=truth, mu=mu, sd=sd,
            metrics=m, timing=tm,
        )
        print(f"  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"EV={m['EV']:.4f}  time={tm['total_s']:.0f}s")

# --- 8. comparison tables ----------------------------------------------------
print(f"\n" + "="*70)
print("  RESULTS SUMMARY")
print("="*70 + "\n")

rows = []
for inst_name in INSTRUMENTS:
    for act_name in ACTIVATIONS:
        r = ALL_RESULTS[act_name][inst_name]
        rows.append(dict(
            instrument = inst_name,
            activation = act_name,
            MAE        = r["metrics"]["MAE"],
            RMSE       = r["metrics"]["RMSE"],
            EV         = r["metrics"]["EV"],
            RelErr_pct = r["metrics"]["RelErr_pct"],
            MaxAbsErr  = r["metrics"]["MaxAbsErr"],
            train_time_s = r["timing"]["total_s"],
        ))

df = pd.DataFrame(rows)
df.to_csv("results/activation_comparison.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 20,
              "display.float_format", lambda v: f"{v:.4g}")
print("\nEuropean put:")
display(df[df.instrument == "european_put"].drop(columns="instrument").set_index("activation"))
print("\nAmerican put:")
display(df[df.instrument == "american_put"].drop(columns="instrument").set_index("activation"))

# save best activation per instrument
best = df.loc[df.groupby("instrument")["MAE"].idxmin(), ["instrument","activation","MAE","RMSE","EV"]]
print("\n Best activation per instrument (by MAE):")
display(best)

# --- 9. figures --------------------------------------------------------------
COLORS = {"tanh": "#e63946", "gelu": "#2196f3", "silu": "#4caf50", "sine": "#ff9800"}
LABELS = {"tanh": "tanh (baseline)", "gelu": "GELU", "silu": "SiLU / Swish", "sine": "Sine (SIREN)"}

for inst_name, (spec, S_grid) in INSTRUMENTS.items():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle(f"Activation function comparison — {inst_name.replace('_',' ').title()}",
                 fontsize=13, fontweight="bold")

    truth = ALL_RESULTS["tanh"][inst_name]["truth"]   # same for all

    # -- panel A: price curves -------------------------------------------------
    ax = axes[0]
    ax.plot(S_grid, truth, color="black", lw=3, label="Ground truth", zorder=5)
    for act_name in ACTIVATIONS:
        mu = ALL_RESULTS[act_name][inst_name]["mu"]
        sd = ALL_RESULTS[act_name][inst_name]["sd"]
        c  = COLORS[act_name]
        ax.plot(S_grid, mu, color=c, lw=1.8, label=LABELS[act_name])
        ax.fill_between(S_grid, mu - sd, mu + sd, color=c, alpha=0.12)
    ax.set(xlabel="Stock price S", ylabel="Option price", title="Price curves (±1σ band)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # -- panel B: absolute error -----------------------------------------------
    ax = axes[1]
    for act_name in ACTIVATIONS:
        mu = ALL_RESULTS[act_name][inst_name]["mu"]
        c  = COLORS[act_name]
        ax.plot(S_grid, np.abs(mu - truth), color=c, lw=1.8, label=LABELS[act_name])
    ax.set(xlabel="Stock price S", ylabel="|error|", title="Absolute error")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # -- panel C: bar chart — MAE by activation --------------------------------
    ax = axes[2]
    act_names = list(ACTIVATIONS.keys())
    maes = [ALL_RESULTS[a][inst_name]["metrics"]["MAE"] for a in act_names]
    bars = ax.bar(act_names, maes,
                  color=[COLORS[a] for a in act_names],
                  edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(maes) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set(xlabel="Activation", ylabel="MAE", title="MAE comparison")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = f"results/act_comparison_{inst_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

# -- training loss curves -----------------------------------------------------
for inst_name in INSTRUMENTS:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    fig.suptitle(f"Training loss — {inst_name.replace('_',' ').title()}", fontsize=12, fontweight="bold")
    for ax, stage, label in zip(axes, ["h1", "h2"], ["Stage 1", "Stage 2"]):
        for act_name in ACTIVATIONS:
            hist = ALL_RESULTS[act_name][inst_name][stage]
            ax.semilogy(hist["epoch"], hist["loss"],
                        color=COLORS[act_name], lw=1.6, label=LABELS[act_name])
        ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title=label)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = f"results/act_loss_{inst_name}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

# -- final summary -------------------------------------------------------------
print("\n" + "="*70)
print("  ALL DONE")
print(f"  Mode: {ACT_MODE}  |  Device: {DEVICE}")
print("  Outputs:")
print("    results/activation_comparison.csv")
for inst_name in INSTRUMENTS:
    print(f"    results/act_comparison_{inst_name}.png")
    print(f"    results/act_loss_{inst_name}.png")
    for act_name in ACTIVATIONS:
        print(f"    results/act_{act_name}_{inst_name[:3]}_ensemble.pt  (+ _stage1.pt)")
print(f"{'='*70}")
print(json.dumps(
    {a: {i: {k: round(v, 5) for k, v in ALL_RESULTS[a][i]["metrics"].items()}
         for i in INSTRUMENTS}
     for a in ACTIVATIONS},
    indent=2
))
