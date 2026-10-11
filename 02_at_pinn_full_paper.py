# 0. Imports & environment
import os, math, time, json, random, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.special import ndtr
from scipy.linalg import solve_banded

try:
    from IPython.display import display
except Exception:          # allows running as a plain script
    display = print

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| device:", DEVICE)
os.makedirs("results", exist_ok=True)

MODE = "quick"          # "quick" | "medium" | "paper"
SEED = 42

_PRESETS = {
    #            stage-1 epochs, stage-2 epochs, ensemble M, baseline epochs
    "quick":  dict(E1=3000,  E2=500,  M=6,  BASE_EPOCHS=25),
    "medium": dict(E1=12000, E2=2000, M=15, BASE_EPOCHS=80),
    "paper":  dict(E1=50000, E2=5000, M=30, BASE_EPOCHS=200),   # §4.2 / §5.1
}

CFG = dict(
    **_PRESETS[MODE],
    # ---- network / optimiser (§4.3) ----
    hidden=(50, 50, 50, 50), lr1=1e-3, lr2=3e-4, lr_final_ratio=0.01,   # exponential decay to 1% of lr
    # ---- sampling (§4.3, §5.1): 150 interior, 256 boundary/terminal ----
    N_INT=150, N_TERM=128, N_BND_EACH=64,          # 128 + 64 + 64 = 256 boundary/terminal points
    S_MIN_FRAC=0.25, S_MAX_FRAC=2.0,               # domain S in [0.25K, 2K]
    KINK_STD=0.15,                                 # half of points concentrated near log K  (paper: "sampling near the payoff kink")
    T_NEAR_MATURITY=True,                          # half of the t-samples biased toward t = T
    # ---- loss weights (Eq. 4) ----
    lam_r=1.0, lam_i=1.0, lam_b=1.0,
    # ---- anchoring (Eq. 6, Alg. 1) ----
    lam_anc=1.0,
    INIT_NOISE_STD=0.0,     # >0 -> each member starts from θ^(1)+N(0,std²) (more member diversity). 0 -> Algorithm 1 exactly.
    ANCHOR_NOISE_STD=0.0,   # 0.0 -> Algorithm 1 exactly (same anchor θ^(1) for every member).  >0 -> Pearce et al. style randomised anchors.
    # ---- output mapping (§5: "bounded-output mapping") ----
    OUTPUT_MAP="bounded",   # "bounded": V = U(S,t)·sigmoid(z) ∈ (0,U), then projection onto [L,U] ; "raw": V = K·net
    # ---- market / paper experiment settings (§5.1–5.2) ----
    K=45.0, SIGMA=0.20, R=0.05, T=0.5,
    PARAM_SETS=[(45., 0.20, 0.5)],   # only 1 set for quick mode
    EVAL_S_LO=0.5, EVAL_S_HI=1.5,                  # evaluation grid S ∈ [0.5K, 1.5K] (Tables 1–2)
    REL_ERR_MIN_PRICE=1.00,                        # relative error only where true price >= $1 (avoids dividing by ~0)
    # ---- baselines (§5.3) ----
    W=60, ALLOW_SYNTHETIC=True,                   # set True ONLY to smoke-test the pipeline without Yahoo data
    DATA_DIR="data",
    # ---- optional anchor-strength study (§6.3): which λ_anc / init noise gives bands whose size matches the error? ----
    RUN_ANCHOR_SWEEP=False, ANCHOR_SWEEP_LAMBDAS=(1.0, 0.1, 0.01, 0.001), ANCHOR_SWEEP_INIT_NOISE=(0.0, 0.02, 0.05),
)
print("MODE =", MODE, {k: CFG[k] for k in ("E1", "E2", "M", "BASE_EPOCHS")})

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

def metrics(y, yhat, rel_min=None):
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    rel_min = CFG["REL_ERR_MIN_PRICE"] if rel_min is None else rel_min
    err = y - yhat
    mask = y >= rel_min
    return dict(
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        EV=float(1.0 - np.var(err, ddof=1) / np.var(y, ddof=1)),
        RelErr_pct=float(100 * np.mean(np.abs(err[mask]) / np.abs(y[mask]))) if mask.any() else float("nan"),
        MaxAbsErr=float(np.max(np.abs(err))),
    )

def bs_price(S, K, tau, r, sigma, kind):
    """Closed-form European price. kind: 'call' | 'put'. tau = time to maturity."""
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
        # spatial operator A (interior rows): A V = a(V+ - 2V + V-) + b(V+ - V-) - rV
        lower, diag, upper = (a - b) * np.ones(N), (-2 * a - r) * np.ones(N), (a + b) * np.ones(N)

        def step(Vold, dt, tau_new):
            ab = np.zeros((3, N))
            ab[0, 1:] = -dt * upper[:-1]            # super-diagonal
            ab[1, :] = 1.0 - dt * diag
            ab[2, :-1] = -dt * lower[1:]            # sub-diagonal
            # Dirichlet boundaries
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
        steps = [dt / 2] * 4 + [dt] * (M - 2)       # Rannacher start-up
        V, tau = payoff.copy(), 0.0
        self.taus, self.Vs = [0.0], [V.copy()]
        for h in steps:
            tau += h; V = step(V, h, tau)
            self.taus.append(tau); self.Vs.append(V.copy())
        self.taus, self.Vs = np.array(self.taus), np.array(self.Vs)

    def price(self, S, tau):
        """Price at stock price(s) S and time-to-maturity tau (tau=0 -> payoff)."""
        n = int(np.argmin(np.abs(self.taus - tau)))
        return np.interp(np.asarray(S, float), self.S, self.Vs[n])

# ---- validation of the FD solver against closed form (must be tiny) ----
_fd_chk = FDSolver(45., .2, .05, .5, "put", american=False)
_S = np.linspace(30, 60, 61)
_e = np.abs(_fd_chk.price(_S, .5) - bs_price(_S, 45., .5, .05, .2, "put"))
print(f"FD vs Black–Scholes (European put): max abs error = {_e.max():.2e}   (should be << 1e-2)")
_fd_am = FDSolver(45., .2, .05, .5, "put", american=True)
_ea = _fd_am.price(_S, .5) - bs_price(_S, 45., .5, .05, .2, "put")
print(f"American − European put: min = {_ea.min():.2e} (must be ≥ 0), max = {_ea.max():.3f} (early-exercise premium)")
assert _e.max() < 1e-2 and _ea.min() > -1e-6, "FD reference failed validation"

@dataclass
class Spec:
    kind: str          # 'european_call' | 'european_put' | 'american_put'
    K: float; sigma: float; r: float; T: float
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
    """No-arbitrage bounds L(S,t) <= V <= U(S,t) (non-dividend)."""
    tau = spec.T - t
    disc = spec.K * torch.exp(-spec.r * tau)
    if spec.kind == "european_call": return torch.relu(S - disc), S
    if spec.kind == "european_put":  return torch.relu(disc - S), disc
    return torch.relu(spec.K - S), torch.full_like(S, spec.K)          # american put

class BatchedMLP(nn.Module):
    """M independent fully-connected tanh networks evaluated in one batched pass (Glorot-uniform init, zero bias)."""
    def __init__(self, M, sizes, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W, self.b = nn.ParameterList(), nn.ParameterList()
        for a, b in zip(sizes[:-1], sizes[1:]):
            lim = math.sqrt(6.0 / (a + b))
            self.W.append(nn.Parameter((torch.rand(M, a, b, generator=g) * 2 - 1) * lim))
            self.b.append(nn.Parameter(torch.zeros(M, 1, b)))
    def forward(self, h):                                   # h: [M, N, in]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            h = torch.tanh(torch.baddbmm(b, h, W))
        return torch.baddbmm(self.b[-1], h, self.W[-1])
    @property
    def n_params_per_model(self):
        return sum(p[0].numel() for p in list(self.W) + list(self.b))

class Pricer(nn.Module):
    """V̂_θ(x=log S, t) for M ensemble members, with the paper's bounded-output mapping."""
    def __init__(self, spec, M=1, seed=0):
        super().__init__()
        self.spec = spec
        self.net = BatchedMLP(M, (2,) + tuple(CFG["hidden"]) + (1,), seed)
        self.xc = 0.5 * (math.log(spec.Smin) + math.log(spec.Smax))
        self.xh = 0.5 * (math.log(spec.Smax) - math.log(spec.Smin))
    def raw(self, x, t):
        inp = torch.cat([(x - self.xc) / self.xh, 2.0 * t / self.spec.T - 1.0], dim=-1)
        return self.net(inp)
    def forward(self, x, t):                                # un-projected network value
        sp = self.spec; z = self.raw(x, t)
        if CFG["OUTPUT_MAP"] == "raw":
            return sp.K * z
        S = torch.exp(x); _, U = bounds_t(sp, S, t)
        return U * torch.sigmoid(z)
    def project(self, V, x, t):
        """Final no-arbitrage projection V ∈ [L, U] (American: V ≥ payoff)."""
        S = torch.exp(x); L, U = bounds_t(self.spec, S, t)
        return torch.maximum(torch.minimum(V, U), L)

def sample_points(spec, seed):
    """Fixed training sets D_r, D_i, D_b (150 interior, 128 terminal, 64+64 boundary)."""
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
        return torch.cat([T * torch.rand(nu, 1, generator=g), T * torch.sqrt(torch.rand(n - nu, 1, generator=g))])
    P = dict(
        int=(sx(CFG["N_INT"]), st(CFG["N_INT"])),
        term=(sx(CFG["N_TERM"]), torch.full((CFG["N_TERM"], 1), T)),
        low=(torch.full((CFG["N_BND_EACH"], 1), xmin), T * torch.rand(CFG["N_BND_EACH"], 1, generator=g)),
        high=(torch.full((CFG["N_BND_EACH"], 1), xmax), T * torch.rand(CFG["N_BND_EACH"], 1, generator=g)),
    )
    return P

def stack_points(spec, seeds):
    sets = [sample_points(spec, s) for s in seeds]
    return {k: tuple(torch.stack([s[k][i] for s in sets]).to(DEVICE) for i in range(2)) for k in sets[0]}

def pde_residual(model, x, t):
    """R_BS[V̂] in (log S,t) form, plus V̂ (Eq. 3). Derivatives by automatic differentiation."""
    x = x.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
    V = model(x, t)
    Vt, Vx = torch.autograd.grad(V.sum(), (t, x), create_graph=True)
    Vxx = torch.autograd.grad(Vx.sum(), x, create_graph=True)[0]
    sp = model.spec
    R = Vt + 0.5 * sp.sigma ** 2 * Vxx + (sp.r - 0.5 * sp.sigma ** 2) * Vx - sp.r * V
    return R, V

def pinn_loss(model, pts, per_member=True):
    """Eq. 4 (all quantities scaled by K so λ's are O(1)); returns tensor [M] (+ components)."""
    sp = model.spec; K = sp.K
    # --- interior: Black–Scholes residual ---
    xi, ti = pts["int"]; R, V = pde_residual(model, xi, ti); R = R / K
    Si = torch.exp(xi)
    if sp.american:
        pay = payoff_t(sp, Si)
        cont = (V.detach() > pay + 1e-3 * K).float()          # continuation region: V strictly above the obstacle
        # continuation: R = 0 ;   exercise region: -(L V) >= 0  i.e. R <= 0  (penalise R > 0)
        Lr = (cont * R ** 2 + (1 - cont) * torch.relu(R) ** 2).mean(dim=(1, 2))
        Lr = Lr + (torch.relu(pay - V) / K).pow(2).mean(dim=(1, 2))      # obstacle  V >= payoff
    else:
        Lr = (R ** 2).mean(dim=(1, 2))
    # --- terminal: V(S,T) = payoff ---
    xt, tt = pts["term"]
    Li = ((model(xt, tt) - payoff_t(sp, torch.exp(xt))) / K).pow(2).mean(dim=(1, 2))
    # --- boundaries ---
    xl, tl = pts["low"]; xh, th = pts["high"]
    Sl, Sh = torch.exp(xl), torch.exp(xh)
    disc_l, disc_h = sp.K * torch.exp(-sp.r * (sp.T - tl)), sp.K * torch.exp(-sp.r * (sp.T - th))
    if sp.kind == "european_call":   hl, hh = torch.zeros_like(Sl), Sh - disc_h
    elif sp.kind == "european_put":  hl, hh = disc_l - Sl, torch.zeros_like(Sh)
    else:                            hl, hh = sp.K - Sl, torch.zeros_like(Sh)
    Lb = 0.5 * (((model(xl, tl) - hl) / K).pow(2).mean(dim=(1, 2)) + ((model(xh, th) - hh) / K).pow(2).mean(dim=(1, 2)))
    total = CFG["lam_r"] * Lr + CFG["lam_i"] * Li + CFG["lam_b"] * Lb
    return total, (Lr.detach(), Li.detach(), Lb.detach())

@torch.no_grad()
def predict(model, S, t, project=True):
    """Predictions of every ensemble member at stock prices S (np array) and time t (float). Returns [M, len(S)]."""
    M = model.net.W[0].shape[0]
    x = torch.log(torch.tensor(np.asarray(S), dtype=torch.float32, device=DEVICE)).view(1, -1, 1).expand(M, -1, -1)
    tt = torch.full_like(x, float(t))
    V = model(x, tt)
    if project: V = model.project(V, x, tt)
    return V.squeeze(-1).cpu().numpy()

def _flat_params(model):
    return list(model.net.W) + list(model.net.b)

def train(model, pts_fn, epochs, lr, anchor=None, tag="", log_every=None, lam_anc=None):
    """Adam + exponential LR decay. `anchor` (list of tensors like the parameters) switches on the Eq.-6 penalty."""
    lam_anc = CFG["lam_anc"] if lam_anc is None else lam_anc
    params = _flat_params(model)
    opt = torch.optim.Adam(params, lr=lr)
    gamma = CFG["lr_final_ratio"] ** (1.0 / epochs)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    pts = pts_fn()
    Ntheta = model.net.n_params_per_model
    log_every = log_every or max(1, epochs // 10)
    hist, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        tot, (Lr, Li, Lb) = pinn_loss(model, pts)               # [M]
        if anchor is not None:
            pen = sum(((p - a) ** 2).sum(dim=tuple(range(1, p.dim()))) for p, a in zip(params, anchor))   # [M]
            tot_all = tot + lam_anc / Ntheta * pen
        else:
            pen, tot_all = torch.zeros_like(tot), tot
        tot_all.sum().backward()
        opt.step(); sched.step()
        if ep == 1 or ep % log_every == 0 or ep == epochs:
            rec = (ep, tot.mean().item(), Lr.mean().item(), Li.mean().item(), Lb.mean().item(), pen.mean().item())
            hist.append(rec)
            el = time.time() - t0
            print(f"{tag} ep {ep:6d}/{epochs} | L {rec[1]:.3e} (res {rec[2]:.2e} term {rec[3]:.2e} bnd {rec[4]:.2e}) | anchor-pen {rec[5]:.2e} | {el:6.0f}s")
    return pd.DataFrame(hist, columns=["epoch", "loss", "res", "term", "bnd", "anchor_pen"])

def run_stage1(spec, E1=None, seed=SEED, tag=""):
    """Algorithm 1, Stage 1: train one PINN with L_PINN (Eq. 4) -> θ^(1)."""
    E1 = E1 or CFG["E1"]; t0 = time.time()
    m1 = Pricer(spec, M=1, seed=seed).to(DEVICE)
    h1 = train(m1, lambda: stack_points(spec, [seed]), E1, CFG["lr1"], tag=f"[{tag} S1]")
    theta1 = [p.detach().clone() for p in _flat_params(m1)]
    return m1, theta1, h1, time.time() - t0

def run_stage2(spec, theta1, M=None, E2=None, seed=SEED, tag="", lam_anc=None, init_noise=None, anchor_noise=None):
    """Algorithm 1, Stage 2: M members start at θ^(1) (optionally + noise), anchored to θ^(1) (optionally + noise), own collocation sets."""
    M, E2 = M or CFG["M"], E2 or CFG["E2"]
    init_noise = CFG["INIT_NOISE_STD"] if init_noise is None else init_noise
    anchor_noise = CFG["ANCHOR_NOISE_STD"] if anchor_noise is None else anchor_noise
    t0 = time.time()
    ens = Pricer(spec, M=M, seed=seed + 1).to(DEVICE)
    g = torch.Generator().manual_seed(seed + 123)
    with torch.no_grad():
        for p, th in zip(_flat_params(ens), theta1):
            p.copy_(th.expand_as(p))
            if init_noise > 0: p.add_(init_noise * torch.randn(p.shape, generator=g).to(DEVICE))
    anchor = [th.expand(M, *th.shape[1:]).clone() for th in theta1]
    if anchor_noise > 0:                                          # optional randomised anchors (Pearce et al.)
        anchor = [a + anchor_noise * torch.randn(a.shape, generator=g).to(DEVICE) for a in anchor]
    member_seeds = [seed * 1000 + 17 * (m + 1) for m in range(M)]
    h2 = train(ens, lambda: stack_points(spec, member_seeds), E2, CFG["lr2"], anchor=anchor, tag=f"[{tag} S2]", lam_anc=lam_anc)
    return ens, h2, time.time() - t0

def run_at_pinn(spec, M=None, E1=None, E2=None, seed=SEED, tag=""):
    """Algorithm 1. Returns (stage-1 model, ensemble model with M members, histories, timings)."""
    m1, theta1, h1, t1 = run_stage1(spec, E1, seed, tag)
    ens, h2, t2 = run_stage2(spec, theta1, M, E2, seed, tag)
    return m1, ens, h1, h2, dict(stage1_s=t1, total_s=t1 + t2)

def ensemble_stats(pred):
    """pred: [M, n]  ->  ȳ (Eq. 7) and σ̂ (Eq. 8, 1/(M−1))."""
    return pred.mean(0), pred.std(0, ddof=1)

def coverage_report(y, mu, sd):
    """Empirical coverage of ȳ ± kσ̂ — a *check*, not a guarantee (the paper does not run a formal calibration study)."""
    z = np.abs(y - mu) / np.maximum(sd, 1e-12)
    return {f"coverage_{k}sigma": float(np.mean(z <= k)) for k in (1, 2)}

def true_price(spec, S, t):
    """Ground truth at (S, t): closed-form BS for European, FD reference for American."""
    tau = spec.T - t
    if spec.kind == "european_call": return bs_price(S, spec.K, tau, spec.r, spec.sigma, "call")
    if spec.kind == "european_put":  return bs_price(S, spec.K, tau, spec.r, spec.sigma, "put")
    return FDSolver(spec.K, spec.sigma, spec.r, spec.T, "put", american=True).price(S, tau)

_FD_CACHE = {}
def truth_fn(spec):
    if spec.kind != "american_put":
        return lambda S, t: true_price(spec, S, t)
    key = (spec.K, spec.sigma, spec.r, spec.T)
    if key not in _FD_CACHE: _FD_CACHE[key] = FDSolver(spec.K, spec.sigma, spec.r, spec.T, "put", american=True)
    fd = _FD_CACHE[key]
    return lambda S, t: fd.price(S, spec.T - t)

RESULTS = {}
spec_eu = Spec("european_put", CFG["K"], CFG["SIGMA"], CFG["R"], CFG["T"])
m1_eu, ens_eu, h1_eu, h2_eu, tm_eu = run_at_pinn(spec_eu, tag="EU-put")
# ── checkpoint: save immediately after training ───────────────────────────────
torch.save([m.state_dict() for m in ens_eu], "results/pinn_european_ensemble.pt")
torch.save(m1_eu.state_dict(),               "results/pinn_european_stage1.pt")
print("[checkpoint] European models saved to results/")
# ─────────────────────────────────────────────────────────────────────────────

S_grid = np.linspace(30.0, 60.0, 121)
truth = truth_fn(spec_eu)(S_grid, 0.0)
pred = predict(ens_eu, S_grid, 0.0)
mu, sd = ensemble_stats(pred)
RESULTS["fig2_european_put"] = {**metrics(truth, mu), **coverage_report(truth, mu, sd), **tm_eu,
                                "mean_sigma": float(sd.mean()),
                                "stage1_only": metrics(truth, predict(m1_eu, S_grid, 0.0)[0])}
print(json.dumps(RESULTS["fig2_european_put"], indent=1))
print("\nPaper reports (representative): MAE≈5.2e-2, RMSE≈7.1e-2, EV≈0.999, rel. err ≤ 1.2%")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(S_grid, truth, label="ground truth", lw=2)
ax[0].plot(S_grid, mu, "--", label="prediction (mean)", lw=2)
ax[0].fill_between(S_grid, mu - 2 * sd, mu + 2 * sd, alpha=.3, label="prediction ±2σ")
ax[0].set(xlabel="Stock Price S", ylabel="Option Price", title="European put, t = 0"); ax[0].legend(); ax[0].grid(alpha=.3)
err = mu - truth
ax[1].plot(S_grid, err, label="error = pred − truth"); ax[1].axhline(0, c="gray", lw=.8)
ax[1].fill_between(S_grid, err - sd, err + sd, alpha=.3, label="error ±1σ")
ax[1].set(xlabel="Stock Price S", ylabel="Error (pred − truth)", title="Signed error"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig2_european.png", dpi=150); plt.show()

spec_am = Spec("american_put", CFG["K"], CFG["SIGMA"], CFG["R"], CFG["T"])
m1_am, ens_am, h1_am, h2_am, tm_am = run_at_pinn(spec_am, tag="AM-put")
# ── checkpoint: save immediately after training ───────────────────────────────
torch.save([m.state_dict() for m in ens_am], "results/pinn_american_ensemble.pt")
torch.save(m1_am.state_dict(),               "results/pinn_american_stage1.pt")
print("[checkpoint] American models saved to results/")
# ─────────────────────────────────────────────────────────────────────────────
truth_am = truth_fn(spec_am)
S_grid_am = np.linspace(38.0, 65.0, 109)
slices = [(0.0, "t = 0"), (spec_am.T / 2, "t = T/2"), (spec_am.T, "t = T")]

fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
all_y, all_mu, per_slice = [], [], {}
for j, (tt, name) in enumerate(slices):
    y = truth_am(S_grid_am, tt); p = predict(ens_am, S_grid_am, tt); mu_, sd_ = ensemble_stats(p)
    all_y.append(y); all_mu.append(mu_); per_slice[name] = metrics(y, mu_, rel_min=0.1) | coverage_report(y, mu_, sd_)
    ax[0, j].plot(S_grid_am, y, label="ground truth (FD)"); ax[0, j].plot(S_grid_am, mu_, "--", label="prediction")
    ax[0, j].fill_between(S_grid_am, mu_ - 2 * sd_, mu_ + 2 * sd_, alpha=.3, label="prediction ±2σ")
    ax[0, j].set_title(name); ax[0, j].grid(alpha=.3)
    e_ = mu_ - y
    ax[1, j].plot(S_grid_am, e_, label="error (pred − truth)"); ax[1, j].axhline(0, c="gray", lw=.8)
    ax[1, j].fill_between(S_grid_am, e_ - sd_, e_ + sd_, alpha=.3, label="error ±1σ"); ax[1, j].set_xlabel("Stock Price S"); ax[1, j].grid(alpha=.3)
ax[0, 0].set_ylabel("Option Price"); ax[1, 0].set_ylabel("Error (pred − truth)"); ax[0, 0].legend(); ax[1, 0].legend()
plt.tight_layout(); plt.savefig("results/fig3_american.png", dpi=150); plt.show()

RESULTS["fig3_american_put"] = {**metrics(np.concatenate(all_y), np.concatenate(all_mu)), "per_slice": per_slice, **tm_am}
print(json.dumps({k: v for k, v in RESULTS["fig3_american_put"].items() if k != "per_slice"}, indent=1))
print("per-slice max |error|:", {k: round(v["MaxAbsErr"], 3) for k, v in per_slice.items()})
print("\nPaper reports: MAE≈8.05e-2, RMSE≈9.11e-2, EV≈0.999; local over-valuation near S≈K up to ≈$0.30 at t=T")

# ---- §6.3 (optional) anchor-strength study: reuses the Stage-1 model of §6.1, only Stage 2 is re-run ----
# The paper's bands are about as large as the error itself (Fig. 2). With λ_anc=1 on a K-scaled loss the quadratic pull is strong and the
# spread can be much smaller than the error. This table shows which (λ_anc, init noise) gives coverage near 68% / 95%. Enable with CFG["RUN_ANCHOR_SWEEP"]=True.
if CFG["RUN_ANCHOR_SWEEP"]:
    S_sw = np.linspace(30.0, 60.0, 121); y_sw = truth_fn(spec_eu)(S_sw, 0.0)
    theta1_eu = [p.detach().clone() for p in _flat_params(m1_eu)]
    rows = []
    for lam in CFG["ANCHOR_SWEEP_LAMBDAS"]:
        for nz in CFG["ANCHOR_SWEEP_INIT_NOISE"]:
            e_, _, _ = run_stage2(spec_eu, theta1_eu, lam_anc=lam, init_noise=nz, tag=f"sweep lam={lam} nz={nz}")
            mu_, sd_ = ensemble_stats(predict(e_, S_sw, 0.0))
            rows.append(dict(lam_anc=lam, init_noise=nz, **metrics(y_sw, mu_), mean_sigma=float(sd_.mean()), **coverage_report(y_sw, mu_, sd_)))
    SWEEP = pd.DataFrame(rows); display(SWEEP)
    best = SWEEP.iloc[(SWEEP["coverage_1sigma"] - 0.68).abs().argsort().iloc[0]]
    print("closest to nominal 1σ coverage:", dict(lam_anc=best["lam_anc"], init_noise=best["init_noise"]), "-> set these in CFG (lam_anc / INIT_NOISE_STD) and re-run.")
else:
    print("anchor sweep skipped (CFG['RUN_ANCHOR_SWEEP']=False)")

def load_yahoo():
    f_opt, f_hist = f"{CFG['DATA_DIR']}/options_clean.csv", f"{CFG['DATA_DIR']}/history_returns.csv"
    if os.path.exists(f_opt) and os.path.exists(f_hist):
        return pd.read_csv(f_opt, parse_dates=["snapshot_date"]), pd.read_csv(f_hist, parse_dates=["date"]), "yahoo"
    if CFG["ALLOW_SYNTHETIC"]:
        print("!!! SYNTHETIC DATA — pipeline smoke test only, NOT a reproduction of the paper's baseline results !!!")
        o, h = make_synthetic_data(); return o, h, "SYNTHETIC"
    raise FileNotFoundError("Run 01_download_yahoo_full.ipynb first (needs data/options_clean.csv and data/history_returns.csv).")

OPT, HIST, DATA_SOURCE = load_yahoo()
print("data source:", DATA_SOURCE, "| option rows:", len(OPT)); display(OPT.groupby(["instrument"]).size().to_frame("rows"))

# Does the Yahoo snapshot cover the four (K, sigma, T) settings of Tables 1-2? (baselines are scored there)
_chk = []
for inst in ("european_call", "american_put"):
    d_ = OPT[OPT["instrument"] == inst]
    for si_, (K_, sg_, T_) in enumerate(CFG["PARAM_SETS"], 1):
        near = d_[(d_["tau"] - T_).abs() <= (0.12 if T_ <= 0.5 else 0.25)]
        _chk.append(dict(instrument=inst, set=f"Set {si_}", T=T_, sigma=sg_, rows_near_T=len(near), rows_near_T_and_sigma=int((near["sigma"] - sg_).abs().le(0.08).sum())))
print("Data support around the paper's parameter sets:"); display(pd.DataFrame(_chk))

def build_sequences(opt, hist, W):
    """[N, W, 2] of [log_ret, rv] ending at the snapshot date of each row; NaN-filled where history is missing."""
    cache, seqs = {}, np.full((len(opt), W, 2), np.nan, dtype=np.float32)
    h = {tk: g.sort_values("date") for tk, g in hist.groupby("ticker")}
    for i, (tk, sd) in enumerate(zip(opt["ticker"].values, opt["snapshot_date"].values)):
        key = (tk, sd)
        if key not in cache:
            g = h.get(tk); arr = None
            if g is not None:
                g = g[g["date"] <= pd.Timestamp(sd)]
                if len(g) >= W: arr = g[["log_ret", "rv"]].values[-W:].astype(np.float32)
            cache[key] = arr
        if cache[key] is not None: seqs[i] = cache[key]
    return seqs

def make_dataset(instrument, seed=SEED):
    d = OPT[OPT["instrument"] == instrument].reset_index(drop=True).copy()
    d = d[(d["U"] - d["L"] > 1e-8) & (d["tau"] > 0) & (d["sigma"] > 0)].reset_index(drop=True)
    X = np.stack([np.log(d["S"] / d["K"]), d["tau"], d["sigma"]], 1).astype(np.float32)
    seq = build_sequences(d, HIST, CFG["W"])
    f = ((d["y"] - d["L"]) / (d["U"] - d["L"])).clip(1e-4, 1 - 1e-4).values
    z = np.log(f / (1 - f)).astype(np.float32)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(d)); n = len(d)
    tr, va, te = idx[: int(.8 * n)], idx[int(.8 * n): int(.9 * n)], idx[int(.9 * n):]
    # train-split moments
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    ok = ~np.isnan(seq[tr]).any(axis=(1, 2))
    if ok.any():
        flat = seq[tr][ok].reshape(-1, 2); sm, ss = flat.mean(0), flat.std(0) + 1e-8
    else:
        sm, ss = np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    seq_n = np.where(np.isnan(seq), 0.0, (seq - sm) / ss).astype(np.float32)       # missing -> train mean (fallback)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    pack = lambda ix: (T((X[ix] - xm) / xs), T(seq_n[ix]), T(z[ix]))
    return dict(train=pack(tr), val=pack(va), test=pack(te), xm=xm, xs=xs, n=n, missing_seq=float(np.isnan(seq).any(axis=(1, 2)).mean()),
                te_meta=(d.iloc[te], X[te]))

def mlp(dims, last_act=True):
    L = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        L.append(nn.Linear(a, b))
        if i < len(dims) - 2 or last_act: L.append(nn.ReLU())
    return nn.Sequential(*L)

class ANN(nn.Module):
    def __init__(self, W, f_seq=2, d=3):
        super().__init__()
        self.seq = mlp([W * f_seq, 128, 64]); self.stat = mlp([d, 32]); self.head = nn.Sequential(mlp([96, 64, 32]), nn.Linear(32, 1))
    def forward(self, xs, seq): return self.head(torch.cat([self.seq(seq.flatten(1)), self.stat(xs)], 1)).squeeze(-1)

class RNN(nn.Module):
    def __init__(self, W, f_seq=2, d=3):
        super().__init__()
        self.lstm = nn.LSTM(f_seq, 64, batch_first=True); self.stat = mlp([d, 32]); self.head = nn.Sequential(mlp([96, 64, 32]), nn.Linear(32, 1))
    def forward(self, xs, seq):
        _, (h, _) = self.lstm(seq); return self.head(torch.cat([h[-1], self.stat(xs)], 1)).squeeze(-1)

class _Tiny1D(nn.Module):
    def __init__(self, H=16):
        super().__init__(); self.net = nn.Sequential(nn.Linear(1, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 1))
    def forward(self, x): return self.net(x)

class KAN(nn.Module):
    """Kolmogorov–Arnold form u(x) ≈ Σ_i g_i( Σ_j h_ij(x_j) ) (+ linear skip). Static covariates only."""
    def __init__(self, W=None, f_seq=2, d=3, n_terms=8, H=16):
        super().__init__()
        self.d, self.n = d, n_terms
        self.h = nn.ModuleList([nn.ModuleList([_Tiny1D(H) for _ in range(d)]) for _ in range(n_terms)])
        self.g = nn.ModuleList([_Tiny1D(H) for _ in range(n_terms)]); self.skip = nn.Linear(d, 1)
    def forward(self, xs, seq=None):
        out = self.skip(xs)
        for i in range(self.n):
            inner = sum(self.h[i][j](xs[:, j:j + 1]) for j in range(self.d))
            out = out + self.g[i](inner)
        return out.squeeze(-1)

def fit_baseline(model, data, epochs, lr=1e-3, bs=256, patience=25, tag=""):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr, Str, Ztr = data["train"]; Xva, Sva, Zva = data["val"]
    best, best_state, bad, t0 = 1e9, None, 0, time.time()
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[b], Str[b]), Ztr[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = nn.functional.mse_loss(model(Xva, Sva), Zva).item()
        if v < best - 1e-6: best, bad, best_state = v, 0, {k: x.clone() for k, x in model.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    else: print('  WARNING: validation loss never improved (NaN?) - using last weights')
    print(f"  {tag:4s} trained {ep} epochs, best val MSE(z) = {best:.4f}, {time.time()-t0:.0f}s")
    return time.time() - t0

BASELINES, BASE_DATA, BASE_TIME = {}, {}, {}
for inst in ("european_call", "american_put"):
    data = make_dataset(inst); BASE_DATA[inst] = data
    print(f"\n[{inst}] rows={data['n']}, rows with missing history (fallback)={data['missing_seq']:.1%}")
    for name, cls in (("ANN", ANN), ("RNN", RNN), ("KAN", KAN)):
        set_seed(SEED)
        m = cls(CFG["W"]); BASE_TIME[(inst, name)] = fit_baseline(m, data, CFG["BASE_EPOCHS"], tag=name); BASELINES[(inst, name)] = m

@torch.no_grad()
def baseline_predict(inst, name, S, K, tau, sigma, r):
    """ŷ = L + (U−L)·sigmoid(z) on a grid of S (static covariates; sequence = train-mean fallback window)."""
    data = BASE_DATA[inst]; m = BASELINES[(inst, name)].eval()
    X = np.stack([np.log(S / K), np.full_like(S, tau), np.full_like(S, sigma)], 1).astype(np.float32)
    xs = torch.tensor((X - data["xm"]) / data["xs"], dtype=torch.float32, device=DEVICE)
    seq = torch.zeros(len(S), CFG["W"], 2, device=DEVICE)
    z = m(xs, seq).cpu().numpy()
    if inst == "european_call": L, U = np.maximum(S - K * np.exp(-r * tau), 0.0), S
    else:                       L, U = np.maximum(K - S, 0.0), np.full_like(S, K)
    return L + (U - L) / (1 + np.exp(-z))

def baseline_test_metrics(inst, name):
    """Hold-out *market-label* error of each baseline (sanity check that the baselines learned something)."""
    data = BASE_DATA[inst]; Xte, Ste, Zte = data["test"]; m = BASELINES[(inst, name)].eval()
    with torch.no_grad(): z = m(Xte, Ste).cpu().numpy()
    d, X = data["te_meta"]; L, U, y = d["L"].values, d["U"].values, d["y"].values
    return metrics(y, L + (U - L) / (1 + np.exp(-z)), rel_min=1.0)

print("Hold-out (market-label) error of baselines:")
display(pd.DataFrame({f"{i}|{n}": baseline_test_metrics(i, n) for i in ("european_call", "american_put") for n in ("ANN", "RNN", "KAN")}).T)

PARAM_SETS = CFG["PARAM_SETS"]
PAPER_ANCHOR = {   # paper Tables 1-2, "Anchor (PINN)" columns, for side-by-side comparison
    "european_call": [(4.5e-2, 6.5e-2, .93), (2.6e-1, 4.4e-1, .93), (8.5e-2, 1.3e-1, .98), (2.3e-1, 3.4e-1, .96)],
    "american_put":  [(6.7e-2, 1.2e-1, .99), (9.1e-2, 1.7e-1, .93), (9.3e-2, 1.3e-1, .98), (2.6e-1, 3.4e-1, .98)],
}
TABLES, CURVES, PINN_TIME = {}, {}, {}
for inst in ("european_call", "american_put"):
    rows = []
    for si, (K, sg, T) in enumerate(PARAM_SETS):
        spec = Spec(inst, K, sg, CFG["R"], T)
        S = np.linspace(CFG["EVAL_S_LO"] * K, CFG["EVAL_S_HI"] * K, 101)
        y = truth_fn(spec)(S, 0.0)
        if inst == "american_put" and (K, sg, T) == (CFG["K"], CFG["SIGMA"], CFG["T"]):
            ens, tm = ens_am, tm_am                                    # identical to the Fig. 3 experiment: reuse
        else:
            _, ens, _, _, tm = run_at_pinn(spec, tag=f"{inst[:3]} set{si+1}")
        mu, sd = ensemble_stats(predict(ens, S, 0.0)); PINN_TIME[(inst, si)] = tm["total_s"]
        preds = {"PINN-anchor": mu, **{n: baseline_predict(inst, n, S, K, T, sg, CFG["R"]) for n in ("KAN", "RNN", "ANN")}}
        CURVES[(inst, si)] = (S, y, preds, sd)
        row = {"set": f"Set {si+1}", "K": K, "sigma": sg, "T": T}
        for n, p in preds.items():
            m_ = metrics(y, p); row |= {f"{n} MAE": m_["MAE"], f"{n} RMSE": m_["RMSE"], f"{n} EV": m_["EV"]}
        row |= {f"PINN cov1σ": coverage_report(y, mu, sd)["coverage_1sigma"], f"PINN cov2σ": coverage_report(y, mu, sd)["coverage_2sigma"]}
        pa = PAPER_ANCHOR[inst][si]; row |= {"paper-PINN MAE": pa[0], "paper-PINN RMSE": pa[1], "paper-PINN EV": pa[2]}
        rows.append(row)
    TABLES[inst] = pd.DataFrame(rows)
    TABLES[inst].to_csv(f"results/table_{'1_european_call' if inst=='european_call' else '2_american_put'}.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:.3g}")
print("TABLE 1 — European call (ours vs. paper's PINN column)"); display(TABLES["european_call"])
print("TABLE 2 — American put  (ours vs. paper's PINN column)");  display(TABLES["american_put"])

# ---- automatic check of the paper's qualitative claims on *our* numbers ----
claims = []
for inst, T_ in TABLES.items():
    for _, r_ in T_.iterrows():
        mae = {n: r_[f"{n} MAE"] for n in ("PINN-anchor", "KAN", "RNN", "ANN")}
        claims.append(dict(instrument=inst, set=r_["set"], best_model=min(mae, key=mae.get),
                           PINN_beats_ANN=mae["PINN-anchor"] < mae["ANN"], PINN_beats_RNN=mae["PINN-anchor"] < mae["RNN"],
                           PINN_beats_KAN=mae["PINN-anchor"] < mae["KAN"]))
claims = pd.DataFrame(claims); display(claims)
print(f"PINN beats ANN in {claims.PINN_beats_ANN.mean():.0%} and RNN in {claims.PINN_beats_RNN.mean():.0%} of the 8 cases (paper: 'PINN-anchor and KAN generally outperform ANN/RNN').")
tt = pd.DataFrame({"KAN train s (european_call)": [BASE_TIME[("european_call", "KAN")]], "ANN": [BASE_TIME[("european_call", "ANN")]], "RNN": [BASE_TIME[("european_call", "RNN")]],
                   "PINN (mean total s per set)": [np.mean([v for (i, _), v in PINN_TIME.items() if i == "european_call"])]})
print("\nTraining time (paper claims KAN ≈ 10× slower than PINN-anchor — hardware/epoch dependent, see note):"); display(tt)

# ---- Fig. 4: price-vs-stock curves at t = 0 for parameter set 2 (K=45, σ=0.2, T=0.5) ----
si = 1
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for a, inst, ttl in ((ax[0], "european_call", "European (call)"), (ax[1], "american_put", "American (put)")):
    S, y, preds, sd = CURVES[(inst, si)]
    a.plot(S, y, c="pink", lw=5, label="Ground Truth", zorder=1)
    a.plot(S, preds["ANN"], c="blue", lw=2, label="ANN"); a.plot(S, preds["RNN"], ".", c="cyan", ms=5, label="RNN")
    a.plot(S, preds["KAN"], "-.", c="darkgreen", lw=2, label="Kolmogorov–Arnold")
    a.plot(S, preds["PINN-anchor"], ":", c="purple", lw=3, label="Anchor (mean)")
    a.set(xlabel="Stock Price", ylabel="Option Price", title=ttl); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/fig4_comparison.png", dpi=150); plt.show()

RESULTS["data_source"] = DATA_SOURCE; RESULTS["mode"] = MODE; RESULTS["cfg"] = {k: (v if not isinstance(v, tuple) else list(v)) for k, v in CFG.items()}
with open("results/summary.json", "w") as f: json.dump(RESULTS, f, indent=1, default=float)
# ── final model save (consolidates all trained objects) ───────────────────────
for _obj, _path, _label in [
    (ens_eu, "results/pinn_european_ensemble.pt", "ens_eu"),
    (ens_am, "results/pinn_american_ensemble.pt", "ens_am"),
    (m1_eu,  "results/pinn_european_stage1.pt",   "m1_eu"),
    (m1_am,  "results/pinn_american_stage1.pt",   "m1_am"),
]:
    try:
        _sd = [m.state_dict() for m in _obj] if isinstance(_obj, list) else _obj.state_dict()
        torch.save(_sd, _path)
        print(f"[save] {_label} -> {_path}")
    except Exception as _e:
        print(f"[save] WARN could not save {_label}: {_e}")
# ─────────────────────────────────────────────────────────────────────────────
print("MODE:", MODE, "| baseline data:", DATA_SOURCE)
if MODE != "paper": print("!! Reduced-budget run: numbers are NOT directly comparable to the paper (50k/5k epochs, M=30).")
if DATA_SOURCE != "yahoo": print("!! Baselines were trained on SYNTHETIC data — not a reproduction of the paper's baseline comparison.")
print("Saved: results/summary.json, results/table_*.csv, results/fig*.png")
print("""
Notes on interpretation (not verifiable from the paper alone):
 * Paper Tables 1 and 2 print IDENTICAL KAN rows (all four sets), although one is for European calls and one for American puts - almost certainly a copy/paste slip, so KAN numbers cannot be matched exactly for both.
 * Baseline numbers depend on the Yahoo snapshot date (quotes change daily), so they can only be compared qualitatively (PINN < RNN/ANN error), not digit-for-digit.
 * Paper's Table 1 (European call, set 2) lists PINN MAE 2.6e-1 while §5.1/Fig. 2 report MAE 5.2e-2 for the same K, σ, T — the two are not mutually consistent,
   so exact agreement with every number is not expected.
 * Coverage of ȳ±kσ̂ is reported as a diagnostic. In a development test (European put, 12k/1.5k epochs, M=8) the ensemble mean reached MAE≈0.046 but the
   ±1σ / ±2σ bands covered only ≈12% / ≈52% of grid points (nominal 68% / 95%): the spread is smaller than the error, so the bands are NOT calibrated.
   Randomised anchors (CFG['ANCHOR_NOISE_STD']=0.02–0.05) did not fix this. The paper's claim of 'calibrated' bands is therefore not supported by this implementation; treat the bands as relative indicators.
""")
