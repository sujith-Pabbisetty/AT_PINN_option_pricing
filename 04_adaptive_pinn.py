#!/usr/bin/env python
"""
04_adaptive_pinn.py
====================
Extension 2: Adaptive PINN variants vs AT-PINN baseline.

Three training strategies are compared on European-put and American-put:

  1. AT-PINN      -- Original paper: two-stage anchored ensemble (fixed weights,
                     fixed collocation, anchoring for diversity).

  2. SA-PINN      -- Self-Adaptive PINN (McClenny & Braga-Neto, 2022):
                     per-component loss weights lambda_r, lambda_i, lambda_b are
                     LEARNED jointly with the network via a max-min adversarial
                     update (network minimises, weights maximise the weighted loss).
                     Collocation points are fixed (same as AT-PINN Stage 1).

  3. RAR-PINN     -- Residual-based Adaptive Resampling PINN:
                     Every `RAR_PERIOD` epochs the interior collocation set is
                     partially replaced with points sampled proportional to the
                     squared PDE residual, concentrating effort in hard regions
                     (near the payoff kink, early-exercise boundary, etc.).

All three use the same MLP architecture (tanh, 50x50x50x50) and total epoch
budget so the comparison is fair. AT-PINN produces an ensemble (M members);
SA-PINN and RAR-PINN produce a single model (no anchoring stage).

Run:
    python 04_adaptive_pinn.py

Set ADAPT_MODE = "medium" or "paper" for longer / paper-accurate runs.
"""

# ============================================================
# 0. Imports & Config
# ============================================================
import os, math, time, json, random, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

ADAPT_MODE = "quick"   # "quick" | "medium" | "paper"
SEED = 42

_PRESETS = {
    # AT-PINN uses E1+E2 total; SA/RAR use TOTAL epochs for fair comparison
    "quick":  dict(E1=3000, E2=500,  M=6,  TOTAL=3500),
    "medium": dict(E1=12000,E2=2000, M=15, TOTAL=14000),
    "paper":  dict(E1=50000,E2=5000, M=30, TOTAL=55000),
}
CFG = dict(
    **_PRESETS[ADAPT_MODE],
    hidden=(50, 50, 50, 50), lr1=1e-3, lr2=3e-4, lr_final_ratio=0.01,
    N_INT=150, N_TERM=128, N_BND_EACH=64,
    S_MIN_FRAC=0.25, S_MAX_FRAC=2.0,
    KINK_STD=0.15, T_NEAR_MATURITY=True,
    # AT-PINN fixed weights
    lam_r=1.0, lam_i=1.0, lam_b=1.0,
    lam_anc=1.0,
    INIT_NOISE_STD=0.0, ANCHOR_NOISE_STD=0.0,
    OUTPUT_MAP="bounded",
    K=45.0, SIGMA=0.20, R=0.05, T=0.5,
    EVAL_S_LO=0.5, EVAL_S_HI=1.5,
    REL_ERR_MIN_PRICE=1.00,
    # SA-PINN specific
    SA_LR_WEIGHTS=1e-3,      # learning rate for the adaptive weight parameters
    SA_INIT_WEIGHT=1.0,      # initial value for each log-weight
    # RAR-PINN specific
    RAR_PERIOD=300,          # resample every N epochs
    RAR_FRAC=0.5,            # fraction of interior pts to replace each cycle
    RAR_CAND_MULT=10,        # candidate pool = RAR_CAND_MULT * N_INT
)
print("ADAPT_MODE =", ADAPT_MODE, {k: CFG[k] for k in ("E1","E2","M","TOTAL")})

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
set_seed(SEED)

# ============================================================
# 1. Shared utilities
# ============================================================
def metrics(y, yhat, rel_min=None):
    y, yhat = np.asarray(y, float).ravel(), np.asarray(yhat, float).ravel()
    rel_min = CFG["REL_ERR_MIN_PRICE"] if rel_min is None else rel_min
    err = y - yhat; mask = y >= rel_min
    return dict(
        MAE        = float(np.mean(np.abs(err))),
        RMSE       = float(np.sqrt(np.mean(err**2))),
        EV         = float(1.0 - np.var(err, ddof=1) / np.var(y, ddof=1)),
        RelErr_pct = float(100*np.mean(np.abs(err[mask])/np.abs(y[mask]))) if mask.any() else float("nan"),
        MaxAbsErr  = float(np.max(np.abs(err))),
    )

def bs_price(S, K, tau, r, sigma, kind):
    S = np.asarray(S, float)
    if tau <= 0:
        return np.maximum(S-K, 0.) if kind=="call" else np.maximum(K-S, 0.)
    d1 = (np.log(S/K)+(r+.5*sigma**2)*tau)/(sigma*np.sqrt(tau)); d2 = d1-sigma*np.sqrt(tau)
    if kind=="call": return S*ndtr(d1)-K*np.exp(-r*tau)*ndtr(d2)
    return K*np.exp(-r*tau)*ndtr(-d2)-S*ndtr(-d1)

class FDSolver:
    def __init__(self, K, sigma, r, T, kind="put", american=False, N=2001, M=2000, lo=0.05, hi=5.0):
        self.K,self.sigma,self.r,self.T,self.kind,self.american = K,sigma,r,T,kind,american
        x = np.linspace(np.log(lo*K),np.log(hi*K),N); self.x,self.S = x,np.exp(x)
        dx = x[1]-x[0]
        payoff = np.maximum(self.S-K,0.) if kind=="call" else np.maximum(K-self.S,0.)
        a=.5*sigma**2/dx**2; b=(r-.5*sigma**2)/(2*dx)
        lower,diag,upper=(a-b)*np.ones(N),(-2*a-r)*np.ones(N),(a+b)*np.ones(N)
        def step(Vold,dt,tau_new):
            ab=np.zeros((3,N)); ab[0,1:]=-dt*upper[:-1]; ab[1,:]=1.-dt*diag; ab[2,:-1]=-dt*lower[1:]
            ab[1,0]=ab[1,-1]=1.; ab[0,1]=0.; ab[2,-2]=0.
            disc=K*np.exp(-r*tau_new)
            vlo=(K-self.S[0]) if american else disc-self.S[0]; vhi=0.
            if kind=="call": vlo=0.; vhi=self.S[-1]-disc
            rhs=Vold.copy(); rhs[0],rhs[-1]=vlo,vhi
            if not american: return solve_banded((1,1),ab,rhs)
            V=Vold.copy(); rho=1e8
            for _ in range(30):
                active=V<payoff; ab2=ab.copy(); ab2[1,1:-1]+=rho*active[1:-1]
                rhs2=rhs.copy(); rhs2[1:-1]+=rho*active[1:-1]*payoff[1:-1]
                Vn=solve_banded((1,1),ab2,rhs2)
                if np.array_equal(Vn<payoff,active): V=Vn; break
                V=Vn
            return V
        dt=T/M; steps=[dt/2]*4+[dt]*(M-2); V,tau=payoff.copy(),0.
        self.taus,self.Vs=[0.],[V.copy()]
        for h in steps:
            tau+=h; V=step(V,h,tau); self.taus.append(tau); self.Vs.append(V.copy())
        self.taus,self.Vs=np.array(self.taus),np.array(self.Vs)
    def price(self,S,tau):
        n=int(np.argmin(np.abs(self.taus-tau))); return np.interp(np.asarray(S,float),self.S,self.Vs[n])

@dataclass
class Spec:
    kind:str; K:float; sigma:float; r:float; T:float
    Smin:float=None; Smax:float=None
    def __post_init__(self):
        self.Smin=CFG["S_MIN_FRAC"]*self.K if self.Smin is None else self.Smin
        self.Smax=CFG["S_MAX_FRAC"]*self.K if self.Smax is None else self.Smax
    @property
    def is_call(self): return self.kind.endswith("call")
    @property
    def american(self): return self.kind.startswith("american")

def payoff_t(spec, S):
    return torch.relu(S-spec.K) if spec.is_call else torch.relu(spec.K-S)

def bounds_t(spec, S, t):
    tau=spec.T-t; disc=spec.K*torch.exp(-spec.r*tau)
    if spec.kind=="european_call": return torch.relu(S-disc),S
    if spec.kind=="european_put":  return torch.relu(disc-S),disc
    return torch.relu(spec.K-S),torch.full_like(S,spec.K)

# ============================================================
# 2. Shared network (BatchedMLP / Pricer) -- tanh, same arch
# ============================================================
class BatchedMLP(nn.Module):
    def __init__(self, M, sizes, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W,self.b = nn.ParameterList(),nn.ParameterList()
        for a,b_ in zip(sizes[:-1],sizes[1:]):
            lim=math.sqrt(6.0/(a+b_))
            self.W.append(nn.Parameter((torch.rand(M,a,b_,generator=g)*2-1)*lim))
            self.b.append(nn.Parameter(torch.zeros(M,1,b_)))
    def forward(self,h):
        for W,b_ in zip(self.W[:-1],self.b[:-1]):
            h=torch.tanh(torch.baddbmm(b_,h,W))
        return torch.baddbmm(self.b[-1],h,self.W[-1])
    @property
    def n_params_per_model(self):
        return sum(p[0].numel() for p in list(self.W)+list(self.b))

class Pricer(nn.Module):
    def __init__(self, spec, M=1, seed=0):
        super().__init__(); self.spec=spec
        self.net=BatchedMLP(M,(2,)+tuple(CFG["hidden"])+(1,),seed)
        self.xc=.5*(math.log(spec.Smin)+math.log(spec.Smax))
        self.xh=.5*(math.log(spec.Smax)-math.log(spec.Smin))
    def raw(self,x,t):
        inp=torch.cat([(x-self.xc)/self.xh, 2.*t/self.spec.T-1.],dim=-1)
        return self.net(inp)
    def forward(self,x,t):
        sp=self.spec; z=self.raw(x,t)
        if CFG["OUTPUT_MAP"]=="raw": return sp.K*z
        S=torch.exp(x); _,U=bounds_t(sp,S,t); return U*torch.sigmoid(z)
    def project(self,V,x,t):
        S=torch.exp(x); L,U=bounds_t(self.spec,S,t)
        return torch.maximum(torch.minimum(V,U),L)

def _flat_params(m): return list(m.net.W)+list(m.net.b)

@torch.no_grad()
def predict(model, S, t, project=True):
    M=model.net.W[0].shape[0]
    x=torch.log(torch.tensor(np.asarray(S),dtype=torch.float32,device=DEVICE)).view(1,-1,1).expand(M,-1,-1)
    tt=torch.full_like(x,float(t)); V=model(x,tt)
    if project: V=model.project(V,x,tt)
    return V.squeeze(-1).cpu().numpy()

def ensemble_stats(pred): return pred.mean(0), pred.std(0,ddof=1)

# ============================================================
# 3. Sampling
# ============================================================
def sample_points(spec, seed, n_int=None):
    n_int=n_int or CFG["N_INT"]
    g=torch.Generator().manual_seed(seed)
    xmin,xmax,xK,T=math.log(spec.Smin),math.log(spec.Smax),math.log(spec.K),spec.T
    def sx(n):
        nu=n//2
        xu=xmin+(xmax-xmin)*torch.rand(nu,1,generator=g)
        xk=(xK+CFG["KINK_STD"]*torch.randn(n-nu,1,generator=g)).clamp(xmin,xmax)
        return torch.cat([xu,xk])
    def st(n):
        nu=n//2
        return torch.cat([T*torch.rand(nu,1,generator=g),
                          T*torch.sqrt(torch.rand(n-nu,1,generator=g))])
    return dict(
        int =(sx(n_int),           st(n_int)),
        term=(sx(CFG["N_TERM"]),   torch.full((CFG["N_TERM"],1),T)),
        low =(torch.full((CFG["N_BND_EACH"],1),xmin), T*torch.rand(CFG["N_BND_EACH"],1,generator=g)),
        high=(torch.full((CFG["N_BND_EACH"],1),xmax), T*torch.rand(CFG["N_BND_EACH"],1,generator=g)),
    )

def stack_points(spec, seeds):
    sets=[sample_points(spec,s) for s in seeds]
    return {k: tuple(torch.stack([s[k][i] for s in sets]).to(DEVICE) for i in range(2)) for k in sets[0]}

def to_device(pts):
    return {k: (a.to(DEVICE),b.to(DEVICE)) for k,(a,b) in pts.items()}

# ============================================================
# 4. PDE residual & loss builders
# ============================================================
def pde_residual(model, x, t):
    x=x.clone().requires_grad_(True); t=t.clone().requires_grad_(True)
    V=model(x,t)
    Vt,Vx=torch.autograd.grad(V.sum(),(t,x),create_graph=True)
    Vxx=torch.autograd.grad(Vx.sum(),x,create_graph=True)[0]
    sp=model.spec
    R=Vt+.5*sp.sigma**2*Vxx+(sp.r-.5*sp.sigma**2)*Vx-sp.r*V
    return R,V

def _pinn_components(model, pts):
    """Returns (Lr, Li, Lb) as [M] tensors, unweighted."""
    sp=model.spec; K=sp.K
    xi,ti=pts["int"]; R,V=pde_residual(model,xi,ti); R=R/K
    Si=torch.exp(xi)
    if sp.american:
        pay=payoff_t(sp,Si)
        cont=(V.detach()>pay+1e-3*K).float()
        Lr=(cont*R**2+(1-cont)*torch.relu(R)**2).mean(dim=(1,2))
        Lr=Lr+(torch.relu(pay-V)/K).pow(2).mean(dim=(1,2))
    else:
        Lr=(R**2).mean(dim=(1,2))
    xt,tt=pts["term"]
    Li=((model(xt,tt)-payoff_t(sp,torch.exp(xt)))/K).pow(2).mean(dim=(1,2))
    xl,tl=pts["low"]; xh,th=pts["high"]
    Sl,Sh=torch.exp(xl),torch.exp(xh)
    disc_l=sp.K*torch.exp(-sp.r*(sp.T-tl)); disc_h=sp.K*torch.exp(-sp.r*(sp.T-th))
    if sp.kind=="european_call":  hl,hh=torch.zeros_like(Sl),Sh-disc_h
    elif sp.kind=="european_put": hl,hh=disc_l-Sl,torch.zeros_like(Sh)
    else:                         hl,hh=sp.K-Sl,torch.zeros_like(Sh)
    Lb=.5*(((model(xl,tl)-hl)/K).pow(2).mean(dim=(1,2))+((model(xh,th)-hh)/K).pow(2).mean(dim=(1,2)))
    return Lr,Li,Lb

def pinn_loss_fixed(model, pts, lam_r, lam_i, lam_b):
    Lr,Li,Lb=_pinn_components(model,pts)
    total=lam_r*Lr+lam_i*Li+lam_b*Lb
    return total,(Lr.detach(),Li.detach(),Lb.detach())

# ============================================================
# 5. AT-PINN training (Algorithm 1 from paper, unchanged)
# ============================================================
def train_at(model, pts_fn, epochs, lr, anchor=None, lam_anc=None, tag=""):
    lam_anc = CFG["lam_anc"] if lam_anc is None else lam_anc
    params=_flat_params(model)
    opt=torch.optim.Adam(params,lr=lr)
    gamma=CFG["lr_final_ratio"]**(1./epochs)
    sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=gamma)
    pts=pts_fn(); Ntheta=model.net.n_params_per_model
    log_every=max(1,epochs//8); hist=[]; t0=time.time()
    for ep in range(1,epochs+1):
        opt.zero_grad(set_to_none=True)
        tot,(Lr,Li,Lb)=pinn_loss_fixed(model,pts,CFG["lam_r"],CFG["lam_i"],CFG["lam_b"])
        if anchor is not None:
            pen=sum(((p-a)**2).sum(dim=tuple(range(1,p.dim()))) for p,a in zip(params,anchor))
            tot_all=tot+lam_anc/Ntheta*pen
        else:
            pen,tot_all=torch.zeros_like(tot),tot
        tot_all.sum().backward(); opt.step(); sched.step()
        if ep==1 or ep%log_every==0 or ep==epochs:
            rec=(ep,tot.mean().item(),Lr.mean().item(),Li.mean().item(),Lb.mean().item(),pen.mean().item())
            hist.append(rec)
            print(f"{tag} ep {ep:6d}/{epochs} | L {rec[1]:.3e} "
                  f"(r {rec[2]:.2e} i {rec[3]:.2e} b {rec[4]:.2e}) | pen {rec[5]:.2e} | {time.time()-t0:.0f}s")
    return pd.DataFrame(hist,columns=["epoch","loss","res","term","bnd","anchor_pen"])

def run_at_pinn(spec, seed=SEED, tag=""):
    E1,E2,M=CFG["E1"],CFG["E2"],CFG["M"]
    # Stage 1
    t0=time.time()
    m1=Pricer(spec,M=1,seed=seed).to(DEVICE)
    h1=train_at(m1,lambda:stack_points(spec,[seed]),E1,CFG["lr1"],tag=f"[{tag}/AT S1]")
    theta1=[p.detach().clone() for p in _flat_params(m1)]
    t1=time.time()-t0
    # Stage 2
    t0=time.time()
    ens=Pricer(spec,M=M,seed=seed+1).to(DEVICE)
    with torch.no_grad():
        for p,th in zip(_flat_params(ens),theta1): p.copy_(th.expand_as(p))
    anchor=[th.expand(M,*th.shape[1:]).clone() for th in theta1]
    member_seeds=[seed*1000+17*(m+1) for m in range(M)]
    h2=train_at(ens,lambda:stack_points(spec,member_seeds),E2,CFG["lr2"],
                anchor=anchor,tag=f"[{tag}/AT S2]")
    t2=time.time()-t0
    # Save
    torch.save([m.state_dict() for m in [ens]], f"results/adapt_at_{tag}_ensemble.pt")
    torch.save(m1.state_dict(),                 f"results/adapt_at_{tag}_stage1.pt")
    return m1, ens, h1, h2, dict(stage1_s=t1, total_s=t1+t2)

# ============================================================
# 6. SA-PINN  (Self-Adaptive loss weights — McClenny 2022)
# ============================================================
class SelfAdaptiveWeights(nn.Module):
    """Three learnable log-weights: log_lam_r, log_lam_i, log_lam_b.
    Updated by gradient ASCENT (weights want to maximise the loss).
    """
    def __init__(self, init=1.0):
        super().__init__()
        # store log so we can use exp() to keep weights > 0
        v=math.log(init)
        self.log_lam_r=nn.Parameter(torch.tensor(v))
        self.log_lam_i=nn.Parameter(torch.tensor(v))
        self.log_lam_b=nn.Parameter(torch.tensor(v))

    def lambdas(self):
        return torch.exp(self.log_lam_r), torch.exp(self.log_lam_i), torch.exp(self.log_lam_b)

def train_sa(model, spec, epochs, lr, tag=""):
    """SA-PINN: jointly learn network params (min) and loss weights (max)."""
    net_params  = _flat_params(model)
    saw         = SelfAdaptiveWeights(CFG["SA_INIT_WEIGHT"]).to(DEVICE)
    # Two separate optimisers: network minimises, weights maximise
    opt_net  = torch.optim.Adam(net_params,         lr=lr)
    opt_wts  = torch.optim.Adam(saw.parameters(),   lr=CFG["SA_LR_WEIGHTS"])
    gamma    = CFG["lr_final_ratio"]**(1./epochs)
    sched    = torch.optim.lr_scheduler.ExponentialLR(opt_net, gamma=gamma)
    pts      = stack_points(spec, [SEED])
    log_every= max(1, epochs//8); hist=[]; t0=time.time()

    for ep in range(1, epochs+1):
        # ---- network step (minimise) ----
        opt_net.zero_grad(set_to_none=True)
        lam_r, lam_i, lam_b = saw.lambdas()
        Lr, Li, Lb = _pinn_components(model, pts)
        loss = (lam_r.detach()*Lr + lam_i.detach()*Li + lam_b.detach()*Lb).sum()
        loss.backward(); opt_net.step(); sched.step()

        # ---- weight step (maximise → negate gradients) ----
        opt_wts.zero_grad(set_to_none=True)
        loss_wts = -(lam_r*Lr.detach() + lam_i*Li.detach() + lam_b*Lb.detach()).sum()   # negate = ascent
        loss_wts.backward(); opt_wts.step()

        if ep==1 or ep%log_every==0 or ep==epochs:
            with torch.no_grad():
                lr_v, li_v, lb_v = [x.item() for x in saw.lambdas()]
                l_tot = (CFG["lam_r"]*Lr+CFG["lam_i"]*Li+CFG["lam_b"]*Lb).mean().item()
            rec=(ep, l_tot, Lr.mean().item(), Li.mean().item(), Lb.mean().item(),
                 lr_v, li_v, lb_v)
            hist.append(rec)
            print(f"{tag} ep {ep:6d}/{epochs} | L {rec[1]:.3e} "
                  f"(r {rec[2]:.2e} i {rec[3]:.2e} b {rec[4]:.2e}) "
                  f"| lam [{lr_v:.2f},{li_v:.2f},{lb_v:.2f}] | {time.time()-t0:.0f}s")

    return pd.DataFrame(hist, columns=["epoch","loss","res","term","bnd",
                                        "lam_r","lam_i","lam_b"]), saw

def run_sa_pinn(spec, seed=SEED, tag=""):
    t0=time.time()
    model=Pricer(spec, M=1, seed=seed).to(DEVICE)
    hist, saw = train_sa(model, spec, CFG["TOTAL"], CFG["lr1"], tag=f"[{tag}/SA]")
    elapsed=time.time()-t0
    torch.save(model.state_dict(), f"results/adapt_sa_{tag}.pt")
    lam_r, lam_i, lam_b = [x.item() for x in saw.lambdas()]
    print(f"  SA-PINN final weights: lam_r={lam_r:.3f}  lam_i={lam_i:.3f}  lam_b={lam_b:.3f}")
    return model, hist, dict(total_s=elapsed, final_lam=(lam_r, lam_i, lam_b))

# ============================================================
# 7. RAR-PINN  (Residual-based Adaptive Resampling)
# ============================================================
def _residual_magnitude(model, x, t):
    """Compute |R_BS| for candidate points (x, t have shape [1, N_cand, 1])."""
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)
        V = model(x, t)
        Vt, Vx = torch.autograd.grad(V.sum(), (t, x), create_graph=True)
        Vxx = torch.autograd.grad(Vx.sum(), x)[0]
        sp = model.spec
        R = Vt + 0.5 * sp.sigma**2 * Vxx + (sp.r - 0.5 * sp.sigma**2) * Vx - sp.r * V
        return R.detach().abs().squeeze()

def _rar_resample(spec, pts, model, seed_offset):
    """Replace RAR_FRAC of interior pts with residual-weighted samples."""
    N = CFG["N_INT"]
    n_replace = max(1, int(N * CFG["RAR_FRAC"]))
    n_cand = N * CFG["RAR_CAND_MULT"]
    g = torch.Generator(device=DEVICE).manual_seed(seed_offset)
    xmin, xmax = math.log(spec.Smin), math.log(spec.Smax); T = spec.T
    xc = xmin + (xmax - xmin) * torch.rand(1, n_cand, 1, generator=g, device=DEVICE)
    tc = T * torch.rand(1, n_cand, 1, generator=g, device=DEVICE)
    try:
        res = _residual_magnitude(model, xc, tc)
        probs = (res.float() ** 2) + 1e-8
        probs = probs / probs.sum()
        idx = torch.multinomial(probs.cpu(), n_replace, replacement=False).to(DEVICE)
    except Exception as e:
        print(f"Resampling fallback: {e}")
        return pts

    xi_old, ti_old = pts["int"]
    g_cpu = torch.Generator().manual_seed(seed_offset)
    keep_perm = torch.randperm(N, generator=g_cpu)[:(N - n_replace)].to(DEVICE)

    new_x = torch.cat([xi_old[:, keep_perm, :], xc[:, idx, :]], dim=1)
    new_t = torch.cat([ti_old[:, keep_perm, :], tc[:, idx, :]], dim=1)
    pts_new = dict(pts)
    pts_new["int"] = (new_x.detach(), new_t.detach())
    return pts_new

def train_rar(model, spec, epochs, lr, tag=""):
    """RAR-PINN: periodically replace high-residual collocation points."""
    params  =_flat_params(model)
    opt     =torch.optim.Adam(params,lr=lr)
    gamma   =CFG["lr_final_ratio"]**(1./epochs)
    sched   =torch.optim.lr_scheduler.ExponentialLR(opt,gamma=gamma)
    pts     =stack_points(spec, [SEED])
    log_every=max(1,epochs//8); hist=[]; t0=time.time()
    resample_count=0

    for ep in range(1,epochs+1):
        # Adaptive resampling
        if ep>1 and (ep-1)%CFG["RAR_PERIOD"]==0:
            pts=_rar_resample(spec,pts,model,seed_offset=SEED+ep)
            resample_count+=1

        opt.zero_grad(set_to_none=True)
        tot,(Lr,Li,Lb)=pinn_loss_fixed(model,pts,CFG["lam_r"],CFG["lam_i"],CFG["lam_b"])
        tot.sum().backward(); opt.step(); sched.step()

        if ep==1 or ep%log_every==0 or ep==epochs:
            rec=(ep,tot.mean().item(),Lr.mean().item(),Li.mean().item(),Lb.mean().item())
            hist.append(rec)
            print(f"{tag} ep {ep:6d}/{epochs} | L {rec[1]:.3e} "
                  f"(r {rec[2]:.2e} i {rec[3]:.2e} b {rec[4]:.2e}) "
                  f"| resamples={resample_count} | {time.time()-t0:.0f}s")

    return pd.DataFrame(hist,columns=["epoch","loss","res","term","bnd"])

def run_rar_pinn(spec, seed=SEED, tag=""):
    t0=time.time()
    model=Pricer(spec,M=1,seed=seed).to(DEVICE)
    hist=train_rar(model,spec,CFG["TOTAL"],CFG["lr1"],tag=f"[{tag}/RAR]")
    elapsed=time.time()-t0
    torch.save(model.state_dict(),f"results/adapt_rar_{tag}.pt")
    return model, hist, dict(total_s=elapsed)

# ============================================================
# 8. Ground truth helpers
# ============================================================
_FD_CACHE={}
def truth_fn(spec):
    if spec.kind!="american_put":
        return lambda S,t: bs_price(S,spec.K,spec.T-t,spec.r,spec.sigma,
                                    "call" if spec.is_call else "put")
    key=(spec.K,spec.sigma,spec.r,spec.T)
    if key not in _FD_CACHE:
        _FD_CACHE[key]=FDSolver(spec.K,spec.sigma,spec.r,spec.T,"put",american=True)
    fd=_FD_CACHE[key]
    return lambda S,t: fd.price(S,spec.T-t)

# ============================================================
# 9. Main experiment
# ============================================================
INSTRUMENTS={
    "european_put":(Spec("european_put",CFG["K"],CFG["SIGMA"],CFG["R"],CFG["T"]),
                    np.linspace(30.,60.,121)),
    "american_put":(Spec("american_put",CFG["K"],CFG["SIGMA"],CFG["R"],CFG["T"]),
                    np.linspace(38.,65.,109)),
}

METHODS=["AT-PINN","SA-PINN","RAR-PINN"]
ALL={}   # method -> instrument -> dict

for inst_name,(spec,S_grid) in INSTRUMENTS.items():
    truth=truth_fn(spec)(S_grid,0.0)
    ALL[inst_name]={}

    print("\n" + "="*72)
    print(f"  INSTRUMENT: {inst_name.upper()}")
    print("="*72)

    # ---- AT-PINN --------------------------------------------------------
    print("\n--- AT-PINN ---")
    m1,ens,h1,h2,tm=run_at_pinn(spec,seed=SEED,tag=inst_name[:3])
    pred=predict(ens,S_grid,0.0); mu,sd=ensemble_stats(pred)
    ALL[inst_name]["AT-PINN"]=dict(mu=mu,sd=sd,truth=truth,
        metrics=metrics(truth,mu),timing=tm,
        hist=pd.concat([h1.assign(stage=1),h2.assign(stage=2)]))
    print(f"  AT-PINN  MAE={ALL[inst_name]['AT-PINN']['metrics']['MAE']:.4f}  "
          f"time={tm['total_s']:.0f}s")

    # ---- SA-PINN --------------------------------------------------------
    print("\n--- SA-PINN ---")
    sa_model,sa_hist,sa_tm=run_sa_pinn(spec,seed=SEED,tag=inst_name[:3])
    sa_pred=predict(sa_model,S_grid,0.0); sa_mu=sa_pred[0]  # M=1
    ALL[inst_name]["SA-PINN"]=dict(mu=sa_mu,sd=np.zeros_like(sa_mu),truth=truth,
        metrics=metrics(truth,sa_mu),timing=sa_tm,hist=sa_hist)
    print(f"  SA-PINN  MAE={ALL[inst_name]['SA-PINN']['metrics']['MAE']:.4f}  "
          f"time={sa_tm['total_s']:.0f}s")

    # ---- RAR-PINN -------------------------------------------------------
    print("\n--- RAR-PINN ---")
    rar_model,rar_hist,rar_tm=run_rar_pinn(spec,seed=SEED,tag=inst_name[:3])
    rar_pred=predict(rar_model,S_grid,0.0); rar_mu=rar_pred[0]
    ALL[inst_name]["RAR-PINN"]=dict(mu=rar_mu,sd=np.zeros_like(rar_mu),truth=truth,
        metrics=metrics(truth,rar_mu),timing=rar_tm,hist=rar_hist)
    print(f"  RAR-PINN MAE={ALL[inst_name]['RAR-PINN']['metrics']['MAE']:.4f}  "
          f"time={rar_tm['total_s']:.0f}s")

# ============================================================
# 10. Results table
# ============================================================
print("\n" + "="*72)
print("  RESULTS SUMMARY")
print("="*72)

rows=[]
for inst_name in INSTRUMENTS:
    for method in METHODS:
        r=ALL[inst_name][method]
        rows.append(dict(instrument=inst_name, method=method,
            **r["metrics"], train_time_s=r["timing"]["total_s"]))

df=pd.DataFrame(rows)
df.to_csv("results/adaptive_pinn_comparison.csv",index=False)
pd.set_option("display.width",200,"display.max_columns",15,
              "display.float_format",lambda v:f"{v:.4g}")
for inst_name in INSTRUMENTS:
    print(f"\n{inst_name}:")
    display(df[df.instrument==inst_name].drop(columns="instrument").set_index("method"))

best=df.loc[df.groupby("instrument")["MAE"].idxmin(),["instrument","method","MAE","RMSE","EV"]]
print("\nBest method per instrument (by MAE):")
display(best)

# ============================================================
# 11. Figures
# ============================================================
COLORS={"AT-PINN":"#e63946","SA-PINN":"#2196f3","RAR-PINN":"#4caf50"}
METHOD_LABELS={"AT-PINN":"AT-PINN (ensemble)","SA-PINN":"SA-PINN (adaptive weights)","RAR-PINN":"RAR-PINN (adaptive sampling)"}

for inst_name,(spec,S_grid) in INSTRUMENTS.items():
    truth=ALL[inst_name]["AT-PINN"]["truth"]
    fig,axes=plt.subplots(1,3,figsize=(17,4.8))
    fig.suptitle(f"Adaptive PINN comparison -- {inst_name.replace('_',' ').title()}",
                 fontsize=13,fontweight="bold")

    # Panel A: price curves
    ax=axes[0]
    ax.plot(S_grid,truth,color="black",lw=3,label="Ground truth",zorder=5)
    for method in METHODS:
        mu=ALL[inst_name][method]["mu"]; sd=ALL[inst_name][method]["sd"]
        c=COLORS[method]
        ax.plot(S_grid,mu,color=c,lw=1.8,label=METHOD_LABELS[method])
        if sd.max()>1e-6:
            ax.fill_between(S_grid,mu-sd,mu+sd,color=c,alpha=0.15)
    ax.set(xlabel="Stock price S",ylabel="Option price",title="Price curves")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: absolute error
    ax=axes[1]
    for method in METHODS:
        mu=ALL[inst_name][method]["mu"]; c=COLORS[method]
        ax.plot(S_grid,np.abs(mu-truth),color=c,lw=1.8,label=METHOD_LABELS[method])
    ax.set(xlabel="Stock price S",ylabel="|error|",title="Absolute error")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel C: MAE bar
    ax=axes[2]
    maes=[ALL[inst_name][m]["metrics"]["MAE"] for m in METHODS]
    bars=ax.bar(METHODS,maes,color=[COLORS[m] for m in METHODS],
                edgecolor="white",linewidth=0.8,width=0.45)
    for bar,val in zip(bars,maes):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(maes)*0.01,
                f"{val:.4f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set(xlabel="Method",ylabel="MAE",title="MAE comparison"); ax.grid(axis="y",alpha=0.3)

    plt.tight_layout()
    fig_path=f"results/adapt_comparison_{inst_name}.png"
    plt.savefig(fig_path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {fig_path}")

# --- training loss curves ---
for inst_name in INSTRUMENTS:
    fig,axes=plt.subplots(1,len(METHODS),figsize=(16,4.2))
    fig.suptitle(f"Training loss -- {inst_name.replace('_',' ').title()}",
                 fontsize=12,fontweight="bold")
    for ax,method in zip(axes,METHODS):
        hist=ALL[inst_name][method]["hist"]; c=COLORS[method]
        ax.semilogy(hist["epoch"],hist["loss"],color=c,lw=1.8)
        ax.set(xlabel="Epoch",ylabel="Loss (log)",title=METHOD_LABELS[method]); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path=f"results/adapt_loss_{inst_name}.png"
    plt.savefig(fig_path,dpi=150,bbox_inches="tight"); plt.close()
    print(f"Saved: {fig_path}")

# --- SA-PINN weight evolution ---
for inst_name in INSTRUMENTS:
    sa_hist=ALL[inst_name]["SA-PINN"]["hist"]
    if all(c in sa_hist.columns for c in ["lam_r","lam_i","lam_b"]):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(sa_hist["epoch"],sa_hist["lam_r"],label="lam_r (PDE res)",color="#e63946")
        ax.plot(sa_hist["epoch"],sa_hist["lam_i"],label="lam_i (terminal)",color="#2196f3")
        ax.plot(sa_hist["epoch"],sa_hist["lam_b"],label="lam_b (boundary)",color="#4caf50")
        ax.set(xlabel="Epoch",ylabel="Adaptive weight",
               title=f"SA-PINN learned weights -- {inst_name.replace('_',' ').title()}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path=f"results/adapt_sa_weights_{inst_name}.png"
        plt.savefig(fig_path,dpi=150,bbox_inches="tight"); plt.close()
        print(f"Saved: {fig_path}")

# ============================================================
# 12. Final summary JSON
# ============================================================
print("\n" + "="*72)
print(f"  ALL DONE  |  Mode: {ADAPT_MODE}  |  Device: {DEVICE}")
print("="*72)
print(json.dumps(
    {inst: {m: {k:round(v,5) for k,v in ALL[inst][m]["metrics"].items()}
            for m in METHODS}
     for inst in INSTRUMENTS},indent=2))
