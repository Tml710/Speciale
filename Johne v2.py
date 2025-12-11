import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class BaselineParams:
    # Technology / green outcome
    a: float = 1.0          # effectiveness of effort in green signal g(e, μ)
    b: float = 1.0          # productivity of effort in financial returns R(e)

    # Cost parameters
    kappa_e: float = 1.0    # curvature of effort cost c(e) = 0.5 * kappa_e * e^2
    kappa_mu: float = 1.0   # curvature of manipulation cost h(μ) = 0.5 * kappa_mu * α * μ^2
    alpha: float = 0.5      # audit probability α ∈ [0,1]
    kappa_gamma: float = 1.0  # curvature of contractual cost γ(C) (if used)

    # Contract / dynamic parameters (baseline – no CAPM)
    delta: float = 0.9      # intertemporal discount factor δ
    rho: float = 0.3        # carry-over / SLB rebate sensitivity from period 1 to 2
    theta: float = 1.0      # warm-glow weight on green outcome θ

    # Project / contract structure
    z1_star: float = 1.0    # project quality threshold in period 1
    z2_star: float = 1.0    # project quality threshold in period 2
    x1: int = 1             # project type indicator in period 1 (1 = green implemented)
    x2: int = 1             # project type indicator in period 2
    G1_bar: float = 1.0     # benchmark/target green outcome in period 1

    # Baseline debt levels (used in D1, D2)
    D1_bar: float = 1.0
    D2_bar: float = 1.0

    # Contractual terms intensity (if you want to use γ(C))
    C1: float = 0.0
    C2: float = 0.0

def cost_effort(e, p: BaselineParams):
    """Effort cost c(e_t) = 0.5 * kappa_e * e_t^2"""
    return 0.5 * p.kappa_e * e**2


def cost_manip(mu, p: BaselineParams):
    """Manipulation cost h(μ_t) = 0.5 * kappa_mu * α * μ_t^2"""
    return 0.5 * p.kappa_mu * p.alpha * mu**2


def cost_contract(C, p: BaselineParams):
    """Contractual cost γ(C_t) = 0.5 * kappa_gamma * C_t^2"""
    return 0.5 * p.kappa_gamma * C**2


def green_signal(e, mu, p: BaselineParams):
    """g_t(e_t, μ_t) = a * e_t + μ_t"""
    return p.a * e + mu


def project_return(e, p: BaselineParams):
    """R(e_t) = b * e_t"""
    return p.b * e

def phi_t(psi_t, z_star):
    """ϕ_t = (1 - ψ_t) * z_t*"""
    return (1.0 - psi_t) * z_star


def D1(e1, mu1, psi1, p: BaselineParams):
    """D1 = D̄1 − ϕ1 x1 − ψ1 g(e1, μ1)"""
    g1 = green_signal(e1, mu1, p)
    phi1 = phi_t(psi1, p.z1_star)
    return p.D1_bar - phi1 * p.x1 - psi1 * g1


def D2(e1, mu1, e2, mu2, psi1, psi2, p: BaselineParams):
    """
    D2(ℐ1) = D̄2 − ϕ2 x2 − ψ2 g(e2, μ2) − ρ (g(e1, μ1) − Ḡ1)
    """
    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)
    phi2 = phi_t(psi2, p.z2_star)
    return p.D2_bar - phi2 * p.x2 - psi2 * g2 - p.rho * (g1 - p.G1_bar)

def xi_term(p: BaselineParams):
    """Ξ = a^2 / kappa_e + 1 / (kappa_mu * alpha)"""
    return (p.a**2) / p.kappa_e + 1.0 / (p.kappa_mu * p.alpha)


def lender_optimal_psi(p: BaselineParams):
    """Closed-form optimal ψ1*, ψ2* (baseline model without CAPM)."""
    Xi = xi_term(p)
    common = (p.a * p.b) / p.kappa_e

    psi1_star = (
        p.theta / 2.0
        - p.delta * p.rho
        + (p.z1_star * p.x1 - common) / (2.0 * Xi)
    )

    psi2_star = (
        p.theta / 2.0
        + (p.z2_star * p.x2 - common) / (2.0 * Xi)
    )

    return psi1_star, psi2_star


def borrower_optimal_e_mu(psi1, psi2, p: BaselineParams):
    """Closed-form optimal e1*, e2*, μ1*, μ2* given ψ1, ψ2."""
    e2_star = (p.b + p.a * psi2) / p.kappa_e
    mu2_star = psi2 / (p.kappa_mu * p.alpha)

    e1_star = (p.b + p.a * (psi1 + p.delta * p.rho)) / p.kappa_e
    mu1_star = (psi1 + p.delta * p.rho) / (p.kappa_mu * p.alpha)

    return e1_star, e2_star, mu1_star, mu2_star

def solve_baseline_model(p: BaselineParams):
    """
    Solve the baseline dynamic green debt model (no CAPM)
    for given parameters p.

    Returns a dictionary with:
        - psi1, psi2
        - e1, e2, mu1, mu2
        - g1, g2
        - D1, D2
        - simple borrower and lender utilities (no default region)
    """
    # 1. Lender chooses SLB intensities
    psi1, psi2 = lender_optimal_psi(p)

    # 2. Borrower best responds with optimal effort and manipulation
    e1, e2, mu1, mu2 = borrower_optimal_e_mu(psi1, psi2, p)

    # 3. Green signals
    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)

    # 4. Repayments (no default)
    D1_star = D1(e1, mu1, psi1, p)
    D2_star = D2(e1, mu1, e2, mu2, psi1, psi2, p)

    # 5. Borrower utilities per period (using your U_B structure, ignoring randomness)
    #    U_B,t = R(e_t) - D_t - c(e_t) - h(μ_t) - γ(C_t)
    UB1 = (
        project_return(e1, p)
        - D1_star
        - cost_effort(e1, p)
        - cost_manip(mu1, p)
        - cost_contract(p.C1, p)
    )
    UB2 = (
        project_return(e2, p)
        - D2_star
        - cost_effort(e2, p)
        - cost_manip(mu2, p)
        - cost_contract(p.C2, p)
    )
    UB_total = UB1 + p.delta * UB2

    # 6. Lender utilities (risk neutral, ignoring default)
    #    U_L = D1 + δ D2 + θ [g1 + δ g2]
    UL = D1_star + p.delta * D2_star + p.theta * (g1 + p.delta * g2)

    return {
        "psi1": psi1,
        "psi2": psi2,
        "e1": e1,
        "e2": e2,
        "mu1": mu1,
        "mu2": mu2,
        "g1": g1,
        "g2": g2,
        "D1": D1_star,
        "D2": D2_star,
        "UB1": UB1,
        "UB2": UB2,
        "UB_total": UB_total,
        "UL": UL,
    }

def comparative_static(param_name, values, base_params: BaselineParams):
    """
    Vary 'param_name' over 'values' and compute equilibrium outcomes.

    Returns a pandas DataFrame with one row per value.
    """
    records = []

    for v in values:
        # copy base parameters and set the chosen parameter
        p = BaselineParams(**vars(base_params))
        setattr(p, param_name, v)

        res = solve_baseline_model(p)
        res[param_name] = v
        records.append(res)

    return pd.DataFrame(records)

if __name__ == "__main__":
    # baseline parameters
    p0 = BaselineParams(
        a=1.0, b=1.0,
        kappa_e=1.0, kappa_mu=1.0, alpha=0.5,
        delta=0.9, rho=0.3, theta=1.0,
        z1_star=1.0, z2_star=1.0,
        x1=1, x2=1, G1_bar=1.0,
        D1_bar=1.0, D2_bar=1.0
    )

    theta_grid = np.linspace(0.0, 4.0, 41)
    df_theta = comparative_static("theta", theta_grid, p0)

    print(df_theta[["theta", "e1", "e2", "mu1", "mu2", "psi1", "psi2"]].head())


# Analysis of effort costs on 
kappa_e_grid = np.linspace(0.1, 3.0, 60)  # avoid 0
df_ke = comparative_static("kappa_e", kappa_e_grid, p0)

plt.figure()
plt.plot(df_ke["kappa_e"], df_ke["e1"])
plt.xlabel(r"$\kappa_e$")
plt.ylabel(r"$e_1^*$")
plt.title("Non-linear response of e1* to effort cost curvature κ_e")
plt.tight_layout()
plt.show()

# analysis of rho on borrower and lender utility
rho_grid = np.linspace(0.0, 1.0, 81)
df_rho = comparative_static("rho", rho_grid, p0)

plt.figure()
plt.plot(df_rho["rho"], df_rho["UB_total"], label="Borrower utility $U_B$")
plt.plot(df_rho["rho"], df_rho["UL"], label="Lender utility $U_L$")
plt.xlabel(r"$\rho$")
plt.ylabel("Utility")
plt.title("Borrower and lender utility vs SLB carry-over $\\rho$")
plt.legend()
plt.tight_layout()
plt.show()

# analysis of greenwashing share vs audit probability alpha
alpha_grid = np.linspace(0.05, 0.95, 50)  # avoid 0 and 1 to keep things stable
records = []

for alpha in alpha_grid:
    p = BaselineParams(**vars(p0))
    p.alpha = alpha
    res = solve_baseline_model(p)

    g1_true = p.a * res["e1"]
    g1_fake = res["mu1"]
    s1 = g1_fake / (g1_true + g1_fake)

    g2_true = p.a * res["e2"]
    g2_fake = res["mu2"]
    s2 = g2_fake / (g2_true + g2_fake)

    res_row = {
        "alpha": alpha,
        "s1": s1,
        "s2": s2,
    }
    records.append(res_row)

df_alpha = pd.DataFrame(records)

plt.figure()
plt.plot(df_alpha["alpha"], df_alpha["s1"], label="Period 1 share $s_1$")
plt.plot(df_alpha["alpha"], df_alpha["s2"], label="Period 2 share $s_2$")
plt.xlabel(r"Audit probability $\alpha$")
plt.ylabel("Share of fake green performance")
plt.title("Greenwashing share vs audit probability")
plt.legend()
plt.tight_layout()
plt.show()

# Intertemporal distortion ratio
ratio_e, ratio_mu = [], []

for rho in rho_grid:
    p = BaselineParams(**vars(p0))
    p.rho = rho
    res = solve_baseline_model(p)
    ratio_e.append(res["e1"]/res["e2"])
    ratio_mu.append(res["mu1"]/res["mu2"])

plt.figure()
plt.plot(rho_grid, ratio_e, label="e1/e2")
plt.axhline(1, color="gray", linestyle="--")
plt.xlabel(r"$\rho$")
plt.ylabel("Effort ratio")
plt.title("Intertemporal distortion in effort: e1/e2 vs ρ")
plt.legend()
plt.show()

plt.figure()
plt.plot(rho_grid, ratio_mu, label="μ1/μ2")
plt.axhline(1, color="gray", linestyle="--")
plt.xlabel(r"$\rho$")
plt.ylabel("Manipulation ratio")
plt.title("Intertemporal distortion in manipulation: μ1/μ2 vs ρ")
plt.legend()
plt.show()

# Greenwashing share for period 1 vs period 2 as a function of ρ
s1_list, s2_list = [], []

for rho in rho_grid:
    p = BaselineParams(**vars(p0))
    p.rho = rho
    res = solve_baseline_model(p)

    s1 = res["mu1"] / (p0.a * res["e1"] + res["mu1"])
    s2 = res["mu2"] / (p0.a * res["e2"] + res["mu2"])

    s1_list.append(s1)
    s2_list.append(s2)

plt.figure()
plt.plot(rho_grid, s1_list, label="s1 (Period 1)")
plt.plot(rho_grid, s2_list, label="s2 (Period 2)")
plt.xlabel(r"$\rho$")
plt.ylabel("Greenwashing share")
plt.title("Greenwashing in period 1 vs period 2 as function of ρ")
plt.legend()
plt.tight_layout()
plt.show()


# --- baseline parameters (adjust if needed) ---
p0 = BaselineParams()  # or BaselineParams(a=..., b=..., ...)

# --- optimal contract and utilities ---
opt = solve_baseline_model(p0)
psi1_opt = opt["psi1"]
psi2_opt = opt["psi2"]
UB_opt   = opt["UB_total"]
UL_opt   = opt["UL"]

print("ψ1* =", psi1_opt, "  ψ2* =", psi2_opt)
print("UB* =", UB_opt,     "  UL* =", UL_opt)

def evaluate_with_given_psi(psi1, psi2, p: BaselineParams):
    """
    Evaluate borrower and lender utilities for given (possibly suboptimal)
    ψ1, ψ2. Borrower responds optimally.
    """
    # Borrower best response
    e1, e2, mu1, mu2 = borrower_optimal_e_mu(psi1, psi2, p)

    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)

    D1_star = D1(e1, mu1, psi1, p)
    D2_star = D2(e1, mu1, e2, mu2, psi1, psi2, p)

    UB1 = (
        project_return(e1, p)
        - D1_star
        - cost_effort(e1, p)
        - cost_manip(mu1, p)
        - cost_contract(p.C1, p)
    )
    UB2 = (
        project_return(e2, p)
        - D2_star
        - cost_effort(e2, p)
        - cost_manip(mu2, p)
        - cost_contract(p.C2, p)
    )
    UB_total = UB1 + p.delta * UB2

    UL = D1_star + p.delta * D2_star + p.theta * (g1 + p.delta * g2)

    return UB_total, UL

# grid around optimum – adjust width if you like
psi1_grid = np.linspace(psi1_opt - 0.4, psi1_opt + 0.4, 100)

UB_vs_psi1 = []

for psi1 in psi1_grid:
    UB, _ = evaluate_with_given_psi(psi1, psi2_opt, p0)
    UB_vs_psi1.append(UB)

plt.figure()
plt.plot(psi1_grid, UB_vs_psi1, label="Borrower utility $U_B$")
plt.axvline(psi1_opt, linestyle="--", label=r"Optimal $\psi_1^*$")
plt.xlabel(r"$\psi_1$ (with $\psi_2 = \psi_2^*$)")
plt.ylabel(r"Borrower utility $U_B$")
plt.title("Borrower utility vs $\\psi_1$")
plt.legend()
plt.tight_layout()
plt.savefig("UBvpsi1.png", dpi=300, bbox_inches="tight")
plt.show()

psi2_grid = np.linspace(psi2_opt - 0.4, psi2_opt + 0.4, 100)

UB_vs_psi2 = []

for psi2 in psi2_grid:
    UB, _ = evaluate_with_given_psi(psi1_opt, psi2, p0)
    UB_vs_psi2.append(UB)

plt.figure()
plt.plot(psi2_grid, UB_vs_psi2, label="Borrower utility $U_B$")
plt.axvline(psi2_opt, linestyle="--", label=r"Optimal $\psi_2^*$")
plt.xlabel(r"$\psi_2$ (with $\psi_1 = \psi_1^*$)")
plt.ylabel(r"Borrower utility $U_B$")
plt.title("Borrower utility vs $\\psi_2$")
plt.legend()
plt.tight_layout()
plt.savefig("UBvpsi2.png", dpi=300, bbox_inches="tight")
plt.show()

UL_vs_psi1 = []

for psi1 in psi1_grid:
    _, UL = evaluate_with_given_psi(psi1, psi2_opt, p0)
    UL_vs_psi1.append(UL)

plt.figure()
plt.plot(psi1_grid, UL_vs_psi1, label="Lender utility $U_L$")
plt.axvline(psi1_opt, linestyle="--", label=r"Optimal $\psi_1^*$")
plt.xlabel(r"$\psi_1$ (with $\psi_2 = \psi_2^*$)")
plt.ylabel(r"Lender utility $U_L$")
plt.title("Lender utility vs $\\psi_1$")
plt.legend()
plt.tight_layout()
plt.savefig("ULvpsi1.png", dpi=300, bbox_inches="tight")
plt.show()

UL_vs_psi2 = []

for psi2 in psi2_grid:
    _, UL = evaluate_with_given_psi(psi1_opt, psi2, p0)
    UL_vs_psi2.append(UL)

plt.figure()
plt.plot(psi2_grid, UL_vs_psi2, label="Lender utility $U_L$")
plt.axvline(psi2_opt, linestyle="--", label=r"Optimal $\psi_2^*$")
plt.xlabel(r"$\psi_2$ (with $\psi_1 = \psi_1^*$)")
plt.ylabel(r"Lender utility $U_L$")
plt.title("Lender utility vs $\\psi_2$")
plt.legend()
plt.tight_layout()
plt.savefig("ULvpsi2.png", dpi=300, bbox_inches="tight")
plt.show()