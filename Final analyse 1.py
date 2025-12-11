import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============================
#     PARAMETER CLASS
# =============================

@dataclass
class BaselineParams:
    a: float = 1.0
    b: float = 1.0
    kappa_e: float = 1.0
    kappa_mu: float = 1.0
    alpha: float = 0.5
    kappa_gamma: float = 1.0
    delta: float = 0.9
    rho: float = 0.3
    theta: float = 1.0
    z1_star: float = 1.0
    z2_star: float = 1.0
    x1: int = 1
    x2: int = 1
    G1_bar: float = 1.0
    D1_bar: float = 1.0
    D2_bar: float = 1.0
    C1: float = 0.0
    C2: float = 0.0
    r_f: float = 0.02
    E_RM: float = 0.08
    beta_green: float = 0.6
    beta_brown: float = 1.0   

# =============================
#     MODEL COMPONENTS
# =============================

def cost_effort(e, p): return 0.5 * p.kappa_e * e**2
def cost_manip(mu, p): return 0.5 * p.kappa_mu * p.alpha * mu**2
def cost_contract(C, p): return 0.5 * p.kappa_gamma * C**2

def green_signal(e, mu, p): return p.a * e + mu
def project_return(e, p): return p.b * e
def phi_t(psi, z): return (1 - psi) * z

def D1(e1, mu1, psi1, p):
    return p.D1_bar - phi_t(psi1, p.z1_star)*p.x1 - psi1 * green_signal(e1, mu1, p)

def D2(e1, mu1, e2, mu2, psi1, psi2, p):
    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)
    return (p.D2_bar
            - phi_t(psi2, p.z2_star)*p.x2
            - psi2 * g2
            - p.rho * (g1 - p.G1_bar))

# =============================
#     OPTIMAL ACTIONS
# =============================

def borrower_optimal_e_mu(psi1, psi2, p):
    e2 = (p.b + p.a*psi2) / p.kappa_e
    mu2 = psi2 / (p.kappa_mu * p.alpha)
    e1 = (p.b + p.a*(psi1 + p.delta*p.rho)) / p.kappa_e
    mu1 = (psi1 + p.delta*p.rho) / (p.kappa_mu * p.alpha)
    return e1, e2, mu1, mu2

def solve_baseline_model(p):
    Xi = (p.a**2)/p.kappa_e + 1/(p.kappa_mu*p.alpha)
    common = (p.a*p.b)/p.kappa_e

    psi1 = p.theta/2 - p.delta*p.rho + (p.z1_star*p.x1 - common)/(2*Xi)
    psi2 = p.theta/2 + (p.z2_star*p.x2 - common)/(2*Xi)

    e1, e2, mu1, mu2 = borrower_optimal_e_mu(psi1, psi2, p)

    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)

    D1_star = D1(e1, mu1, psi1, p)
    D2_star = D2(e1, mu1, e2, mu2, psi1, psi2, p)

    UB1 = project_return(e1, p) - D1_star - cost_effort(e1, p) - cost_manip(mu1, p)
    UB2 = project_return(e2, p) - D2_star - cost_effort(e2, p) - cost_manip(mu2, p)
    UB_total = UB1 + p.delta*UB2

    UL = D1_star + p.delta*D2_star + p.theta*(g1 + p.delta*g2)

    return {"psi1": psi1, "psi2": psi2, "UB_total": UB_total, "UL": UL}

# =============================
#     UTILITY EVALUATION
# =============================

def evaluate_with_given_psi(psi1, psi2, p):
    e1, e2, mu1, mu2 = borrower_optimal_e_mu(psi1, psi2, p)
    g1 = green_signal(e1, mu1, p)
    g2 = green_signal(e2, mu2, p)

    D1_star = D1(e1, mu1, psi1, p)
    D2_star = D2(e1, mu1, e2, mu2, psi1, psi2, p)

    UB1 = project_return(e1, p) - D1_star - cost_effort(e1, p) - cost_manip(mu1, p)
    UB2 = project_return(e2, p) - D2_star - cost_effort(e2, p) - cost_manip(mu2, p)
    UB_total = UB1 + p.delta*UB2

    UL = D1_star + p.delta*D2_star + p.theta*(g1 + p.delta*g2)

    return UB_total, UL

# ============================================================
#     ANALYSIS OF IMPORTANCE OF OPTIMAL CONTRACT FOMULATION
# ============================================================

p0 = BaselineParams()
opt = solve_baseline_model(p0)
psi1_opt, psi2_opt = opt["psi1"], opt["psi2"]

psi1_grid = np.linspace(psi1_opt - 0.4, psi1_opt + 0.4, 100)
psi2_grid = np.linspace(psi2_opt - 0.4, psi2_opt + 0.4, 100)

# ---- Plot 1: UB vs ψ1 ----
UB_vs_psi1 = [evaluate_with_given_psi(psi1, psi2_opt, p0)[0] for psi1 in psi1_grid]
plt.figure()
plt.plot(psi1_grid, UB_vs_psi1)
plt.axvline(psi1_opt, linestyle="--", label=r"Optimal $\psi_1^*$")
plt.title("Borrower utility vs ψ1")
plt.tight_layout()
plt.show()

# ---- Plot 2: UB vs ψ2 ----
UB_vs_psi2 = [evaluate_with_given_psi(psi1_opt, psi2, p0)[0] for psi2 in psi2_grid]
plt.figure()
plt.plot(psi2_grid, UB_vs_psi2)
plt.axvline(psi2_opt, linestyle="--")
plt.title("Borrower utility vs ψ2")
plt.tight_layout()
plt.show()

# ---- Plot 3: UL vs ψ1 ----
UL_vs_psi1 = [evaluate_with_given_psi(psi1, psi2_opt, p0)[1] for psi1 in psi1_grid]
plt.figure()
plt.plot(psi1_grid, UL_vs_psi1)
plt.axvline(psi1_opt, linestyle="--")
plt.title("Lender utility vs ψ1")
plt.tight_layout()
plt.show()

# ---- Plot 4: UL vs ψ2 ----
UL_vs_psi2 = [evaluate_with_given_psi(psi1_opt, psi2, p0)[1] for psi2 in psi2_grid]
plt.figure()
plt.plot(psi2_grid, UL_vs_psi2)
plt.axvline(psi2_opt, linestyle="--")
plt.title("Lender utility vs ψ2")
plt.tight_layout()
plt.show()

# ============================================================
#     SIMULATION OF ENDOGENOUS PROJECT CHOICE
# ============================================================


def capm_discount(beta, r_f, E_RM):
    r = r_f + beta * (E_RM - r_f)
    return 1.0 / (1.0 + r)

def lender_optimal_psi_green(p, q2):
    Xi = p.a**2 / p.kappa_e + 1.0 / (p.kappa_mu * p.alpha)
    psi1 = (p.theta/2 
            - q2 * p.rho 
            + (p.z1_star * p.x1 - p.a*p.b/p.kappa_e) / (2*Xi))
    psi2 = (p.theta/2 
            + (p.z2_star * p.x2 - p.a*p.b/p.kappa_e) / (2*Xi))

    return np.clip(psi1, 0.0, 1.0), np.clip(psi2, 0.0, 1.0)

# def lender_optimal_psi_green(p, q2):
#     a, b = p["a"], p["b"]
#     k_e, k_mu = p["k_e"], p["k_mu"]
#     alpha, rho, theta = p["alpha_audit"], p["rho"], p["theta"]
#     z1, z2 = p["z1"], p["z2"]
#     Xi = a**2 / k_e + 1.0 / (k_mu * alpha)
#     psi1 = theta/2 - q2*rho + (z1 - a*b/k_e) / (2*Xi)
#     psi2 = theta/2 + (z2 - a*b/k_e) / (2*Xi)
#     psi1 = np.clip(psi1, 0.0, 1.0)
#     psi2 = np.clip(psi2, 0.0, 1.0)
#     return psi1, psi2

# def borrower_optima_green(p, psi1, psi2, q2):
#     a, b = p["a"], p["b"]
#     k_e, k_mu = p["k_e"], p["k_mu"]
#     alpha, rho = p["alpha_audit"], p["rho"]

#     # Period 2 (static)
#     e2 = (b + a*psi2) / k_e
#     mu2 = psi2 / (k_mu * alpha)

#     # Period 1 (dynamic with carry-over)
#     e1 = (b + a*(psi1 + q2*rho)) / k_e
#     mu1 = (psi1 + q2*rho) / (k_mu * alpha)

#     return e1, mu1, e2, mu2

def borrower_optima_green(p, psi1, psi2, q2):
    # Period 2
    e2  = (p.b + p.a*psi2) / p.kappa_e
    mu2 = psi2 / (p.kappa_mu * p.alpha)

    # Period 1 (dynamic)
    e1  = (p.b + p.a*(psi1 + q2*p.rho)) / p.kappa_e
    mu1 = (psi1 + q2*p.rho) / (p.kappa_mu * p.alpha)

    return e1, mu1, e2, mu2

def debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2):
    g1 = p.a * e1 + mu1
    g2 = p.a * e2 + mu2

    phi1 = (1 - psi1) * p.z1_star * p.x1
    phi2 = (1 - psi2) * p.z2_star * p.x2

    D1 = p.D1_bar - phi1 - psi1*g1
    D2 = p.D2_bar - phi2 - psi2*g2 - p.rho*(g1 - p.G1_bar)

    return D1, D2, g1, g2

# def debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2):
#     a = p["a"]
#     z1, z2 = p["z1"], p["z2"]
#     rho = p["rho"]
#     G1_bar = p["G1_bar"]
#     D1_bar, D2_bar = p["D1_bar"], p["D2_bar"]
#     g1 = a*e1 + mu1
#     g2 = a*e2 + mu2
#     phi1 = (1.0 - psi1) * z1
#     phi2 = (1.0 - psi2) * z2
#     D1 = D1_bar - phi1 - psi1 * g1
#     D2 = D2_bar - phi2 - psi2 * g2 - rho * (g1 - G1_bar)

    # return D1, D2, g1, g2

# def borrower_utility_green(p):
#     q2 = capm_discount(p["beta_green"], p["r_f"], p["E_RM"])

#     psi1, psi2 = lender_optimal_psi_green(p, q2)
#     e1, mu1, e2, mu2 = borrower_optima_green(p, psi1, psi2, q2)
#     D1, D2, g1, g2 = debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2)

#     b, k_e, k_mu = p["b"], p["k_e"], p["k_mu"]
#     alpha = p["alpha_audit"]

#     U1 = b*e1 - D1 - 0.5*k_e*e1**2 - 0.5*k_mu*alpha*mu1**2 - p["gamma_g1"]
#     U2 = b*e2 - D2 - 0.5*k_e*e2**2 - 0.5*k_mu*alpha*mu2**2 - p["gamma_g2"]

#     NB_green = U1 + q2 * U2
#     return {"NB_green": NB_green}

def borrower_utility_green(p):
    q2 = capm_discount(p.beta_green, p.r_f, p.E_RM)

    psi1, psi2 = lender_optimal_psi_green(p, q2)
    e1, mu1, e2, mu2 = borrower_optima_green(p, psi1, psi2, q2)
    D1, D2, g1, g2 = debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2)

    U1 = p.b*e1 - D1 - 0.5*p.kappa_e*e1**2 - 0.5*p.kappa_mu*p.alpha*mu1**2 - p.C1
    U2 = p.b*e2 - D2 - 0.5*p.kappa_e*e2**2 - 0.5*p.kappa_mu*p.alpha*mu2**2 - p.C2

    return U1 + q2 * U2


# def borrower_utility_brown(p):
#     q2 = capm_discount(p["beta_brown"], p["r_f"], p["E_RM"])
#     b, k_e = p["b"], p["k_e"]

#     e1 = b / k_e
#     e2 = b / k_e
#     D1, D2 = p["D1_bar"], p["D2_bar"]

#     U1 = b*e1 - D1 - 0.5*k_e*e1**2 - p["gamma_b1"]
#     U2 = b*e2 - D2 - 0.5*k_e*e2**2 - p["gamma_b2"]

#     NB_brown = U1 + q2 * U2
#     return {"NB_brown": NB_brown}

def borrower_utility_brown(p):
    q2 = capm_discount(p.beta_brown, p.r_f, p.E_RM)

    e1 = p.b / p.kappa_e
    e2 = p.b / p.kappa_e

    D1 = p.D1_bar
    D2 = p.D2_bar

    U1 = p.b*e1 - D1 - 0.5*p.kappa_e*e1**2 - p.C1
    U2 = p.b*e2 - D2 - 0.5*p.kappa_e*e2**2 - p.C2

    return U1 + q2 * U2


# def evaluate_projects(p):
#     res_g = borrower_utility_green(p)
#     res_b = borrower_utility_brown(p)
#     NB_g = res_g["NB_green"]
#     NB_b = res_b["NB_brown"]
#     dNB = NB_g - NB_b
#     return {
#         "NB_green": NB_g,
#         "NB_brown": NB_b,
#         "Delta_NB": dNB
#     }
def evaluate_projects(p):
    NB_g = borrower_utility_green(p)
    NB_b = borrower_utility_brown(p)
    return NB_g, NB_b, NB_g - NB_b


# def plot_NB_vs_k_e(p, k_e_grid=np.linspace(0.0, 8.0, 80)):
#     NB_g, NB_b = [], []

#     for ke in k_e_grid:
#         p = p.copy()
#         p["k_e"] = ke
#         res = evaluate_projects(p)
#         NB_g.append(res["NB_green"])
#         NB_b.append(res["NB_brown"])
#     NB_g = np.array(NB_g)
#     NB_b = np.array(NB_b)

#     plt.figure(figsize=(7,5))
#     plt.plot(k_e_grid, NB_g, label="NB_green")
#     plt.plot(k_e_grid, NB_b, label="NB_brown")
#     plt.plot(k_e_grid, NB_g - NB_b, label="ΔNB")
#     plt.axhline(0, color='black', linestyle='--')
#     plt.xlabel(r"Effort cost $\kappa_e$")
#     plt.ylabel("Net Benefit")
#     plt.title("NB as a function of effort cost κ_e")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
def plot_NB_vs_k_e(p, k_e_grid=np.linspace(0.1, 8.0, 80)):
    NB_g, NB_b = [], []
    for ke in k_e_grid:
        p.kappa_e = ke
        g, b, _ = evaluate_projects(p)
        NB_g.append(g)
        NB_b.append(b)

    NB_g = np.array(NB_g)
    NB_b = np.array(NB_b)

    plt.figure()
    plt.plot(k_e_grid, NB_g, label="NB_green")
    plt.plot(k_e_grid, NB_b, label="NB_brown")
    plt.plot(k_e_grid, NB_g - NB_b, label="ΔNB")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"$\kappa_e$")
    plt.ylabel("Net Benefit")
    plt.title("NB as a function of κ_e")
    plt.legend()
    plt.show()


def plot_NB_vs_k_mu(params, k_mu_grid=np.linspace(0.0, 8.0, 80)):
    NB_g, NB_b = [], []
    for kmu in k_mu_grid:
        p = params.copy()
        p["k_mu"] = kmu
        res = evaluate_projects(p)
        NB_g.append(res["NB_green"])
        NB_b.append(res["NB_brown"])
    NB_g = np.array(NB_g)
    NB_b = np.array(NB_b)

    plt.figure(figsize=(7,5))
    plt.plot(k_mu_grid, NB_g, label="NB_green")
    plt.plot(k_mu_grid, NB_b, label="NB_brown")
    plt.plot(k_mu_grid, NB_g - NB_b, label="ΔNB")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"Manipulation cost $\kappa_\mu$")
    plt.ylabel("Net Benefit")
    plt.title("NB as a function of manipulation cost κ_μ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_NB_vs_rho(params, rho_grid=np.linspace(0.0, 1.0, 80)):
    NB_g, NB_b = [], []
    for r in rho_grid:
        p = params.copy()
        p["rho"] = r
        res = evaluate_projects(p)
        NB_g.append(res["NB_green"])
        NB_b.append(res["NB_brown"])
    NB_g = np.array(NB_g)
    NB_b = np.array(NB_b)

    plt.figure(figsize=(7,5))
    plt.plot(rho_grid, NB_g, label="NB_green")
    plt.plot(rho_grid, NB_b, label="NB_brown")
    plt.plot(rho_grid, NB_g - NB_b, label="ΔNB")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"Carry-over intensity $\rho$")
    plt.ylabel("Net Benefit")
    plt.title("NB as a function of carry-over intensity ρ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_NB_vs_contract_term_difference(p,
                                        DeltaC_grid=np.linspace(0.0, 2.0, 120)):
    """
    Picture 6: contractual term difference
    ΔC = C̄_{t,1} - C̄_{t,0}, implemented as extra burdens on the green project:
        gamma_g1 = gamma_b1 + ΔC
        gamma_g2 = gamma_b2 + ΔC
    """

    NB_g_list = []
    NB_b_list = []
    Delta_list = []

    gamma_b1 = params["gamma_b1"]
    gamma_b2 = params["gamma_b2"]

    for DeltaC in DeltaC_grid:
        p = params.copy()
        p["gamma_g1"] = gamma_b1 + DeltaC
        p["gamma_g2"] = gamma_b2 + DeltaC
        res = evaluate_projects(p)
        NB_g_list.append(res["NB_green"])
        NB_b_list.append(res["NB_brown"])
        Delta_list.append(res["Delta_NB"])
    NB_g = np.array(NB_g_list)
    NB_b = np.array(NB_b_list)
    Delta = np.array(Delta_list)

    plt.figure(figsize=(7,5))
    plt.plot(DeltaC_grid, NB_g,  label=r"$NB_{\text{green}}$")
    plt.plot(DeltaC_grid, NB_b,  label=r"$NB_{\text{brown}}$")
    plt.plot(DeltaC_grid, Delta, label=r"$\Delta NB$")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel(r"Contractual term difference $\Delta C = \bar C_{t,1} - \bar C_{t,0}$")
    plt.ylabel("Net Benefit")
    plt.title("Effect of contractual-term differences on project choice")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

plot_NB_vs_k_e(params)
plot_NB_vs_k_mu(params)
plot_NB_vs_rho(params)
plot_NB_vs_contract_term_difference(params)