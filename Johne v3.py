# import numpy as np
# import matplotlib.pyplot as plt

# # ===========================
# # 1. BASELINE PARAMETERS
# # ===========================

# params = {
#     # Technology / green signal
#     "a": 1.0,            # effectiveness of effort in green outcome y(e) = a e
#     "b": 1.0,            # productivity of effort in financial return R(e) = b e

#     # Cost parameters
#     "k_e": 1.0,          # κ_e, effort cost curvature
#     "k_mu": 1.0,         # κ_μ, manipulation cost curvature
#     "alpha_audit": 0.5,  # α, probability / intensity of being caught

#     # Contract / incentive parameters
#     "rho": 0.3,          # carry-over intensity ρ
#     "theta": 1.0,        # warm-glow θ
#     "z1": 1.0,           # project-quality threshold z_1^*
#     "z2": 1.0,           # project-quality threshold z_2^*
#     "G1_bar": 1.0,       # target green benchmark in period 1, \bar{G}_1

#     # Baseline face values of debt
#     "D1_bar": 0.5,       # baseline coupon / face value in period 1
#     "D2_bar": 0.5,       # baseline face value in period 2

#     # CAPM parameters
#     "r_f": 0.02,         # risk-free rate
#     "E_RM": 0.08,        # expected market return E[R_M]
#     "beta_green": 0.6,   # β_1 for green project
#     "beta_brown": 1.0,   # β_0 for brown project

#     # Contractual-term disutilities γ( C̄_t,x ) -- you can treat these as given
#     "gamma_g1": 0.1,     # period-1 green contract cost
#     "gamma_g2": 0.1,     # period-2 green contract cost
#     "gamma_b1": 0.02,    # period-1 brown contract cost (smaller)
#     "gamma_b2": 0.02     # period-2 brown contract cost
# }

# def capm_discount(beta, r_f, E_RM):
#     """
#     CAPM discount factor q = 1 / (1 + r),
#     with r = r_f + beta * (E[R_M] - r_f).
#     """
#     r = r_f + beta * (E_RM - r_f)
#     return 1.0 / (1.0 + r)

# def lender_optimal_psi_green(p, q2_green):
#     a = p["a"]
#     b = p["b"]
#     k_e = p["k_e"]
#     k_mu = p["k_mu"]
#     alpha = p["alpha_audit"]
#     rho = p["rho"]
#     theta = p["theta"]
#     z1 = p["z1"]
#     z2 = p["z2"]

#     Xi = a**2 / k_e + 1.0 / (k_mu * alpha)

#     psi1 = theta / 2.0 - q2_green * rho + (z1 - a * b / k_e) / (2.0 * Xi)
#     psi2 = theta / 2.0 + (z2 - a * b / k_e) / (2.0 * Xi)

#     psi1 = min(max(psi1, 0.0), 1.0)
#     psi2 = min(max(psi2, 0.0), 1.0)

#     return psi1, psi2

# def borrower_optima_green(p, psi1, psi2, q2_green):
#     a = p["a"]
#     b = p["b"]
#     k_e = p["k_e"]
#     k_mu = p["k_mu"]
#     alpha = p["alpha_audit"]
#     rho = p["rho"]

#     # Period 2 (static)
#     e2 = (b + a * psi2) / k_e
#     mu2 = psi2 / (k_mu * alpha)

#     # Period 1 (dynamic with carry-over)
#     e1 = (b + a * (psi1 + q2_green * rho)) / k_e
#     mu1 = (psi1 + q2_green * rho) / (k_mu * alpha)

#     return e1, mu1, e2, mu2

# def debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2):
#     a = p["a"]
#     z1 = p["z1"]
#     z2 = p["z2"]
#     rho = p["rho"]
#     G1_bar = p["G1_bar"]
#     D1_bar = p["D1_bar"]
#     D2_bar = p["D2_bar"]

#     # Green signals
#     g1 = a * e1 + mu1
#     g2 = a * e2 + mu2

#     # Project-based components
#     phi1 = (1.0 - psi1) * z1
#     phi2 = (1.0 - psi2) * z2

#     # Dynamic debt structure
#     D1 = D1_bar - phi1 - psi1 * g1
#     D2 = D2_bar - phi2 - psi2 * g2 - rho * (g1 - G1_bar)

#     return D1, D2, g1, g2

# def borrower_utility_green(p):
#     # Discounts
#     q2_green = capm_discount(p["beta_green"], p["r_f"], p["E_RM"])

#     # Lender chooses ψ_1*, ψ_2* for the green project
#     psi1, psi2 = lender_optimal_psi_green(p, q2_green)

#     # Borrower best response
#     e1, mu1, e2, mu2 = borrower_optima_green(p, psi1, psi2, q2_green)

#     # Debt payments
#     D1, D2, g1, g2 = debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2)

#     # Utilities per period
#     b = p["b"]
#     k_e = p["k_e"]
#     k_mu = p["k_mu"]
#     alpha = p["alpha_audit"]

#     gamma_g1 = p["gamma_g1"]
#     gamma_g2 = p["gamma_g2"]

#     U1 = b * e1 - D1 - 0.5 * k_e * e1**2 - 0.5 * k_mu * alpha * mu1**2 - gamma_g1
#     U2 = b * e2 - D2 - 0.5 * k_e * e2**2 - 0.5 * k_mu * alpha * mu2**2 - gamma_g2

#     NB_green = U1 + q2_green * U2

#     # Return detailed result for diagnostics
#     return {
#         "NB_green": NB_green,
#         "U1": U1, "U2": U2,
#         "q2_green": q2_green,
#         "psi1": psi1, "psi2": psi2,
#         "e1": e1, "e2": e2,
#         "mu1": mu1, "mu2": mu2,
#         "D1": D1, "D2": D2,
#         "g1": g1, "g2": g2
#     }

# def borrower_utility_brown(p):
#     q2_brown = capm_discount(p["beta_brown"], p["r_f"], p["E_RM"])

#     b = p["b"]
#     k_e = p["k_e"]

#     # Brown project behavioural choices
#     e1 = b / k_e
#     e2 = b / k_e
#     mu1 = 0.0
#     mu2 = 0.0

#     # Brown debt: just baseline payments
#     D1 = p["D1_bar"]
#     D2 = p["D2_bar"]

#     gamma_b1 = p["gamma_b1"]
#     gamma_b2 = p["gamma_b2"]

#     U1 = b * e1 - D1 - 0.5 * k_e * e1**2 - gamma_b1
#     U2 = b * e2 - D2 - 0.5 * k_e * e2**2 - gamma_b2

#     NB_brown = U1 + q2_brown * U2

#     return {
#         "NB_brown": NB_brown,
#         "U1": U1, "U2": U2,
#         "q2_brown": q2_brown,
#         "e1": e1, "e2": e2,
#         "mu1": mu1, "mu2": mu2,
#         "D1": D1, "D2": D2
#     }
# def evaluate_projects(p):
#     res_g = borrower_utility_green(p)
#     res_b = borrower_utility_brown(p)

#     NB_g = res_g["NB_green"]
#     NB_b = res_b["NB_brown"]
#     dNB = NB_g - NB_b
#     choice = "green" if dNB > 0 else "brown"

#     return {
#         "NB_green": NB_g,
#         "NB_brown": NB_b,
#         "Delta_NB": dNB,
#         "choice": choice,
#         "green_details": res_g,
#         "brown_details": res_b
#     }

# # Example single evaluation
# res0 = evaluate_projects(params)
# print("NB_green:", res0["NB_green"])
# print("NB_brown:", res0["NB_brown"])
# print("Delta NB:", res0["Delta_NB"])
# print("Endogenous choice:", res0["choice"])


# def plot_DeltaNB_vs_green_premium(params,
#                                   premium_grid=np.linspace(-1.0, 1.0, 200)):

#     beta_b = params["beta_brown"]

#     Delta_list = []
#     NB_g_list = []
#     NB_b_list = []

#     for prem in premium_grid:
#         p = params.copy()
#         p["beta_green"] = beta_b + prem
#         res = evaluate_projects(p)
#         Delta_list.append(res["Delta_NB"])
#         NB_g_list.append(res["NB_green"])
#         NB_b_list.append(res["NB_brown"])

#     plt.figure(figsize=(7,5))
#     plt.plot(premium_grid, Delta_list, label="ΔNB = NB_g - NB_b")
#     plt.axhline(0, color="black", linestyle="--")
#     plt.xlabel("Green premium  β_green - β_brown")
#     plt.ylabel("ΔNB")
#     plt.title("Borrower project choice as a function of the green premium")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. BASELINE PARAMETERS
# ===========================

params = {
    # Technology / green signal
    "a": 1.0,            # effectiveness of effort in green outcome y(e) = a e
    "b": 1.0,            # productivity of effort in financial return R(e) = b e

    # Cost parameters
    "k_e": 1.0,          # κ_e, effort cost curvature
    "k_mu": 1.0,         # κ_μ, manipulation cost curvature
    "alpha_audit": 0.5,  # α, probability / intensity of being caught

    # Contract / incentive parameters
    "rho": 0.3,          # carry-over intensity ρ
    "theta": 1.0,        # warm-glow θ (you can set this to 0 if you want)
    "z1": 1.0,           # project-quality threshold z_1^*
    "z2": 1.0,           # project-quality threshold z_2^*
    "G1_bar": 1.0,       # target green benchmark in period 1, \bar{G}_1

    # Baseline face values of debt
    "D1_bar": 0.5,       # baseline coupon / face value in period 1
    "D2_bar": 0.5,       # baseline face value in period 2

    # CAPM parameters
    "r_f": 0.02,         # risk-free rate
    "E_RM": 0.08,        # expected market return E[R_M]
    "beta_green": 0.6,   # β_1 for green project
    "beta_brown": 1.0,   # β_0 for brown project

    # Contractual-term disutilities γ( C̄_t,x )
    "gamma_g1": 0.1,     # period-1 green contract cost
    "gamma_g2": 0.1,     # period-2 green contract cost
    "gamma_b1": 0.02,    # period-1 brown contract cost
    "gamma_b2": 0.02     # period-2 brown contract cost
}


def capm_discount(beta, r_f, E_RM):
    """CAPM discount factor q = 1 / (1 + r),
    with r = r_f + beta * (E[R_M] - r_f)."""
    r = r_f + beta * (E_RM - r_f)
    return 1.0 / (1.0 + r)


def lender_optimal_psi_green(p, q2_green):
    a = p["a"]
    b = p["b"]
    k_e = p["k_e"]
    k_mu = p["k_mu"]
    alpha = p["alpha_audit"]
    rho = p["rho"]
    theta = p["theta"]
    z1 = p["z1"]
    z2 = p["z2"]

    Xi = a**2 / k_e + 1.0 / (k_mu * alpha)

    psi1 = theta / 2.0 - q2_green * rho + (z1 - a * b / k_e) / (2.0 * Xi)
    psi2 = theta / 2.0 + (z2 - a * b / k_e) / (2.0 * Xi)

    # Enforce 0 <= psi <= 1
    psi1 = min(max(psi1, 0.0), 1.0)
    psi2 = min(max(psi2, 0.0), 1.0)

    return psi1, psi2


def borrower_optima_green(p, psi1, psi2, q2_green):
    a = p["a"]
    b = p["b"]
    k_e = p["k_e"]
    k_mu = p["k_mu"]
    alpha = p["alpha_audit"]
    rho = p["rho"]

    # Period 2 (static)
    e2 = (b + a * psi2) / k_e
    mu2 = psi2 / (k_mu * alpha)

    # Period 1 (dynamic with carry-over)
    e1 = (b + a * (psi1 + q2_green * rho)) / k_e
    mu1 = (psi1 + q2_green * rho) / (k_mu * alpha)

    return e1, mu1, e2, mu2


def debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2):
    a = p["a"]
    z1 = p["z1"]
    z2 = p["z2"]
    rho = p["rho"]
    G1_bar = p["G1_bar"]
    D1_bar = p["D1_bar"]
    D2_bar = p["D2_bar"]

    # Green signals
    g1 = a * e1 + mu1
    g2 = a * e2 + mu2

    # Project-based components
    phi1 = (1.0 - psi1) * z1
    phi2 = (1.0 - psi2) * z2

    # Dynamic debt structure
    D1 = D1_bar - phi1 - psi1 * g1
    D2 = D2_bar - phi2 - psi2 * g2 - rho * (g1 - G1_bar)

    return D1, D2, g1, g2


def borrower_utility_green(p):
    # Discounts
    q2_green = capm_discount(p["beta_green"], p["r_f"], p["E_RM"])

    # Lender chooses ψ_1*, ψ_2*
    psi1, psi2 = lender_optimal_psi_green(p, q2_green)

    # Borrower best response
    e1, mu1, e2, mu2 = borrower_optima_green(p, psi1, psi2, q2_green)

    # Debt payments
    D1, D2, g1, g2 = debt_payments_green(p, psi1, psi2, e1, mu1, e2, mu2)

    # Utilities per period
    b = p["b"]
    k_e = p["k_e"]
    k_mu = p["k_mu"]
    alpha = p["alpha_audit"]

    gamma_g1 = p["gamma_g1"]
    gamma_g2 = p["gamma_g2"]

    U1 = b * e1 - D1 - 0.5 * k_e * e1**2 - 0.5 * k_mu * alpha * mu1**2 - gamma_g1
    U2 = b * e2 - D2 - 0.5 * k_e * e2**2 - 0.5 * k_mu * alpha * mu2**2 - gamma_g2

    NB_green = U1 + q2_green * U2

    return {
        "NB_green": NB_green,
        "U1": U1, "U2": U2,
        "q2_green": q2_green,
        "psi1": psi1, "psi2": psi2,
        "e1": e1, "e2": e2,
        "mu1": mu1, "mu2": mu2,
        "D1": D1, "D2": D2,
        "g1": g1, "g2": g2
    }


def borrower_utility_brown(p):
    q2_brown = capm_discount(p["beta_brown"], p["r_f"], p["E_RM"])

    b = p["b"]
    k_e = p["k_e"]

    # Brown project behavioural choices
    e1 = b / k_e
    e2 = b / k_e
    mu1 = 0.0
    mu2 = 0.0

    # Brown debt: just baseline payments
    D1 = p["D1_bar"]
    D2 = p["D2_bar"]

    gamma_b1 = p["gamma_b1"]
    gamma_b2 = p["gamma_b2"]

    U1 = b * e1 - D1 - 0.5 * k_e * e1**2 - gamma_b1
    U2 = b * e2 - D2 - 0.5 * k_e * e2**2 - gamma_b2

    NB_brown = U1 + q2_brown * U2

    return {
        "NB_brown": NB_brown,
        "U1": U1, "U2": U2,
        "q2_brown": q2_brown,
        "e1": e1, "e2": e2,
        "mu1": mu1, "mu2": mu2,
        "D1": D1, "D2": D2
    }


def evaluate_projects(p):
    res_g = borrower_utility_green(p)
    res_b = borrower_utility_brown(p)

    NB_g = res_g["NB_green"]
    NB_b = res_b["NB_brown"]
    dNB = NB_g - NB_b
    choice = "green" if dNB > 0 else "brown"

    return {
        "NB_green": NB_g,
        "NB_brown": NB_b,
        "Delta_NB": dNB,
        "choice": choice,
        "green_details": res_g,
        "brown_details": res_b
    }


def plot_DeltaNB_vs_green_premium(params,
                                  premium_grid=np.linspace(-5.0, 95.0, 5000)):

    beta_b = params["beta_brown"]

    Delta_list = []
    NB_g_list = []
    NB_b_list = []

    for prem in premium_grid:
        p = params.copy()
        p["beta_green"] = beta_b + prem
        res = evaluate_projects(p)
        Delta_list.append(res["Delta_NB"])
        NB_g_list.append(res["NB_green"])
        NB_b_list.append(res["NB_brown"])

    plt.figure(figsize=(7, 5))
    plt.plot(premium_grid, Delta_list, label="ΔNB = NB_g - NB_b")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel("Green premium  β_green - β_brown")
    plt.ylabel("ΔNB")
    plt.title("Borrower project choice as a function of the green premium")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================
# Run everything
# ================
# Run directly (works in notebooks and ChatGPT)
res0 = evaluate_projects(params)
print("NB_green:", res0["NB_green"])
print("NB_brown:", res0["NB_brown"])
print("Delta NB:", res0["Delta_NB"])
print("Endogenous choice:", res0["choice"])

plot_DeltaNB_vs_green_premium(params)

# NB v kappa e

def plot_NB_vs_k_e(params, k_e_grid=np.linspace(0.0, 8.0, 80)):
    NB_g, NB_b = [], []

    for ke in k_e_grid:
        p = params.copy()
        p["k_e"] = ke
        res = evaluate_projects(p)
        NB_g.append(res["NB_green"])
        NB_b.append(res["NB_brown"])

    NB_g = np.array(NB_g)
    NB_b = np.array(NB_b)



    plt.figure(figsize=(7,5))
    plt.plot(k_e_grid, NB_g, label="NB_green")
    plt.plot(k_e_grid, NB_b, label="NB_brown")
    plt.plot(k_e_grid, NB_g - NB_b, label="ΔNB")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel(r"Effort cost $\kappa_e$")
    plt.ylabel("Net Benefit")
    plt.title("NB as a function of effort cost κ_e")
    plt.legend()
    plt.tight_layout()
    plt.savefig("NBvkappae.png", dpi=300, bbox_inches="tight")
    plt.show()

# NB v kappa mu
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
    plt.savefig("NBvkappamu.png", dpi=300, bbox_inches="tight")
    plt.show()

# NB v rho
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
    plt.savefig("NBvrho.png", dpi=300, bbox_inches="tight")
    plt.show()

# # NB v contractual cost
# def plot_NB_vs_gamma_g(params,
#                        gamma_grid=np.linspace(0.0, 8.0, 80)):
#     NB_g1, NB_g2, NB_b = [], [], None

#     for gam in gamma_grid:
#         p1 = params.copy()
#         p1["gamma_g1"] = gam

#         p2 = params.copy()
#         p2["gamma_g2"] = gam

#         # green NB for varying γ_g1
#         NB_g1.append(evaluate_projects(p1)["NB_green"])
#         # green NB for varying γ_g2
#         NB_g2.append(evaluate_projects(p2)["NB_green"])

#     # Brown NB is constant w.r.t. green contractual burdens
#     NB_b = evaluate_projects(params)["NB_brown"]
#     NB_b_line = np.full_like(gamma_grid, NB_b)

#     plt.figure(figsize=(7,5))
#     plt.plot(gamma_grid, NB_g1, label="NB_green (varying γ_g1)")
#     plt.plot(gamma_grid, NB_g2, label="NB_green (varying γ_g2)")
#     plt.plot(gamma_grid, NB_b_line, label="NB_brown")
#     plt.xlabel(r"Green contract burdens $\gamma_{g1}$ or $\gamma_{g2}$")
#     plt.ylabel("Net Benefit")
#     plt.title("NB as a function of green contractual burdens γ_g1 and γ_g2")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("NBvgammas.png", dpi=300, bbox_inches="tight")
#     plt.show()

def plot_NB_vs_gamma_g(params,
                       gamma_grid=np.linspace(0.0, 8.0, 120)):

    NB_g1, NB_g2, Delta_g1, Delta_g2 = [], [], [], []

    # Brown NB is constant with respect to green contractual burdens
    NB_b = evaluate_projects(params)["NB_brown"]
    NB_b_line = np.full_like(gamma_grid, NB_b)

    for gam in gamma_grid:

        # Case varying γ_g1
        p1 = params.copy()
        p1["gamma_g1"] = gam
        res1 = evaluate_projects(p1)
        NB_g1.append(res1["NB_green"])
        Delta_g1.append(res1["Delta_NB"])

        # Case varying γ_g2
        p2 = params.copy()
        p2["gamma_g2"] = gam
        res2 = evaluate_projects(p2)
        NB_g2.append(res2["NB_green"])
        Delta_g2.append(res2["Delta_NB"])

    NB_g1 = np.array(NB_g1)
    NB_g2 = np.array(NB_g2)
    Delta_g1 = np.array(Delta_g1)
    Delta_g2 = np.array(Delta_g2)

    plt.figure(figsize=(7,5))

    # Main NB curves
    plt.plot(gamma_grid, NB_g1, label=r"$NB_{\text{green}}(\gamma_{g1})$")
    plt.plot(gamma_grid, NB_g2, label=r"$NB_{\text{green}}(\gamma_{g2})$")
    plt.plot(gamma_grid, NB_b_line, label=r"$NB_{\text{brown}}$")

    # ΔNB curves
    plt.plot(gamma_grid, Delta_g1, label=r"$\Delta NB(\gamma_{g1})$", linestyle=":")
    plt.plot(gamma_grid, Delta_g2, label=r"$\Delta NB(\gamma_{g2})$", linestyle=":")

    # Zero line (NB = 0)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    plt.xlabel(r"Green contractual burdens $\gamma_{g1}, \gamma_{g2}$")
    plt.ylabel("Net Benefit")
    plt.title("NB and ΔNB as functions of γ_g1 and γ_g2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("NBvgammas.png", dpi=300, bbox_inches="tight")
    plt.show()


plot_NB_vs_k_e(params)
plot_NB_vs_k_mu(params)
plot_NB_vs_rho(params)
plot_NB_vs_gamma_g(params)

def plot_NB_vs_contract_term_difference(params,
                                        DeltaC_grid=np.linspace(0.0, 2.0, 120)):
    """
    Increase ONLY the difference in contractual terms:
        ΔC = C̄_{t,1} - C̄_{t,0}

    Implemented as:
        gamma_g1 = gamma_b1 + ΔC
        gamma_g2 = gamma_b2 + ΔC

    This varies contractual *amounts* but not the cost structure γ.
    """

    NB_g_list = []
    NB_b_list = []
    Delta_list = []

    gamma_b1 = params["gamma_b1"]
    gamma_b2 = params["gamma_b2"]

    for DeltaC in DeltaC_grid:

        # Green contract becomes more complex than brown
        p = params.copy()
        p["gamma_g1"] = gamma_b1 + DeltaC
        p["gamma_g2"] = gamma_b2 + DeltaC

        res = evaluate_projects(p)
        NB_g_list.append(res["NB_green"])
        NB_b_list.append(res["NB_brown"])
        Delta_list.append(res["Delta_NB"])

    NB_g  = np.array(NB_g_list)
    NB_b  = np.array(NB_b_list)
    Delta = np.array(Delta_list)

    plt.figure(figsize=(7,5))
    plt.plot(DeltaC_grid, NB_g,  label=r"$NB_{\text{green}}$")
    plt.plot(DeltaC_grid, NB_b,  label=r"$NB_{\text{brown}}$")
    plt.plot(DeltaC_grid, Delta, label=r"$\Delta NB$")

    # ΔNB = 0 line
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)

    plt.xlabel(r"Contractual term difference $\Delta C = \bar C_{t,1} - \bar C_{t,0}$")
    plt.ylabel("Net Benefit")
    plt.title("Effect of contractual-term amount differences on project choice")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("NBvbarCs.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_NB_vs_contract_term_difference(params)