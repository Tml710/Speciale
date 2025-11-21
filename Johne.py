import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ==============================================================
# 1. MODEL PARAMETERS (BASELINE DYNAMIC MODEL, NO CAPM)
# ==============================================================

a        = 1.0
b        = 1.0
alpha    = 0.5
delta    = 0.9
rho      = 0.5
theta    = 1.0
z        = 1.0     # verifiable project threshold z_t* x_t
Gbar     = 1.0

# ==============================================================
# 2. BORROWER OPTIMA (Baseline dynamic model)
# ==============================================================

def borrower_optima(psi1, psi2, k_e, k_mu):
    """
    Computes optimal effort and manipulation under given psi_t.
    """
    e1 = (b + a*(psi1 + delta*rho)) / k_e
    e2 = (b + a*psi2) / k_e

    mu1 = (psi1 + delta*rho) / (k_mu * alpha)
    mu2 = psi2 / (k_mu * alpha)

    return e1, e2, mu1, mu2


# ==============================================================
# 3. OPTIMAL ψ1*, ψ2* FROM YOUR CLOSED-FORM SOLUTIONS
# ==============================================================

def psi_stars(k_e, k_m):
    """
    Your exact closed-form solution for psi*_1 and psi*_2.
    """
    denom = 2*(a**2/k_e + 1.0/(k_m*alpha))
    num   = z - a*b/k_e

    psi1 = theta/2 - delta*rho + num/denom
    psi2 = theta/2 + num/denom

    return max(0, psi1), max(0, psi2)


# ==============================================================
# 4. REPAYMENT FUNCTIONS
# ==============================================================

def D1(psi1, e1, mu1):
    """
    Correct period-1 repayment from your model:
    D1 = D̄1 - (1-psi1)z - psi1 * g1
    """
    g1 = a*e1 + mu1
    Dbar1 = 1.0  # baseline, can be changed
    return Dbar1 - (1-psi1)*z - psi1*g1


def D2(psi2, psi1, e2, mu2, e1, mu1):
    """
    Correct period-2 repayment INCLUDING information dependence:
    D2 = D̄2 - (1-psi2)z - psi2 * g2 - rho(g1 - Ḡ1)
    """
    g2 = a*e2 + mu2
    g1 = a*e1 + mu1
    Dbar2 = 1.0   # baseline, can be changed
    return Dbar2 - (1-psi2)*z - psi2*g2 - rho*(g1 - Gbar)


# ==============================================================
# 5. LENDER UTILITY (CORRECTLY IMPLEMENTED)
# ==============================================================

def lender_utility(psi1, psi2, k_e, k_mu):
    """
    Lender utility = D1 + δ·D2 + θ(g1 + δ g2)
    Using CORRECT D1 and D2.
    """
    # Borrower behaviour
    e1, e2, mu1, mu2 = borrower_optima(psi1, psi2, k_e, k_mu)

    # Correct repayments
    D1_val = D1(psi1, e1, mu1)
    D2_val = D2(psi2, psi1, e2, mu2, e1, mu1)

    g1 = a*e1 + mu1
    g2 = a*e2 + mu2

    return D1_val + delta*D2_val + theta*(g1 + delta*g2)


# ==============================================================
# 6. CONTRACT UTILITIES (GB, SLB, HYBRID)
# ==============================================================

def U_GB(k_e, k_m):
    psi1 = 0
    psi2 = 0
    return lender_utility(psi1, psi2, k_e, k_m)

def U_SLB(k_e, k_m):
    psi1, psi2 = psi_stars(k_e, k_m)
    return lender_utility(psi1, psi2, k_e, k_m)

def U_Hybrid(k_e, k_m):
    # In your model, Hybrid uses the exact ψ* solutions
    psi1, psi2 = psi_stars(k_e, k_m)
    return lender_utility(psi1, psi2, k_e, k_m)


# ==============================================================
# 7. GRID OVER (κ_e, κ_μ)
# ==============================================================

ke_vals  = np.linspace(0.2, 5, 200)
kmu_vals = np.linspace(0.2, 5, 200)

KE, KMU = np.meshgrid(ke_vals, kmu_vals)

region1 = np.zeros_like(KE, dtype=int)
region2 = np.zeros_like(KE, dtype=int)


# ==============================================================
# 8. PERIOD-SEPARATE UTILITIES
# ==============================================================

def period1_utility(psi1, k_e, k_mu):
    e1, e2, mu1, mu2 = borrower_optima(psi1, 0, k_e, k_mu)
    g1 = a*e1 + mu1
    D1_val = D1(psi1, e1, mu1)
    return D1_val + theta*g1

def period2_utility(psi2, k_e, k_mu):
    # psi1 irrelevant here except for continuation term in D2
    e1, e2, mu1, mu2 = borrower_optima(0, psi2, k_e, k_mu)
    g2 = a*e2 + mu2

    # For period 2 we need g1, so compute with psi1=0
    g1 = a*e1 + mu1

    D2_val = D2(psi2, 0, e2, mu2, e1, mu1)
    return D2_val + theta*g2

# ==============================================================
# 9. CLASSIFICATION FOR PERIOD 1 AND PERIOD 2
# ==============================================================

for i in range(len(kmu_vals)):
    for j in range(len(ke_vals)):

        ke = KE[i,j]
        km = KMU[i,j]

        psi1, psi2 = psi_stars(ke, km)

        # ----- PERIOD 1 -----
        Ugb1  = period1_utility(0, ke, km)
        Uhyp1 = period1_utility(psi1, ke, km)
        Uslb1 = Uhyp1  # SLB and Hybrid identical in period 1 utility

        region1[i,j] = np.argmax([Ugb1, Uhyp1, Uslb1])

        # ----- PERIOD 2 -----
        Ugb2  = period2_utility(0, ke, km)
        Uhyp2 = period2_utility(psi2, ke, km)
        Uslb2 = Uhyp2

        region2[i,j] = np.argmax([Ugb2, Uhyp2, Uslb2])

# # ==============================================================
# # ✔ SIMPLE SIMULATION: Heatmaps of e1 and mu1 over (k_e, k_mu)
# # ==============================================================

# e1_grid  = np.zeros_like(KE)
# mu1_grid = np.zeros_like(KE)

# for i in range(len(kmu_vals)):
#     for j in range(len(ke_vals)):

#         ke  = KE[i,j]
#         kmu = KMU[i,j]

#         # compute psi* using your own function
#         psi1, psi2 = psi_stars(ke, kmu)

#         # compute borrower optimal behaviour
#         e1, e2, mu1, mu2 = borrower_optima(psi1, psi2, ke, kmu)

#         e1_grid[i,j]  = e1
#         mu1_grid[i,j] = mu1


# # ==============================================================
# #  PLOTTING THE TWO HEATMAPS
# # ==============================================================

# fig, ax = plt.subplots(1, 2, figsize=(14,6))

# im1 = ax[0].imshow(e1_grid, origin="lower",
#                    extent=[0.2,5,0.2,5], aspect="auto")
# ax[0].set_title("Effort in Period 1  $e_1$")
# ax[0].set_xlabel(r"Effort cost $\kappa_e$")
# ax[0].set_ylabel(r"Manipulation cost $\kappa_\mu$")
# plt.colorbar(im1, ax=ax[0])

# im2 = ax[1].imshow(mu1_grid, origin="lower",
#                    extent=[0.2,5,0.2,5], aspect="auto")
# ax[1].set_title("Manipulation in Period 1  $\mu_1$")
# ax[1].set_xlabel(r"Effort cost $\kappa_e$")
# ax[1].set_ylabel(r"Manipulation cost $\kappa_\mu$")
# plt.colorbar(im2, ax=ax[1])

# plt.tight_layout()
# plt.show()

# ==============================================================
# ✔ PROPOSAL 4: Vary delta and plot (e1, mu1)
# ==============================================================

# Range for delta in [0, 1]
delta_vals = np.linspace(0, 1, 200)

e1_list  = []
mu1_list = []

for d in delta_vals:
    # update global delta **temporarily**
    delta = d

    # compute psi*
    psi1, psi2 = psi_stars(k_e=1.0, k_m=1.0)  # choose baseline ke, k_mu

    # compute borrower choices
    e1, e2, mu1, mu2 = borrower_optima(psi1, psi2, k_e=1.0, k_mu=1.0)

    e1_list.append(e1)
    mu1_list.append(mu1)

# restore baseline delta if needed
delta = 0.9  


# ==============================================================
#  PLOT
# ==============================================================

plt.figure(figsize=(8,5))
plt.plot(delta_vals, e1_list, label=r"$e_1$", linewidth=2)
plt.plot(delta_vals, mu1_list, label=r"$\mu_1$", linewidth=2)

plt.xlabel(r"Carry-over parameter $\delta$")
plt.ylabel("Level")
plt.title(r"Effect of $\delta$ on effort and manipulation in period 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()