import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. BASELINE PARAMETERS (TRUE TECHNOLOGY & PREFERENCES)
# ============================================================
# These are arbitrary but consistent with your model structure.
# You can replace them with the exact baseline used in your thesis.

a_true   = 1.0    # green productivity parameter a
b_true   = 1.0    # financial productivity parameter b
k_e      = 1.0    # effort cost curvature κ_e
k_mu     = 1.0    # manipulation cost curvature κ_μ
alpha    = 0.5    # audit probability α
delta    = 0.9    # discount factor δ
rho      = 0.3    # carry-over intensity ρ
theta    = 1.0    # warm-glow parameter θ
z1       = 1.0    # project quality threshold z*_1
z2       = 1.0    # project quality threshold z*_2
x1       = 1.0    # assume project implemented in both periods
x2       = 1.0

Dbar1    = 1.0    # baseline coupon / face value period 1
Dbar2    = 1.0    # baseline coupon / face value period 2
Gbar1    = 1.0    # period 1 green threshold Ḡ_1

k_gamma  = 0.0    # cost of contractual terms (set to 0 here to keep focus on a,b)
Cbar1    = 0.0
Cbar2    = 0.0

# ============================================================
# 2. BASIC MODEL FUNCTIONS (AS IN YOUR THESIS)
# ============================================================

def g(e, mu, a):
    """Green signal g_t(e_t, μ_t) = a e_t + μ_t (eq. 4.1.6)."""
    return a * e + mu

def c(e, k_e):
    """Effort cost c(e_t) = 1/2 κ_e e_t^2 (eq. 4.1.1)."""
    return 0.5 * k_e * e**2

def h(mu, k_mu, alpha):
    """Manipulation cost h(μ_t) = 1/2 κ_μ α μ_t^2 (eq. 4.1.2)."""
    return 0.5 * k_mu * alpha * mu**2

def gamma_cost(Cbar, k_gamma):
    """Contractual term cost γ(C̄_t) = 1/2 κ_γ C̄_t^2 (eq. 4.1.3)."""
    return 0.5 * k_gamma * Cbar**2

def R(e, b):
    """Project return R(e_t) = b e_t (eq. 4.1.10)."""
    return b * e

# ============================================================
# 3. BORROWER OPTIMAL BEHAVIOUR (NO-DEFAULT, EQ. 4.2.1–4.2.2)
# ============================================================

def borrower_optima(psi1, psi2, a, b, k_e, k_mu, alpha, delta, rho):
    """
    Returns optimal efforts and manipulations (e1,e2,mu1,mu2)
    for given ψ1, ψ2 and parameters, using eq. (4.2.1)–(4.2.2).
    """
    e2  = (b + a * psi2) / k_e
    mu2 = psi2 / (k_mu * alpha)

    e1  = (b + a * (psi1 + delta * rho)) / k_e
    mu1 = (psi1 + delta * rho) / (k_mu * alpha)

    return e1, e2, mu1, mu2

# ============================================================
# 4. LENDER OPTIMAL ψ* GIVEN PERCEIVED (a_hat, b_hat) (EQ. 4.2.5)
# ============================================================

def lender_optimal_psi(a_hat, b_hat, k_e, k_mu, alpha, theta,
                       delta, rho, z1, z2, x1, x2):
    """
    Compute ψ1* and ψ2* using the lender’s FOC (eq. 4.2.5),
    but with perceived technology parameters (a_hat, b_hat).

    ψ1* = θ/2 − δρ + (z1* x1 − (a b)/κ_e) / (2 (a^2 / κ_e + 1/(κ_μ α)))
    ψ2* = θ/2       + (z2* x2 − (a b)/κ_e) / (2 (a^2 / κ_e + 1/(κ_μ α)))
    """
    Xi = (a_hat**2) / k_e + 1.0 / (k_mu * alpha)

    num1 = z1 * x1 - (a_hat * b_hat) / k_e
    num2 = z2 * x2 - (a_hat * b_hat) / k_e

    psi1 = theta / 2.0 - delta * rho + num1 / (2.0 * Xi)
    psi2 = theta / 2.0 + num2 / (2.0 * Xi)

    # Enforce feasibility (pure GB / pure SLB / hybrid)
    psi1 = np.clip(psi1, 0.0, 1.0)
    psi2 = np.clip(psi2, 0.0, 1.0)

    return psi1, psi2

# ============================================================
# 5. REPAYMENTS D1, D2 AND UTILITIES UB, UL
#    (ADAPTED FROM EQ. 4.1.7–4.1.9 & 4.2.4, 4.3.5–4.3.9)
# ============================================================

def repayments(psi1, psi2, e1, e2, mu1, mu2,
               a, b, k_e, k_mu, alpha,
               Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1):
    """
    Compute D1, D2 for the green project using your repayment
    structure (eq. 4.1.7–4.1.9) and the closed-form expression
    structure in eq. (4.3.8)–(4.3.9), but keeping δ separate.
    """

    # φ_t = (1 - ψ_t) z*_t  (eq. 4.1.9)
    phi1 = (1.0 - psi1) * z1
    phi2 = (1.0 - psi2) * z2

    g1 = g(e1, mu1, a)
    g2 = g(e2, mu2, a)

    # Using the explicit forms (4.3.8)–(4.3.9) without CAPM (q2),
    # i.e. the same structure but with δ not inside ψ* here.
    # Note: e*, μ* and ψ* already reflect δ, so we do NOT reinsert δ again.

    D1 = Dbar1 \
         - phi1 * x1 \
         - psi1 * (a * e1 + mu1)

    D2 = Dbar2 \
         - phi2 * x2 \
         - psi2 * (a * e2 + mu2) \
         - rho * (g1 - Gbar1)

    return D1, D2, g1, g2

def borrower_utility(psi1, psi2, e1, e2, mu1, mu2,
                     a, b, k_e, k_mu, alpha,
                     D1, D2, delta, k_gamma, Cbar1, Cbar2):
    """
    Borrower utility UB = UB1 + δ UB2, using R(e), c(e), h(μ) and γ(C̄)
    as in the dynamic model (eq. 3.1.18, 3.1.19), no-default case.
    """
    UB1 = R(e1, b) - D1 - c(e1, k_e) - h(mu1, k_mu, alpha) - gamma_cost(Cbar1, k_gamma)
    UB2 = R(e2, b) - D2 - c(e2, k_e) - h(mu2, k_mu, alpha) - gamma_cost(Cbar2, k_gamma)
    return UB1 + delta * UB2

def lender_utility(D1, D2, g1, g2, theta, delta):
    """
    Lender utility UL = D1 + δ D2 + θ (g1 + δ g2) (eq. 4.2.4).
    """
    return D1 + delta * D2 + theta * (g1 + delta * g2)

# ============================================================
# 6. SENSITIVITY ANALYSIS: MISPERCEPTION OF a (η_a)
# ============================================================

# Range of misestimation: η_a in [-0.5, 0.5] = [-50%, +50%]
eta_a_grid = np.linspace(-0.5, 0.5, 201)

UB_list  = []
UL_list  = []
psi1_list = []
psi2_list = []

# First compute the TRUE baseline (η_a = 0) for later comparison
psi1_base, psi2_base = lender_optimal_psi(
    a_true, b_true, k_e, k_mu, alpha, theta,
    delta, rho, z1, z2, x1, x2
)
e1_base, e2_base, mu1_base, mu2_base = borrower_optima(
    psi1_base, psi2_base, a_true, b_true, k_e, k_mu, alpha, delta, rho
)
D1_base, D2_base, g1_base, g2_base = repayments(
    psi1_base, psi2_base, e1_base, e2_base, mu1_base, mu2_base,
    a_true, b_true, k_e, k_mu, alpha,
    Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1
)
UB_base = borrower_utility(
    psi1_base, psi2_base, e1_base, e2_base, mu1_base, mu2_base,
    a_true, b_true, k_e, k_mu, alpha,
    D1_base, D2_base, delta, k_gamma, Cbar1, Cbar2
)
UL_base = lender_utility(D1_base, D2_base, g1_base, g2_base, theta, delta)

for eta_a in eta_a_grid:
    # Lender’s misperceived technology (only a distorted here)
    a_hat = a_true * (1.0 + eta_a)
    b_hat = b_true  # keep b correct for this first analysis

    # Lender chooses ψ* based on misperceived (a_hat, b_hat)
    psi1, psi2 = lender_optimal_psi(
        a_hat, b_hat, k_e, k_mu, alpha, theta,
        delta, rho, z1, z2, x1, x2
    )

    # Borrower reacts using TRUE technology
    e1, e2, mu1, mu2 = borrower_optima(
        psi1, psi2, a_true, b_true, k_e, k_mu, alpha, delta, rho
    )

    # Repayments and signals under true technology and chosen ψ*
    D1, D2, g1, g2 = repayments(
        psi1, psi2, e1, e2, mu1, mu2,
        a_true, b_true, k_e, k_mu, alpha,
        Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1
    )

    # Utilities
    UB = borrower_utility(
        psi1, psi2, e1, e2, mu1, mu2,
        a_true, b_true, k_e, k_mu, alpha,
        D1, D2, delta, k_gamma, Cbar1, Cbar2
    )
    UL = lender_utility(D1, D2, g1, g2, theta, delta)

    UB_list.append(UB)
    UL_list.append(UL)
    psi1_list.append(psi1)
    psi2_list.append(psi2)

UB_arr  = np.array(UB_list)
UL_arr  = np.array(UL_list)
psi1_arr = np.array(psi1_list)
psi2_arr = np.array(psi2_list)

# ============================================================
# 7. PLOTS: EFFECT OF MISPERCEIVED PROJECT QUALITY a
# ============================================================

# (1) Borrower and lender utility vs η_a
plt.figure()
plt.plot(eta_a_grid, UB_arr, label="Borrower utility $U_B$")
plt.plot(eta_a_grid, UL_arr, label="Lender utility $U_L$")
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.axhline(UB_base, linestyle=":", linewidth=1, label="$U_B$ baseline")
plt.axhline(UL_base, linestyle=":", linewidth=1, label="$U_L$ baseline")
plt.xlabel(r"Relative misestimation of $a$: $\eta_a$ (lender uses $\hat a = a(1+\eta_a)$)")
plt.ylabel("Utility")
plt.title("Effect of misperceived green productivity $a$ on $U_B$ and $U_L$")
plt.legend()
plt.tight_layout()
plt.savefig("BorrowerLenderUTIL.png", dpi=300, bbox_inches="tight")

# (2) Utility *differences* vs η_a (clearer sensitivity)
plt.figure()
plt.plot(eta_a_grid, UB_arr - UB_base, label=r"$U_B(\eta_a) - U_B^{baseline}$")
plt.plot(eta_a_grid, UL_arr - UL_base, label=r"$U_L(\eta_a) - U_L^{baseline}$")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel(r"Relative misestimation of $a$: $\eta_a$")
plt.ylabel("Utility difference vs baseline")
plt.title("Sensitivity of utilities to project quality misperception")
plt.legend()
plt.tight_layout()
plt.savefig("UTILDiffs.png", dpi=300, bbox_inches="tight")

# (3) ψ1* and ψ2* as a function of misperceived a
plt.figure()
plt.plot(eta_a_grid, psi1_arr, label=r"$\psi_1^*(\eta_a)$")
plt.plot(eta_a_grid, psi2_arr, label=r"$\psi_2^*(\eta_a)$")
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel(r"Relative misestimation of $a$: $\eta_a$")
plt.ylabel(r"SLB intensity $\psi_t$")
plt.title("Effect of project quality misperception on optimal SLB intensities")
plt.legend()
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("Maybe.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# COMPUTE EFFECT OF 5% MISPERCEPTION IN a (η_a = 0.05)
# ============================================================

eta_test = 0.05
a_hat = a_true * (1 + eta_test)

# Step 1: Lender chooses psi* under misperceived a
psi1_test, psi2_test = lender_optimal_psi(
    a_hat, b_true, k_e, k_mu, alpha, theta,
    delta, rho, z1, z2, x1, x2
)

# Step 2: Borrower reacts using TRUE technology
e1_test, e2_test, mu1_test, mu2_test = borrower_optima(
    psi1_test, psi2_test, a_true, b_true, k_e, k_mu, alpha, delta, rho
)

# Step 3: Compute repayments and utilities under true a,b
D1_test, D2_test, g1_test, g2_test = repayments(
    psi1_test, psi2_test,
    e1_test, e2_test, mu1_test, mu2_test,
    a_true, b_true, k_e, k_mu, alpha,
    Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1
)

UB_test = borrower_utility(
    psi1_test, psi2_test,
    e1_test, e2_test, mu1_test, mu2_test,
    a_true, b_true, k_e, k_mu, alpha,
    D1_test, D2_test, delta, k_gamma, Cbar1, Cbar2
)

UL_test = lender_utility(D1_test, D2_test, g1_test, g2_test, theta, delta)

# ============================================================
# PRINT RESULTS
# ============================================================

print("=== Effect of 5% lender overestimation of a (η_a = 0.05) ===")
print()

print(f"Baseline ψ₁*: {psi1_base:.4f},  New ψ₁*: {psi1_test:.4f},  Change = {psi1_test - psi1_base:.4f}")
print(f"Baseline ψ₂*: {psi2_base:.4f},  New ψ₂*: {psi2_test:.4f},  Change = {psi2_test - psi2_base:.4f}")
print()

print(f"Borrower utility UB_baseline: {UB_base:.4f}")
print(f"Borrower utility UB_test:     {UB_test:.4f}")
print(f"Absolute change in UB:        {UB_test - UB_base:.4f}")
print(f"Percent change in UB:         {(UB_test - UB_base) / UB_base * 100:.2f}%")
print()

print(f"Lender utility UL_baseline:   {UL_base:.4f}")
print(f"Lender utility UL_test:       {UL_test:.4f}")
print(f"Absolute change in UL:        {UL_test - UL_base:.4f}")
print(f"Percent change in UL:         {(UL_test - UL_base) / UL_base * 100:.2f}%")

# ============================================================
# COMPUTE EFFECT OF -5% MISPERCEPTION IN a (η_a = -0.05)
# ============================================================

eta_test = -0.05
a_hat = a_true * (1 + eta_test)

# Step 1: Lender chooses psi* under misperceived a
psi1_test, psi2_test = lender_optimal_psi(
    a_hat, b_true, k_e, k_mu, alpha, theta,
    delta, rho, z1, z2, x1, x2
)

# Step 2: Borrower reacts using TRUE technology
e1_test, e2_test, mu1_test, mu2_test = borrower_optima(
    psi1_test, psi2_test, a_true, b_true, k_e, k_mu, alpha, delta, rho
)

# Step 3: Compute repayments and utilities under true a,b
D1_test, D2_test, g1_test, g2_test = repayments(
    psi1_test, psi2_test,
    e1_test, e2_test, mu1_test, mu2_test,
    a_true, b_true, k_e, k_mu, alpha,
    Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1
)

UB_test = borrower_utility(
    psi1_test, psi2_test,
    e1_test, e2_test, mu1_test, mu2_test,
    a_true, b_true, k_e, k_mu, alpha,
    D1_test, D2_test, delta, k_gamma, Cbar1, Cbar2
)

UL_test = lender_utility(D1_test, D2_test, g1_test, g2_test, theta, delta)

# ============================================================
# PRINT RESULTS
# ============================================================

print("=== Effect of 5% lender underestimation of a (η_a = -0.05) ===")
print()

print(f"Baseline ψ₁*: {psi1_base:.4f},  New ψ₁*: {psi1_test:.4f},  Change = {psi1_test - psi1_base:.4f}")
print(f"Baseline ψ₂*: {psi2_base:.4f},  New ψ₂*: {psi2_test:.4f},  Change = {psi2_test - psi2_base:.4f}")
print()

print(f"Borrower utility UB_baseline: {UB_base:.4f}")
print(f"Borrower utility UB_test:     {UB_test:.4f}")
print(f"Absolute change in UB:        {UB_test - UB_base:.4f}")
print(f"Percent change in UB:         {(UB_test - UB_base) / UB_base * 100:.2f}%")
print()

print(f"Lender utility UL_baseline:   {UL_base:.4f}")
print(f"Lender utility UL_test:       {UL_test:.4f}")
print(f"Absolute change in UL:        {UL_test - UL_base:.4f}")
print(f"Percent change in UL:         {(UL_test - UL_base) / UL_base * 100:.2f}%")

# ============================================================
# 8. SENSITIVITY ANALYSIS: MISPERCEPTION OF b (η_b)
# ============================================================

# Range of misestimation: η_b in [-0.5, 0.5] = [-50%, +50%]
eta_b_grid = np.linspace(-0.5, 0.5, 201)

UB_b_list  = []
UL_b_list  = []
psi1_b_list = []
psi2_b_list = []

for eta_b in eta_b_grid:
    # Lender’s misperceived technology (now only b distorted)
    a_hat = a_true            # lender gets a right
    b_hat = b_true * (1.0 + eta_b)

    # Lender chooses ψ* based on misperceived (a_hat, b_hat)
    psi1_b, psi2_b = lender_optimal_psi(
        a_hat, b_hat, k_e, k_mu, alpha, theta,
        delta, rho, z1, z2, x1, x2
    )

    # Borrower reacts using TRUE technology (a_true, b_true)
    e1_b, e2_b, mu1_b, mu2_b = borrower_optima(
        psi1_b, psi2_b, a_true, b_true, k_e, k_mu, alpha, delta, rho
    )

    # Repayments and signals under true technology
    D1_b, D2_b, g1_b, g2_b = repayments(
        psi1_b, psi2_b, e1_b, e2_b, mu1_b, mu2_b,
        a_true, b_true, k_e, k_mu, alpha,
        Dbar1, Dbar2, z1, z2, x1, x2, rho, Gbar1
    )

    # Utilities under true technology
    UB_b = borrower_utility(
        psi1_b, psi2_b, e1_b, e2_b, mu1_b, mu2_b,
        a_true, b_true, k_e, k_mu, alpha,
        D1_b, D2_b, delta, k_gamma, Cbar1, Cbar2
    )
    UL_b = lender_utility(D1_b, D2_b, g1_b, g2_b, theta, delta)

    UB_b_list.append(UB_b)
    UL_b_list.append(UL_b)
    psi1_b_list.append(psi1_b)
    psi2_b_list.append(psi2_b)

UB_b_arr   = np.array(UB_b_list)
UL_b_arr   = np.array(UL_b_list)
psi1_b_arr = np.array(psi1_b_list)
psi2_b_arr = np.array(psi2_b_list)

plt.figure()
plt.plot(eta_b_grid, UB_b_arr, label="Borrower utility $U_B$")
plt.plot(eta_b_grid, UL_b_arr, label="Lender utility $U_L$")
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.axhline(UB_base, linestyle=":", linewidth=1, label="$U_B$ baseline")
plt.axhline(UL_base, linestyle=":", linewidth=1, label="$U_L$ baseline")
plt.xlabel(r"Relative misestimation of $b$: $\eta_b$ (lender uses $\hat b = b(1+\eta_b)$)")
plt.ylabel("Utility")
plt.title("Effect of misperceived repayment productivity $b$ on $U_B$ and $U_L$")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(eta_b_grid, UB_b_arr - UB_base, label=r"$U_B(\eta_b) - U_B^{baseline}$")
plt.plot(eta_b_grid, UL_b_arr - UL_base, label=r"$U_L(\eta_b) - U_L^{baseline}$")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel(r"Relative misestimation of $b$: $\eta_b$")
plt.ylabel("Utility difference vs baseline")
plt.title("Sensitivity of utilities to misperceived repayment productivity $b$")
plt.legend()
plt.tight_layout()
plt.savefig("UTILDiffsBbB.png", dpi=300, bbox_inches="tight")

plt.figure()
plt.plot(eta_b_grid, psi1_b_arr, label=r"$\psi_1^*(\eta_b)$")
plt.plot(eta_b_grid, psi2_b_arr, label=r"$\psi_2^*(\eta_b)$")
plt.axvline(0.0, linestyle="--", linewidth=1)
plt.xlabel(r"Relative misestimation of $b$: $\eta_b$")
plt.ylabel(r"SLB intensity $\psi_t$")
plt.title("Effect of repayment productivity misperception on optimal SLB intensities")
plt.legend()
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()

# ============================================================
# COMPUTE EFFECT OF 5% MISPERCEPTION IN b (η_b = 0.05)
# ============================================================

eta_b_test = 0.05
a_hat = a_true                    # lender perceives a correctly
b_hat = b_true * (1 + eta_b_test)  # lender overestimates b

# Step 1: Lender chooses psi* under misperceived b
psi1_btest, psi2_btest = lender_optimal_psi(
    a_hat, b_hat,
    k_e, k_mu, alpha, theta,
    delta, rho, z1, z2, x1, x2
)

# Step 2: Borrower reacts using TRUE technology (a_true, b_true)
e1_btest, e2_btest, mu1_btest, mu2_btest = borrower_optima(
    psi1_btest, psi2_btest,
    a_true, b_true,
    k_e, k_mu, alpha, delta, rho
)

# Step 3: Compute repayments and utilities under TRUE technology
D1_btest, D2_btest, g1_btest, g2_btest = repayments(
    psi1_btest, psi2_btest,
    e1_btest, e2_btest, mu1_btest, mu2_btest,
    a_true, b_true,
    k_e, k_mu, alpha,
    Dbar1, Dbar2,
    z1, z2, x1, x2,
    rho, Gbar1
)

UB_btest = borrower_utility(
    psi1_btest, psi2_btest,
    e1_btest, e2_btest, mu1_btest, mu2_btest,
    a_true, b_true, k_e, k_mu, alpha,
    D1_btest, D2_btest,
    delta, k_gamma, Cbar1, Cbar2
)

UL_btest = lender_utility(
    D1_btest, D2_btest,
    g1_btest, g2_btest,
    theta, delta
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("=== Effect of 5% lender overestimation of b (η_b = 0.05) ===")
print()

print(f"Baseline ψ₁*: {psi1_base:.4f},  New ψ₁*: {psi1_btest:.4f},  Change = {psi1_btest - psi1_base:.4f}")
print(f"Baseline ψ₂*: {psi2_base:.4f},  New ψ₂*: {psi2_btest:.4f},  Change = {psi2_btest - psi2_base:.4f}")
print()

print(f"Borrower utility UB_baseline: {UB_base:.4f}")
print(f"Borrower utility UB_test:     {UB_btest:.4f}")
print(f"Absolute change in UB:        {UB_btest - UB_base:.4f}")
print(f"Percent change in UB:         {(UB_btest - UB_base) / UB_base * 100:.2f}%")
print()

print(f"Lender utility UL_baseline:   {UL_base:.4f}")
print(f"Lender utility UL_test:       {UL_btest:.4f}")
print(f"Absolute change in UL:        {UL_btest - UL_base:.4f}")
print(f"Percent change in UL:         {(UL_btest - UL_base) / UL_base * 100:.2f}%")

# ============================================================
# COMPUTE EFFECT OF -5% MISPERCEPTION IN b (η_b = -0.05)
# ============================================================

eta_b_test = -0.05
a_hat = a_true                    # lender perceives a correctly
b_hat = b_true * (1 + eta_b_test)  # lender overestimates b

# Step 1: Lender chooses psi* under misperceived b
psi1_btest, psi2_btest = lender_optimal_psi(
    a_hat, b_hat,
    k_e, k_mu, alpha, theta,
    delta, rho, z1, z2, x1, x2
)

# Step 2: Borrower reacts using TRUE technology (a_true, b_true)
e1_btest, e2_btest, mu1_btest, mu2_btest = borrower_optima(
    psi1_btest, psi2_btest,
    a_true, b_true,
    k_e, k_mu, alpha, delta, rho
)

# Step 3: Compute repayments and utilities under TRUE technology
D1_btest, D2_btest, g1_btest, g2_btest = repayments(
    psi1_btest, psi2_btest,
    e1_btest, e2_btest, mu1_btest, mu2_btest,
    a_true, b_true,
    k_e, k_mu, alpha,
    Dbar1, Dbar2,
    z1, z2, x1, x2,
    rho, Gbar1
)

UB_btest = borrower_utility(
    psi1_btest, psi2_btest,
    e1_btest, e2_btest, mu1_btest, mu2_btest,
    a_true, b_true, k_e, k_mu, alpha,
    D1_btest, D2_btest,
    delta, k_gamma, Cbar1, Cbar2
)

UL_btest = lender_utility(
    D1_btest, D2_btest,
    g1_btest, g2_btest,
    theta, delta
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("=== Effect of 5% lender underestimation of b (η_b = -0.05) ===")
print()

print(f"Baseline ψ₁*: {psi1_base:.4f},  New ψ₁*: {psi1_btest:.4f},  Change = {psi1_btest - psi1_base:.4f}")
print(f"Baseline ψ₂*: {psi2_base:.4f},  New ψ₂*: {psi2_btest:.4f},  Change = {psi2_btest - psi2_base:.4f}")
print()

print(f"Borrower utility UB_baseline: {UB_base:.4f}")
print(f"Borrower utility UB_test:     {UB_btest:.4f}")
print(f"Absolute change in UB:        {UB_btest - UB_base:.4f}")
print(f"Percent change in UB:         {(UB_btest - UB_base) / UB_base * 100:.2f}%")
print()

print(f"Lender utility UL_baseline:   {UL_base:.4f}")
print(f"Lender utility UL_test:       {UL_btest:.4f}")
print(f"Absolute change in UL:        {UL_btest - UL_base:.4f}")
print(f"Percent change in UL:         {(UL_btest - UL_base) / UL_base * 100:.2f}%")