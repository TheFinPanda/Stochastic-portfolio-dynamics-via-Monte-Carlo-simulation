import numpy as np

def simulate_portfolio_returns(
    mu: float,
    sigma: float,
    num_days: int,
    num_simulations: int,
    initial_value: float
):
    """
    terminal portfolio values using i.i.d. Gaussian returns
    """
    returns = np.random.normal(mu, sigma, size=(num_simulations, num_days))
    growth_factors = 1.0 + returns
    terminal_values = initial_value * np.prod(growth_factors, axis=1)
    return terminal_values


terminal_values = simulate_portfolio_returns(
    mu=0.00027,
    sigma=0.0105,
    num_days=365,
    num_simulations=10_000,
    initial_value=100_000
)


#Loss based VaR: Risk
#note: risk always should be scale aware
def compute_var(terminal_values, initial_value, alpha=0.95):
    pnl = terminal_values - initial_value
    var = -np.percentile(pnl, 100 * (1 - alpha))
    return var

var_95 = compute_var(
    terminal_values=terminal_values,
    initial_value=100_000,
    alpha=0.95
)
print(f"95% Risk P&L based: ${var_95:,.2f}")



#moving from arithmetic returns to log return as a next step: namely, geometric brownian motion with Euler–Maruyama discretization

def simulate_gbm(
        mu: float,
        sigma: float,
        T: float,
        num_steps: int,
        num_simulations: int,
        initial_value: float
):
    """
    Simulating terminal portfolio values using Geometric Brownian Motion.
    """
    dt = T / num_steps

    Z = np.random.normal(size=(num_simulations, num_steps))

    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_returns = drift + diffusion

    terminal_values = initial_value * np.exp(np.sum(log_returns, axis=1))

    return terminal_values


terminal_values_gbm = simulate_gbm(
    mu=0.1,
    sigma=0.2,
    T=1.0,
    num_steps=365,
    num_simulations=10_000,
    initial_value=100_000
)

var_95_gbm = compute_var(
    terminal_values_gbm,
    initial_value=100_000
)

print(f"95% VaR (GBM model): ${var_95_gbm:,.2f}")

### going beyond VaR, to CVaR

def compute_cvar(terminal_values, initial_value, alpha=0.95):
    pnl = terminal_values - initial_value
    var_threshold = np.percentile(pnl, 100 * (1 - alpha))

    tail_losses = pnl[pnl <= var_threshold]

    cvar = -np.mean(tail_losses)

    return cvar

cvar_95 = compute_cvar(
    terminal_values,
    initial_value=100_000,
    alpha=0.95
)

print(f"95% CVaR expected shortfall: ${cvar_95:,.2f}")


# ===============================
# MULTI-ASSET GBM WITH CORRELATION
# ===============================

def simulate_multi_asset_gbm(
    mu: np.ndarray,                 # expected returns (vector)
    sigma: np.ndarray,              # volatilities (vector)
    corr_matrix: np.ndarray,        # correlation matrix
    weights: np.ndarray,            # portfolio weights
    T: float,
    num_steps: int,
    num_simulations: int,
    initial_value: float
):
    """
    Basically simulateing correlated multi-asset GBM
    returns: terminal portfolio values.
    """

    num_assets = len(mu)
    dt = T / num_steps

    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    L = np.linalg.cholesky(cov_matrix)

    Z = np.random.normal(size=(num_simulations, num_steps, num_assets)) #indep shock
    correlated_shocks = np.einsum("ijk,kl->ijl", Z, L)

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = correlated_shocks * np.sqrt(dt)

    log_returns = drift + diffusion
    cumulative_log_returns = np.sum(log_returns, axis=1)

    asset_terminal_values = np.exp(cumulative_log_returns)
    portfolio_terminal = initial_value * np.dot(asset_terminal_values, weights)

    return portfolio_terminal

#portfolio structure I give now
mu = np.array([0.08, 0.12, 0.05])
sigma = np.array([0.15, 0.25, 0.10])

corr_matrix = np.array([
    [1.0, 0.6, 0.2],
    [0.6, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])

weights = np.array([0.4, 0.4, 0.2])

terminal_values_multi = simulate_multi_asset_gbm(
    mu=mu,
    sigma=sigma,
    corr_matrix=corr_matrix,
    weights=weights,
    T=1.0,
    num_steps=365,
    num_simulations=10_000,
    initial_value=100_000
)

var_95_multi = compute_var(
    terminal_values_multi,
    initial_value=100_000
)

cvar_95_multi = compute_cvar(
    terminal_values_multi,
    initial_value=100_000
)

print(f"95% VaR (Multi-Asset Portfolio): ${var_95_multi:,.2f}")
print(f"95% CVaR (Multi-Asset Portfolio): ${cvar_95_multi:,.2f}")