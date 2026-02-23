import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    mu: np.ndarray,
    sigma: np.ndarray,
    corr_matrix: np.ndarray,
    weights: np.ndarray,
    T: float,
    num_steps: int,
    num_simulations: int,
    initial_value: float
):

    num_assets = len(mu)
    dt = T / num_steps

    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    cov_matrix += 1e-10 * np.eye(num_assets)
    L = np.linalg.cholesky(cov_matrix)

    Z = np.random.normal(size=(num_simulations, num_steps, num_assets))
    correlated_shocks = np.einsum("ijk,kl->ijl", Z, L)

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = correlated_shocks * np.sqrt(dt)

    log_returns = drift + diffusion
    cumulative_log_returns = np.sum(log_returns, axis=1)

    asset_terminal_values = np.exp(cumulative_log_returns)
    portfolio_terminal = initial_value * np.dot(asset_terminal_values, weights)

    return portfolio_terminal


mu = np.array([0.08, 0.12, 0.05])
sigma = np.array([0.15, 0.25, 0.10])

corr_matrix = np.array([
    [1.0, 0.6, 0.2],
    [0.6, 1.0, 0.4],
    [0.2, 0.4, 1.0]
])

corr_high = np.array([
    [1.0, 0.9, 0.9],
    [0.9, 1.0, 0.9],
    [0.9, 0.9, 1.0]
])

corr_zero = np.eye(3)

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


terminal_high_corr = simulate_multi_asset_gbm(
    mu, sigma, corr_high, weights,
    T=1.0,
    num_steps=365,
    num_simulations=10_000,
    initial_value=100_000
)

print("High Correlation VaR:",
      compute_var(terminal_high_corr, 100_000))


terminal_zero_corr = simulate_multi_asset_gbm(
    mu, sigma, corr_zero, weights,
    T=1.0,
    num_steps=365,
    num_simulations=10_000,
    initial_value=100_000
)

print("Zero Correlation VaR:",
      compute_var(terminal_zero_corr, 100_000))


rho_values = []
var_values = []

for rho in np.linspace(0, 0.99, 25):
    corr_test = np.array([
        [1.0, rho, rho],
        [rho, 1.0, rho],
        [rho, rho, 1.0]
    ])

    terminal = simulate_multi_asset_gbm(
        mu, sigma, corr_test, weights,
        T=1.0,
        num_steps=365,
        num_simulations=10_000,
        initial_value=100_000
    )

    var = compute_var(terminal, 100_000)

    rho_values.append(rho)
    var_values.append(var)

    print(f"rho={rho:.2f} → VaR={var:,.0f}")

plt.figure(figsize=(10, 6))
plt.plot(rho_values, var_values)
plt.xlabel("Correlation (rho)")
plt.ylabel("95% VaR")
plt.title("Impact of Correlation on Portfolio VaR")
plt.grid(True)
plt.show()


# Now let's deal with real life data: working with SPY

tickers = ["SPY"]

data = yf.download(tickers, start="2015-01-01", auto_adjust=True)
prices = data["Close"] #important to close

returns = np.log(prices / prices.shift(1)).dropna()

mu = returns.mean() * 252
cov_matrix = returns.cov() * 252
sigma = np.sqrt(np.diag(cov_matrix))
corr_matrix = returns.corr()

weights = np.array([1.0])

terminal_real = simulate_multi_asset_gbm(
    mu.values,
    sigma,
    corr_matrix.values,
    weights,
    T=1.0,
    num_steps=252,
    num_simulations=10_000,
    initial_value=100_000
)

var_real = compute_var(terminal_real, 100_000)
cvar_real = compute_cvar(terminal_real, 100_000)

print(f"Real Data VaR: ${var_real:,.2f}")
print(f"Real Data CVaR: ${cvar_real:,.2f}")


#some trends to understand this ETF
plt.figure(figsize=(10,6))
plt.plot(prices["SPY"])
plt.title("SPY Adjusted Price History")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()


rolling_vol = returns["SPY"].rolling(252).std() * np.sqrt(252)

plt.figure(figsize=(10,6))
plt.plot(rolling_vol)
plt.title("SPY Rolling 1-Year Annualized Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.show()


#comparing with my MC
historical_pnl = returns["SPY"] * 100_000
historical_var = -np.percentile(historical_pnl, 5)

print(f"Historical 95% Daily VaR (SPY): ${historical_var:,.2f}")
