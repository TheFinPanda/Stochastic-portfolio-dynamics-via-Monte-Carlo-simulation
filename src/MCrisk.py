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
    mu=0.0005,
    sigma=0.02,
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