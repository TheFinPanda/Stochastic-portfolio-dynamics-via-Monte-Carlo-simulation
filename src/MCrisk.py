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