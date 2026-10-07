import numpy as np
from scipy import stats

TRADING_DAYS = 252  # 365 was calendar days; markets trade ~252 days a year


# ---------- Step 1: arithmetic model, now with consistent units ----------
def simulate_arithmetic(mu, sigma, num_days, num_simulations, initial_value, rng):
    """i.i.d. Gaussian daily returns. mu and sigma are DAILY."""
    returns = rng.normal(mu, sigma, size=(num_simulations, num_days))
    return initial_value * np.prod(1.0 + returns, axis=1)


# ---------- Step 2: GBM, now keeping the whole path ----------
def simulate_gbm_paths(mu, sigma, T, num_steps, num_simulations, initial_value, rng):
    """mu and sigma are ANNUAL, T is in years. Returns shape (sims, steps + 1)."""
    dt = T / num_steps
    Z = rng.normal(size=(num_simulations, num_steps))

    # we step log S, not S, so this is exact for any dt (Euler on S itself would not be)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.cumsum(log_returns, axis=1)
    paths = initial_value * np.exp(log_paths)
    start = np.full((num_simulations, 1), initial_value)
    return np.hstack([start, paths])


# ---------- Step 3: risk metrics ----------
# risk should always be scale aware: dollars here, and the horizon matters
def compute_var(values, initial_value, alpha=0.95):
    pnl = values - initial_value
    return -np.percentile(pnl, 100 * (1 - alpha))


def compute_cvar(values, initial_value, alpha=0.95):
    # average loss in the worst (1 - alpha) tail, always >= VaR
    pnl = values - initial_value
    cutoff = np.percentile(pnl, 100 * (1 - alpha))
    return -pnl[pnl <= cutoff].mean()


def var_std_error(values, initial_value, alpha=0.95, n_boot=200, rng=None):
    # monte carlo error: how much would VaR move if we reran with other random draws?
    rng = rng or np.random.default_rng()
    n = len(values)
    est = [compute_var(rng.choice(values, n), initial_value, alpha) for _ in range(n_boot)]
    return np.std(est, ddof=1)


def max_drawdown(paths):
    running_peak = np.maximum.accumulate(paths, axis=1)
    return ((paths - running_peak) / running_peak).min(axis=1)  # negative numbers


# ---------- Step 4: sanity check against the exact answer ----------
def gbm_var_exact(mu, sigma, T, initial_value, alpha=0.95):
    # S_T is lognormal, so its quantile has a closed form
    z = stats.norm.ppf(1 - alpha)
    q = initial_value * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return initial_value - q


if __name__ == "__main__":
    rng = np.random.default_rng(42)  # seeded so results are reproducible
    S0 = 100_000
    mu, sigma = 0.10, 0.20  # annual

    # arithmetic model, daily params chosen to match the annual ones
    tv_arith = simulate_arithmetic(
        mu / TRADING_DAYS, sigma / np.sqrt(TRADING_DAYS), TRADING_DAYS, 10_000, S0, rng
    )
    print(f"Arithmetic 1y 95% VaR: ${compute_var(tv_arith, S0):,.0f}")

    # GBM
    paths = simulate_gbm_paths(mu, sigma, 1.0, TRADING_DAYS, 10_000, S0, rng)
    tv = paths[:, -1]

    sim_var = compute_var(tv, S0)
    se = var_std_error(tv, S0, rng=rng)
    exact = gbm_var_exact(mu, sigma, 1.0, S0)
    print(f"GBM 1y 95% VaR:        ${sim_var:,.0f}  (+/- ${se:,.0f}),  exact: ${exact:,.0f}")
    print(f"GBM 1y 95% CVaR:       ${compute_cvar(tv, S0):,.0f}")
    print(f"Mean max drawdown:     {max_drawdown(paths).mean():.1%}")

    # same model, short horizon: this is the scale risk desks actually use
    one_day = paths[:, 1]
    print(f"GBM 1-day 95% VaR:     ${compute_var(one_day, S0):,.0f}  "
          f"(exact: ${gbm_var_exact(mu, sigma, 1 / TRADING_DAYS, S0):,.0f})")