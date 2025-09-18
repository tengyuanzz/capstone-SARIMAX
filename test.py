import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import invgamma
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def time_series_train_test_split(y, test_size=0.2):
    """
    Split a pandas Series into train/test sets by time order.
    """
    split = int(len(y) * (1 - test_size))
    return y.iloc[:split], y.iloc[split:]

def compute_forecast_metrics(y_true, y_pred):
    """
    Compute MSE, RMSE, MAE, and MAPE for forecast evaluation.
    """
    mse = mean_squared_error(y_true, y_pred)                         # :contentReference[oaicite:7]{index=7}
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)                        # :contentReference[oaicite:8]{index=8}
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100       # :contentReference[oaicite:9]{index=9}
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def bayesian_local_level(y, n_iter=2000, burn_in=500, alpha0=0.01, beta0=0.01):
    """
    Gibbs sampler for a local level model:
      - State draw via simulation smoother
      - Variance draws from inverse-Gamma posteriors
    Returns posterior state draws and variance samples.
    """
    model = sm.tsa.UnobservedComponents(y, level='llevel')           # :contentReference[oaicite:10]{index=10}
    # Initialize from MLE
    res      = model.fit(disp=False)
    obs_var  = res.params['sigma2.irregular']
    state_var= res.params['sigma2.level']

    T = len(y)
    state_draws       = np.zeros((n_iter - burn_in, T))
    obs_var_samples  = np.zeros(n_iter)
    state_var_samples= np.zeros(n_iter)

    for i in range(n_iter):
        model.update([obs_var, state_var])
        sim = model.simulation_smoother()                            # :contentReference[oaicite:11]{index=11}
        sim.simulate()
        alpha = sim.simulated_state[0]

        rss_obs = np.sum((y - alpha)**2)
        obs_var = invgamma.rvs(a=alpha0 + T/2, scale=beta0 + rss_obs/2)  # :contentReference[oaicite:12]{index=12}

        rss_trend = np.sum(np.diff(alpha)**2)
        state_var = invgamma.rvs(a=alpha0 + (T-1)/2, scale=beta0 + rss_trend/2)

        # Store
        obs_var_samples[i]   = obs_var
        state_var_samples[i] = state_var
        if i >= burn_in:
            state_draws[i - burn_in, :] = alpha

    return state_draws, obs_var_samples, state_var_samples

# ——— Main workflow ———

# 1. Load data
path = r'datasets/AAPL_xlstm_date.csv'
df = pd.read_csv(path, header=None,
                 names=["Date","Open","High","Low","Close","Volume"])
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
y = df['Close']

# 2. Train/test split
split     = int(len(y) * 0.8)
y_train   = y.iloc[:split]
y_test    = y.iloc[split:]

# 3. Fit a local‑linear‑trend model
model_trend = sm.tsa.UnobservedComponents(
    y_train,
    level='lltrend',    # <-- local‑level + trend
    seasonal=None       # or seasonal=12 if needed
)
res_trend   = model_trend.fit(disp=False)

# 4. Forecast the test period
pred_trend = res_trend.get_forecast(steps=len(y_test)).predicted_mean

# 5. Compute evaluation metrics
mse  = mean_squared_error(y_test, pred_trend)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, pred_trend)
mape = mean_absolute_percentage_error(y_test, pred_trend) * 100
print("LLTrend Forecast Metrics:",
      {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

# 6. Plot only ground truth vs. trending forecast
plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test,        label='Actual (Test)',    linewidth=2)
plt.plot(y_test.index, pred_trend,    label='LLTrend Forecast',  linestyle='--', linewidth=2)
plt.title("Test‑Set: Actual vs. Local‑Linear‑Trend Forecast")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.show()