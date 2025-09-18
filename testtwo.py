import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
# tqdm fallback if not installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import invgamma
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# Optional for auto ARIMA: catch binary incompatibility errors
try:
    import pmdarima as pm
    AUTO_ARIMA = True
except (ImportError, ValueError):
    AUTO_ARIMA = False

# ─────── CONFIG DATACLASS ───────
@dataclass
class SARIMAXConfig:
    p: int = 2
    d: int = 1
    q: int = 2
    P: int = 1
    D: int = 1
    Q: int = 1
    S: int = 5  # season length (e.g., 5 for business week)
    test_size: float = 0.2
    n_iter: int = 3000
    burn_in: int = 1000
    alpha0: float = 0.01
    beta0: float = 0.01
    random_seed: int = 42
    data_path: str = 'datasets/AAPL_xlstm_date.csv'

# ─────── METRIC FUNCTIONS ───────
def compute_metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# ─────── TRAIN/TEST SPLIT ───────
def time_series_train_test_split(y, test_size):
    split = int(len(y) * (1 - test_size))
    return y.iloc[:split], y.iloc[split:]

# ─────── MAIN PIPELINE ───────
def main(config: SARIMAXConfig):
    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger(__name__)
    np.random.seed(config.random_seed)

    # 1. Load data
    df = pd.read_csv(config.data_path, header=None,
                     names=['Date','Open','High','Low','Close','Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    y = df['Close']

    # 2. (Optional) auto ARIMA for orders
    if AUTO_ARIMA:
        log.info('Running auto_arima to select orders...')
        arima_res = pm.auto_arima(y, seasonal=True, m=config.S,
                                  suppress_warnings=True, stepwise=True)
        config.p, config.d, config.q = arima_res.order
        config.P, config.D, config.Q = arima_res.seasonal_order[:-1]
        log.info(f"Auto ARIMA selected order={arima_res.order} seasonal_order={arima_res.seasonal_order}")

    # 3. Train/test split
    y_train, y_test = time_series_train_test_split(y, config.test_size)
    horizon = len(y_test)

    # 4. MLE SARIMAX fit
    log.info('Fitting MLE SARIMAX...')
    mle_model = sm.tsa.SARIMAX(
        y_train,
        order=(config.p, config.d, config.q),
        seasonal_order=(config.P, config.D, config.Q, config.S),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res_mle = mle_model.fit(disp=False)
    pred_mle = res_mle.get_forecast(steps=horizon).predicted_mean
    metrics_mle = compute_metrics(y_test, pred_mle)
    log.info(f"MLE SARIMAX metrics: {metrics_mle}")

    # 5. Residual diagnostics
    lb_test = sm.stats.acorr_ljungbox(res_mle.resid, lags=[10], return_df=True)
    log.info('Ljung-Box (lag=10) p-value: {:.4f}'.format(lb_test['lb_pvalue'].iloc[0]))

    # 6. Bayesian sampling of sigma2 (vectorized)
    T = len(y_train)
    resid = y_train - res_mle.fittedvalues
    alpha_post = config.alpha0 + T/2
    beta_post_const = config.beta0 + 0.5 * np.sum(resid**2)

    log.info('Sampling sigma^2 posterior...')
    sigma2_all = invgamma.rvs(
        a=alpha_post,
        scale=beta_post_const,
        size=config.n_iter
    )
    sigma2_samples = sigma2_all[config.burn_in:]

    # 7. Simulate forecast paths
    log.info('Simulating forecast paths...')
    all_paths = np.zeros((len(sigma2_samples), horizon))
    mle_params = res_mle.params.copy()
    sim_model = sm.tsa.SARIMAX(
        y_train,
        order=(config.p, config.d, config.q),
        seasonal_order=(config.P, config.D, config.Q, config.S),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    init_state = res_mle.filtered_state[:, -1]
    for i, sigma2 in enumerate(tqdm(sigma2_samples, desc='Bayesian sims')):
        params_i = mle_params.copy()
        params_i['sigma2'] = sigma2
        path = sim_model.simulate(
            params=params_i,
            nsimulations=horizon,
            initial_state=init_state,
            anchor='end'
        )
        all_paths[i, :] = path

    pred_bayes = all_paths.mean(axis=0)
    metrics_bayes = compute_metrics(y_test, pred_bayes)
    log.info(f"Bayesian SARIMAX metrics: {metrics_bayes}")

    # 8. Plot results
    fig, ax = plt.subplots(figsize=(12,6))
    idx = y_test.index
    ax.plot(idx, y_test, label='Actual', linewidth=2)
    ax.plot(idx, pred_mle, '--', label='MLE Forecast')
    ax.plot(idx, pred_bayes, ':', label='Bayesian Mean')
    # lower = np.percentile(all_paths, 2.5, axis=0)
    # upper = np.percentile(all_paths, 97.5, axis=0)
    # ax.fill_between(idx, lower, upper, alpha=0.3, label='95% Credible Interval')
    ax.set_title('SARIMAX Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.xaxis.set_tick_params(rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    cfg = SARIMAXConfig()
    main(cfg)
