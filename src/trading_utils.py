import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from rl_env import ActiveCryptoEnv


def compute_indicators(df, fit_hmm=False, hmm_artifact_path="models/hmm_regime.pkl", hmm_model=None, state_map=None):
    """
    Compute all technical indicators required by the RL environment.
    Avoids backward filling to prevent future data leakage.
    """
    df = df.copy()
    df["sma_7"] = df["close"].rolling(window=7, min_periods=1).mean()
    df["sma_25"] = df["close"].rolling(window=25, min_periods=1).mean()
    df["sma_99"] = df["close"].rolling(window=99, min_periods=1).mean()

    # RSI 14 — Wilder's EWM smoothing (alpha=1/14)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = np.where(loss == 0.0, np.nan, gain / loss)
    rsi = np.where(loss == 0.0, np.where(gain > 0.0, 100.0, 50.0), 100.0 - (100.0 / (1.0 + rs)))
    df["rsi_14"] = pd.Series(rsi, index=df.index).fillna(50.0)

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Volume SMA & ROC
    df["volume_sma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["roc_12"] = df["close"].pct_change(periods=12).fillna(0.0)

    # 5-Regime Classifier (deterministic inference, no implicit fitting)
    df["market_regime"] = compute_market_regimes(
        df,
        fit_hmm=fit_hmm,
        hmm_artifact_path=hmm_artifact_path,
        hmm_model=hmm_model,
        state_map=state_map
    )

    return df


def fit_market_regimes_hmm(df, window=30, n_components=5, random_state=42):
    """
    Fit a Gaussian Hidden Markov Model strictly on training data.
    Uses deterministic clipping for low-volatility stability (no random noise).
    Returns (hmm_model, state_map).
    """
    from hmmlearn.hmm import GaussianHMM

    log_returns = np.log(df["close"] / df["close"].shift(1).replace(0, np.nan)).fillna(0)
    rolling_vol = log_returns.rolling(window=window, min_periods=1).std().fillna(0)

    # Clip volatility to a deterministic lower bound to prevent singular covariance
    vol_stable = np.maximum(rolling_vol.values, 1e-6)
    X = np.column_stack([log_returns.values, vol_stable])

    hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=random_state)
    hmm_model.fit(X)

    sorted_states = np.argsort(hmm_model.means_[:, 1])
    state_map = {
        sorted_states[0]: 0.0,   # Lowest vol -> Sideways
        sorted_states[1]: 1.0,   # Low vol -> Consolidating Bull
        sorted_states[2]: -1.0,  # Medium vol -> Consolidating Bear
        sorted_states[3]: 2.0,   # High vol -> Runaway Bull
        sorted_states[4]: -2.0   # Highest vol -> Panic Bear
    }
    return hmm_model, state_map


def predict_market_regimes_hmm(df, hmm_model, state_map, window=30):
    """
    Deterministically infer market regimes via causal forward filtering.
    Zero random noise is injected during inference.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1).replace(0, np.nan)).fillna(0)
    rolling_vol = log_returns.rolling(window=window, min_periods=1).std().fillna(0)
    vol_stable = np.maximum(rolling_vol.values, 1e-6)
    X = np.column_stack([log_returns.values, vol_stable])

    n_samples, n_features = X.shape
    n_components = hmm_model.n_components

    log_B = np.zeros((n_samples, n_components))
    for i in range(n_components):
        mean = hmm_model.means_[i]
        cov = hmm_model.covars_[i]
        diff = X - mean
        if cov.ndim == 1:
            log_prob = -0.5 * n_features * np.log(2.0 * np.pi) - 0.5 * np.sum(np.log(cov)) - 0.5 * np.sum(diff**2 / cov, axis=1)
        else:
            sign, logdet = np.linalg.slogdet(cov)
            inv_cov = np.linalg.inv(cov)
            mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)
            log_prob = -0.5 * n_features * np.log(2.0 * np.pi) - 0.5 * logdet - 0.5 * mahalanobis
        log_B[:, i] = log_prob

    log_alpha = np.zeros((n_samples, n_components))
    log_startprob = np.log(np.clip(hmm_model.startprob_, 1e-30, None))
    log_transmat = np.log(np.clip(hmm_model.transmat_, 1e-30, None))

    log_alpha[0] = log_startprob + log_B[0]
    for t in range(1, n_samples):
        temp = log_alpha[t-1, :, None] + log_transmat
        temp_max = np.max(temp, axis=0)
        log_alpha[t] = log_B[t] + temp_max + np.log(np.sum(np.exp(temp - temp_max), axis=0))

    states = np.argmax(log_alpha, axis=1)
    regime = np.array([state_map[s] for s in states], dtype=np.float32)
    return regime


def compute_market_regimes(df, window=30, fit_hmm=False, hmm_artifact_path="models/hmm_regime.pkl", hmm_model=None, state_map=None):
    """
    Classifies market into 5 distinct regimes using Hidden Markov Model (GaussianHMM).
    Separates fitting from inference. Evaluation must never fit or overwrite an HMM implicitly.
    """
    import joblib
    import os

    if hmm_model is not None and state_map is not None:
        return predict_market_regimes_hmm(df, hmm_model, state_map, window=window)

    if fit_hmm:
        hmm_model, state_map = fit_market_regimes_hmm(df, window=window)
        os.makedirs(os.path.dirname(hmm_artifact_path), exist_ok=True)
        joblib.dump({"model": hmm_model, "map": state_map}, hmm_artifact_path)
        return predict_market_regimes_hmm(df, hmm_model, state_map, window=window)
    else:
        if not os.path.exists(hmm_artifact_path):
            raise FileNotFoundError(
                f"Required HMM artifact not found at '{hmm_artifact_path}'. "
                "Evaluation must never fit an HMM implicitly. Fit on training data first."
            )
        saved = joblib.load(hmm_artifact_path)
        return predict_market_regimes_hmm(df, saved["model"], saved["map"], window=window)


def run_single_fold_backtest(
    model, test_df, initial_balance=10000.0, is_daily=False, deterministic=True, **env_kwargs
):
    """
    Run a highly optimized deterministic backtest on a 5-action Long/Short space.
    0: Neutral, 1: Buy (Long), 2: Exit Long, 3: Sell (Short), 4: Exit Short
    Bypasses SB3 wrapper overhead to run ~12x faster on CPU.
    """
    # Seed random generators to ensure 100% reproducible stochastic backtest runs
    import torch
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    # Enforce realistic commission and slippage by default (0.1% fee, 0.05% slippage)
    if "commission_rate" not in env_kwargs:
        env_kwargs["commission_rate"] = 0.001
    if "slippage_rate" not in env_kwargs:
        env_kwargs["slippage_rate"] = 0.0005
    # Disable drawdown-based early termination by default during evaluation backtests
    if "max_drawdown_cap" not in env_kwargs:
        env_kwargs["max_drawdown_cap"] = 1.0

    # Auto-extract parameters from model attributes if not explicitly passed
    if "cooldown_steps" not in env_kwargs and hasattr(model, "cooldown_steps"):
        env_kwargs["cooldown_steps"] = model.cooldown_steps
    if "inactivity_penalty" not in env_kwargs and hasattr(model, "inactivity_penalty"):
        env_kwargs["inactivity_penalty"] = model.inactivity_penalty
    if "negative_pnl_penalty" not in env_kwargs and hasattr(model, "negative_pnl_penalty"):
        env_kwargs["negative_pnl_penalty"] = model.negative_pnl_penalty

    # Auto-detect obs_dim from model if it is a standard SB3 model or state dict
    use_regime = True
    use_context = True
    model_obs_dim = 17
    if hasattr(model, "observation_space"):
        model_obs_dim = model.observation_space.shape[0] // 8
        use_context = (model_obs_dim in (16, 17))
        use_regime = (model_obs_dim in (12, 17))  # 12=old regime, 17=new regime
    elif hasattr(model, "policy"):
        for name, param in model.policy.state_dict().items():
            if "features_extractor.embedding.weight" in name:
                model_obs_dim = param.shape[1]
                use_context = (model_obs_dim in (16, 17))
                use_regime = (model_obs_dim in (12, 17))  # 12=old regime, 17=new regime
                break
                
    env_kwargs["include_regime"] = use_regime
    env_kwargs["include_context"] = use_context

    # Instantiate raw environment directly to bypass vectorized wrappers overhead
    test_env = ActiveCryptoEnv(
        test_df, initial_balance=initial_balance, render_mode=None, random_start=False, max_steps=len(test_df), **env_kwargs
    )
    
    trades = []
    net_worth_history = [initial_balance]

    # Pre-extract numpy array for 10-50x faster indexing than pandas .loc inside the loop
    np_close = test_df["close"].to_numpy(dtype=np.float32)

    # Initialize raw frame-stack buffer for high-performance sequence tracking
    n_stack = 8
    obs_dim = model_obs_dim
    buffer = np.zeros((n_stack, obs_dim), dtype=np.float32)
    
    obs, _ = test_env.reset()
    # Match VecFrameStack.reset(): earlier frames are zero, newest is reset obs.
    buffer[-1] = obs

    import torch
    import contextlib

    use_optimized = hasattr(model, "policy") and hasattr(model.policy, "_predict")
    original_device = None
    if use_optimized:
        policy = model.policy
        # Direct distribution calls must disable training-time dropout, like SB3 predict().
        policy.set_training_mode(False)
        # Use the policy's existing device to avoid thread-safety issues from mutating a shared global model's device
        device = next(policy.parameters()).device
        context = torch.inference_mode()
    else:
        device = torch.device("cpu")
        context = contextlib.nullcontext()

    total_reward = 0.0
    policy_action_counts = [0, 0, 0]

    dropped_rows = len(test_df) - test_env.n_rows

    with context:
        for i in range(test_env.n_rows - 1):
            # Flatten buffer to shape (1, 88) matching VecFrameStack structure
            flat_obs = buffer.reshape(1, -1)
            
            if use_optimized:
                # Normalize observation if model has a normalizer wrapper, aligning with SB3 predict() behavior
                vec_norm = model.get_vec_normalize_env() if hasattr(model, "get_vec_normalize_env") else None
                flat_obs_norm = vec_norm.normalize_obs(flat_obs) if vec_norm is not None else flat_obs
                obs_tensor = torch.from_numpy(flat_obs_norm).to(device=device, dtype=torch.float32)
                
                # Check if environment supports action masking
                if hasattr(test_env, "action_masks"):
                    mask = test_env.action_masks()
                    mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
                    dist = policy.get_distribution(obs_tensor)
                    logits = dist.distribution.logits.clone()
                    logits[:, ~mask_tensor] = -1e9
                    if deterministic:
                        action = int(torch.argmax(logits, dim=-1)[0].item())
                    else:
                        masked_dist = torch.distributions.Categorical(logits=logits)
                        action = int(masked_dist.sample()[0].item())
                else:
                    # Direct policy prediction to completely bypass SB3 wrapper layers overhead
                    action_tensor = policy._predict(obs_tensor, deterministic=deterministic)
                    action = int(action_tensor[0].item())
            else:
                if hasattr(test_env, "action_masks"):
                    action_masks = test_env.action_masks()
                    try:
                        action, _ = model.predict(flat_obs, action_masks=action_masks, deterministic=deterministic)
                    except TypeError:
                        action, _ = model.predict(flat_obs, deterministic=deterministic)
                else:
                    action, _ = model.predict(flat_obs, deterministic=deterministic)
                if isinstance(action, np.ndarray):
                    action = int(action[0])
                else:
                    action = int(action)

            # Retrieve price from active environment np_close array to align with NaN dropping
            price = test_env.np_close[i]

            # Track what happened for logs/charts
            policy_action_counts[action] += 1
            executed = False
            action_name = ""
            execution_reason = "policy_target"

            if action == 0:
                if test_env.is_long:
                    executed, action_name = True, "EXIT LONG"
                elif test_env.is_short:
                    executed, action_name = True, "EXIT SHORT"
            elif action == 1:
                if test_env.is_short and test_env.cooldown == 0:
                    executed, action_name = True, "FLIP TO LONG"
                elif not test_env.is_long and not test_env.is_short and test_env.cooldown == 0:
                    executed, action_name = True, "BUY (LONG)"
            elif action == 2:
                if test_env.is_long and test_env.cooldown == 0:
                    executed, action_name = True, "FLIP TO SHORT"
                elif not test_env.is_short and not test_env.is_long and test_env.cooldown == 0:
                    executed, action_name = True, "SELL (SHORT)"

            # Step the raw environment directly
            next_obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward

            # Capture duration-based force closes as explicit exit events
            if "duration_force_closed" in info:
                action_name = "EXIT LONG" if info["duration_force_closed"] == "LONG" else "EXIT SHORT"
                executed = True
                execution_reason = "maximum_holding_duration"

            current_nw = info["net_worth"]
            net_worth_history.append(current_nw)

            if executed:
                trades.append(
                    {
                        "Step": i + dropped_rows,
                        "Action": action_name,
                        "Reason": execution_reason,
                        "PolicyTarget": action,
                        "Price": price,
                        "Net Worth": current_nw,
                    }
                )

            # Update the sliding observation stack buffer
            buffer[:-1] = buffer[1:]
            buffer[-1] = next_obs

            if done:
                break

    # --- Force-Close Logging ---
    if test_env.is_long or test_env.is_short:
        final_idx = len(test_df) - 1
        final_price = np_close[-1]
        final_action = "EXIT LONG" if test_env.is_long else "EXIT SHORT"
        trades.append(
            {
                "Step": final_idx,
                "Action": final_action,
                "Price": final_price,
                "Net Worth": net_worth_history[-1],
            }
        )

    # Prefix net_worth_history with [initial_balance] for dropped NaN rows
    if dropped_rows > 0:
        net_worth_history = [initial_balance] * dropped_rows + net_worth_history
    
    # In case of early termination, pad the end with the last known net worth to match len(test_df)
    if len(net_worth_history) < len(test_df):
        padding_needed = len(test_df) - len(net_worth_history)
        net_worth_history += [net_worth_history[-1]] * padding_needed

    # --- Compute Metrics ---
    final_worth = net_worth_history[-1]
    roi = ((final_worth - initial_balance) / initial_balance) * 100
    bh_return = (
        (test_df.loc[len(test_df) - 1, "close"] - test_df.loc[0, "close"])
        / test_df.loc[0, "close"]
    ) * 100

    nw_series = pd.Series(net_worth_history)
    step_returns = nw_series.pct_change().dropna()

    # Timeframe multiplier (8760 for 1H, 365 or 252 for Daily)
    # Crypto trades 24/7, so 365 for daily.
    multiplier = 365 if is_daily else 8760

    if len(step_returns) > 1 and step_returns.std() > 0:
        # Annualized Sharpe
        sharpe = (step_returns.mean() / step_returns.std()) * np.sqrt(multiplier)
        sharpe_raw = step_returns.mean() / step_returns.std()
        
        # Sortino Ratio (fix downside math using RMS of all returns below MAR=0)
        # N = len(step_returns), Denominator is N-1
        downside_sq = np.minimum(step_returns, 0.0)**2
        downside_rms = np.sqrt(downside_sq.sum() / (len(step_returns) - 1)) if len(step_returns) > 1 else 0.0
        
        sortino = (
            (step_returns.mean() / downside_rms) * np.sqrt(multiplier)
            if downside_rms > 0
            else 0.0
        )
        sortino_raw = (
            step_returns.mean() / downside_rms
            if downside_rms > 0
            else 0.0
        )
        cumulative = (1 + step_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown_series = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown_series.min() * 100
        calmar = (roi * (multiplier / len(test_df))) / abs(max_dd) if max_dd != 0 else 0.0
    else:
        sharpe, sharpe_raw, sortino, sortino_raw, max_dd, calmar = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Restore the policy to its original CUDA device if it was temporarily offloaded to CPU
    if original_device is not None:
        policy.to(original_device)

    return {
        "ROI %": round(roi, 2),
        "B&H %": round(bh_return, 2),
        "Sharpe": round(sharpe, 2),
        "Sharpe (Raw)": round(sharpe_raw, 4),
        "Sortino": round(sortino, 2),
        "Sortino (Raw)": round(sortino_raw, 4),
        "Max DD %": round(max_dd, 2),
        "Calmar": round(calmar, 2),
        "Trades": len(trades),
        "Final $": round(final_worth, 2),
        "TradeHistory": trades,
        "PolicyActionCounts": dict(zip(["cash", "long", "short"], policy_action_counts)),
        "NetWorthHistory": net_worth_history,
        "ValidationReward": total_reward,
    }


def compute_deflated_sharpe_ratio(returns, n_trials=10, benchmark_sharpe=0.0):
    """
    Computes Marcos López de Prado's Deflated Sharpe Ratio (DSR) to adjust 
    the estimated Sharpe Ratio for multiple testing trial inflation and non-normality.
    """
    try:
        from utils.dsr import deflated_sharpe_ratio
        dsr_res = deflated_sharpe_ratio(returns, n_trials=n_trials, benchmark_sharpe=benchmark_sharpe, annualization_factor=8760)
        return round(dsr_res.sharpe_ratio, 2), round(dsr_res.dsr_probability, 4)
    except Exception:
        import scipy.stats as stats
        returns = np.array(returns)
        N = len(returns)
        if N < 2 or np.std(returns) == 0:
            return 0.0, 0.0

        sr_period = np.mean(returns) / np.std(returns)
        sr_hat = sr_period * np.sqrt(8760)
        
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns, fisher=False)) # Pearson kurtosis (normal = 3)
        
        mertens_factor = 1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period ** 2)
        sr_var = max(mertens_factor / (N - 1.0), 1e-12)
        sr_std = np.sqrt(sr_var) * np.sqrt(8760)
        
        em_const = 0.5772156649
        exp_max_sr = (1 - em_const) * stats.norm.ppf(1 - 1/n_trials) + em_const * stats.norm.ppf(1 - 1/(n_trials * np.e))
        
        dsr_stat = (sr_hat - exp_max_sr) / max(sr_std, 1e-8)
        p_value = float(stats.norm.cdf(dsr_stat))
        
        return round(sr_hat, 2), round(p_value, 4)
