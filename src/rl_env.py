"""
Reinforcement Learning Trading Environment
========================================================================
A 3-target-position environment supporting Cash, Long, and Short:
  0: Cash (Flat)
  1: Target Long
  2: Target Short
Reward is based on the step-by-step differential log-wealth return of portfolio net worth.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ActiveCryptoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        churn_penalty=0.01,
        render_mode=None,
        random_start=True,
        max_steps=2160,
        max_trade_duration=48,
        cooldown_steps=1,
        inactivity_penalty=0.0005,
        negative_pnl_penalty=0.0,
        volatility_scaling=0.5,
        drawdown_penalty_coef=0.1,
        max_drawdown_cap=0.15,
        include_regime=True,
        include_context=True,
        dsr_eta=0.01,
        use_dsor=False,
        use_atr_stop_loss=False,
        atr_multiplier=2.5,
        warm_start_dsr=False,
        gradual_drawdown=False,
        drawdown_cap_penalty=5.0,
    ):
        super(ActiveCryptoEnv, self).__init__()
        self.render_mode = render_mode
        self.df = df.dropna().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.churn_penalty = churn_penalty
        self.random_start = random_start
        self.max_steps = max_steps
        self.max_trade_duration = max_trade_duration
        self.cooldown_steps = cooldown_steps
        self.inactivity_penalty = inactivity_penalty
        self.negative_pnl_penalty = negative_pnl_penalty
        self.volatility_scaling = volatility_scaling
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.max_drawdown_cap = max_drawdown_cap
        self.gradual_drawdown = gradual_drawdown
        self.drawdown_cap_penalty = drawdown_cap_penalty
        self.include_regime = include_regime
        self.include_context = include_context
        self.dsr_eta = dsr_eta
        self.use_dsor = use_dsor
        self.use_atr_stop_loss = use_atr_stop_loss
        self.atr_multiplier = atr_multiplier
        self.warm_start_dsr = warm_start_dsr
        self.trade_closed = False
        self.realized_pnl = 0.0

        # --- PRE-EXTRACT NUMPY ARRAYS (10-30x faster than pandas .loc inside step()) ---
        # OHLC approach: raw stationary candle ratios instead of lagging technical indicators
        self.np_open  = self.df["open"].to_numpy(dtype=np.float32)
        self.np_high  = self.df["high"].to_numpy(dtype=np.float32)
        self.np_low   = self.df["low"].to_numpy(dtype=np.float32)
        self.np_close = self.df["close"].to_numpy(dtype=np.float32)
        self.np_volume = self.df["volume"].to_numpy(dtype=np.float32)
        vol_sma_col = (
            self.df["volume_sma_20"].to_numpy(dtype=np.float32)
            if "volume_sma_20" in self.df.columns
            else np.ones(len(self.df), dtype=np.float32)
        )
        vol_sma_col = np.where(
            (np.isnan(vol_sma_col)) | (vol_sma_col == 0), 1.0, vol_sma_col
        )
        self.np_vol_sma = vol_sma_col
        self.n_rows = len(self.np_close)

        # Safely extract or compute market regime local calculation to avoid circular imports
        if self.include_regime:
            if "market_regime" not in self.df.columns:
                # Local dynamic 5-regime classifier calculation if missing
                import trading_utils
                self.df["market_regime"] = trading_utils.compute_market_regimes(self.df)
            
            self.np_regime = self.df["market_regime"].to_numpy(dtype=np.float32)
        else:
            self.np_regime = np.zeros(self.n_rows, dtype=np.float32)

        # Vectorized precomputation of technical candle features to achieve 10-20x validation speedup
        open_ratio_vec = np.where(self.np_close != 0, (self.np_open - self.np_close) / self.np_close * 200.0, 0.0)
        self.np_open_ratio = np.clip(open_ratio_vec, -10.0, 10.0)

        high_ratio_vec = np.where(self.np_close != 0, (self.np_high - self.np_close) / self.np_close * 200.0, 0.0)
        self.np_high_ratio = np.clip(high_ratio_vec, 0.0, 10.0)

        low_ratio_vec = np.where(self.np_close != 0, (self.np_low - self.np_close) / self.np_close * 200.0, 0.0)
        self.np_low_ratio = np.clip(low_ratio_vec, -10.0, 0.0)

        volatility_vec = np.where(self.np_close != 0, (self.np_high - self.np_low) / self.np_close * 200.0, 0.0)
        self.np_volatility = np.clip(volatility_vec, 0.0, 10.0)

        close_prev = np.zeros_like(self.np_close)
        close_prev[1:] = self.np_close[:-1]
        close_change_vec = np.zeros_like(self.np_close)
        valid_mask_1 = close_prev != 0
        close_change_vec[valid_mask_1] = (self.np_close[valid_mask_1] - close_prev[valid_mask_1]) / close_prev[valid_mask_1] * 200.0
        self.np_close_change = np.clip(close_change_vec, -10.0, 10.0)

        close_prev_2 = np.zeros_like(self.np_close)
        close_prev_2[2:] = self.np_close[:-2]
        close_prev_1 = np.zeros_like(self.np_close)
        close_prev_1[2:] = self.np_close[1:-1]
        close_change_prev_vec = np.zeros_like(self.np_close)
        valid_mask_2 = close_prev_2 != 0
        close_change_prev_vec[valid_mask_2] = (close_prev_1[valid_mask_2] - close_prev_2[valid_mask_2]) / close_prev_2[valid_mask_2] * 200.0
        self.np_close_change_prev = np.clip(close_change_prev_vec, -10.0, 10.0)

        volume_norm_vec = np.where(self.np_vol_sma != 0, self.np_volume / self.np_vol_sma / 5.0, 0.0)
        self.np_volume_norm = np.clip(volume_norm_vec, 0.0, 2.0)

        # --- MULTI-SCALE CONTEXT FEATURES (Reduces Critic partial observability) ---
        # 1. trend_sma20: short-term trend position relative to 20-period SMA
        sma_20 = pd.Series(self.np_close).rolling(window=20, min_periods=1).mean().to_numpy(dtype=np.float32)
        trend_sma20_vec = np.where(sma_20 != 0, (self.np_close - sma_20) / sma_20 * 100.0, 0.0)
        self.np_trend_sma20 = np.clip(trend_sma20_vec, -5.0, 5.0).astype(np.float32)

        # 2. trend_sma99: medium-term trend position relative to 99-period SMA
        sma_99_arr = pd.Series(self.np_close).rolling(window=99, min_periods=1).mean().to_numpy(dtype=np.float32)
        trend_sma99_vec = np.where(sma_99_arr != 0, (self.np_close - sma_99_arr) / sma_99_arr * 100.0, 0.0)
        self.np_trend_sma99 = np.clip(trend_sma99_vec, -5.0, 5.0).astype(np.float32)

        # 3. roc_24: 24-period rate of change (1-day momentum for hourly data)
        close_prev_24 = np.empty_like(self.np_close)
        if len(self.np_close) > 0:
            close_prev_24[24:] = self.np_close[:-24]
            close_prev_24[:24] = self.np_close[0]  # fill warmup with first value
            roc_24_vec = np.where(close_prev_24 != 0, (self.np_close - close_prev_24) / close_prev_24 * 100.0, 0.0)
        else:
            roc_24_vec = np.zeros_like(self.np_close)
        self.np_roc_24 = np.clip(roc_24_vec, -5.0, 5.0).astype(np.float32)


        # 4. rolling_vol_ratio: current volatility vs its own rolling median (continuous regime signal)
        log_ret = np.zeros_like(self.np_close)
        valid_prev = self.np_close[:-1] != 0
        log_ret[1:] = np.where(valid_prev, np.log(self.np_close[1:] / np.where(valid_prev, self.np_close[:-1], 1.0)), 0.0)
        rolling_vol_arr = pd.Series(log_ret).rolling(window=30, min_periods=1).std().to_numpy(dtype=np.float32)
        rolling_vol_arr = np.nan_to_num(rolling_vol_arr, nan=0.0)  # std of single value = NaN
        vol_median_arr = pd.Series(rolling_vol_arr).rolling(window=max(1, min(500, self.n_rows)), min_periods=1).median().to_numpy(dtype=np.float32)

        vol_median_arr = np.nan_to_num(vol_median_arr, nan=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio_vec = np.where(vol_median_arr != 0, rolling_vol_arr / vol_median_arr, 1.0)
        self.np_vol_ratio = np.clip(np.nan_to_num(vol_ratio_vec, nan=1.0), 0.0, 3.0).astype(np.float32)

        # 5. drawdown_from_peak: distance from rolling 168-period high (macro health signal)
        rolling_max_arr = pd.Series(self.np_close).rolling(window=168, min_periods=1).max().to_numpy(dtype=np.float32)
        dd_vec = np.where(rolling_max_arr != 0, (self.np_close - rolling_max_arr) / rolling_max_arr * 100.0, 0.0)
        self.np_drawdown = np.clip(dd_vec, -20.0, 0.0).astype(np.float32)

        # Actions: 0=Cash (Flat), 1=Target Long, 2=Target Short
        self.action_space = spaces.Discrete(3)

        # Shape is:
        # include_context=True, include_regime=True -> 17
        # include_context=True, include_regime=False -> 16
        # include_context=False, include_regime=True -> 12
        # include_context=False, include_regime=False -> 11
        obs_shape = 7 + (5 if self.include_context else 0) + (1 if self.include_regime else 0) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

        # Auto-detect interval for daily vs hourly to scale trade frequency threshold
        is_daily = False
        if "datetime" in self.df.columns and len(self.df) > 1:
            try:
                first_time = pd.to_datetime(self.df["datetime"].iloc[0])
                second_time = pd.to_datetime(self.df["datetime"].iloc[1])
                diff_hours = (second_time - first_time).total_seconds() / 3600.0
                if diff_hours > 12:
                     is_daily = True
            except Exception:
                pass
        self.trade_frequency_threshold = 1 if is_daily else 24

        # Precompute initial empirical return moments if warm starting DSR
        if self.warm_start_dsr and self.n_rows > 1:
            valid_prev = self.np_close[:-1] != 0
            log_rets = np.where(
                valid_prev,
                np.log(self.np_close[1:] / np.where(valid_prev, self.np_close[:-1], 1.0)),
                0.0,
            )
            self._init_ema_A = float(np.mean(log_rets))
            self._init_ema_B = float(np.mean(log_rets ** 2))
        else:
            self._init_ema_A = 0.0
            self._init_ema_B = 0.0

        self.reset()

    def _get_observation(self):
        if self.n_rows == 0:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        i = max(0, min(self.current_step, self.n_rows - 1))


        # --- 4 PORTFOLIO STATE FEATURES ---
        pos_side = 1.0 if self.is_long else (-1.0 if self.is_short else 0.0)

        unrealized_pnl = 0.0
        if self.is_long or self.is_short:
            unrealized_pnl = (
                (self.net_worth - self.last_balance) / self.last_balance
                if self.last_balance != 0
                else 0.0
            )

        # idle_norm: how long has the agent been sitting flat? Normalized to [0, 1] over 48 steps.
        # Replaces cooldown_norm which was always 0 when cooldown_steps=1 (dead feature).
        idle_norm = np.clip(self.steps_since_last_trade / 48.0, 0.0, 1.0)
        # duration_norm: scaled 2x so it reaches ±1 at half the max duration (better variance vs market features)
        duration_norm = np.clip(self.trade_duration / self.max_trade_duration * 2.0, 0.0, 1.0) if self.max_trade_duration > 0 else 0.0

        obs_list = [
            # Precomputed stationary technical candle features (7)
            self.np_open_ratio[i] / 10.0,
            self.np_high_ratio[i] / 10.0,
            self.np_low_ratio[i] / 10.0,
            self.np_volatility[i] / 10.0,
            self.np_close_change[i] / 10.0,
            self.np_close_change_prev[i] / 10.0,
            self.np_volume_norm[i] / 2.0,
        ]

        if self.include_context:
            obs_list.extend([
                # Multi-scale context features (5): give the Critic macro regime awareness
                self.np_trend_sma20[i] / 5.0,
                self.np_trend_sma99[i] / 5.0,
                self.np_roc_24[i] / 5.0,
                self.np_vol_ratio[i] / 3.0,
                self.np_drawdown[i] / 20.0,
            ])

        if self.include_regime:
            regime_norm = self.np_regime[i] / 2.0
            obs_list.append(regime_norm)

        obs_list.extend([
            # Portfolio state (4): scaled to match market feature variance (~0.5-0.65 std)
            pos_side,
            np.clip(unrealized_pnl * 20.0, -1.0, 1.0),  # boosted 2x: reaches ±1 with smaller moves
            idle_norm,                                    # replaces dead cooldown_norm
            duration_norm,                               # scaled 2x for better gradient signal
        ])

        return np.array(obs_list, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """Return boolean mask of legal target positions for the current state.
        
        Indices:
            0: Cash / Flat (always legal)
            1: Target Long (legal if already Long, or if cooldown <= 1)
            2: Target Short (legal if already Short, or if cooldown <= 1)
        """
        mask = np.zeros(3, dtype=bool)
        mask[0] = True  # Cash is always legal

        # In step(), self.cooldown is decremented by 1 before checking if self.cooldown == 0.
        # Therefore, opening a new position or flipping is allowed if self.cooldown <= 1.
        can_enter = (self.cooldown <= 1)

        mask[1] = self.is_long or can_enter
        mask[2] = self.is_short or can_enter

        return mask

    def step(self, action):
        if self.n_rows == 0:
            return self._get_observation(), 0.0, True, True, {"error": "empty_environment"}

        prev_net_worth = self.net_worth
        
        # Execute trades at the OPEN price of the next candle to prevent look-ahead bias.
        # This simulates placing an order at the exact moment the current candle closes.
        next_step_idx = max(0, min(self.current_step + 1, self.n_rows - 1))
        current_price = self.np_open[next_step_idx]


        
        # Decrement cooldown BEFORE processing the current step's action
        if self.cooldown > 0:
            self.cooldown -= 1

        info = {}
        invalid_action = False
        self.trade_closed = False
        self.realized_pnl = 0.0

        # Action 0: Target CASH (FLAT)
        if action == 0:
            if self.is_long:
                # Liquidate Long
                exit_price = current_price * (1.0 - self.slippage_rate)
                pnl = (exit_price - self.entry_price) / self.entry_price
                self.balance *= 1.0 + pnl
                self.balance *= 1.0 - self.commission_rate
                self.is_long = False
                self.entry_price = 0.0
                self.cooldown = self.cooldown_steps
                self.last_balance = self.balance
                self.trade_closed = True
                self.realized_pnl = pnl
                self.trade_duration = 0
            elif self.is_short:
                # Liquidate Short
                exit_price = current_price * (1.0 + self.slippage_rate)
                pnl = (self.entry_price - exit_price) / self.entry_price
                self.balance *= 1.0 + pnl
                self.balance *= 1.0 - self.commission_rate
                self.is_short = False
                self.entry_price = 0.0
                self.cooldown = self.cooldown_steps
                self.last_balance = self.balance
                self.trade_closed = True
                self.realized_pnl = pnl
                self.trade_duration = 0
            else:
                # Already flat in Cash: valid no-op, $0 fee
                pass

        # Action 1: Target LONG
        elif action == 1:
            if self.is_long:
                # Already Long: maintain position with $0 fee (no trade increment)
                pass
            elif self.is_short:
                if self.cooldown == 0:
                    # Flip: Close Short first at current price
                    exit_price = current_price * (1.0 + self.slippage_rate)
                    pnl = (self.entry_price - exit_price) / self.entry_price
                    self.balance *= 1.0 + pnl
                    self.balance *= 1.0 - self.commission_rate
                    self.is_short = False
                    self.trade_closed = True
                    self.realized_pnl = pnl

                    # Immediately Open Long
                    self.is_long = True
                    self.entry_price = current_price * (1.0 + self.slippage_rate)
                    self.balance *= 1.0 - self.commission_rate
                    self.trade_duration = 0
                    self.last_balance = self.balance
                    self.trade_count += 1
                else:
                    invalid_action = True
            else:
                # From Cash to Long
                if self.cooldown == 0:
                    self.is_long = True
                    self.entry_price = current_price * (1.0 + self.slippage_rate)
                    self.balance *= 1.0 - self.commission_rate
                    self.trade_duration = 0
                    self.last_balance = self.balance
                    self.trade_count += 1
                else:
                    invalid_action = True

        # Action 2: Target SHORT
        elif action == 2:
            if self.is_short:
                # Already Short: maintain position with $0 fee (no trade increment)
                pass
            elif self.is_long:
                if self.cooldown == 0:
                    # Flip: Close Long first at current price
                    exit_price = current_price * (1.0 - self.slippage_rate)
                    pnl = (exit_price - self.entry_price) / self.entry_price
                    self.balance *= 1.0 + pnl
                    self.balance *= 1.0 - self.commission_rate
                    self.is_long = False
                    self.trade_closed = True
                    self.realized_pnl = pnl

                    # Immediately Open Short
                    self.is_short = True
                    self.entry_price = current_price * (1.0 - self.slippage_rate)
                    self.balance *= 1.0 - self.commission_rate
                    self.trade_duration = 0
                    self.last_balance = self.balance
                    self.trade_count += 1
                else:
                    invalid_action = True
            else:
                # From Cash to Short
                if self.cooldown == 0:
                    self.is_short = True
                    self.entry_price = current_price * (1.0 - self.slippage_rate)
                    self.balance *= 1.0 - self.commission_rate
                    self.trade_duration = 0
                    self.last_balance = self.balance
                    self.trade_count += 1
                else:
                    invalid_action = True

        self.current_step += 1

        # Track and limit trade duration (maximum holding period / time-in-market constraint)
        if self.is_long or self.is_short:
            self.trade_duration += 1
            if self.max_trade_duration > 0 and self.trade_duration >= self.max_trade_duration:
                # Force close at the end-of-step price
                new_price = self.np_close[min(self.current_step, self.n_rows - 1)]
                if self.is_long:
                    exit_price = new_price * (1.0 - self.slippage_rate)
                    pnl = (exit_price - self.entry_price) / self.entry_price
                    self.balance *= 1.0 + pnl
                    self.balance *= 1.0 - self.commission_rate
                    self.is_long = False
                    self.entry_price = 0.0
                    self.cooldown = self.cooldown_steps
                    self.last_balance = self.balance
                    self.trade_closed = True
                    self.realized_pnl = pnl
                    info["duration_force_closed"] = "LONG"
                elif self.is_short:
                    exit_price = new_price * (1.0 + self.slippage_rate)
                    pnl = (self.entry_price - exit_price) / self.entry_price
                    self.balance *= 1.0 + pnl
                    self.balance *= 1.0 - self.commission_rate
                    self.is_short = False
                    self.entry_price = 0.0
                    self.cooldown = self.cooldown_steps
                    self.last_balance = self.balance
                    self.trade_closed = True
                    self.realized_pnl = pnl
                    info["duration_force_closed"] = "SHORT"

        steps_taken = self.current_step - self.start_step
        
        # Split done into Gym standard truncated and terminated
        truncated = self.current_step >= len(self.df) - 1 or steps_taken >= self.max_steps
        terminated = self.net_worth <= 0.1 * self.initial_balance # Margin call / bankruptcy
        done = truncated or terminated

        # Force-close any open position at the end of episode
        if done:
            final_price = self.np_close[min(self.current_step, self.n_rows - 1)]
            if self.is_long:
                exit_price = final_price * (1.0 - self.slippage_rate)
                pnl = (exit_price - self.entry_price) / self.entry_price
                self.balance *= 1.0 + pnl
                self.balance *= 1.0 - self.commission_rate
                self.is_long = False
                self.entry_price = 0.0
                self.last_balance = self.balance
                info["force_closed"] = "LONG"
            elif self.is_short:
                exit_price = final_price * (1.0 + self.slippage_rate)
                pnl = (self.entry_price - exit_price) / self.entry_price
                self.balance *= 1.0 + pnl
                self.balance *= 1.0 - self.commission_rate
                self.is_short = False
                self.entry_price = 0.0
                self.last_balance = self.balance
                info["force_closed"] = "SHORT"

        # Calculate current net worth (Mark-to-Market Liquidation Value)
        # Fix #2: Remove simulated exit commission from mark-to-market calculation.
        # The exit fee is only deducted when the trade is actually closed (actions 2 & 4).
        # Including it here caused a double-commission penalty on the entry step, creating
        # a sharp reward cliff that made the policy excessively trade-averse.
        new_price = self.np_close[min(self.current_step, self.n_rows - 1)]
        if self.is_long:
            pnl = (new_price - self.entry_price) / self.entry_price
            self.net_worth = self.balance * (1.0 + pnl)
        elif self.is_short:
            pnl = (self.entry_price - new_price) / self.entry_price
            self.net_worth = self.balance * (1.0 + pnl)
        else:
            self.net_worth = self.balance

        # Track peak net worth for drawdown calculations
        self.peak_net_worth = max(self.peak_net_worth, self.net_worth)

        # --- DIRECT LOG-WEALTH UTILITY REWARD FUNCTION ---
        # Objective: Maximize cumulative log wealth R_t = ln(W_t / W_{t-1}).
        # Holding cash yields strictly 0.0 reward, eliminating DSR cash-hoarding gravity wells.
        if self.net_worth > 0 and prev_net_worth > 0:
            step_log_return = float(np.log(self.net_worth / prev_net_worth))
            # Scale log return (10.0x) so typical ~1% moves produce ~0.10 gradients.
            # Bounded to [-1.0, 1.0] for neural network stability.
            reward = float(np.clip(step_log_return * 10.0, -1.0, 1.0))
        else:
            reward = -10.0 # Heavy penalty for blowing up the account

        # 2. Peak-Based Drawdown Penalty and Hard Cap
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth if self.peak_net_worth > 0 else 0.0
        self.episode_max_drawdown = max(self.episode_max_drawdown, current_drawdown)
        
        # Continuous step-level drawdown penalty
        if current_drawdown > 0:
            reward -= self.drawdown_penalty_coef * current_drawdown
            
        # Drawdown safeguard (hard cap vs gradual safeguard)
        if self.max_drawdown_cap > 0 and current_drawdown >= self.max_drawdown_cap:
            if self.gradual_drawdown:
                reward -= self.drawdown_cap_penalty
                info["force_closed"] = "DRAWDOWN_SAFEGUARD"
                # Liquidate position and enter cooldown, but preserve episode continuation
                if self.is_long or self.is_short:
                    self.is_long = False
                    self.is_short = False
                    self.entry_price = 0.0
                    self.last_balance = self.net_worth
                    self.balance = self.net_worth
                    self.cooldown = self.cooldown_steps
                    # Reset peak net worth to current capital so subsequent steps in cash don't re-trigger
                    self.peak_net_worth = self.net_worth
            else:
                reward -= self.drawdown_cap_penalty
                terminated = True
                done = True
                info["force_closed"] = "DRAWDOWN_CAP"
                # Liquidate position for final metrics
                if self.is_long or self.is_short:
                    self.is_long = False
                    self.is_short = False
                    self.entry_price = 0.0
                    self.last_balance = self.net_worth
                    self.balance = self.net_worth

        # Penalty for invalid actions (e.g. attempting to enter during active cooldown)
        if invalid_action:
            reward -= 0.05

        # Dense feedback: if net worth is in a drawdown state below starting capital,
        # apply a small continuous step-level penalty to encourage drawdown mitigation.
        # This is dynamically scaled by negative_pnl_penalty and disabled if it is set to 0.0.
        if self.negative_pnl_penalty > 0.0 and self.net_worth < self.initial_balance:
            drawdown_pct = (self.initial_balance - self.net_worth) / self.initial_balance
            # Scaled down 10x from 0.01 to match log returns magnitude
            reward -= 0.001 * self.negative_pnl_penalty * drawdown_pct

        # Soft warning penalty near duration limits to encourage active self-liquidation
        if self.max_trade_duration > 0 and (self.is_long or self.is_short) and self.trade_duration >= 0.8 * self.max_trade_duration:
            warning_fraction = (self.trade_duration - 0.8 * self.max_trade_duration) / (0.2 * self.max_trade_duration)
            # Scaled down 10x from 0.02 to match log returns magnitude
            reward -= 0.002 * warning_fraction

        # Profit realization bonus: positive reinforcement for locking in profitable trades
        if self.trade_closed:
            if self.realized_pnl > 0.0:
                reward += min(0.05, self.realized_pnl * 2.0)
            self.trade_closed = False
            self.realized_pnl = 0.0

        # Dense directional shaping: small bonus for moving with price trend while holding a position
        prev_idx = max(0, min(self.current_step - 1, self.n_rows - 1))
        curr_idx = min(self.current_step, self.n_rows - 1)
        if self.np_close[prev_idx] > 0:
            price_change = (self.np_close[curr_idx] - self.np_close[prev_idx]) / self.np_close[prev_idx]
            if self.is_long and price_change > 0:
                reward += min(0.005, price_change * 0.5)
            elif self.is_short and price_change < 0:
                reward += min(0.005, abs(price_change) * 0.5)

        # Track flat inactivity steps and apply escalating opportunity cost beyond grace window
        if not (self.is_long or self.is_short):
            self.steps_since_last_trade += 1
            grace_period = 6  # 6 hours of patience before penalizing flat cash
            if self.inactivity_penalty > 0.0 and self.steps_since_last_trade > grace_period:
                excess_idle = self.steps_since_last_trade - grace_period
                idle_mult = min(10.0, 1.0 + excess_idle / 12.0)
                reward -= self.inactivity_penalty * idle_mult
        else:
            self.steps_since_last_trade = 0

        # Apply end-of-episode negative PnL penalty to make validation score go way down on overall losses
        if done and self.negative_pnl_penalty > 0.0:
            total_pnl_pct = (self.net_worth - self.initial_balance) / self.initial_balance
            if total_pnl_pct < 0.0:
                # Continuous, proportional penalty to avoid discontinuous cliffs and agent paralysis
                # Scaled down from 100.0 to 10.0 to prevent value target explosions in the Critic
                reward -= self.negative_pnl_penalty * abs(total_pnl_pct) * 10.0

        info["net_worth"] = self.net_worth
        if done:
            info["episode_metrics"] = {
                "roi": float((self.net_worth - self.initial_balance) / self.initial_balance * 100.0),
                "trades": int(self.trade_count),
                "max_drawdown": float(self.episode_max_drawdown * 100.0),
                "final_capital": float(self.net_worth),
            }
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize the starting point so the agent sees different regimes every episode!
        if self.random_start:
            max_start = len(self.df) - self.max_steps - 1
            if max_start <= 0:
                self.start_step = 0
            else:
                # Use Gym's seeded random generator instead of global np.random
                self.start_step = int(self.np_random.integers(0, max_start))
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.last_balance = (
            self.initial_balance
        )  # Track balance between trades for reward
        self.is_long = False
        self.is_short = False
        self.entry_price = 0.0
        self.cooldown = 0
        self.net_worth = self.initial_balance
        self.trade_duration = 0
        self.steps_since_last_trade = 0
        self.trade_count = 0
        self.trade_closed = False
        self.realized_pnl = 0.0

        # New attributes for advanced reward engineering
        self.peak_net_worth = self.initial_balance
        self.episode_max_drawdown = 0.0
        
        # Differential Sharpe Ratio (DSR) EMAs
        self.ema_A = self._init_ema_A
        self.ema_B = self._init_ema_B

        return self._get_observation(), {}


class TinyTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, n_stack=8):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        
        # Shape validation guard to prevent runtime reshaping crashes
        obs_dim = observation_space.shape[0]
        assert obs_dim % n_stack == 0, (
            f"Observation space dimension ({obs_dim}) must be a multiple of n_stack ({n_stack}). "
            "Verify environment is wrapped in VecFrameStack!"
        )
        self.input_dim = obs_dim // n_stack

        # 1. Feature Embedding
        self.embedding = nn.Linear(self.input_dim, features_dim)

        # 2. Learned Positional Encoding
        self.position_embedding = nn.Parameter(torch.randn(1, n_stack, features_dim))

        # 3. Transformer (Pre-LN architecture with 0.15 dropout for noisy tabular time series)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=2,
            dim_feedforward=128,
            batch_first=True,
            norm_first=True,
            dropout=0.15,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)

    def forward(self, observations):
        batch_size = observations.shape[0]

        # Reshape to (batch, sequence_length, features) using reshape to handle non-contiguous memory layouts safely
        x = observations.reshape(batch_size, self.n_stack, self.input_dim)

        # Map features up to dimensions
        x = self.embedding(x)

        # Inject positional awareness!
        x = x + self.position_embedding

        # Pass through Transformer
        x = self.transformer(x)

        # Robust sequence aggregation: combine historical mean with the high-impact final step
        # to preserve temporal recency and prevent information dilution, while maintaining strict
        # backward-compatibility with older checkpoint weights by avoiding new learnable parameters.
        return 0.3 * x.mean(dim=1) + 0.7 * x[:, -1, :]


class DeepTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_stack=8):
        super().__init__(observation_space, features_dim)
        self.n_stack = n_stack
        
        # Shape validation guard to prevent runtime reshaping crashes
        obs_dim = observation_space.shape[0]
        assert obs_dim % n_stack == 0, (
            f"Observation space dimension ({obs_dim}) must be a multiple of n_stack ({n_stack}). "
            "Verify environment is wrapped in VecFrameStack!"
        )
        self.input_dim = obs_dim // n_stack

        self.embedding = nn.Linear(self.input_dim, features_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, n_stack, features_dim))

        # Give it 8 attention heads and a massive 1024 feedforward network (Pre-LN + dropout 0.15)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            norm_first=True,
            dropout=0.15,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4, enable_nested_tensor=False)

    def forward(self, observations):
        batch_size = observations.shape[0]
        x = observations.reshape(batch_size, self.n_stack, self.input_dim)
        x = self.embedding(x)
        x = x + self.position_embedding
        x = self.transformer(x)

        # Robust sequence aggregation: combine historical mean with the high-impact final step
        # to preserve temporal recency and prevent information dilution, while maintaining strict
        # backward-compatibility with older checkpoint weights by avoiding new learnable parameters.
        return 0.3 * x.mean(dim=1) + 0.7 * x[:, -1, :]
