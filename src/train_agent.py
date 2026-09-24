import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch

torch.set_num_threads(1)
import time
import datetime
from functools import partial
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

# Ensure src directory is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from rl_env import ActiveCryptoEnv, TinyTransformerExtractor, DeepTransformerExtractor
from trading_utils import compute_indicators


class CLICallback(BaseCallback):
    def __init__(self, total_steps, eval_callback=None, verbose=1):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.eval_callback = eval_callback
        self.start_timesteps = 0
        self.start_time = 0

        # Phase-Aware Timing variables
        self.total_pause_time = 0.0
        self.pause_start_time = 0.0
        self.pause_count = 0
        self.avg_pause_duration = 0.0

        # Rollout tracking variables
        self.completed_episodes = []
        self.rollout_count = 0

    def _on_training_start(self):
        self.start_timesteps = self.model.num_timesteps
        self.start_time = time.time()

    def _on_rollout_end(self):
        # Triggers the instant rollout ends (policy update pause starts)
        self.pause_start_time = time.time()
        self.rollout_count += 1
        
        # Print rollout report if episodes finished
        if self.completed_episodes:
            rois = [ep["roi"] for ep in self.completed_episodes]
            trades = [ep["trades"] for ep in self.completed_episodes]
            capitals = [ep["final_capital"] for ep in self.completed_episodes]
            
            avg_roi = sum(rois) / len(rois)
            avg_trades = sum(trades) / len(trades)
            win_rate = (sum(1 for r in rois if r > 0) / len(rois)) * 100.0
            best_cap = max(capitals)
            worst_cap = min(capitals)
            
            # Helper to construct perfectly aligned visual rows of 56 visual characters wide
            def make_row(label, value):
                pad_len = 26 - len(value)
                content = f"  {label:<28}{value}{' ' * max(0, pad_len)}"
                return f"|{content}|"

            title_str = f"ROLLOUT #{self.rollout_count} REPORT"
            title_pad_left = (56 - len(title_str)) // 2
            title_pad_right = 56 - len(title_str) - title_pad_left
            header_row = f"|{' ' * title_pad_left}{title_str}{' ' * title_pad_right}|"

            print("\n" + "=" * 58)
            print(header_row)
            print("=" * 58)
            print(make_row("Episodes Completed:", f"{len(self.completed_episodes)}"))
            print(make_row("Avg Training Return:", f"{avg_roi:+.2f}%"))
            print(make_row("Avg Trades Executed:", f"{avg_trades:.1f}"))
            print(make_row("Win Rate (ROI > 0):", f"{win_rate:.1f}%"))
            print(make_row("Best Final Capital:", f"${best_cap:,.2f}"))
            print(make_row("Worst Final Capital:", f"${worst_cap:,.2f}"))
            print("=" * 58 + "\n")
            
            # Trigger validation immediately after training episodes complete
            if self.eval_callback is not None:
                self.eval_callback.run_validation()
            
            self.completed_episodes.clear()
        else:
            print(f"\n[Rollout #{self.rollout_count}] Rollout phase complete. Policy weights updated.\n")

    def _on_rollout_start(self):
        # Triggers the instant policy update ends (rollout starts again)
        if self.pause_start_time > 0:
            pause_dur = time.time() - self.pause_start_time
            self.total_pause_time += pause_dur
            self.pause_count += 1
            self.avg_pause_duration = self.total_pause_time / self.pause_count
            self.pause_start_time = 0.0

    def _on_step(self):
        # Collect finished episode metrics from the vectorized environments
        infos = self.locals.get("infos")
        if infos is not None:
            for info in infos:
                if "episode" in info and "episode_metrics" in info:
                    self.completed_episodes.append(info["episode_metrics"])

        # Overwrite the same line every 1024 steps (no clutter, no overhead)
        if self.num_timesteps % 1024 == 0:
            relative_steps = self.num_timesteps - self.start_timesteps
            progress = (relative_steps / self.total_steps) * 100

            # 1. Total elapsed time
            total_elapsed = time.time() - self.start_time

            # 2. Pure rollout time (excluding pauses)
            rollout_elapsed = total_elapsed - self.total_pause_time

            # Calculate pure rollout speed (steps/sec)
            if rollout_elapsed > 0:
                rollout_speed = relative_steps / rollout_elapsed
            else:
                rollout_speed = 0.0

            remaining_steps = self.total_steps - relative_steps

            if remaining_steps > 0:
                # 3. Calculate remaining rollout time
                remaining_rollout_time = (
                    remaining_steps / rollout_speed if rollout_speed > 0 else 0.0
                )

                # 4. Calculate remaining policy updates
                n_steps = getattr(self.model, "n_steps", 2048)
                n_envs = self.model.env.num_envs if self.model.env is not None else 8
                steps_per_update = n_steps * n_envs

                remaining_pauses = remaining_steps / steps_per_update

                # 5. Estimate remaining pause time
                remaining_pause_time = remaining_pauses * self.avg_pause_duration

                # 6. Total projected remaining time (ETA)
                eta_seconds = remaining_rollout_time + remaining_pause_time
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_str = "0:00:00"

            if not sys.stdout.isatty():
                print(
                    f"Optimization: {relative_steps:,} / {self.total_steps:,} steps | [{progress:.1f}%] | ETA: {eta_str}",
                    flush=True,
                )
            else:
                print(
                    f"\rOptimization: {relative_steps:,} / {self.total_steps:,} steps | [{progress:.1f}%] | ETA: {eta_str}    ",
                    end="",
                    flush=True,
                )
        return True


class EntropyDecayCallback(BaseCallback):
    def __init__(
        self, initial_ent_coef, final_ent_coef=0.001, total_steps=100000, verbose=0
    ):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_steps = total_steps
        self.start_timesteps = 0

    def _on_training_start(self):
        self.start_timesteps = self.model.num_timesteps

    def _on_step(self):
        relative_steps = self.num_timesteps - self.start_timesteps
        progress = relative_steps / self.total_steps
        progress = min(1.0, max(0.0, progress))

        # Linearly decay entropy from initial to final value
        current_ent_coef = self.initial_ent_coef - progress * (
            self.initial_ent_coef - self.final_ent_coef
        )

        # Inject the updated entropy coefficient back into the PPO model
        self.model.ent_coef = current_ent_coef

        # Record it so it shows up in tensorboard if enabled
        self.logger.record("train/ent_coef", current_ent_coef)
        return True


class FeeCurriculumCallback(BaseCallback):
    """
    Transaction Cost Curriculum Scheduler:
    - Phase 1 (Warmup): Fees = 0.0 to discover directional alpha without friction penalty.
    - Phase 2 (Ramping): Fees ramp linearly from 0.0 to target rates (e.g. 0.1% comm, 0.05% slip).
    - Phase 3 (Real Fees): Full realistic transaction fees to master real-world execution.
    """
    def __init__(
        self,
        target_commission=0.001,
        target_slippage=0.0005,
        total_steps=500000,
        warmup_ratio=0.20,
        ramp_ratio=0.30,
        verbose=1,
    ):
        super().__init__(verbose)
        self.target_commission = target_commission
        self.target_slippage = target_slippage
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.ramp_steps = int(total_steps * ramp_ratio)
        self.current_commission = 0.0
        self.current_slippage = 0.0
        self.last_reported_step = -25000

    def _update_env_fees(self, comm, slip):
        self.current_commission = comm
        self.current_slippage = slip
        vec_env = self.training_env
        while hasattr(vec_env, "venv"):
            vec_env = vec_env.venv
        if hasattr(vec_env, "envs"):
            for env in vec_env.envs:
                target = env.unwrapped if hasattr(env, "unwrapped") else env
                target.commission_rate = comm
                target.slippage_rate = slip

    def _on_training_start(self):
        self._update_env_fees(0.0, 0.0)
        if self.verbose:
            print(f"\n[FeeCurriculum] Initialized: 0% fees for first {self.warmup_steps:,} steps.")
            print(f"[FeeCurriculum] Ramping to {self.target_commission*100:.2f}% comm / {self.target_slippage*100:.2f}% slip from step {self.warmup_steps:,} to {self.warmup_steps + self.ramp_steps:,}.\n")

    def _on_step(self):
        step = self.num_timesteps
        if step < self.warmup_steps:
            comm = 0.0
            slip = 0.0
            phase_label = "Phase 1 (Zero Fees)"
        elif step < (self.warmup_steps + self.ramp_steps):
            progress = (step - self.warmup_steps) / max(1, self.ramp_steps)
            comm = progress * self.target_commission
            slip = progress * self.target_slippage
            phase_label = f"Phase 2 (Ramping {progress*100:.1f}%)"
        else:
            comm = self.target_commission
            slip = self.target_slippage
            phase_label = "Phase 3 (Real Fees)"

        if abs(comm - self.current_commission) > 1e-6 or abs(slip - self.current_slippage) > 1e-6:
            self._update_env_fees(comm, slip)

        if self.verbose and (step - self.last_reported_step >= 25000):
            self.last_reported_step = step
            roundtrip = (comm + slip) * 200.0
            print(f"\n>>> [FeeCurriculum] Step {step:,} | {phase_label} | Comm: {comm*100:.3f}% | Slip: {slip*100:.3f}% | Roundtrip: {roundtrip:.3f}%\n", flush=True)

        return True


class FastEvalCallback(BaseCallback):
    def __init__(
        self,
        df_val,
        eval_freq,
        save_path,
        stop_callback=None,
        commission=0.001,
        slippage=0.0005,
        cooldown=12,
        inactivity_penalty=0.0,
        negative_pnl_penalty=0.0,
        volatility_scaling=0.5,
        drawdown_penalty_coef=0.1,
        max_drawdown_cap=0.15,
        max_trade_duration=48,
        gradual_drawdown=False,
        drawdown_cap_penalty=5.0,
        val_windows=None,
        verbose=1,
    ):
        super().__init__(verbose)
        self.df_val = df_val
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.stop_callback = stop_callback
        self.commission = commission
        self.slippage = slippage
        self.cooldown = cooldown
        self.inactivity_penalty = inactivity_penalty
        self.negative_pnl_penalty = negative_pnl_penalty
        self.volatility_scaling = volatility_scaling
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.max_drawdown_cap = max_drawdown_cap
        self.max_trade_duration = max_trade_duration
        self.gradual_drawdown = gradual_drawdown
        self.drawdown_cap_penalty = drawdown_cap_penalty
        self.val_windows = val_windows
        
        self.best_mean_reward = -999.0
        self.last_mean_reward = 0.0

    def init_callback(self, model) -> None:
        super().init_callback(model)
        if self.stop_callback is not None:
            self.stop_callback.init_callback(self.model)
            self.stop_callback.parent = self

    def run_validation(self) -> bool:
        """Run a validation backtest and print the report. Called by CLICallback after episodes complete."""
        from trading_utils import run_single_fold_backtest

        if self.val_windows is not None and len(self.val_windows) > 0:
            window_scores = []
            print("\n" + "+" + "-" * 74 + "+")
            print(f"|{'MULTI-REGIME VALIDATION REPORT':^74}|")
            print("+" + "-" * 74 + "+")
            step_str = f"Step: {self.num_timesteps:,}"
            print(f"|  {step_str:<70}  |")
            print("+" + "-" * 74 + "+")

            for win_name, df_win in self.val_windows:
                metrics = run_single_fold_backtest(
                    self.model,
                    df_win,
                    is_daily=False,
                    deterministic=True,
                    commission_rate=self.commission,
                    slippage_rate=self.slippage,
                    cooldown_steps=self.cooldown,
                    inactivity_penalty=self.inactivity_penalty,
                    negative_pnl_penalty=self.negative_pnl_penalty,
                    volatility_scaling=self.volatility_scaling,
                    drawdown_penalty_coef=self.drawdown_penalty_coef,
                    max_drawdown_cap=self.max_drawdown_cap,
                    max_trade_duration=self.max_trade_duration,
                    gradual_drawdown=self.gradual_drawdown,
                    drawdown_cap_penalty=self.drawdown_cap_penalty,
                )
                w_roi = metrics["ROI %"]
                w_trades = metrics["Trades"]
                w_dd = metrics["Max DD %"]
                w_bh = metrics.get("B&H %", metrics.get("Buy_Hold_Return %", 0.0))
                w_alpha = w_roi - w_bh
                w_sortino = metrics.get("Sortino", metrics.get("Sharpe", 0.0))
                w_risk_mult = 1.0 + max(0.0, min(3.0, w_sortino))

                if w_bh >= 0:
                    w_obj = (w_roi + max(0.0, w_alpha) * 0.5) * w_risk_mult
                else:
                    if w_roi >= 0:
                        w_obj = (w_roi + w_alpha * 0.5) * w_risk_mult
                    else:
                        w_obj = (w_alpha * 0.5 + w_roi * 0.5) * w_risk_mult

                window_scores.append(w_obj)
                line = f"[{win_name:<8}] ROI: {w_roi:+6.2f}% | B&H: {w_bh:+6.2f}% | DD: {w_dd:5.2f}% | Trades: {w_trades:<3} | Score: {w_obj:+6.2f}"
                print(f"|  {line:<70}  |")

            objective_score = float(np.mean(window_scores))
            comp_line = f"Composite Multi-Regime Score: {objective_score:+.2f}"
            print("+" + "-" * 74 + "+")
            print(f"|  {comp_line:<70}  |")
            print("+" + "-" * 74 + "+\n")

            if objective_score > self.best_mean_reward:
                self.best_mean_reward = objective_score
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(
                    {
                        "policy": self.model.policy.state_dict(),
                        "optimizer": self.model.policy.optimizer.state_dict(),
                        "total_steps": self.model.num_timesteps,
                        "ent_coef": self.model.ent_coef,
                        "cooldown_steps": self.cooldown,
                        "inactivity_penalty": self.inactivity_penalty,
                        "negative_pnl_penalty": self.negative_pnl_penalty,
                        "best_mean_reward": self.best_mean_reward,
                    },
                    self.save_path,
                )
                vec_norm = self.model.get_vec_normalize_env()
                if vec_norm is not None:
                    normalizer_path = self.save_path.replace(".pth", "_normalizer.pkl")
                    vec_norm.save(normalizer_path)
                if self.verbose > 0:
                    print(f"[FastEvalCallback] New best model saved to {self.save_path}!")

            if self.stop_callback is not None:
                return self.stop_callback.on_step()
            return True

        is_daily = False
        if "datetime" in self.df_val.columns and len(self.df_val) > 1:
            delta = pd.to_datetime(self.df_val["datetime"]).diff().dropna().iloc[0]
            if delta.total_seconds() > 12 * 3600:
                is_daily = True

        metrics = run_single_fold_backtest(
            self.model,
            self.df_val,
            is_daily=is_daily,
            deterministic=True,
            commission_rate=self.commission,
            slippage_rate=self.slippage,
            cooldown_steps=self.cooldown,
            inactivity_penalty=self.inactivity_penalty,
            negative_pnl_penalty=self.negative_pnl_penalty,
            volatility_scaling=self.volatility_scaling,
            drawdown_penalty_coef=self.drawdown_penalty_coef,
            max_drawdown_cap=self.max_drawdown_cap,
            max_trade_duration=self.max_trade_duration,
            gradual_drawdown=self.gradual_drawdown,
            drawdown_cap_penalty=self.drawdown_cap_penalty,
        )

        val_roi = metrics["ROI %"]
        trades_count = metrics["Trades"]
        max_dd = metrics["Max DD %"]
        final_capital = metrics["Final $"]
        mean_reward = metrics.get("ValidationReward", 0.0)
        
        self.last_mean_reward = mean_reward

        # Helper to construct perfectly aligned visual rows of 56 visual characters wide
        def make_row(label, value):
            pad_len = 26 - len(value)
            content = f"  {label:<28}{value}{' ' * max(0, pad_len)}"
            return f"|{content}|"

        step_str = f"Step: {self.num_timesteps:,}"
        step_pad = 56 - 2 - len(step_str)
        step_row = f"|  {step_str}{' ' * max(0, step_pad)}|"

        title_str = "VALIDATION REPORT"
        title_pad_left = (56 - len(title_str)) // 2
        title_pad_right = 56 - len(title_str) - title_pad_left
        header_row = f"|{' ' * title_pad_left}{title_str}{' ' * title_pad_right}|"

        print("\n" + "+" + "-" * 56 + "+")
        print(header_row)
        print("+" + "-" * 56 + "+")
        print(step_row)
        print(
            make_row(
                "Real Portfolio Return:",
                f"{val_roi:+.2f}% ({'Daily' if is_daily else 'Hourly'})",
            )
        )
        print(make_row("Final Capital:", f"${final_capital:,.2f}"))
        print(
            make_row(
                "Sharpe Ratio:",
                f"{metrics['Sharpe']:.2f} | Sortino: {metrics['Sortino']:.2f}",
            )
        )
        print(make_row("Max Drawdown:", f"{max_dd:.2f}%"))
        print(make_row("Trades Executed:", f"{trades_count:,}"))
        bh_return = metrics.get("B&H %", metrics.get("Buy_Hold_Return %", 0.0))
        alpha = val_roi - bh_return  # Excess return over Buy & Hold
        cash_excess = val_roi        # Excess return over Cash (which returns 0%)

        # Fair multi-benchmark scoring against Cash and Buy-and-Hold (no artificial trade quotas).
        # Downside risk is penalized via Sortino ratio.
        sortino = metrics.get("Sortino", metrics.get("Sharpe", 0.0))
        risk_mult = 1.0 + max(0.0, min(3.0, sortino))

        if bh_return >= 0:
            # Bull market: Cash (0%) is baseline. Reward positive ROI and alpha over B&H.
            objective_score = (cash_excess + max(0.0, alpha) * 0.5) * risk_mult
        else:
            # Bear market: Cash (0%) is the hurdle.
            if val_roi >= 0:
                # Outperformed both Cash and Market
                objective_score = (cash_excess + alpha * 0.5) * risk_mult
            else:
                # Losing money: reward mitigating crash vs B&H, but penalize losing vs cash
                objective_score = (alpha * 0.5 + val_roi * 0.5) * risk_mult

        print(
            make_row(
                "Validation Score:", f"{mean_reward:+.2f} | Obj: {objective_score:+.2f}"
            )
        )
        print("+" + "-" * 56 + "+\n")

        # Save checkpoint if objective score improves (fair evaluation, no forced trade quota)
        if objective_score > self.best_mean_reward:
            self.best_mean_reward = objective_score
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(
                {
                    "policy": self.model.policy.state_dict(),
                    "optimizer": self.model.policy.optimizer.state_dict(),
                    "total_steps": self.model.num_timesteps,
                    "ent_coef": self.model.ent_coef,
                    "cooldown_steps": self.cooldown,
                    "inactivity_penalty": self.inactivity_penalty,
                    "negative_pnl_penalty": self.negative_pnl_penalty,
                    "best_mean_reward": self.best_mean_reward,
                },
                self.save_path,
            )
            # Also save the VecNormalize reward stats so the best checkpoint is always paired
            # with its normalizer. Uses SB3's built-in method to access the wrapper.
            vec_norm = self.model.get_vec_normalize_env()
            if vec_norm is not None:
                normalizer_path = self.save_path.replace(".pth", "_normalizer.pkl")
                vec_norm.save(normalizer_path)
            if self.verbose > 0:
                print(f"[FastEvalCallback] New best model saved to {self.save_path}!")


        if self.stop_callback is not None:
            return self.stop_callback.on_step()

        return True

    def _on_step(self) -> bool:
        # Validation is now triggered by CLICallback after training episodes complete.
        # This removes the old fixed-timer approach that produced useless mid-episode reports.
        return True


def get_asset_file(asset_symbol, interval="1H"):
    iv = "1h" if interval == "1H" else "daily"
    symbol = asset_symbol.lower()

    if "btc" in symbol:
        return f"btc_usdt_{iv}.csv"
    if "eth" in symbol:
        return f"eth_usdt_{iv}.csv"
    if "doge" in symbol:
        return f"doge_usdt_{iv}.csv"
    if "spy" in symbol:
        return f"spy_{iv}.csv"
    if "qqq" in symbol:
        return f"qqq_{iv}.csv"
    if "gld" in symbol:
        return f"gld_{iv}.csv"
    if "wti" in symbol:
        return f"wti_{iv}.csv"
    if "gas" in symbol:
        return f"natgas_{iv}.csv"
    if "gold" in symbol:
        return f"gold_{iv}.csv"

    return f"{symbol}_{iv}.csv"


def make_env_global(
    df,
    commission_rate=0.001,
    slippage_rate=0.0005,
    cooldown_steps=12,
    inactivity_penalty=0.0005,
    negative_pnl_penalty=0.0,
    volatility_scaling=0.5,
    drawdown_penalty_coef=0.1,
    max_drawdown_cap=0.15,
    include_regime=True,
    include_context=True,
    max_trade_duration=48,
    max_steps=2160,
    warm_start_dsr=True,
    use_dsor=False,
    gradual_drawdown=False,
    drawdown_cap_penalty=5.0,
):
    env = ActiveCryptoEnv(
        df,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        cooldown_steps=cooldown_steps,
        max_steps=max_steps,
        inactivity_penalty=inactivity_penalty,
        negative_pnl_penalty=negative_pnl_penalty,
        volatility_scaling=volatility_scaling,
        drawdown_penalty_coef=drawdown_penalty_coef,
        max_drawdown_cap=max_drawdown_cap,
        render_mode=None,
        include_regime=include_regime,
        include_context=include_context,
        max_trade_duration=max_trade_duration,
        warm_start_dsr=warm_start_dsr,
        use_dsor=use_dsor,
        gradual_drawdown=gradual_drawdown,
        drawdown_cap_penalty=drawdown_cap_penalty,
    )
    return ActionMasker(Monitor(env), lambda e: e.unwrapped.action_masks())


def make_val_env(
    df,
    commission_rate=0.001,
    slippage_rate=0.0005,
    cooldown_steps=12,
    inactivity_penalty=0.0005,
    negative_pnl_penalty=0.0,
    volatility_scaling=0.5,
    drawdown_penalty_coef=0.1,
    max_drawdown_cap=0.15,
    include_regime=True,
    include_context=True,
    max_trade_duration=48,
    warm_start_dsr=False,
    use_dsor=False,
    gradual_drawdown=False,
    drawdown_cap_penalty=5.0,
):
    env = ActiveCryptoEnv(
        df,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        cooldown_steps=cooldown_steps,
        random_start=False,  # MUST BE FALSE for deterministic chronological validation
        max_steps=len(df),  # MUST BE FULL LENGTH to match true backtest PnL
        inactivity_penalty=inactivity_penalty,
        negative_pnl_penalty=negative_pnl_penalty,
        volatility_scaling=volatility_scaling,
        drawdown_penalty_coef=drawdown_penalty_coef,
        max_drawdown_cap=max_drawdown_cap,
        render_mode=None,
        include_regime=include_regime,
        include_context=include_context,
        max_trade_duration=max_trade_duration,
        warm_start_dsr=warm_start_dsr,
        use_dsor=use_dsor,
        gradual_drawdown=gradual_drawdown,
        drawdown_cap_penalty=drawdown_cap_penalty,
    )
    return Monitor(env)


def linear_schedule(initial_value: float, floor: float = 0.2):
    """
    Linear learning rate schedule with a customizable floor.
    Prevents LR from hitting exactly 0.0, which freezes the policy.
    :param initial_value: (float) Initial learning rate.
    :param floor: (float) Minimum baseline floor for the learning rate schedule.
    :return: (function)
    """

    def func(progress_remaining: float) -> float:
        """
        Progress remaining decreases from 1.0 (start of training) to 0.0 (end of training).
        Floor ensures the optimizer can always update weights.
        """
        return max(progress_remaining, floor) * initial_value

    return func


def train_agent(
    symbol,
    interval,
    steps,
    entropy,
    override_entropy=False,
    reset_weights=False,
    device="cpu",
    extractor="deep",
    commission=0.001,
    slippage=0.0005,
    cooldown=1,
    inactivity_penalty=0.0005,
    negative_pnl_penalty=0.0,
    volatility_scaling=0.5,
    drawdown_penalty_coef=0.1,
    max_drawdown_cap=0.15,
    lr_floor=0.2,
    regime=True,
    context=True,
    lr=None,
    max_trade_duration=48,
    gamma=0.97,
    gae_lambda=0.92,
    target_kl=0.02,
    clip_range_vf=None,
    warm_start_dsr=True,
    use_dsor=False,
    purge_window=48,
    save_path=None,
    seed=None,
    gradual_drawdown=False,
    drawdown_cap_penalty=5.0,
    multi_regime_val=False,
):
    print(f"--- Starting Agent Training Pipeline ---")
    print(
        f"Asset: {symbol} | Interval: {interval} | Steps: {steps} | Entropy: {entropy} | Device: {device} | Extractor: {extractor}"
    )
    print(
        f"Commission: {commission} | Slippage: {slippage} | Cooldown: {cooldown} | Max Trade Duration: {max_trade_duration} | Inactivity Penalty: {inactivity_penalty} | Negative PnL Penalty: {negative_pnl_penalty} | Regime Feature: {regime} | Context Features: {context}"
    )
    print(
        f"PPO Tuning: gamma={gamma} | gae_lambda={gae_lambda} | target_kl={target_kl} | clip_range_vf={clip_range_vf} | warm_start_dsr={warm_start_dsr} | use_dsor={use_dsor} | purge_window={purge_window} | Multi-Regime Val: {multi_regime_val}"
    )

    # 1. Load Data
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", get_asset_file(symbol, interval)
    )
    if not os.path.exists(data_path):
        print(
            f"Error: Data file {data_path} not found. Please sync data via the app first."
        )
        return

    df_full = pd.read_csv(data_path)

    if multi_regime_val:
        # Strict 3-way partition:
        # 1. Train Partition: 2020-01-04 to 2025-03-31 (45,886 rows, 5+ years of diverse data)
        # 2. Multi-Regime Validation Windows (used solely to select best checkpoint):
        #    - Val Bull: 2025-04-01 to 2025-10-31 (5,136 rows, +32.6% B&H, -17.0% Max DD)
        #    - Val Bear: 2025-11-01 to 2026-05-31 (5,088 rows, -32.9% B&H, -43.4% Max DD)
        # 3. Retrospective evaluation: 2026-06-01 onward. Earlier experiments
        # already used this period for validation; it is not a sealed test.
        dt_col = pd.to_datetime(df_full["datetime"])
        train_mask = dt_col < "2025-04-01"
        df_train_raw = df_full[train_mask].reset_index(drop=True)
        if purge_window > 0 and len(df_train_raw) > purge_window:
            df_train_raw = df_train_raw.iloc[:-purge_window].reset_index(drop=True)
        df_train = compute_indicators(df_train_raw, fit_hmm=True)

        df_bull_raw = df_full[(dt_col >= "2025-04-01") & (dt_col < "2025-11-01")].reset_index(drop=True)
        df_val_bull = compute_indicators(df_bull_raw, fit_hmm=False)

        df_bear_raw = df_full[(dt_col >= "2025-11-01") & (dt_col < "2026-06-01")].reset_index(drop=True)
        df_val_bear = compute_indicators(df_bear_raw, fit_hmm=False)

        val_windows = [
            ("Val Bull", df_val_bull),
            ("Val Bear", df_val_bear),
        ]
        df_val = df_val_bear
        df_val_eval = df_val_bear
        print(
            f"[Multi-Regime Mode] Training on {len(df_train)} rows (before April 2025, with purge), Val Bull={len(df_val_bull)} rows, Val Bear={len(df_val_bear)} rows. June 2026 onward is retrospective evaluation, not an untouched test."
        )
    else:
        # Standard chronological 80/20 train/val split
        test_size = max(50, int(len(df_full) * 0.2))
        purge_gap = min(purge_window, max(0, int(test_size * 0.1)))
        train_end = max(100, len(df_full) - test_size - purge_gap)
        df_train = compute_indicators(df_full.iloc[:train_end].reset_index(drop=True), fit_hmm=True)
        df_val = compute_indicators(df_full.iloc[-test_size:].reset_index(drop=True), fit_hmm=False)
        df_val_eval = df_val.reset_index(drop=True)
        val_windows = None
        print(
            f"Data Loaded: {len(df_full)} total rows. Training on {len(df_train)} rows (0:{train_end}), purged {purge_gap} rows, validating on {len(df_val)} rows."
        )

    if seed is not None:
        import random
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # 3. Model Setup
    USE_DEVICE = device
    persistent_weights = (
        save_path
        if save_path is not None
        else os.path.join(os.path.dirname(__file__), "..", "models", "persistent_brain.pth")
    )
    normalizer_path = persistent_weights.replace(".pth", "_normalizer.pkl")
    is_continuing = os.path.exists(persistent_weights) and not reset_weights

    # Architecture configuration
    if extractor in ("mlp", "small"):
        extractor_class = None
        features_dim = 64
        is_deep = False
        POLICY_KWARGS = dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
    elif extractor == "tiny":
        extractor_class = TinyTransformerExtractor
        features_dim = 64
        is_deep = False
        POLICY_KWARGS = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
            share_features_extractor=False,
            net_arch=dict(pi=[128, 128], vf=[256, 256]),
        )
    else:
        extractor_class = DeepTransformerExtractor
        features_dim = 256
        is_deep = True
        POLICY_KWARGS = dict(
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
            share_features_extractor=False,
            net_arch=dict(pi=[128, 128], vf=[256, 256]),
        )

    checkpoint = None
    include_regime = regime
    include_context = context
    if is_continuing:
        print(f"Loading existing weights from {persistent_weights}...")
        checkpoint = torch.load(
            persistent_weights, map_location=torch.device(USE_DEVICE), weights_only=False
        )
        state_dict = checkpoint["policy"] if "policy" in checkpoint else checkpoint

        # Self-describe the extractor and features size based on stored weight shapes to ensure zero crash during loading
        for key in state_dict.keys():
            if "features_extractor.embedding.weight" in key:
                out_features = state_dict[key].shape[0]
                in_features = state_dict[key].shape[1]
                include_regime = (in_features in (12, 17))  # 12=old regime, 17=new regime
                include_context = (in_features in (16, 17))
                
                if out_features == 64:
                    extractor_class = TinyTransformerExtractor
                    features_dim = 64
                    is_deep = False
                    POLICY_KWARGS = dict(
                        features_extractor_class=extractor_class,
                        features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
                        share_features_extractor=False,
                        net_arch=dict(pi=[128, 128], vf=[256, 256]),
                    )
                    print(
                        f"Auto-detected TinyTransformerExtractor architecture (include_regime={include_regime}, include_context={include_context}) from persistent checkpoint."
                    )
                elif out_features == 256:
                    extractor_class = DeepTransformerExtractor
                    features_dim = 256
                    is_deep = True
                    POLICY_KWARGS = dict(
                        features_extractor_class=extractor_class,
                        features_extractor_kwargs=dict(features_dim=features_dim, n_stack=8),
                        share_features_extractor=False,
                        net_arch=dict(pi=[128, 128], vf=[256, 256]),
                    )
                    print(
                        f"Auto-detected DeepTransformerExtractor architecture (include_regime={include_regime}, include_context={include_context}) from persistent checkpoint."
                    )
                break

    is_gpu = "cuda" in USE_DEVICE.lower()

    # Setup Environment using detected include_regime shape
    # Dynamically scale parallel environments to maximize GPU utilization and CPU throughput
    cpu_count = os.cpu_count() or 4
    if is_gpu:
        num_parallel_envs = min(12, max(6, cpu_count - 2))
    else:
        num_parallel_envs = min(8, max(4, cpu_count - 2))

    # Scale max trade holding period by data interval only if not explicitly set to 0 (disabled)
    if max_trade_duration != 0:
        max_trade_duration = 72 if interval == "1H" else 30
    print(f"Max trade duration: {max_trade_duration} steps")

    env_funcs = [
        partial(
            make_env_global,
            df_train,
            commission_rate=commission,
            slippage_rate=slippage,
            cooldown_steps=cooldown,
            inactivity_penalty=inactivity_penalty,
            negative_pnl_penalty=negative_pnl_penalty,
            volatility_scaling=volatility_scaling,
            drawdown_penalty_coef=drawdown_penalty_coef,
            max_drawdown_cap=max_drawdown_cap,
            include_regime=include_regime,
            include_context=include_context,
            max_trade_duration=max_trade_duration,
            max_steps=10995,
            warm_start_dsr=warm_start_dsr,
            use_dsor=use_dsor,
            gradual_drawdown=gradual_drawdown,
            drawdown_cap_penalty=drawdown_cap_penalty,
        )
        for _ in range(num_parallel_envs)
    ]

    train_env = DummyVecEnv(env_funcs)
    train_env = VecFrameStack(train_env, n_stack=8)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=gamma)

    if is_deep and is_gpu:
        base_lr = 1e-4 if lr is None else lr
        optimized_n_steps = 1024
        optimized_batch_size = 512
        optimized_epochs = 8
        print(
            f"\n[DEEP GPU MODE ACTIVATED]: n_steps={optimized_n_steps}, batch_size={optimized_batch_size}, lr={base_lr:.2e}, num_envs={num_parallel_envs}"
        )
    elif is_deep:  # Deep CPU
        base_lr = 1e-5 if lr is None else lr
        optimized_n_steps = 2048
        optimized_batch_size = 1024
        optimized_epochs = 4
        print(
            f"\n[DEEP CPU MODE ACTIVATED]: n_steps={optimized_n_steps}, batch_size={optimized_batch_size}, lr={base_lr:.2e}"
        )
    elif extractor in ("mlp", "small"):
        base_lr = 3e-4 if lr is None else lr
        optimized_n_steps = 2048
        optimized_batch_size = 512 if is_gpu else 256
        optimized_epochs = 4
        print(
            f"\n[SMALL MLP PPO MODE ACTIVATED]: n_steps={optimized_n_steps}, batch_size={optimized_batch_size}, lr={base_lr:.2e}"
        )
    else:  # Tiny model
        base_lr = 1e-4 if lr is None else lr
        optimized_n_steps = 8192 if is_gpu else 2048
        optimized_batch_size = 4096 if is_gpu else 1024
        optimized_epochs = 3 if is_gpu else 4
        print(
            f"\n[TINY MODE ACTIVATED]: n_steps={optimized_n_steps}, batch_size={optimized_batch_size}, lr={base_lr:.2e}"
        )

    # 2. Entropy: Respect the user's defined entropy parameter.
    # We no longer auto-boost to 0.10 because it was found to overpower the policy gradient
    # and force the agent into a perfectly uniform distribution trap.
    print(f"Using starting exploration entropy of {entropy:.4f}")

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=POLICY_KWARGS,
        verbose=0,
        learning_rate=linear_schedule(
            base_lr, floor=lr_floor
        ),  # Optimized: Dynamic LR base + decay schedule
        n_steps=optimized_n_steps,
        batch_size=optimized_batch_size,
        n_epochs=optimized_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range_vf=clip_range_vf,
        target_kl=target_kl,
        vf_coef=0.5,
        ent_coef=entropy,
        seed=seed,
        device=USE_DEVICE,
        tensorboard_log="./tensorboard_logs/",
    )

    if is_continuing and checkpoint is not None:
        model.policy.load_state_dict(checkpoint["policy"])
        model.policy.optimizer.load_state_dict(checkpoint["optimizer"])
        model.num_timesteps = checkpoint.get("total_steps", 0)
        # Allow the user to override/boost the entropy from the GUI/CLI parameter if they wish,
        # otherwise default to the saved decayed entropy from the checkpoint.
        saved_ent_coef = checkpoint.get("ent_coef", entropy)
        if not override_entropy and saved_ent_coef is not None:
            # If we are continuing and not explicitly overriding, use the saved decayed entropy
            model.ent_coef = saved_ent_coef
            entropy = saved_ent_coef
            print(
                f"Resuming from step {model.num_timesteps:,} (ent_coef={saved_ent_coef:.4f})"
            )
        else:
            model.ent_coef = entropy
            print(
                f"Resuming from step {model.num_timesteps:,}. Starting entropy set to user override: {entropy:.4f} (saved checkpoint was {saved_ent_coef:.4f})"
            )
    else:
        print("Starting fresh optimization from scratch.")

    # Maintain an exploration floor (at least 0.015) to prevent logit saturation and single-action collapse
    final_ent_coef = max(0.015, min(entropy * 0.4, 0.05 if is_deep else 0.025))
    print(f"Entropy decay target floor set to: {final_ent_coef:.6f}")

    # Load normalizer stats on resume so reward scaling is consistent across training sessions.
    # VecNormalize.load() restores the full running mean/variance object saved by vec_norm.save().
    if is_continuing and os.path.exists(normalizer_path):
        train_env = VecNormalize.load(normalizer_path, train_env)
        train_env.training = True   # Keep updating running stats during continued training
        train_env.norm_reward = True
        # CRITICAL: VecNormalize.load() returns a NEW wrapper object. The model was bound
        # to the old env during PPO(...) construction, so we must rebind it here.
        model.set_env(train_env)
        print(f"Reward normalizer stats loaded from {normalizer_path}")
    elif is_continuing:
        print("No normalizer stats found — starting normalizer from scratch (first run with VecNormalize).")
    else:
        print("Fresh normalizer stats — reward normalization will adapt over first rollout.")


    # 4. Train
    print("Optimization started...")

    # Scale eval_freq by num_parallel_envs because the callback counts environment steps (calls), not total timesteps
    raw_eval_freq = max(5000, steps // 10)
    eval_freq = max(1, raw_eval_freq // num_parallel_envs)

    # FastEvalCallback evaluates the agent on the unseen validation set and saves only the BEST weights
    fast_eval_callback = FastEvalCallback(
        df_val=df_val_eval,
        eval_freq=eval_freq,
        save_path=persistent_weights,
        commission=commission,
        slippage=slippage,
        cooldown=cooldown,
        inactivity_penalty=inactivity_penalty,
        negative_pnl_penalty=negative_pnl_penalty,
        volatility_scaling=volatility_scaling,
        drawdown_penalty_coef=drawdown_penalty_coef,
        max_drawdown_cap=max_drawdown_cap,
        max_trade_duration=max_trade_duration,
        gradual_drawdown=gradual_drawdown,
        drawdown_cap_penalty=drawdown_cap_penalty,
        val_windows=val_windows,
        verbose=1,
    )

    cli_callback = CLICallback(steps, eval_callback=fast_eval_callback)
    entropy_callback = EntropyDecayCallback(
        initial_ent_coef=entropy, final_ent_coef=final_ent_coef, total_steps=steps
    )

    # Load the best objective score from checkpoint upon resume to prevent overwriting with a worse score
    if is_continuing and checkpoint is not None:
        best_score = checkpoint.get("best_mean_reward", -999.0)
        if best_score == -float("inf") or best_score is None:
            best_score = -999.0
        print(f"Resuming: loaded best validation objective score of {best_score:.2f} from checkpoint.")
        fast_eval_callback.best_mean_reward = best_score

    # Using reset_num_timesteps=True ensures that resuming starts a fresh learning rate schedule
    # from the start of the budget, avoiding sudden learning rate jumps (Scenario B).
    model.learn(
        total_timesteps=steps,
        callback=[cli_callback, entropy_callback, fast_eval_callback],
        reset_num_timesteps=True,
    )

    # 5. Save final checkpoint and fallback separately
    final_weights_path = persistent_weights.replace(".pth", "_final.pth")
    torch.save(
        {
            "policy": model.policy.state_dict(),
            "optimizer": model.policy.optimizer.state_dict(),
            "total_steps": model.num_timesteps,
            "ent_coef": model.ent_coef,
            "cooldown_steps": cooldown,
            "inactivity_penalty": inactivity_penalty,
            "negative_pnl_penalty": negative_pnl_penalty,
            "best_mean_reward": fast_eval_callback.best_mean_reward,
        },
        final_weights_path,
    )
    # Save normalizer stats alongside the final checkpoint
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        vec_norm.save(normalizer_path)

    # Fallback: if EvalCallback never saved (e.g. extremely short steps or <5 trades),
    # preserve initial weights if no model exists on disk yet
    if not os.path.exists(persistent_weights):
        torch.save(
            {
                "policy": model.policy.state_dict(),
                "optimizer": model.policy.optimizer.state_dict(),
                "total_steps": model.num_timesteps,
                "ent_coef": model.ent_coef,
                "cooldown_steps": cooldown,
                "inactivity_penalty": inactivity_penalty,
                "negative_pnl_penalty": negative_pnl_penalty,
                "best_mean_reward": fast_eval_callback.best_mean_reward,
            },
            persistent_weights,
        )

    print(f"\nTraining Complete! Final model saved to {final_weights_path}")
    print(f"Best performing model preserved at {persistent_weights}")
    print(f"Total model timesteps: {model.num_timesteps:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the RL Trading Agent.")
    parser.add_argument(
        "--symbol", type=str, default="BTC", help="Asset symbol (e.g., BTC, ETH, SPY)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1H",
        choices=["1H", "Daily"],
        help="Data interval",
    )
    parser.add_argument(
        "--steps", type=int, default=64000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--entropy",
        type=float,
        default=0.05,
        help="Exploration entropy (Decays to 0.001 automatically)",
    )
    parser.add_argument(
        "--override-entropy",
        action="store_true",
        help="Force override saved checkpoint entropy with the user override value",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start training from scratch (ignore persistent weights)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="tiny",
        choices=["deep", "tiny"],
        help="Model neural extractor architecture",
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Trading commission fee rate (e.g. 0.001 = 0.1%%, set to 0.0 for fee holiday)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0005,
        help="Trading slippage rate (e.g. 0.0005 = 0.05%%, set to 0.0 for fee holiday)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=12,
        help="Remaining cooldown steps after position exits before opening a new one",
    )
    parser.add_argument(
        "--inactivity-penalty",
        type=float,
        default=0.0005,
        help="Step-by-step penalty for staying flat to encourage active trading (default: 0.0005)",
    )
    parser.add_argument(
        "--negative-pnl-penalty",
        type=float,
        default=0.0,
        help="End-of-episode penalty if net worth is below starting balance",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Include dynamic 5-regime classification feature in RL observation space",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Include macro context features in RL observation space",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Custom learning rate (overrides default optimized architecture base learning rates)",
    )

    parser.add_argument(
        "--lr-floor",
        type=float,
        default=0.2,
        help="Minimum baseline floor for the learning rate schedule (e.g. 0.2 = 20%% floor)",
    )
    parser.add_argument(
        "--volatility-scaling",
        type=float,
        default=0.5,
        help="Scaling factor for dampening step returns during high market volatility",
    )
    parser.add_argument(
        "--drawdown-penalty-coef",
        type=float,
        default=0.1,
        help="Continuous step-level penalty coefficient for holding trailing drawdowns",
    )
    parser.add_argument(
        "--max-drawdown-cap",
        type=float,
        default=0.15,
        help="Maximum drawdown threshold before episode is terminated early (stop loss)",
    )
    parser.add_argument(
        "--max-trade-duration",
        type=int,
        default=48,
        help="Maximum trade holding duration in steps before position is force-closed (0 = unlimited)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.97,
        help="PPO & VecNormalize discount factor (default: 0.97 for 1H crypto momentum horizon)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.92,
        help="Generalized Advantage Estimation lambda (default: 0.92)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="Target KL divergence for early stopping of policy updates (default: 0.02, None to disable)",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=None,
        help="Value function loss clipping threshold (default: None)",
    )
    parser.add_argument(
        "--use-dsor",
        action="store_true",
        help="Use Differential Sortino Ratio instead of Differential Sharpe Ratio",
    )
    parser.add_argument(
        "--purge-window",
        type=int,
        default=48,
        help="Purge gap between training and validation data (default: 48 steps)",
    )
    parser.add_argument(
        "--gradual-drawdown",
        action="store_true",
        help="Use continuous gradual drawdown penalty without catastrophic cliff termination",
    )
    parser.add_argument(
        "--drawdown-cap-penalty",
        type=float,
        default=5.0,
        help="Penalty deducted when max drawdown cap is reached (default: 5.0, gradual: 0.25)",
    )
    parser.add_argument(
        "--multi-regime-val",
        action="store_true",
        help="Use 3-way partition: Train to 2025-03-31, multi-regime validation (Bull & Bear), and reserved 2026 test set",
    )

    args = parser.parse_args()

    regime_bool = args.regime.lower() == "true"
    context_bool = args.context.lower() == "true"
    train_agent(
        args.symbol,
        args.interval,
        args.steps,
        args.entropy,
        args.override_entropy,
        args.reset,
        args.device,
        args.extractor,
        args.commission,
        args.slippage,
        args.cooldown,
        args.inactivity_penalty,
        args.negative_pnl_penalty,
        volatility_scaling=args.volatility_scaling,
        drawdown_penalty_coef=args.drawdown_penalty_coef,
        max_drawdown_cap=args.max_drawdown_cap,
        lr_floor=args.lr_floor,
        regime=regime_bool,
        context=context_bool,
        lr=args.lr,
        max_trade_duration=args.max_trade_duration,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        target_kl=args.target_kl,
        clip_range_vf=args.clip_range_vf,
        use_dsor=args.use_dsor,
        purge_window=args.purge_window,
        gradual_drawdown=args.gradual_drawdown,
        drawdown_cap_penalty=args.drawdown_cap_penalty,
        multi_regime_val=args.multi_regime_val,
    )
