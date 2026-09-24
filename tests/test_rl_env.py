import pytest
import numpy as np
import pandas as pd
import os
import sys
from typing import Any, Dict, Tuple, Optional

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from rl_env import ActiveCryptoEnv

# Action constants (3-state target positions)
ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2

# Backward compatibility aliases for tests
ACTION_HOLD = ACTION_CASH
ACTION_OPEN_LONG = ACTION_LONG
ACTION_CLOSE_LONG = ACTION_CASH
ACTION_OPEN_SHORT = ACTION_SHORT
ACTION_CLOSE_SHORT = ACTION_CASH

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal synthetic DataFrame that mirrors the real data schema."""
    np.random.seed(42)
    n = 50
    prices = 50000 + np.cumsum(np.random.randn(n) * 100)  # Random walk around 50k
    df = pd.DataFrame({
        'open':   prices + np.random.randn(n) * 10,
        'high':   prices + abs(np.random.randn(n) * 50),
        'low':    prices - abs(np.random.randn(n) * 50),
        'close':  prices,
        'volume': np.random.randint(100, 10000, size=n).astype(float),
        'sma_25': pd.Series(prices).rolling(5, min_periods=1).mean(),
        'rsi_14': 50 + np.random.randn(n) * 15,
        'macd':   np.random.randn(n) * 10,
    })
    df['volume_sma_20'] = df['volume'].rolling(5, min_periods=1).mean()
    return df


@pytest.fixture
def env(sample_df: pd.DataFrame) -> ActiveCryptoEnv:
    """Create a fresh environment instance."""
    return ActiveCryptoEnv(sample_df, initial_balance=10000.0, random_start=False, cooldown_steps=4, include_regime=False, include_context=False)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_initial_balance(self, env: ActiveCryptoEnv) -> None:
        """Environment should start with the specified initial balance."""
        assert env.balance == 10000.0

    def test_initial_no_position(self, env: ActiveCryptoEnv) -> None:
        """Agent should not hold any position at the start."""
        assert env.is_long is False
        assert env.is_short is False

    def test_initial_net_worth(self, env: ActiveCryptoEnv) -> None:
        """Net worth should equal initial balance at the start."""
        assert env.net_worth == 10000.0

    def test_initial_cooldown(self, env: ActiveCryptoEnv) -> None:
        """Cooldown should be zero at the start."""
        assert env.cooldown == 0

    def test_action_space(self, env: ActiveCryptoEnv) -> None:
        """Action space should be Discrete(3): Cash, Target Long, Target Short."""
        assert env.action_space.n == 3

    def test_observation_space_shape(self, env: ActiveCryptoEnv) -> None:
        assert env.observation_space.shape == (11,)

    def test_initial_entry_price(self, env: ActiveCryptoEnv) -> None:
        """Entry price should be zero at the start (no position)."""
        assert env.entry_price == 0.0


# ---------------------------------------------------------------------------
# Observation Tests
# ---------------------------------------------------------------------------

class TestObservation:
    def test_observation_dtype(self, env: ActiveCryptoEnv) -> None:
        """Observations should be float32 numpy arrays."""
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_observation_shape(self, env: ActiveCryptoEnv) -> None:
        """Observations should match the observation space shape."""
        obs, _ = env.reset()
        assert obs.shape == (11,)

    def test_observation_no_nan(self, env: ActiveCryptoEnv) -> None:
        """Observations should never contain NaN values."""
        obs, _ = env.reset()
        for _ in range(min(20, len(env.df) - 2)):
            assert not np.any(np.isnan(obs)), f"NaN found in observation: {obs}"
            obs, _, term, trunc, _ = env.step(ACTION_HOLD)
            done = term or trunc
            if done:
                break

    def test_observation_no_inf(self, env: ActiveCryptoEnv) -> None:
        """Observations should never contain infinite values."""
        obs, _ = env.reset()
        for _ in range(min(20, len(env.df) - 2)):
            assert not np.any(np.isinf(obs)), f"Inf found in observation: {obs}"
            obs, _, term, trunc, _ = env.step(ACTION_HOLD)
            done = term or trunc
            if done:
                break

    def test_observation_position_side_flat(self, env: ActiveCryptoEnv) -> None:
        """Position side (obs[7]) should be 0.0 when flat (no position)."""
        obs, _ = env.reset()
        assert obs[7] == 0.0

    def test_observation_position_side_long(self, env: ActiveCryptoEnv) -> None:
        """Position side (obs[7]) should be 1.0 when holding a long position."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        obs = env._get_observation()
        assert obs[7] == 1.0

    def test_observation_position_side_short(self, env: ActiveCryptoEnv) -> None:
        """Position side (obs[7]) should be -1.0 when holding a short position."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        obs = env._get_observation()
        assert obs[7] == -1.0


# ---------------------------------------------------------------------------
# Long Position Tests
# ---------------------------------------------------------------------------

class TestOpenLong:
    def test_open_long_sets_position(self, env: ActiveCryptoEnv) -> None:
        """Opening a long should set is_long to True."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert env.is_long is True
        assert env.is_short is False

    def test_open_long_records_entry_price(self, env: ActiveCryptoEnv) -> None:
        """Entry price should be recorded with slippage when opening a long."""
        env.reset()
        market_price = env.np_open[1]
        expected_price = market_price * (1.0 + env.slippage_rate)
        env.step(ACTION_OPEN_LONG)
        assert abs(env.entry_price - expected_price) < 0.001

    def test_open_long_no_cooldown(self, env: ActiveCryptoEnv) -> None:
        """After opening a long, cooldown should NOT be set."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert env.cooldown == 0

    def test_open_long_deducts_commission(self, env: ActiveCryptoEnv) -> None:
        """Opening a long should deduct the entry commission from the balance."""
        env.reset()
        initial_balance = env.balance
        env.step(ACTION_OPEN_LONG)
        expected_balance = initial_balance * (1.0 - env.commission_rate)
        assert abs(env.balance - expected_balance) < 0.01


class TestCloseLong:
    def test_close_long_clears_position(self, env: ActiveCryptoEnv) -> None:
        """Closing a long should set is_long to False."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        for _ in range(4):
            env.step(ACTION_LONG)
        env.step(ACTION_CASH)
        assert env.is_long is False

    def test_close_long_resets_entry_price(self, env: ActiveCryptoEnv) -> None:
        """After closing a long, entry_price should be reset to 0."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        for _ in range(4):
            env.step(ACTION_LONG)
        env.step(ACTION_CASH)
        assert env.entry_price == 0.0

    def test_close_long_sets_cooldown(self, env: ActiveCryptoEnv) -> None:
        """After closing a long, cooldown should be set to cooldown_steps."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_CASH)
        assert env.cooldown == env.cooldown_steps

    def test_close_long_updates_balance(self, env: ActiveCryptoEnv) -> None:
        """Closing a long should update balance to reflect realized PnL after slippage and commissions."""
        env.reset()
        initial_balance = env.balance
        env.step(ACTION_OPEN_LONG)
        entry_price = env.entry_price
        for _ in range(4):
            env.step(ACTION_LONG)
        close_price = env.np_open[6]
        exit_price = close_price * (1.0 - env.slippage_rate)
        env.step(ACTION_CASH)
        
        val_at_entry = initial_balance * (1.0 - env.commission_rate)
        pnl = (exit_price - entry_price) / entry_price
        val_after_pnl = val_at_entry * (1.0 + pnl)
        expected_balance = val_after_pnl * (1.0 - env.commission_rate)
        
        assert abs(env.balance - expected_balance) < 0.01


# ---------------------------------------------------------------------------
# Short Position Tests
# ---------------------------------------------------------------------------

class TestOpenShort:
    def test_open_short_sets_position(self, env: ActiveCryptoEnv) -> None:
        """Opening a short should set is_short to True."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        assert env.is_short is True
        assert env.is_long is False

    def test_open_short_records_entry_price(self, env: ActiveCryptoEnv) -> None:
        """Entry price should be recorded with slippage when opening a short."""
        env.reset()
        market_price = env.np_open[1]
        expected_price = market_price * (1.0 - env.slippage_rate)
        env.step(ACTION_OPEN_SHORT)
        assert abs(env.entry_price - expected_price) < 0.001

    def test_open_short_no_cooldown(self, env: ActiveCryptoEnv) -> None:
        """After opening a short, cooldown should NOT be set."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        assert env.cooldown == 0


class TestCloseShort:
    def test_close_short_clears_position(self, env: ActiveCryptoEnv) -> None:
        """Closing a short should set is_short to False."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        for _ in range(4):
            env.step(ACTION_SHORT)
        env.step(ACTION_CASH)
        assert env.is_short is False

    def test_close_short_resets_entry_price(self, env: ActiveCryptoEnv) -> None:
        """After closing a short, entry_price should be reset to 0."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        for _ in range(4):
            env.step(ACTION_SHORT)
        env.step(ACTION_CASH)
        assert env.entry_price == 0.0

    def test_close_short_sets_cooldown(self, env: ActiveCryptoEnv) -> None:
        """After closing a short, cooldown should be set to cooldown_steps."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        env.step(ACTION_CASH)
        assert env.cooldown == env.cooldown_steps


# ---------------------------------------------------------------------------
# Hold Action Tests
# ---------------------------------------------------------------------------

class TestHoldAction:
    def test_hold_preserves_balance(self, env: ActiveCryptoEnv) -> None:
        """Holding cash should not change the balance."""
        env.reset()
        initial_balance = env.balance
        env.step(ACTION_HOLD)
        assert env.balance == initial_balance

    def test_hold_preserves_no_position(self, env: ActiveCryptoEnv) -> None:
        """Holding cash should not open any position."""
        env.reset()
        env.step(ACTION_HOLD)
        assert env.is_long is False
        assert env.is_short is False

    def test_maintain_long_preserves_balance(self, env: ActiveCryptoEnv) -> None:
        """Maintaining a long position incurs zero transaction fees."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        balance_after_entry = env.balance
        env.step(ACTION_OPEN_LONG)  # Target Long again
        assert env.balance == balance_after_entry

    def test_maintain_short_preserves_balance(self, env: ActiveCryptoEnv) -> None:
        """Maintaining a short position incurs zero transaction fees."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        balance_after_entry = env.balance
        env.step(ACTION_OPEN_SHORT)  # Target Short again
        assert env.balance == balance_after_entry


# ---------------------------------------------------------------------------
# Reward and Differential PnL Tests
# ---------------------------------------------------------------------------

class TestRewards:
    def test_cash_gives_zero_reward_when_flat(self, env: ActiveCryptoEnv) -> None:
        """Holding cash when flat yields strictly 0.0 reward, preventing DSR cash-hoarding gravity wells."""
        env.reset()
        obs, reward, _, _, _ = env.step(ACTION_HOLD)
        assert reward == 0.0

    def test_illegal_action_during_cooldown_penalized(self, env: ActiveCryptoEnv) -> None:
        """Attempting to enter a position while cooldown is active incurs an invalid action penalty."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_CLOSE_LONG)  # sets cooldown = 4
        assert env.cooldown == 4
        obs, reward, _, _, _ = env.step(ACTION_OPEN_LONG)  # cooldown decrements to 3, still > 0
        assert reward <= -0.05

    def test_cannot_open_long_during_cooldown(self, env: ActiveCryptoEnv) -> None:
        """Trying to Open Long during cooldown should be treated as Hold (no-op)."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_CLOSE_LONG)  # sets cooldown to 4
        prev_net_worth = env.net_worth
        # Try to open long again while cooldown is active
        env.step(ACTION_OPEN_LONG)
        assert env.is_long is False  # entry blocked by cooldown
        assert env.net_worth == prev_net_worth  # balance unchanged

    def test_flip_long_to_short(self, env: ActiveCryptoEnv) -> None:
        """Selecting ACTION_OPEN_SHORT while in a long position should automatically flip to short."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert env.is_long is True
        env.step(ACTION_OPEN_SHORT)
        assert env.is_long is False
        assert env.is_short is True

    def test_flip_short_to_long(self, env: ActiveCryptoEnv) -> None:
        """Selecting ACTION_OPEN_LONG while in a short position should automatically flip to long."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        assert env.is_short is True
        env.step(ACTION_OPEN_LONG)
        assert env.is_short is False
        assert env.is_long is True

    def test_flip_does_not_set_cooldown(self, env: ActiveCryptoEnv) -> None:
        """Flipping positions should not set any cooldown."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert env.cooldown == 0
        env.step(ACTION_OPEN_SHORT)
        assert env.cooldown == 0
        env.step(ACTION_OPEN_LONG)
        assert env.cooldown == 0

    def test_differential_reward_pnl(self, env: ActiveCryptoEnv) -> None:
        """Reward should represent the scaled log-wealth return adjusted for drawdown penalties."""
        env.reset()
        prev_net_worth = env.net_worth
        _, reward, _, _, _ = env.step(ACTION_OPEN_LONG)
        
        step_log_return = np.log(env.net_worth / prev_net_worth)
        expected_reward = np.clip(step_log_return * 10.0, -1.0, 1.0)
        
        current_drawdown = (env.peak_net_worth - env.net_worth) / env.peak_net_worth if env.peak_net_worth > 0 else 0.0
        if current_drawdown > 0:
            expected_reward -= env.drawdown_penalty_coef * current_drawdown
            
        assert abs(reward - expected_reward) < 0.0001

    def test_inactivity_penalty(self, sample_df: pd.DataFrame) -> None:
        """Inactivity penalty should be subtracted from the reward when flat beyond the 24-step grace period."""
        custom_env = ActiveCryptoEnv(
            sample_df,
            initial_balance=10000.0,
            random_start=False,
            include_regime=False,
            include_context=False,
            inactivity_penalty=0.05,
        )
        custom_env.reset()
        # Step through the 6-step flat inactivity grace period
        for _ in range(6):
            custom_env.step(ACTION_HOLD)
        
        # Step 7: Exceeds grace period (excess_idle = 1), so escalating penalty applies
        _, reward, _, _, _ = custom_env.step(ACTION_HOLD)
        expected_penalty = 0.05 * (1.0 + 1.0 / 12.0)
        assert abs(reward - (-expected_penalty)) < 0.0001


# ---------------------------------------------------------------------------
# Cooldown Tests
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_decrements(self, env: ActiveCryptoEnv) -> None:
        """Cooldown should decrement by 1 each step."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_CLOSE_LONG)  # Cooldown set to 4
        assert env.cooldown == 4
        env.step(ACTION_HOLD)        # Cooldown decrements to 3
        assert env.cooldown == 3
        env.step(ACTION_HOLD)        # Cooldown decrements to 2
        assert env.cooldown == 2
        env.step(ACTION_HOLD)        # Cooldown decrements to 1
        assert env.cooldown == 1
        env.step(ACTION_HOLD)        # Cooldown decrements to 0
        assert env.cooldown == 0

    def test_can_close_long_immediately(self, env: ActiveCryptoEnv) -> None:
        """Agent should be able to close a long position immediately on the next step."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert env.is_long is True
        env.step(ACTION_CLOSE_LONG)
        assert env.is_long is False

    def test_can_close_short_immediately(self, env: ActiveCryptoEnv) -> None:
        """Agent should be able to close a short position immediately on the next step."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        assert env.is_short is True
        env.step(ACTION_CLOSE_SHORT)
        assert env.is_short is False


# ---------------------------------------------------------------------------
# Episode / Done Tests
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_episode_terminates(self, env: ActiveCryptoEnv) -> None:
        """Episode should terminate after stepping through all data."""
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = env.step(ACTION_HOLD)
            done = term or trunc
            steps += 1
        assert done is True
        assert steps == len(env.df) - 1

    def test_reset_restores_state(self, env: ActiveCryptoEnv) -> None:
        """Reset should restore the environment to its initial state."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_HOLD)
        env.step(ACTION_HOLD)
        obs, info = env.reset()
        assert env.balance == 10000.0
        assert env.is_long is False
        assert env.is_short is False
        assert env.entry_price == 0.0
        assert env.current_step == 0
        assert env.cooldown == 0
        assert env.net_worth == 10000.0

    def test_info_contains_net_worth(self, env: ActiveCryptoEnv) -> None:
        """Info dict should always contain 'net_worth'."""
        env.reset()
        _, _, _, _, info = env.step(ACTION_HOLD)
        assert 'net_worth' in info

    def test_force_close_long_at_episode_end(self, env: ActiveCryptoEnv) -> None:
        """Open long positions should be force-closed at episode end."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(ACTION_HOLD)
            done = term or trunc
        assert env.is_long is False

    def test_force_close_short_at_episode_end(self, env: ActiveCryptoEnv) -> None:
        """Open short positions should be force-closed at episode end."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(ACTION_HOLD)
            done = term or trunc
        assert env.is_short is False


# ---------------------------------------------------------------------------
# Net Worth Conservation Tests
# ---------------------------------------------------------------------------

class TestNetWorth:
    def test_net_worth_after_open_long(self, env: ActiveCryptoEnv) -> None:
        """Net worth should approximately equal initial balance right after opening a long."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        assert abs(env.net_worth - 10000.0) < 500.0

    def test_net_worth_after_full_long_cycle(self, env: ActiveCryptoEnv) -> None:
        """Net worth after open long then close long should equal balance."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        for _ in range(4):
            env.step(ACTION_LONG)
        env.step(ACTION_CASH)
        assert abs(env.net_worth - env.balance) < 0.01

    def test_net_worth_after_full_short_cycle(self, env: ActiveCryptoEnv) -> None:
        """Net worth after open short then close short should equal balance."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        for _ in range(4):
            env.step(ACTION_SHORT)
        env.step(ACTION_CASH)
        assert abs(env.net_worth - env.balance) < 0.01

    def test_net_worth_flat_equals_balance(self, env: ActiveCryptoEnv) -> None:
        """When flat (no position), net worth should always equal balance."""
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)
        assert env.net_worth == env.balance


# ---------------------------------------------------------------------------
# Action Masks and Invalid Action Tests
# ---------------------------------------------------------------------------

class TestActionMasksAndInvalidActions:
    def test_action_masks_flat(self, env: ActiveCryptoEnv) -> None:
        """When flat with 0 cooldown, Cash, Target Long, and Target Short must all be valid."""
        env.reset()
        mask = env.action_masks()
        # [Cash, Target Long, Target Short]
        assert np.array_equal(mask, [True, True, True])

    def test_action_masks_long(self, env: ActiveCryptoEnv) -> None:
        """When in a Long position with 0 cooldown, Cash, Target Long (maintain), and Target Short (flip) are valid."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        mask = env.action_masks()
        assert np.array_equal(mask, [True, True, True])

    def test_action_masks_short(self, env: ActiveCryptoEnv) -> None:
        """When in a Short position with 0 cooldown, Cash, Target Long (flip), and Target Short (maintain) are valid."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        mask = env.action_masks()
        assert np.array_equal(mask, [True, True, True])

    def test_action_masks_cooldown(self, env: ActiveCryptoEnv) -> None:
        """When flat during cooldown, only Cash is valid."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        env.step(ACTION_CLOSE_LONG)  # sets cooldown = 4
        assert env.cooldown == 4
        mask = env.action_masks()
        assert np.array_equal(mask, [True, False, False])

    def test_maintain_long_valid_and_not_penalized(self, env: ActiveCryptoEnv) -> None:
        """Calling Target Long when already Long maintains the position with zero penalty."""
        env.reset()
        env.step(ACTION_OPEN_LONG)
        _, reward, _, _, _ = env.step(ACTION_OPEN_LONG)
        assert env.is_long is True

    def test_maintain_short_valid_and_not_penalized(self, env: ActiveCryptoEnv) -> None:
        """Calling Target Short when already Short maintains the position with zero penalty."""
        env.reset()
        env.step(ACTION_OPEN_SHORT)
        _, reward, _, _, _ = env.step(ACTION_OPEN_SHORT)
        assert env.is_short is True

    def test_warm_start_dsr_initialization(self, sample_df: pd.DataFrame) -> None:
        """warm_start_dsr=True must initialize non-zero return moments."""
        warm_env = ActiveCryptoEnv(sample_df, warm_start_dsr=True)
        assert warm_env._init_ema_B > 0.0
        warm_env.reset()
        assert warm_env.ema_B == warm_env._init_ema_B

    def test_gradual_drawdown_safeguard(self) -> None:
        """gradual_drawdown=True closes position and enters cooldown without terminating episode."""
        crash_df = pd.DataFrame({
            "open": [100.0, 100.0, 80.0, 80.0, 80.0],
            "high": [105.0, 105.0, 85.0, 85.0, 85.0],
            "low": [95.0, 95.0, 75.0, 75.0, 75.0],
            "close": [100.0, 100.0, 80.0, 80.0, 80.0],
            "volume": [1000.0] * 5,
            "timestamp": range(5),
            "datetime": pd.date_range("2025-01-01", periods=5, freq="1h"),
        })
        env = ActiveCryptoEnv(
            crash_df,
            max_drawdown_cap=0.15,
            gradual_drawdown=True,
            drawdown_cap_penalty=0.25,
            random_start=False,
            cooldown_steps=2,
        )
        env.reset()
        # Step 1: Open Long at 100
        env.step(ACTION_OPEN_LONG)
        assert env.is_long is True

        # Step 2: Next step price drops to 80 (-20% drawdown, exceeds 15% cap)
        obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
        assert not terminated
        assert env.is_long is False
        assert info.get("force_closed") == "DRAWDOWN_SAFEGUARD"
        assert env.cooldown == 2
        assert env.peak_net_worth == env.net_worth

