import re
import numpy as np
import pandas as pd
import json

class XAIEngine:
    def __init__(self, coin_ticker, current_price, rsi, macd, volume_ratio, action_name, confidence, probs, shap_values=None, history=None, feature_names=None):
        self.coin_ticker = coin_ticker
        self.current_price = current_price
        self.rsi = rsi
        self.macd = macd
        self.volume_ratio = volume_ratio
        self.action_name = action_name
        self.confidence = confidence
        self.probs = probs
        self.shap_values = shap_values
        self.history = history if history else []
        
        # Feature names mapping for SHAP grounding
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            obs_dim = 17  # Default observation dimension
            if shap_values is not None:
                try:
                    if isinstance(shap_values, list) and len(shap_values) > 0:
                        total_features = shap_values[0].shape[1]
                    else:
                        total_features = shap_values.shape[1]
                    # Stacked dimension is obs_dim * 8. If shape is (1, obs_dim, 5), it is just obs_dim.
                    if total_features > 40:
                        obs_dim = total_features // 8
                    else:
                        obs_dim = total_features
                except Exception:
                    pass
            
            core_features = ["Open Ratio", "High Ratio", "Low Ratio", "Volatility", "Price Change", "Price Change Prev", "Normalized Volume"]
            context_features = ["Short Trend (SMA20)", "Medium Trend (SMA99)", "Momentum (ROC24)", "Volatility Ratio", "Macro Drawdown"]
            regime_features = ["Market Regime"]
            portfolio_features = ["Portfolio Position", "Unrealized P&L", "Idle Steps", "Holding Duration"]
            
            if obs_dim == 17:
                self.feature_names = core_features + context_features + regime_features + portfolio_features
            elif obs_dim == 16:
                self.feature_names = core_features + context_features + portfolio_features
            elif obs_dim == 12:
                self.feature_names = core_features + regime_features + portfolio_features
            elif obs_dim == 11:
                self.feature_names = core_features + portfolio_features
            else:
                self.feature_names = core_features + context_features + regime_features + portfolio_features
                if len(self.feature_names) > obs_dim:
                    self.feature_names = self.feature_names[:obs_dim]
                elif len(self.feature_names) < obs_dim:
                    self.feature_names += [f"Feature {k}" for k in range(len(self.feature_names), obs_dim)]

    def classify_intent(self, user_input):
        user_input = user_input.lower()
        
        # --- PHASE 1: Multi-word phrase matching (highest priority) ---
        # These catch suggested-question phrases BEFORE single keywords can misroute them.
        if any(phrase in user_input for phrase in ["profit/loss", "profit and loss", "p&l", "pnl", "show my profit"]):
            return "performance"
        if any(phrase in user_input for phrase in ["explain indicators", "explain the indicators", "what indicators", "which indicators"]):
            return "help"
        if any(phrase in user_input for phrase in ["when will you trade", "when will you buy", "when will you sell", "what are you waiting for"]):
            return "rationale"
        if any(phrase in user_input for phrase in ["where is the momentum", "momentum going", "trend direction"]):
            return "explain_macd"
        
        # --- PHASE 2: Single keyword matching ---
        # 1. Help / Capabilities
        if any(word in user_input for word in ["help", "what can you do", "commands", "how to use", "options"]):
            return "help"
            
        # 2. Risk / Safety
        if any(word in user_input for word in ["risk", "dangerous", "safe", "drawdown", "downside", "stop loss", "lose"]):
            return "risk"
            
        # 3. Confidence / Trust
        if any(word in user_input for word in ["confident", "sure", "trust", "accuracy", "certain", "believe"]):
            return "confidence"
            
        # 4. Performance / Profit
        if any(word in user_input for word in ["perform", "profit", "doing", "return", "roi", "money", "win rate", "loss"]):
            return "performance"
            
        # 5. Specific Indicators (Prioritize over general 'explain')
        if "rsi" in user_input: return "explain_rsi"
        if "macd" in user_input: return "explain_macd"
        if "volume" in user_input: return "explain_volume"

        # 6. Rationale / Why
        if any(word in user_input for word in ["why", "reason", "because", "logic", "explain", "decide", "choice", "thinking"]):
            return "rationale"
            
        # Default fallback
        return "general"

    def generate_response(self, user_input, backtest_metrics=None):
        intent = self.classify_intent(user_input)
        
        if intent == "rationale":
            return self._explain_rationale()
        elif intent == "performance":
            return self._explain_performance(backtest_metrics)
        elif intent == "risk":
            return self._explain_risk(backtest_metrics)
        elif intent == "confidence":
            return self._explain_confidence()
        elif intent == "help":
            return self._explain_help()
        elif intent == "explain_rsi":
            return self._explain_rsi()
        elif intent == "explain_macd":
            return self._explain_macd()
        elif intent == "explain_volume":
            return self._explain_volume()
        else:
            return (
                f"I'm currently focused on **{self.coin_ticker}** at **${self.current_price:,.2f}**. "
                f"My active recommendation is **{self.action_name}** with **{self.confidence:.1f}% confidence**."
            )

    def _get_top_shap_features(self, top_n=3):
        """
        Extract the most influential features from SHAP values for the current recommendation.
        This is the core 'Hallucination Guard'.
        """
        if self.shap_values is None:
            return None
            
        # Canonical 3-action mapping
        action_idx_map = {
            "NEUTRAL": 0, "HOLD": 0, "CASH": 0,
            "BUY (LONG)": 1, "BUY": 1, "LONG": 1,
            "SELL (SHORT)": 2, "SELL": 2, "SHORT": 2,
            "EXIT LONG": 0, "EXIT SHORT": 0
        }
        action_idx = action_idx_map.get(self.action_name, 0)

        # Handle both list and array formats
        if isinstance(self.shap_values, list):
            target_idx = action_idx if action_idx < len(self.shap_values) else 0
            vals = self.shap_values[target_idx][0]
        else:
            # SHAP 0.51+ format: (N_samples, N_features, N_classes)
            target_idx = action_idx if action_idx < self.shap_values.shape[-1] else 0
            vals = self.shap_values[0, :, target_idx]
            
        # Determine observation stride dynamically from the SHAP values shape
        if isinstance(self.shap_values, list):
            total_features = self.shap_values[0].shape[1]
        else:
            total_features = self.shap_values.shape[1]
        
        if total_features > 40:
            stride = total_features // 8
        else:
            stride = total_features
            
        # Aggregate across 8 frame-stacked steps dynamically using the data stride
        num_features = len(self.feature_names)
        agg_vals = np.zeros(num_features)
        for i in range(num_features):
            agg_vals[i] = np.sum(vals[i::stride])
            
        # Get indices of top_n features by absolute magnitude
        top_indices = np.argsort(np.abs(agg_vals))[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            influence = "positive" if agg_vals[idx] > 0 else "negative"
            results.append({
                "feature": self.feature_names[idx],
                "impact": agg_vals[idx],
                "direction": influence
            })
        return results

    def _explain_rationale(self):
        shap_features = self._get_top_shap_features()
        
        action_idx_map = {
            "NEUTRAL": 0, "HOLD": 0, "CASH": 0,
            "BUY (LONG)": 1, "BUY": 1, "LONG": 1,
            "SELL (SHORT)": 2, "SELL": 2, "SHORT": 2,
            "EXIT LONG": 0, "EXIT SHORT": 0
        }
        action_idx = action_idx_map.get(self.action_name, 0)

        # Advisory Introduction for Canonical 3-Action Space
        if action_idx == 0:
            base_msg = f"### 💡 Advisor Recommendation: Target Cash / Neutral on {self.coin_ticker}\n\nI recommend maintaining **CASH (Neutral)** and holding off on market exposure at the current price of **${self.current_price:,.2f}**. "
        elif action_idx == 1:
            base_msg = f"### 💡 Advisor Recommendation: Target Long on {self.coin_ticker}\n\nI recommend opening or maintaining a **LONG (Buy)** position on **{self.coin_ticker}** at the current price of **${self.current_price:,.2f}**. "
        elif action_idx == 2:
            base_msg = f"### 💡 Advisor Recommendation: Target Short on {self.coin_ticker}\n\nI recommend opening or maintaining a **SHORT (Sell)** position on **{self.coin_ticker}** at the current price of **${self.current_price:,.2f}**. "
        else:
            base_msg = f"### 💡 Advisor Recommendation: {self.action_name}\n\n"

        if shap_features:
            features_text = "\n".join([f"- **{f['feature']}**: This metric pushed me {'positively' if f['direction'] == 'positive' else 'negatively'} toward this specific recommendation relative to the explainer baseline (attribution: {f['impact']:+.3e})." for f in shap_features])
            msg = (
                f"{base_msg}"
                f"My reasoning is driven by the following key market factors:\n\n"
                f"{features_text}\n\n"
                f"These attributions explain this model output relative to its baseline; they do not establish a market trend or predict profit."
            )
        else:
            # Fallback if SHAP is missing or unavailable
            rsi_text = "overbought" if self.rsi > 70 else "oversold" if self.rsi < 30 else "neutral"
            msg = (
                f"{base_msg}"
                f"*(Note: Deep SHAP attributions are currently unavailable; providing technical indicator summary).* \n\n"
                f"The policy selected this target, but its feature-level reasons cannot be verified here. "
                f"The RSI is currently **{rsi_text} ({self.rsi:.1f})** and the MACD histogram is **{self.macd:.4f}**. "
                f"These indicators provide context, not evidence that they caused the policy decision."
            )
        return msg

    def _explain_risk(self, metrics):
        max_dd = metrics.get('Max DD %', 0) if metrics else "Unknown"
        
        action_idx_map = {
            "NEUTRAL": 0, "HOLD": 0, "CASH": 0,
            "BUY (LONG)": 1, "BUY": 1, "LONG": 1,
            "SELL (SHORT)": 2, "SELL": 2, "SHORT": 2,
            "EXIT LONG": 0, "EXIT SHORT": 0
        }
        action_idx = action_idx_map.get(self.action_name, 0)

        msg = f"### 🛡️ Risk Assessment & Stop-Loss Advice\n\n"
        if action_idx == 0:
            msg += "Current risk is **Low**. By holding cash (flat), the portfolio preserves capital and eliminates market downside volatility."
        elif action_idx == 1:
            msg += f"Current risk is **Moderate to High**. Holding a LONG exposure at **${self.current_price:,.2f}** exposes your portfolio to market volatility and downward price drops. "
            if isinstance(max_dd, (int, float)):
                msg += f"Historically, strategy drawdowns have reached **{abs(max_dd):.2f}%**; maintaining stop-loss discipline is advised. "
        elif action_idx == 2:
            msg += f"Current risk is **High**. Holding a SHORT exposure at **${self.current_price:,.2f}** exposes your portfolio to market volatility, upward price rallies and squeeze risk. "
            if isinstance(max_dd, (int, float)):
                msg += f"Short trades carry asymmetric risk during market breakouts; strict risk limits are essential. "
        else:
            msg += "Risk is managed according to strategy constraints."
            
        msg += f"\n\n**Market Activity:** Trading volume is **{self.volume_ratio:.1f}x** its reference average. Volume alone does not establish price direction or momentum."
        return msg

    def _explain_confidence(self):
        level = "High" if self.confidence > 80 else "Moderate" if self.confidence > 50 else "Low"
        
        # Safely compute the gap between top and second-best action probabilities
        try:
            if len(self.probs) >= 2:
                second_best = float(np.partition(self.probs, -2)[-2] * 100)
                gap_text = f"The 'gap' between this action and the second-best choice is **{self.confidence - second_best:.1f}%**."
            else:
                gap_text = ""
        except Exception:
            gap_text = ""
            
        return (
            f"### 🎯 Confidence Analysis\n\n"
            f"I have **{level} confidence ({self.confidence:.1f}%)** in the **{self.action_name}** signal. "
            f"\n\nThis confidence score is derived from the probability distribution across all 3 discrete actions (Cash, Long, Short). "
            f"It is not a calibrated probability of profit or a measured prediction accuracy. "
            f"{gap_text}"
        )

    def _explain_performance(self, metrics):
        if not metrics:
            return "I don't have access to my historical performance metrics right now. Please run a backtest first."
            
        roi = metrics.get('ROI %', 0)
        bh = metrics.get('B&H %', 0)
        sharpe = metrics.get('Sharpe', 0)
        
        status = "beating" if roi > bh else "trailing"
        return (
            f"### 📈 Performance Audit\n\n"
            f"In the most recent evaluation, I am **{status}** the market:\n\n"
            f"- **Agent ROI:** `{roi:+.2f}%`\n"
            f"- **Buy & Hold:** `{bh:+.2f}%`\n"
            f"- **Sharpe Ratio:** `{sharpe:.2f}`\n\n"
            f"A Sharpe ratio of `{sharpe:.2f}` indicates that my returns are {'excellent' if sharpe > 2 else 'solid' if sharpe > 1 else 'volatile'} relative to the risk taken."
        )

    def _explain_help(self):
        return (
            f"### 🛠️ How to Interact with me\n\n"
            f"You can ask me various questions about my current state or historical performance:\n\n"
            f"1. **Rationale:** 'Why did you choose this?', 'Explain your logic'\n"
            f"2. **Risk:** 'How risky is this?', 'What is the drawdown?'\n"
            f"3. **Confidence:** 'How sure are you?', 'Is this a strong signal?'\n"
            f"4. **Performance:** 'How are you doing?', 'Show my ROI'\n"
            f"5. **Indicators:** 'What is RSI?', 'Explain MACD'\n\n"
            f"Try clicking one of the **suggested questions** above the chat box!"
        )

    def _explain_rsi(self):
        state = "Overbought" if self.rsi > 70 else "Oversold" if self.rsi < 30 else "Neutral"
        return (
            f"### 📊 Indicator: RSI ({state})\n\n"
            f"The Relative Strength Index (RSI) for **{self.coin_ticker}** is **{self.rsi:.1f}**. "
            f"\n\n- **> 70:** Market is 'hot' and may be due for a downward correction.\n"
            f"- **< 30:** Market is 'fearful' and may be due for an upward bounce.\n"
            f"- **Current:** My model uses this to gauge momentum exhaustion."
        )

    def _explain_macd(self):
        trend = "Bullish" if self.macd > 0 else "Bearish"
        return (
            f"### 📊 Indicator: MACD ({trend})\n\n"
            f"The Moving Average Convergence Divergence histogram is **{self.macd:.4f}**. "
            f"\n\nThis confirms that the short-term trend is currently **{trend.lower()}**. "
            f"My neural network weights this against RSI to ensure we aren't buying into a fake-out."
        )

    def _explain_volume(self):
        return (
            f"### 📊 Indicator: Volume Momentum\n\n"
            f"Trading volume is **{self.volume_ratio:.1f}x** the 20-period moving average. "
            f"\n\nHigh volume (**> 1.0x**) validates price moves. Low volume indicates a lack of conviction from major market participants."
        )
