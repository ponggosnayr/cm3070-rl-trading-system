# CM3070 Final Project: Video Demonstration Script (Readable A/V Layout)

**Target Duration:** 4:20 – 4:40 *(Module lecture benchmark: 3:00–5:00. Confirm exact cohort limit on live Coursera).*  
**Spoken Word Count:** Approximately 400 words (~3.1 minutes spoken at 125–130 WPM, plus screen actions and pauses).  
**Recording Mode:** 1080p MP4 screen recording with genuine student voiceover (no AI narration).

**Before recording:** Open `docs/presentation_slides/slides.html` on Slide 1 and the app at `http://localhost:5173/#/dashboard` in another tab. Wait for the asset list to appear. Keep the teleprompter outside the captured screen. Use the Right Arrow key to advance slides. The time stamps are cues, not deadlines; wait for the advisor response and trim dead time later if needed. Leave the optional Monte Carlo simulation untouched during this short demo.

---

## 🎬 Two-Column Teleprompter Script (Action vs. Speech)

| Time & Visual Action | Spoken Narration (Read Aloud Naturally) |
|---|---|
| **0:00**<br>`[SHOW: slides.html, Slide 1]` | "Hi, I'm **Ryan**. This is my final project: an explainable trading advisor built with reinforcement learning."<br>`[PAUSE 1s]` |
| **0:15**<br>`[PRESS: Right Arrow → Slide 2]` | "Overfitting and transaction costs are major challenges in algorithmic trading that often degrade live market performance.<br><br>My objective was to train an agent's policy within a Markov Decision Process for **downside capital preservation**, showing users **SHAP attributions and plain-English explanations**."<br>`[PAUSE 1s]` |
| **0:40**<br>`[SWITCH: prepared app tab → Dashboard]` | "The system connects a **FastAPI backend** for model inference with a **Vite React frontend**.<br><br>In **Simple Mode**, non-technical users receive clear, question-led decision framing."<br>`[PAUSE 1s]` |
| **0:55**<br>`[CLICK: Bitcoin (BTC) in left asset rail]`<br>`[SHOW: main asset page—signal, probabilities, chart, quote/model timestamps]` | "This is Bitcoin's main overview: the current signal, action probabilities, price chart, and separate quote and model timestamps. The probabilities describe the policy's choices, not chances of profit." |
| **1:15**<br>`[CLICK: Explore → beneath the chart]`<br>`[SHOW: Simple Mode Cards 1 and 2; wait for decision drivers]` | "Now I open Strategy Lab. **Card 1** shows the model's current target allocation; **Card 2** shows the main SHAP decision drivers and explains them in plain English." |
| **1:35**<br>`[CLICK: top suggested Why this signal? beneath the advisor conversation]`<br>`[WAIT: advisor reply appears]` | "In the Copilot pane, we request an explanation to inspect the model's reasoning in natural language, grounded by an action-consistency check."<br>`[PAUSE 1s]` |
| **1:55**<br>`[IN CARD 3: click Fold 1 → Fold 3 → Fold 2; leave Fold 2 selected]` | "**Card 3** displays historical market case studies across walk-forward folds—showing Fold 1 expansion, Fold 3 cash inaction, and Fold 2 downturn outperformance. Monte Carlo risk estimates are available on request."<br>`[PAUSE 1s]` |
| **2:15**<br>`[CLICK: Advanced (Quant) above the cards]` | "Switching to **Advanced Mode** reveals the full quantitative dashboard.<br><br>The policy network is trained using **Proximal Policy Optimization** with a custom 4-layer Transformer self-attention extractor processing an 8-step frame-stacked observation tensor." |
| **2:35**<br>`[CLICK: Backtest & Equity tab → Market Climate tab]` | "To reduce look-ahead bias in execution timing, actions at step T execute against **Open T plus one**, with 0.10% commission and 0.05% slippage per side—totaling **0.30% round-trip friction**.<br><br>Market climate is identified by an unsupervised **Gaussian Hidden Markov Model**, classifying trends, consolidation, and crashes."<br>`[PAUSE 1s]` |
| **3:00**<br>`[CLICK: SHAP Explainability tab; show feature chart]` | "For interpretability, **KernelExplainer calculates feature attributions**, while the explanation layer generates plain-English text with an action-consistency check.<br><br>In our evaluation, **synthetic checks covered all three actions**; the **192-state real-model scan covered LONG only**, with cash postures observed in separate evaluations."<br>`[PAUSE 1s]` |
| **3:20**<br>`[CLICK: Multi-Year Validation tab; show five-fold results]` | "Across **five walk-forward folds**, mean return was **11.11% versus 24.67% for Buy-and-Hold**. Fold 2 gained **44.51%** in a downturn, but the agent made no trades in Folds 3 and 4, missing bull-market gains.<br><br>Cross-market tests were positive on Ethereum and Dogecoin, but poor on equities and commodities."<br>`[PAUSE 1s]` |
| **3:50**<br>`[SWITCH: slides.html tab; press Right Arrow → Slide 3]` | "The latest full repository audit passed **240 automated tests**. An author-led heuristic review, assisted by AI, identified and resolved **eight interface defects**."<br>`[PAUSE 1s]` |
| **4:05**<br>`[STAY: Slide 3; point to Where it failed]` | "Diagnostic runs also showed cash inaction; its precise cause remains unresolved."<br>`[PAUSE 1s]` |
| **4:15**<br>`[STAY: Slide 3 for conclusion]` | "In conclusion, the project demonstrates an explainable reinforcement learning trading advisor and exposes the trade-offs of a risk-averse policy.<br><br>Thank you."<br>`[FADE OUT]` |

---

## 📋 Clean Teleprompter Text (Copy & Paste Ready)

```text
Hi, I'm Ryan. This is my final project: an explainable trading advisor built with reinforcement learning.

Overfitting and transaction costs are major challenges in algorithmic trading that often degrade live market performance. My objective was to train an agent's policy within a Markov Decision Process for downside capital preservation, showing users SHAP attributions and plain-English explanations.

The system connects a FastAPI backend for model inference with a Vite React frontend. In Simple Mode, non-technical users receive clear, question-led decision framing.

This is Bitcoin's main overview: the current signal, action probabilities, price chart, and separate quote and model timestamps. The probabilities describe the policy's choices, not chances of profit.

Now I open Strategy Lab. Card 1 shows the model's current target allocation; Card 2 shows the main SHAP decision drivers and explains them in plain English. In the Copilot pane, we request an explanation to inspect the model's reasoning in natural language, grounded by an action-consistency check.

Card 3 displays historical market case studies across walk-forward folds—showing Fold 1 expansion, Fold 3 cash inaction, and Fold 2 downturn outperformance. Monte Carlo risk estimates are available on request.

Switching to Advanced Mode reveals the full quantitative dashboard. The policy network is trained using Proximal Policy Optimization with a custom 4-layer Transformer self-attention extractor processing an 8-step frame-stacked observation tensor.

To reduce look-ahead bias in execution timing, actions at step T execute against Open T plus one, with 0.10% commission and 0.05% slippage per side—totaling 0.30% round-trip friction. Market climate is identified by an unsupervised Gaussian Hidden Markov Model, classifying trends, consolidation, and crashes.

For interpretability, KernelExplainer calculates feature attributions, while the explanation layer generates plain-English text with an action-consistency check. In our evaluation, synthetic checks covered all three actions; the 192-state real-model scan covered LONG only, with cash postures observed in separate evaluations.

Across five walk-forward folds, mean return was 11.11% versus 24.67% for Buy-and-Hold. Fold 2 gained 44.51% in a downturn, but the agent made no trades in Folds 3 and 4, missing bull-market gains. Cross-market tests were positive on Ethereum and Dogecoin, but poor on equities and commodities.

The latest full repository audit passed 240 automated tests. An author-led heuristic review, assisted by AI, identified and resolved eight interface defects.

Diagnostic runs also showed cash inaction; its precise cause remains unresolved.

In conclusion, the project demonstrates an explainable reinforcement learning trading advisor and exposes the trade-offs of a risk-averse policy.

Thank you.
```

---

## 🎙️ Voice & Pronunciation Guide

| Term | Phonetic Guide | Spoken Phrasing |
|---|---|---|
| **RL** | *R-L* | "Reinforcement Learning" |
| **MDP** | *M-D-P* | "Markov Decision Process" |
| **PPO** | *P-P-O* | "Proximal Policy Optimization" |
| **HMM** | *H-M-M* | "Gaussian Hidden Markov Model" |
| **SHAP** | *SHAP* (rhymes with map) | "SHAP feature attributions" |
| **DSR** | *D-S-R* | "Deflated Sharpe Ratio" |
| **0.30%** | *Zero-point-three-zero percent* | "Thirty basis points or 0.30% round-trip friction" |
| **Open T+1** | *Open T-plus-one* | "Open T plus one" |
| **11.11% vs. 24.67%** | *Eleven-point-one-one percent vs. twenty-four-point-six-seven percent* | "Eleven point one one percent versus twenty-four point six seven percent" |
