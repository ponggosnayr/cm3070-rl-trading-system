import os
import sys
import re
import time
import threading
import subprocess
import webbrowser
import psutil
import torch
import customtkinter as ctk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ModernRLOptimizerGUI(ctk.CTk):
    """
    Modern RL Policy Retraining & Optimization Workstation.
    Standalone developer GUI for model training, real-time telemetry, and rollout convergence analysis.
    """

    def __init__(self):
        super().__init__()

        self.title("Adaptive RL Policy Retraining & Optimization Studio")
        self.geometry("1380x920")
        self.minsize(1120, 780)
        self.configure(fg_color="#090D16")

        # Subprocess and Execution tracking
        self.process = None
        self.execution_thread = None
        self.log_line_count = 0
        self.auto_scroll = True

        # Dynamic Training Timer & Progress tracking
        self.train_start_time = None
        self.timer_running = False
        self.timer_after_id = None
        self.telemetry_after_id = None
        self.current_eta = None
        self.current_steps = 0
        self.target_steps = 500000
        self.current_pct = 0.0

        # Rollout tracking data structures
        self.rollout_data = {
            "steps": [],
            "returns": [],
            "win_rates": [],
            "val_returns": [],
        }
        self.current_rollout_record = {}

        # Build UI layout
        self._build_ui()
        self.detect_existing_model()
        self._start_telemetry_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_window_closing)

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ---------------- 1. TOP HEADER & TELEMETRY BAR ----------------
        self._build_header()

        # ---------------- 2. KPI STATUS CARDS ROW ----------------
        self._build_kpi_cards()

        # ---------------- 3. MAIN WORKSPACE CONTAINER ----------------
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 8))
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)  # Left controls
        main_container.grid_columnconfigure(1, weight=1)  # Right visualization

        # Build Left Control Center
        self._build_left_controls(main_container)

        # Build Right Visualization Center
        self._build_right_workspace(main_container)

        # ---------------- 4. SYSTEM STATUS FOOTER ----------------
        self._build_footer()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="#0B0F19", corner_radius=0, height=72)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_propagate(False)

        # Left Branding
        brand_frame = ctk.CTkFrame(header, fg_color="transparent")
        brand_frame.pack(side="left", padx=20, pady=10)

        title_box = ctk.CTkFrame(brand_frame, fg_color="transparent")
        title_box.pack(anchor="w")

        icon_badge = ctk.CTkLabel(
            title_box,
            text="⚡",
            font=ctk.CTkFont(size=20),
            text_color="#38BDF8",
        )
        icon_badge.pack(side="left", padx=(0, 8))

        title_lbl = ctk.CTkLabel(
            title_box,
            text="RL Policy Retraining Studio",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#F8FAFC",
        )
        title_lbl.pack(side="left")

        arch_tag = ctk.CTkLabel(
            title_box,
            text="PPO Temporal Transformer",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1E293B",
            text_color="#38BDF8",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        arch_tag.pack(side="left", padx=12)

        subtitle_lbl = ctk.CTkLabel(
            brand_frame,
            text="Autonomous Multi-Asset Retraining, Walk-Forward Validation & Telemetry Monitor",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#64748B",
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Right Telemetry Bar (CPU, RAM, GPU)
        telemetry_box = ctk.CTkFrame(header, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        telemetry_box.pack(side="right", padx=20, pady=12)

        # CPU Metric
        self.telem_cpu_lbl = ctk.CTkLabel(
            telemetry_box,
            text="CPU: ...%",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#94A3B8",
            padx=10,
        )
        self.telem_cpu_lbl.pack(side="left")

        # Divider
        ctk.CTkLabel(telemetry_box, text="|", text_color="#334155").pack(side="left")

        # RAM Metric
        self.telem_ram_lbl = ctk.CTkLabel(
            telemetry_box,
            text="RAM: ...",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#94A3B8",
            padx=10,
        )
        self.telem_ram_lbl.pack(side="left")

        # Divider
        ctk.CTkLabel(telemetry_box, text="|", text_color="#334155").pack(side="left")

        # GPU Metric
        self.telem_gpu_lbl = ctk.CTkLabel(
            telemetry_box,
            text="GPU: Detecting...",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#34D399",
            padx=10,
        )
        self.telem_gpu_lbl.pack(side="left")

    def _build_kpi_cards(self):
        cards_frame = ctk.CTkFrame(self, fg_color="transparent")
        cards_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(10, 10))
        cards_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Card 1: Checkpoint Status
        card1 = ctk.CTkFrame(cards_frame, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        card1.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=0)
        c1_t = ctk.CTkLabel(card1, text="ACTIVE MODEL CHECKPOINT", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B")
        c1_t.pack(anchor="w", padx=12, pady=(8, 2))
        self.stat_brain_val = ctk.CTkLabel(card1, text="Checking checkpoint...", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38BDF8")
        self.stat_brain_val.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 2: Active Target
        card2 = ctk.CTkFrame(cards_frame, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        card2.grid(row=0, column=1, sticky="ew", padx=3, pady=0)
        c2_t = ctk.CTkLabel(card2, text="TARGET MARKET & TIMEFRAME", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B")
        c2_t.pack(anchor="w", padx=12, pady=(8, 2))
        self.stat_target_val = ctk.CTkLabel(card2, text="BTCUSDT • 1H • 500k Steps", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC")
        self.stat_target_val.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 3: Engine State
        card3 = ctk.CTkFrame(cards_frame, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        card3.grid(row=0, column=2, sticky="ew", padx=3, pady=0)
        c3_t = ctk.CTkLabel(card3, text="TRAINING ENGINE STATUS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B")
        c3_t.pack(anchor="w", padx=12, pady=(8, 2))
        self.stat_engine_val = ctk.CTkLabel(card3, text="● IDLE • Ready to Train", font=ctk.CTkFont(size=13, weight="bold"), text_color="#10B981")
        self.stat_engine_val.pack(anchor="w", padx=12, pady=(0, 8))

        # Card 4: Elapsed & ETA
        card4 = ctk.CTkFrame(cards_frame, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        card4.grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=0)
        c4_t = ctk.CTkLabel(card4, text="TRAINING ELAPSED & ETA", font=ctk.CTkFont(size=10, weight="bold"), text_color="#64748B")
        c4_t.pack(anchor="w", padx=12, pady=(6, 1))

        c4_vals = ctk.CTkFrame(card4, fg_color="transparent")
        c4_vals.pack(fill="x", padx=12, pady=(0, 6))

        self.stat_timer_val = ctk.CTkLabel(c4_vals, text="00:00:00", font=ctk.CTkFont(size=13, weight="bold"), text_color="#38BDF8")
        self.stat_timer_val.pack(side="left")

        self.stat_eta_val = ctk.CTkLabel(c4_vals, text="ETA: --:--:--", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981")
        self.stat_eta_val.pack(side="right")

    def _build_left_controls(self, parent):
        left_col = ctk.CTkFrame(parent, width=440, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=0)
        left_col.grid_rowconfigure(0, weight=1)
        left_col.grid_columnconfigure(0, weight=1)

        # Tabbed Control Box
        self.config_tabs = ctk.CTkTabview(
            left_col,
            width=430,
            fg_color="#0B0F19",
            segmented_button_fg_color="#0F172A",
            segmented_button_selected_color="#2563EB",
            segmented_button_selected_hover_color="#1D4ED8",
            segmented_button_unselected_color="#1E293B",
            segmented_button_unselected_hover_color="#334155",
            corner_radius=10,
        )
        self.config_tabs.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        tab_data = self.config_tabs.add("Data & Model")
        tab_hyper = self.config_tabs.add("Hyperparameters")
        tab_risk = self.config_tabs.add("Risk & Friction")
        tab_presets = self.config_tabs.add("Presets")

        # Scrollable wrappers for tabs
        scroll_data = ctk.CTkScrollableFrame(tab_data, fg_color="transparent")
        scroll_data.pack(fill="both", expand=True, padx=2, pady=2)
        self._bind_mousewheel(scroll_data)

        scroll_hyper = ctk.CTkScrollableFrame(tab_hyper, fg_color="transparent")
        scroll_hyper.pack(fill="both", expand=True, padx=2, pady=2)
        self._bind_mousewheel(scroll_hyper)

        scroll_risk = ctk.CTkScrollableFrame(tab_risk, fg_color="transparent")
        scroll_risk.pack(fill="both", expand=True, padx=2, pady=2)
        self._bind_mousewheel(scroll_risk)

        scroll_presets = ctk.CTkScrollableFrame(tab_presets, fg_color="transparent")
        scroll_presets.pack(fill="both", expand=True, padx=2, pady=2)
        self._bind_mousewheel(scroll_presets)

        # ================= TAB 1: DATA & MODEL =================
        self._add_section_header(scroll_data, "Market Dataset Selection")
        self._add_field_label(scroll_data, "Target Asset", "Select asset dataset for policy optimization")
        self.asset_option = ctk.CTkOptionMenu(
            scroll_data,
            values=[
                "BTC (Bitcoin USDT)",
                "ETH (Ethereum USDT)",
                "DOGE (Dogecoin USDT)",
                "SPY (S&P 500 ETF)",
                "QQQ (Nasdaq 100 ETF)",
                "GLD (SPDR Gold Shares)",
                "WTI (Crude Oil Future)",
            ],
            command=self._update_stats_summary,
            fg_color="#1E293B",
            button_color="#334155",
            dropdown_fg_color="#0F172A",
        )
        self.asset_option.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_data, "Timeframe Resolution", "Candle frequency for environment rollouts")
        self.interval_option = ctk.CTkOptionMenu(
            scroll_data,
            values=["1H (Hourly Candles)", "Daily (1D Candles)"],
            command=self._update_stats_summary,
            fg_color="#1E293B",
            button_color="#334155",
            dropdown_fg_color="#0F172A",
        )
        self.interval_option.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_data, "Training Timesteps", "Total environment steps for PPO optimization")
        self.steps_entry = ctk.CTkEntry(scroll_data, placeholder_text="500000", fg_color="#1E293B", border_color="#334155")
        self.steps_entry.insert(0, "500000")
        self.steps_entry.pack(fill="x", padx=12, pady=(0, 4))
        self.steps_entry.bind("<KeyRelease>", self._update_stats_summary)

        # Quick Step Steppers
        stepper_box = ctk.CTkFrame(scroll_data, fg_color="transparent")
        stepper_box.pack(fill="x", padx=12, pady=(0, 10))
        stepper_box.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ctk.CTkButton(stepper_box, text="50k", height=24, fg_color="#1E293B", hover_color="#334155", command=lambda: self._set_steps(50000)).grid(row=0, column=0, padx=2)
        ctk.CTkButton(stepper_box, text="100k", height=24, fg_color="#1E293B", hover_color="#334155", command=lambda: self._set_steps(100000)).grid(row=0, column=1, padx=2)
        ctk.CTkButton(stepper_box, text="250k", height=24, fg_color="#1E293B", hover_color="#334155", command=lambda: self._set_steps(250000)).grid(row=0, column=2, padx=2)
        ctk.CTkButton(stepper_box, text="500k", height=24, fg_color="#1E293B", hover_color="#334155", command=lambda: self._set_steps(500000)).grid(row=0, column=3, padx=2)

        self._add_section_header(scroll_data, "Neural Architecture & Device")
        self._add_field_label(scroll_data, "Transformer Architecture", "Deep (256-dim, ~6.5M params) or Tiny (64-dim, ~410k params)")
        self.extractor_option = ctk.CTkOptionMenu(
            scroll_data,
            values=["deep", "tiny"],
            command=self._on_extractor_change,
            fg_color="#1E293B",
            button_color="#334155",
            dropdown_fg_color="#0F172A",
        )
        self.extractor_option.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_data, "Compute Accelerator", "Hardware device for neural network forward/backward updates")
        cuda_avail = torch.cuda.is_available()
        default_dev = "cuda" if cuda_avail else "cpu"
        self.device_option = ctk.CTkOptionMenu(
            scroll_data,
            values=["cuda", "cpu"] if cuda_avail else ["cpu"],
            fg_color="#1E293B",
            button_color="#334155",
            dropdown_fg_color="#0F172A",
        )
        self.device_option.set(default_dev)
        self.device_option.pack(fill="x", padx=12, pady=(0, 10))

        # ================= TAB 2: HYPERPARAMETERS =================
        self._add_section_header(scroll_hyper, "PPO Optimization Parameters")
        self._add_field_label(scroll_hyper, "Learning Rate", "Adam optimizer learning rate (base: 0.0001)")
        self.lr_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.0001", fg_color="#1E293B", border_color="#334155")
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_hyper, "Discount Factor (Gamma)", "Reward temporal discount factor (default: 0.97)")
        self.gamma_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.97", fg_color="#1E293B", border_color="#334155")
        self.gamma_entry.insert(0, "0.97")
        self.gamma_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_hyper, "GAE Lambda", "Generalized Advantage Estimation parameter (default: 0.92)")
        self.gae_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.92", fg_color="#1E293B", border_color="#334155")
        self.gae_entry.insert(0, "0.92")
        self.gae_entry.pack(fill="x", padx=12, pady=(0, 6))

        self.override_entropy_var = ctk.StringVar(value="off")
        self.entropy_switch = ctk.CTkSwitch(
            scroll_hyper,
            text="Override Exploration Entropy",
            variable=self.override_entropy_var,
            onvalue="on",
            offvalue="off",
            command=self._toggle_entropy_entry,
            progress_color="#2563EB",
        )
        self.entropy_switch.pack(anchor="w", padx=12, pady=(8, 4))

        self.entropy_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.02", fg_color="#0F172A", border_color="#1E293B")
        self.entropy_entry.insert(0, "0.02")
        self.entropy_entry.configure(state="disabled", text_color="#64748B")
        self.entropy_entry.pack(fill="x", padx=12, pady=(0, 8))

        self._add_section_header(scroll_hyper, "Reward Shaping & Regularization")
        self._add_field_label(scroll_hyper, "Inactivity Penalty", "Opportunity cost for idling flat in cash (e.g. 0.0005)")
        self.inactivity_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.0005", fg_color="#1E293B", border_color="#334155")
        self.inactivity_entry.insert(0, "0.0005")
        self.inactivity_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_hyper, "Drawdown Penalty Coef", "Continuous penalty for holding unrealized drawdown")
        self.drawdown_coef_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.1", fg_color="#1E293B", border_color="#334155")
        self.drawdown_coef_entry.insert(0, "0.1")
        self.drawdown_coef_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_hyper, "Volatility Scaling", "Dampening coefficient during high-volatility regimes")
        self.vol_scaling_entry = ctk.CTkEntry(scroll_hyper, placeholder_text="0.5", fg_color="#1E293B", border_color="#334155")
        self.vol_scaling_entry.insert(0, "0.5")
        self.vol_scaling_entry.pack(fill="x", padx=12, pady=(0, 10))

        # ================= TAB 3: RISK & FRICTION =================
        self._add_section_header(scroll_risk, "Trading Execution Friction")
        self._add_field_label(scroll_risk, "Commission Rate", "Transaction fee ratio (e.g. 0.001 = 0.10%)")
        self.comm_entry = ctk.CTkEntry(scroll_risk, placeholder_text="0.001", fg_color="#1E293B", border_color="#334155")
        self.comm_entry.insert(0, "0.001")
        self.comm_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_risk, "Slippage Rate", "Execution slippage ratio (e.g. 0.0005 = 0.05%)")
        self.slip_entry = ctk.CTkEntry(scroll_risk, placeholder_text="0.0005", fg_color="#1E293B", border_color="#334155")
        self.slip_entry.insert(0, "0.0005")
        self.slip_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_risk, "Position Cooldown", "Minimum steps to wait between trade reallocations")
        self.cooldown_entry = ctk.CTkEntry(scroll_risk, placeholder_text="12", fg_color="#1E293B", border_color="#334155")
        self.cooldown_entry.insert(0, "12")
        self.cooldown_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_section_header(scroll_risk, "Risk Constraints & Guardrails")
        self._add_field_label(scroll_risk, "Max Drawdown Cap", "Episode termination stop-loss threshold (e.g. 0.15 = 15%)")
        self.max_dd_entry = ctk.CTkEntry(scroll_risk, placeholder_text="0.15", fg_color="#1E293B", border_color="#334155")
        self.max_dd_entry.insert(0, "0.15")
        self.max_dd_entry.pack(fill="x", padx=12, pady=(0, 6))

        self._add_field_label(scroll_risk, "Max Trade Duration", "Maximum holding steps before forced position close (0 = inf)")
        self.max_dur_entry = ctk.CTkEntry(scroll_risk, placeholder_text="48", fg_color="#1E293B", border_color="#334155")
        self.max_dur_entry.insert(0, "48")
        self.max_dur_entry.pack(fill="x", padx=12, pady=(0, 8))

        self.reset_var = ctk.StringVar(value="off")
        self.reset_checkbox = ctk.CTkCheckBox(
            scroll_risk,
            text="Reset Model Weights (Cold Start from Scratch)",
            variable=self.reset_var,
            onvalue="on",
            offvalue="off",
            hover_color="#DC2626",
            border_color="#EF4444",
            text_color="#F8FAFC",
        )
        self.reset_checkbox.pack(anchor="w", padx=12, pady=(10, 10))

        # ================= TAB 4: PRESETS =================
        self._add_section_header(scroll_presets, "Curated Execution Presets")
        self._build_preset_card(
            scroll_presets,
            title="🚀 Grader Fast Prototype (CPU-Optimized)",
            desc="64k steps • Tiny Extractor (64d) • CPU device • 2-3 minute fast verification run for examiners.",
            action=self._apply_preset_grader,
            color="#2563EB",
        )
        self._build_preset_card(
            scroll_presets,
            title="⚡ High-Performance GPU Retrain",
            desc="250k steps • Deep Transformer (256d) • CUDA • Tuned dynamic entropy decay for rapid convergence.",
            action=self._apply_preset_gpu,
            color="#10B981",
        )
        self._build_preset_card(
            scroll_presets,
            title="🏆 Production Alpha Retrain (Full 500k)",
            desc="500k steps • Deep Transformer • Rigorous walk-forward validation and active checkpoint preservation.",
            action=self._apply_preset_production,
            color="#8B5CF6",
        )
        self._build_preset_card(
            scroll_presets,
            title="🛡️ Conservative Capital Preserver",
            desc="Higher drawdown penalty • Volatility scaling 0.75 • Tight stop loss (10% max DD cap).",
            action=self._apply_preset_conservative,
            color="#F59E0B",
        )

        # ================= ACTIONS PANEL =================
        actions_panel = ctk.CTkFrame(left_col, fg_color="#0B0F19", corner_radius=10, border_width=1, border_color="#1E293B")
        actions_panel.grid(row=1, column=0, sticky="ew", pady=(0, 0))

        # Progress bar
        self.training_progress_bar = ctk.CTkProgressBar(
            actions_panel,
            fg_color="#1E293B",
            progress_color="#2563EB",
            height=8,
            corner_radius=4,
        )
        self.training_progress_bar.set(0.0)
        self.training_progress_bar.pack(fill="x", padx=12, pady=(10, 8))

        # Primary Launch Button
        self.start_btn = ctk.CTkButton(
            actions_panel,
            text="🚀 LAUNCH TRAINING PIPELINE",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            height=40,
            command=self.start_training,
        )
        self.start_btn.pack(fill="x", padx=12, pady=(0, 6))

        # Secondary Button Row
        btn_row = ctk.CTkFrame(actions_panel, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))
        btn_row.grid_columnconfigure((0, 1, 2), weight=1)

        self.stop_btn = ctk.CTkButton(
            btn_row,
            text="⏹ Interrupt",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            height=32,
            state="disabled",
            command=self.stop_training,
        )
        self.stop_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.tb_btn = ctk.CTkButton(
            btn_row,
            text="📊 TensorBoard",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            height=32,
            command=self.open_tensorboard,
        )
        self.tb_btn.grid(row=0, column=1, sticky="ew", padx=3)

        self.folder_btn = ctk.CTkButton(
            btn_row,
            text="📁 Models",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            height=32,
            command=self.open_models_dir,
        )
        self.folder_btn.grid(row=0, column=2, sticky="ew", padx=(3, 0))

    def _build_right_workspace(self, parent):
        right_panel = ctk.CTkFrame(parent, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(
            right_panel,
            fg_color="#0B0F19",
            segmented_button_fg_color="#0F172A",
            segmented_button_selected_color="#2563EB",
            segmented_button_selected_hover_color="#1D4ED8",
            segmented_button_unselected_color="#1E293B",
            segmented_button_unselected_hover_color="#334155",
            corner_radius=10,
        )
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Tab 1: Live Performance Curves (Embedded Matplotlib)
        self.tab_charts = self.tabview.add("📈 Live Return & Convergence Curves")
        self._build_charts_tab(self.tab_charts)

        # Tab 2: Rollout Evaluation Table
        self.tab_table = self.tabview.add("📊 Rollout Evaluation Table")
        self._build_table_tab(self.tab_table)

        # Tab 3: Execution Trade Visualizer
        self.tab_trades = self.tabview.add("🎯 Execution Trade Visualizer")
        self._build_trades_tab(self.tab_trades)

        # Tab 4: Terminal Console
        self.tab_console = self.tabview.add("💻 Live Terminal Console")
        self._build_console_tab(self.tab_console)

    def _build_charts_tab(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        plt.style.use("dark_background")
        self.fig, (self.ax_return, self.ax_winrate) = plt.subplots(
            2, 1, figsize=(6.5, 6.0), facecolor="#090D16", dpi=100
        )
        self.fig.tight_layout(pad=3.2)

        self._render_empty_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _render_empty_plots(self):
        self.ax_return.clear()
        self.ax_winrate.clear()

        self.ax_return.set_facecolor("#0F172A")
        self.ax_winrate.set_facecolor("#0F172A")

        self.ax_return.set_title("Training Return (%) & Out-of-Sample Validation Return", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_return.set_ylabel("Return %", fontsize=10, color="#94A3B8")
        self.ax_return.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_return.axhline(0, color="#64748B", linewidth=1.2, linestyle=":")

        self.ax_winrate.set_title("Rollout Win Rate (%) Trajectory Across Optimization", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_winrate.set_xlabel("Environment Timesteps", fontsize=10, color="#94A3B8")
        self.ax_winrate.set_ylabel("Win Rate %", fontsize=10, color="#94A3B8")
        self.ax_winrate.set_ylim(-5, 105)
        self.ax_winrate.grid(True, linestyle="--", alpha=0.25, color="#334155")

    def update_live_charts(self):
        if not self.rollout_data["steps"]:
            return

        self.ax_return.clear()
        self.ax_winrate.clear()

        self.ax_return.set_facecolor("#0F172A")
        self.ax_winrate.set_facecolor("#0F172A")

        steps = self.rollout_data["steps"]

        # Subplot 1: Returns
        if self.rollout_data["returns"]:
            r_len = len(self.rollout_data["returns"])
            self.ax_return.plot(
                steps[:r_len],
                self.rollout_data["returns"],
                marker="o",
                markersize=4,
                color="#10B981",
                linewidth=2.0,
                label="Avg Training ROI %",
            )

        if self.rollout_data["val_returns"]:
            v_len = len(self.rollout_data["val_returns"])
            self.ax_return.plot(
                steps[:v_len],
                self.rollout_data["val_returns"],
                marker="s",
                markersize=4,
                color="#38BDF8",
                linewidth=2.0,
                linestyle="--",
                label="Validation ROI %",
            )

        self.ax_return.axhline(0, color="#64748B", linewidth=1.0, linestyle=":")
        self.ax_return.set_title("Training Return (%) & Out-of-Sample Validation Return", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_return.set_ylabel("Return %", fontsize=10, color="#94A3B8")
        self.ax_return.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_return.legend(loc="upper left", fontsize=9, facecolor="#0B0F19", edgecolor="#334155")

        # Subplot 2: Win Rate
        if self.rollout_data["win_rates"]:
            w_len = len(self.rollout_data["win_rates"])
            self.ax_winrate.plot(
                steps[:w_len],
                self.rollout_data["win_rates"],
                marker="^",
                markersize=4,
                color="#A855F7",
                linewidth=2.0,
                label="Win Rate (ROI > 0) %",
            )

        self.ax_winrate.set_title("Rollout Win Rate (%) Trajectory Across Optimization", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_winrate.set_xlabel("Environment Timesteps", fontsize=10, color="#94A3B8")
        self.ax_winrate.set_ylabel("Win Rate %", fontsize=10, color="#94A3B8")
        self.ax_winrate.set_ylim(-5, 105)
        self.ax_winrate.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_winrate.legend(loc="upper left", fontsize=9, facecolor="#0B0F19", edgecolor="#334155")

        self.fig.tight_layout(pad=2.8)
        self.canvas.draw_idle()

    def _build_table_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        t_header = ctk.CTkFrame(parent, fg_color="#0F172A", corner_radius=6, height=36)
        t_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        headers = ["Rollout #", "Step Count", "Avg Train ROI", "Win Rate %", "Best Capital", "Val Return %", "Checkpoint Status"]
        for h in headers:
            lbl = ctk.CTkLabel(t_header, text=h, font=ctk.CTkFont(size=11, weight="bold"), text_color="#38BDF8")
            lbl.pack(side="left", expand=True, fill="x", padx=4)

        self.table_scroll = ctk.CTkScrollableFrame(parent, fg_color="#090D16", corner_radius=6)
        self.table_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def add_table_row(self, rollout_num, step, train_ret, win_rate, best_cap, val_ret, saved):
        row_frame = ctk.CTkFrame(self.table_scroll, fg_color="#0F172A", corner_radius=4, height=32)
        row_frame.pack(fill="x", padx=4, pady=3)

        ret_color = "#10B981" if train_ret >= 0 else "#EF4444"
        val_color = "#10B981" if val_ret >= 0 else "#EF4444"
        saved_str = "✅ persistent_brain.pth" if saved else "—"

        vals = [
            f"#{rollout_num}",
            f"{step:,}",
            f"{train_ret:+.2f}%",
            f"{win_rate:.1f}%",
            f"${best_cap:,.2f}",
            f"{val_ret:+.2f}%",
            saved_str,
        ]
        colors = ["#F8FAFC", "#94A3B8", ret_color, "#A855F7", "#F8FAFC", val_color, "#38BDF8" if saved else "#64748B"]

        for idx, (v, col) in enumerate(zip(vals, colors)):
            lbl = ctk.CTkLabel(
                row_frame,
                text=v,
                font=ctk.CTkFont(size=11, weight="bold" if idx in [2, 3, 5] else "normal"),
                text_color=col,
            )
            lbl.pack(side="left", expand=True, fill="x", padx=4)

    def _build_trades_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        t_ctrl = ctk.CTkFrame(parent, fg_color="#0F172A", corner_radius=6, height=38)
        t_ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        c_title = ctk.CTkLabel(t_ctrl, text="Backtest Trade Execution Overlay", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC")
        c_title.pack(side="left", padx=12)

        self.trade_metric_lbl = ctk.CTkLabel(t_ctrl, text="Ready • Click 'Plot Trades' to test active brain", font=ctk.CTkFont(size=11), text_color="#94A3B8")
        self.trade_metric_lbl.pack(side="left", padx=10)

        self.trade_plot_btn = ctk.CTkButton(
            t_ctrl,
            text="⚡ Plot Trades",
            width=110,
            height=24,
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.run_and_plot_trades,
        )
        self.trade_plot_btn.pack(side="right", padx=10)

        plt.style.use("dark_background")
        self.fig_trades, (self.ax_trade_price, self.ax_trade_equity) = plt.subplots(
            2, 1, figsize=(6.5, 6.0), facecolor="#090D16", dpi=100, gridspec_kw={"height_ratios": [2.5, 1]}
        )
        self.fig_trades.tight_layout(pad=2.8)

        self._render_empty_trades_plot()

        self.canvas_trades = FigureCanvasTkAgg(self.fig_trades, master=parent)
        self.canvas_trades.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _render_empty_trades_plot(self):
        self.ax_trade_price.clear()
        self.ax_trade_equity.clear()

        self.ax_trade_price.set_facecolor("#0F172A")
        self.ax_trade_equity.set_facecolor("#0F172A")

        self.ax_trade_price.set_title("Out-of-Sample Price Action with Buy/Sell Executions", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_trade_price.set_ylabel("Price ($)", fontsize=10, color="#94A3B8")
        self.ax_trade_price.grid(True, linestyle="--", alpha=0.25, color="#334155")

        self.ax_trade_equity.set_title("Portfolio Equity Growth ($10,000 Starting Capital)", fontsize=10, color="#38BDF8", pad=6, weight="bold")
        self.ax_trade_equity.set_xlabel("Evaluation Timesteps", fontsize=10, color="#94A3B8")
        self.ax_trade_equity.set_ylabel("Capital ($)", fontsize=10, color="#94A3B8")
        self.ax_trade_equity.grid(True, linestyle="--", alpha=0.25, color="#334155")

    def run_and_plot_trades(self):
        self.trade_plot_btn.configure(state="disabled")
        self.trade_metric_lbl.configure(text="Running backtest on active checkpoint...", text_color="#38BDF8")
        threading.Thread(target=self._worker_backtest_trades, daemon=True).start()

    def _worker_backtest_trades(self):
        try:
            import pandas as pd
            from api import get_asset_csv_path, compute_indicators, ActiveCryptoEnv, get_dynamic_policy_kwargs, MODEL_PATH
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
            from trading_utils import run_single_fold_backtest

            symbol_raw = self.asset_option.get()
            asset_label = symbol_raw.split("(")[0].strip()
            timeframe = "1H" if "1H" in self.interval_option.get() else "Daily"

            csv_path = get_asset_csv_path(asset_label, timeframe)
            if not csv_path or not os.path.exists(csv_path):
                self.after(0, lambda: self.trade_metric_lbl.configure(text=f"Error: Dataset {csv_path} not found", text_color="#EF4444"))
                self.after(0, lambda: self.trade_plot_btn.configure(state="normal"))
                return

            if not os.path.exists(MODEL_PATH):
                self.after(0, lambda: self.trade_metric_lbl.configure(text="No model at models/persistent_brain.pth", text_color="#EF4444"))
                self.after(0, lambda: self.trade_plot_btn.configure(state="normal"))
                return

            df = pd.read_csv(csv_path)
            sub_df = df.iloc[-1500:].reset_index(drop=True)
            df_indicators = compute_indicators(sub_df, fit_hmm=False).ffill().bfill()
            test_df = df_indicators.iloc[-1000:].reset_index(drop=True)

            dynamic_kwargs, state_dict, inc_reg, inc_ctx, in_f = get_dynamic_policy_kwargs(MODEL_PATH)
            
            dummy_env = DummyVecEnv([lambda: ActiveCryptoEnv(
                test_df.iloc[:50], render_mode=None, include_regime=inc_reg, include_context=inc_ctx, random_start=False
            )])
            dummy_env = VecFrameStack(dummy_env, n_stack=8)

            model = PPO("MlpPolicy", dummy_env, policy_kwargs=dynamic_kwargs, device="cpu")
            model.policy.load_state_dict(state_dict)

            metrics = run_single_fold_backtest(
                model,
                test_df,
                initial_balance=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0005,
                include_regime=inc_reg,
                include_context=inc_ctx,
            )

            self.after(0, self._render_trade_markers, test_df, metrics, asset_label)
        except Exception as e:
            self.after(0, lambda err=str(e): self.trade_metric_lbl.configure(text=f"Backtest error: {err[:50]}", text_color="#EF4444"))
            self.after(0, lambda: self.trade_plot_btn.configure(state="normal"))

    def _render_trade_markers(self, test_df, metrics, asset_label):
        self.trade_plot_btn.configure(state="normal")
        self.ax_trade_price.clear()
        self.ax_trade_equity.clear()

        self.ax_trade_price.set_facecolor("#0F172A")
        self.ax_trade_equity.set_facecolor("#0F172A")

        prices = test_df["close"].to_numpy()
        steps_x = list(range(len(prices)))
        self.ax_trade_price.plot(steps_x, prices, color="#38BDF8", linewidth=1.4, label=f"{asset_label} Price")

        trade_list = metrics.get("trade_list", [])
        buy_x, buy_p = [], []
        sell_x, sell_p = [], []
        exit_x, exit_p = [], []

        for tr in trade_list:
            step_idx = tr.get("Step", 0)
            act = tr.get("Action", "")
            pr = tr.get("Price", 0)
            if 0 <= step_idx < len(prices):
                if "BUY" in act or ("LONG" in act and "EXIT" not in act):
                    buy_x.append(step_idx)
                    buy_p.append(pr)
                elif "SELL" in act or ("SHORT" in act and "EXIT" not in act):
                    sell_x.append(step_idx)
                    sell_p.append(pr)
                elif "EXIT" in act:
                    exit_x.append(step_idx)
                    exit_p.append(pr)

        if buy_x:
            self.ax_trade_price.scatter(buy_x, buy_p, marker="^", color="#10B981", s=85, label="Buy (Long)", zorder=6)
        if sell_x:
            self.ax_trade_price.scatter(sell_x, sell_p, marker="v", color="#EF4444", s=85, label="Sell (Short)", zorder=6)
        if exit_x:
            self.ax_trade_price.scatter(exit_x, exit_p, marker="x", color="#F59E0B", s=65, label="Exit Position", zorder=6)

        self.ax_trade_price.set_title(f"Trade Execution Overlay: {asset_label} (Out-of-Sample Test Fold)", fontsize=11, color="#38BDF8", pad=8, weight="bold")
        self.ax_trade_price.set_ylabel("Price ($)", fontsize=10, color="#94A3B8")
        self.ax_trade_price.grid(True, linestyle="--", alpha=0.25, color="#334155")
        self.ax_trade_price.legend(loc="upper left", fontsize=9, facecolor="#0B0F19", edgecolor="#334155")

        # Subplot 2: Portfolio Equity Curve
        net_worth = metrics.get("net_worth_history", [])
        if net_worth:
            nw_x = list(range(len(net_worth)))
            final_color = "#10B981" if metrics.get("roi", 0) >= 0 else "#EF4444"
            self.ax_trade_equity.plot(nw_x, net_worth, color=final_color, linewidth=1.8, label="Net Worth ($)")
            self.ax_trade_equity.axhline(10000.0, color="#64748B", linestyle=":", linewidth=1.0)
            self.ax_trade_equity.fill_between(nw_x, 10000.0, net_worth, where=[w >= 10000.0 for w in net_worth], color="#10B981", alpha=0.15)
            self.ax_trade_equity.fill_between(nw_x, 10000.0, net_worth, where=[w < 10000.0 for w in net_worth], color="#EF4444", alpha=0.15)

        self.ax_trade_equity.set_title("Portfolio Equity Growth ($10,000 Starting Capital)", fontsize=10, color="#38BDF8", pad=6, weight="bold")
        self.ax_trade_equity.set_xlabel("Test Timesteps (Hours)", fontsize=10, color="#94A3B8")
        self.ax_trade_equity.set_ylabel("Equity ($)", fontsize=10, color="#94A3B8")
        self.ax_trade_equity.grid(True, linestyle="--", alpha=0.25, color="#334155")

        roi_str = f"{metrics.get('roi', 0.0):+.2f}%"
        win_str = f"{metrics.get('win_rate', 0.0):.1f}%"
        sharpe_str = f"{metrics.get('sharpe', 0.0):.2f}"
        n_trades = len(trade_list)
        stat_summary = f"ROI: {roi_str} | Trades: {n_trades} | Win Rate: {win_str} | Sharpe: {sharpe_str}"
        self.trade_metric_lbl.configure(text=stat_summary, text_color="#10B981" if metrics.get('roi', 0) >= 0 else "#EF4444")

        self.fig_trades.tight_layout(pad=2.6)
        self.canvas_trades.draw_idle()

    def _build_console_tab(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        c_header = ctk.CTkFrame(parent, fg_color="#0F172A", corner_radius=6, height=36)
        c_header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

        c_title = ctk.CTkLabel(c_header, text="Console Output Stream", font=ctk.CTkFont(size=12, weight="bold"), text_color="#F8FAFC")
        c_title.pack(side="left", padx=12)

        self.line_count_lbl = ctk.CTkLabel(c_header, text="0 lines", font=ctk.CTkFont(size=11), text_color="#64748B")
        self.line_count_lbl.pack(side="left", padx=6)

        c_clear_btn = ctk.CTkButton(
            c_header,
            text="Clear",
            width=60,
            height=24,
            fg_color="#1E293B",
            hover_color="#334155",
            font=ctk.CTkFont(size=11),
            command=self.clear_console,
        )
        c_clear_btn.pack(side="right", padx=10)

        self.auto_scroll_var = ctk.StringVar(value="on")
        self.auto_scroll_check = ctk.CTkCheckBox(
            c_header,
            text="Auto-scroll",
            variable=self.auto_scroll_var,
            onvalue="on",
            offvalue="off",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
            checkbox_height=18,
            checkbox_width=18,
        )
        self.auto_scroll_check.pack(side="right", padx=10)

        self.console_text = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="#090D16",
            text_color="#E2E8F0",
            corner_radius=6,
            wrap="word",
        )
        self.console_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.console_text.configure(state="disabled")

        self.log_to_console("===============================================================================\n")
        self.log_to_console("  ⚡ ADAPTIVE RL TRADING BOT — MODEL RETRAINING & OPTIMIZATION STUDIO READY\n")
        self.log_to_console("===============================================================================\n")
        self.log_to_console("• Select an asset pair, timeframe, and target timesteps on the left control pane.\n")
        self.log_to_console("• Choose between Deep Transformer (256d) and Tiny Transformer (64d, CPU-optimized).\n")
        self.log_to_console("• Click '🚀 LAUNCH TRAINING PIPELINE' to start real-time optimization.\n")
        self.log_to_console("• Switch to '📈 Live Return & Convergence Curves' tab for real-time graphs.\n\n")

    def _build_footer(self):
        status_frame = ctk.CTkFrame(self, fg_color="#0B0F19", corner_radius=0, height=28)
        status_frame.grid(row=3, column=0, sticky="ew", padx=0, pady=0)

        self.status_lbl = ctk.CTkLabel(
            status_frame,
            text="System Status: Ready • Standby",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#94A3B8",
        )
        self.status_lbl.pack(side="left", padx=15, pady=4)

        self.status_time_lbl = ctk.CTkLabel(
            status_frame,
            text="Duration: 00:00:00",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color="#64748B",
        )
        self.status_time_lbl.pack(side="left", padx=10, pady=4)

        python_exe = self.get_python_executable()
        self.venv_lbl = ctk.CTkLabel(
            status_frame,
            text=f"Active Environment: {python_exe}",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color="#475569",
        )
        self.venv_lbl.pack(side="right", padx=15, pady=4)

    # ---------------- TELEMETRY & HARDWARE MONITORING ----------------
    def _start_telemetry_loop(self):
        self._update_telemetry()

    def _update_telemetry(self):
        try:
            # CPU non-blocking
            cpu_pct = psutil.cpu_percent(interval=None)
            self.telem_cpu_lbl.configure(text=f"CPU: {cpu_pct:.1f}%")

            # RAM usage
            ram = psutil.virtual_memory()
            used_gb = ram.used / (1024**3)
            total_gb = ram.total / (1024**3)
            self.telem_ram_lbl.configure(text=f"RAM: {used_gb:.1f}/{total_gb:.1f} GB ({ram.percent}%)")

            # GPU / VRAM
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_gpu_gb = (total_bytes - free_bytes) / (1024**3)
                total_gpu_gb = total_bytes / (1024**3)
                raw_gpu = torch.cuda.get_device_name(0)
                clean_gpu = raw_gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()
                self.telem_gpu_lbl.configure(
                    text=f"GPU: {clean_gpu} • {used_gpu_gb:.1f}/{total_gpu_gb:.1f} GB",
                    text_color="#34D399",
                )
            else:
                self.telem_gpu_lbl.configure(text="GPU: CPU Execution Mode", text_color="#FDBA74")
        except Exception:
            pass

        self.telemetry_after_id = self.after(1500, self._update_telemetry)

    # ---------------- HELPER METHODS & WIDGET BUILDERS ----------------
    def _add_section_header(self, parent, text):
        hdr = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#38BDF8",
        )
        hdr.pack(anchor="w", padx=12, pady=(12, 4))

    def _add_field_label(self, parent, title, desc):
        l_frame = ctk.CTkFrame(parent, fg_color="transparent")
        l_frame.pack(fill="x", padx=12, pady=(4, 2))

        t = ctk.CTkLabel(l_frame, text=title, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color="#E2E8F0")
        t.pack(anchor="w")

        d = ctk.CTkLabel(l_frame, text=desc, font=ctk.CTkFont(family="Segoe UI", size=9), text_color="#64748B")
        d.pack(anchor="w")

    def _build_preset_card(self, parent, title, desc, action, color):
        card = ctk.CTkFrame(parent, fg_color="#0F172A", corner_radius=8, border_width=1, border_color="#1E293B")
        card.pack(fill="x", padx=12, pady=5)

        t_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"), text_color="#F8FAFC")
        t_lbl.pack(anchor="w", padx=12, pady=(8, 2))

        d_lbl = ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(family="Segoe UI", size=10), text_color="#94A3B8", justify="left")
        d_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        btn = ctk.CTkButton(
            card,
            text="Load Preset",
            height=26,
            fg_color=color,
            hover_color="#1E293B",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=action,
        )
        btn.pack(anchor="e", padx=12, pady=(0, 8))

    def _set_steps(self, steps_val):
        self.steps_entry.delete(0, "end")
        self.steps_entry.insert(0, str(steps_val))
        self._update_stats_summary()

    def _update_stats_summary(self, *args):
        asset = self.asset_option.get().split("(")[0].strip()
        interval = "1H" if "1H" in self.interval_option.get() else "Daily"
        steps = self.steps_entry.get().strip() or "500000"
        self.stat_target_val.configure(text=f"{asset} • {interval} • {steps} Steps")

    def _on_extractor_change(self, selection):
        if selection == "deep":
            if torch.cuda.is_available():
                self.device_option.set("cuda")
        else:
            self.device_option.set("cpu")

    def _toggle_entropy_entry(self):
        if self.override_entropy_var.get() == "on":
            self.entropy_entry.configure(state="normal", fg_color="#1E293B", text_color="#E2E8F0")
        else:
            self.entropy_entry.configure(state="disabled", fg_color="#0F172A", text_color="#64748B")

    # ---------------- PRESET CONFIGURATIONS ----------------
    def _apply_preset_grader(self):
        self.steps_entry.delete(0, "end")
        self.steps_entry.insert(0, "64000")
        self.extractor_option.set("tiny")
        self.device_option.set("cpu")
        self.lr_entry.delete(0, "end")
        self.lr_entry.insert(0, "0.0003")
        self.inactivity_entry.delete(0, "end")
        self.inactivity_entry.insert(0, "0.0005")
        self._update_stats_summary()
        self.log_to_console("🚀 Applied Preset: Grader Fast Prototype (64k Steps, Tiny Extractor, CPU, LR 3e-4)\n")

    def _apply_preset_gpu(self):
        self.steps_entry.delete(0, "end")
        self.steps_entry.insert(0, "250000")
        self.extractor_option.set("deep")
        if torch.cuda.is_available():
            self.device_option.set("cuda")
        self.lr_entry.delete(0, "end")
        self.lr_entry.insert(0, "0.00008")
        self.override_entropy_var.set("on")
        self._toggle_entropy_entry()
        self.entropy_entry.delete(0, "end")
        self.entropy_entry.insert(0, "0.02")
        self._update_stats_summary()
        self.log_to_console("⚡ Applied Preset: High-Performance GPU Retrain (250k Steps, Deep Transformer, CUDA)\n")

    def _apply_preset_production(self):
        self.steps_entry.delete(0, "end")
        self.steps_entry.insert(0, "500000")
        self.extractor_option.set("deep")
        if torch.cuda.is_available():
            self.device_option.set("cuda")
        self.lr_entry.delete(0, "end")
        self.lr_entry.insert(0, "0.00003")
        self.inactivity_entry.delete(0, "end")
        self.inactivity_entry.insert(0, "0.0008")
        self.override_entropy_var.set("on")
        self._toggle_entropy_entry()
        self.entropy_entry.delete(0, "end")
        self.entropy_entry.insert(0, "0.04")
        self._update_stats_summary()
        self.log_to_console("🏆 Applied Preset: Production Alpha Retrain (500k Steps, Deep Transformer, Strict Guardrails)\n")

    def _apply_preset_conservative(self):
        self.drawdown_coef_entry.delete(0, "end")
        self.drawdown_coef_entry.insert(0, "0.25")
        self.vol_scaling_entry.delete(0, "end")
        self.vol_scaling_entry.insert(0, "0.75")
        self.max_dd_entry.delete(0, "end")
        self.max_dd_entry.insert(0, "0.10")
        self._update_stats_summary()
        self.log_to_console("🛡️ Applied Preset: Conservative Capital Preserver (Drawdown Coef 0.25, Vol Scaling 0.75, DD Cap 10%)\n")

    # ---------------- CHECKPOINT DETECTION ----------------
    def detect_existing_model(self):
        path = "models/persistent_brain.pth"
        if not os.path.exists(path):
            self.stat_brain_val.configure(text="FRESH (No Existing Weights)", text_color="#10B981")
            return

        try:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint.get("policy", checkpoint)
            total_steps = checkpoint.get("total_steps", 0)

            detected_type = "DEEP (256d)"
            for key in state_dict.keys():
                if "features_extractor.embedding.weight" in key:
                    out_features = state_dict[key].shape[0]
                    detected_type = "TINY (64d)" if out_features == 64 else "DEEP (256d)"
                    break

            mtime = os.path.getmtime(path)
            dt_str = time.strftime("%b %d %H:%M", time.localtime(mtime))
            self.stat_brain_val.configure(
                text=f"{detected_type} • {total_steps:,} steps ({dt_str})",
                text_color="#38BDF8",
            )
        except Exception:
            self.stat_brain_val.configure(text="Checkpoint Error / Corrupted", text_color="#EF4444")

    # ---------------- PROCESS & TIMER MANAGEMENT ----------------
    def get_python_executable(self):
        local_env_python = os.path.abspath("env/Scripts/python.exe")
        if os.path.exists(local_env_python):
            return local_env_python
        return sys.executable

    def open_tensorboard(self):
        tb_script = os.path.abspath("run_tensorboard.bat")
        if os.path.exists(tb_script):
            subprocess.Popen(["cmd.exe", "/c", "start", tb_script], shell=True)
            self.log_to_console("📊 Launching TensorBoard background server...\n")
        webbrowser.open("http://localhost:6006")

    def open_models_dir(self):
        models_dir = os.path.abspath("models")
        if os.path.exists(models_dir):
            if os.name == "nt":
                os.startfile(models_dir)
            else:
                subprocess.Popen(["xdg-open", models_dir])

    def _tick_timer(self):
        if self.timer_running and self.train_start_time is not None:
            elapsed = time.perf_counter() - self.train_start_time
            hours, rem = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(rem, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.stat_timer_val.configure(text=time_str, text_color="#38BDF8")

            if self.current_eta:
                eta_display = self.current_eta
            elif self.current_steps > 0 and elapsed > 4:
                speed = self.current_steps / elapsed
                remaining = max(0, self.target_steps - self.current_steps)
                rem_sec = int(remaining / speed) if speed > 0 else 0
                h_e, rem_m = divmod(rem_sec, 3600)
                m_e, s_e = divmod(rem_m, 60)
                eta_display = f"{h_e:02d}:{m_e:02d}:{s_e:02d}"
            else:
                eta_display = "Calculating..."

            pct_suffix = f" • {self.current_pct:.1f}%" if self.current_pct > 0 else ""
            self.stat_eta_val.configure(text=f"ETA: {eta_display}{pct_suffix}", text_color="#10B981")
            self.status_time_lbl.configure(text=f"Duration: {time_str} | ETA: {eta_display}", text_color="#38BDF8")
            self.timer_after_id = self.after(500, self._tick_timer)

    def _stop_timer(self):
        self.timer_running = False
        if self.timer_after_id:
            try:
                self.after_cancel(self.timer_after_id)
            except Exception:
                pass
            self.timer_after_id = None

    def start_training(self):
        symbol_raw = self.asset_option.get()
        symbol = (
            symbol_raw.split("(")[0].strip() + "USDT"
            if "BTC" in symbol_raw or "ETH" in symbol_raw or "DOGE" in symbol_raw
            else symbol_raw.split("(")[0].strip()
        )

        try:
            steps = int(self.steps_entry.get().strip())
            if steps <= 0:
                raise ValueError()
        except ValueError:
            self.log_to_console("❌ Error: Steps must be a positive integer.\n")
            return

        try:
            lr = float(self.lr_entry.get().strip())
            if lr <= 0:
                raise ValueError()
        except ValueError:
            self.log_to_console("❌ Error: Learning Rate must be a positive float.\n")
            return

        try:
            commission = float(self.comm_entry.get().strip())
            if commission < 0:
                raise ValueError()
        except ValueError:
            self.log_to_console("❌ Error: Commission must be a non-negative float.\n")
            return

        try:
            slippage = float(self.slip_entry.get().strip())
            if slippage < 0:
                raise ValueError()
        except ValueError:
            self.log_to_console("❌ Error: Slippage must be a non-negative float.\n")
            return

        try:
            drawdown_coef = float(self.drawdown_coef_entry.get().strip())
            if drawdown_coef < 0:
                raise ValueError()
        except ValueError:
            self.log_to_console("❌ Error: Drawdown Penalty Coef must be a non-negative float.\n")
            return

        try:
            inactivity = float(self.inactivity_entry.get().strip())
            if inactivity < 0:
                raise ValueError()
        except ValueError:
            inactivity = 0.0005

        try:
            gamma = float(self.gamma_entry.get().strip())
            if gamma <= 0 or gamma > 1.0:
                raise ValueError()
        except ValueError:
            gamma = 0.97

        try:
            gae_lambda = float(self.gae_entry.get().strip())
            if gae_lambda <= 0 or gae_lambda > 1.0:
                raise ValueError()
        except ValueError:
            gae_lambda = 0.92

        try:
            vol_scaling = float(self.vol_scaling_entry.get().strip())
            if vol_scaling < 0:
                raise ValueError()
        except ValueError:
            vol_scaling = 0.5

        try:
            cooldown = int(self.cooldown_entry.get().strip())
            if cooldown < 0:
                raise ValueError()
        except ValueError:
            cooldown = 12

        try:
            max_dd = float(self.max_dd_entry.get().strip())
            if max_dd <= 0 or max_dd > 1.0:
                raise ValueError()
        except ValueError:
            max_dd = 0.15

        try:
            max_dur = int(self.max_dur_entry.get().strip())
            if max_dur < 0:
                raise ValueError()
        except ValueError:
            max_dur = 48

        entropy = self.entropy_entry.get().strip() if self.override_entropy_var.get() == "on" else "0.01"

        python_exe = self.get_python_executable()
        interval_val = "Daily" if "daily" in self.interval_option.get().lower() else "1H"

        cmd = [
            python_exe,
            "-u",
            "src/train_agent.py",
            "--symbol", symbol,
            "--interval", interval_val,
            "--steps", str(steps),
            "--entropy", entropy,
            "--device", self.device_option.get(),
            "--extractor", self.extractor_option.get(),
            "--commission", str(commission),
            "--slippage", str(slippage),
            "--drawdown-penalty-coef", str(drawdown_coef),
            "--inactivity-penalty", str(inactivity),
            "--volatility-scaling", str(vol_scaling),
            "--cooldown", str(cooldown),
            "--max-drawdown-cap", str(max_dd),
            "--max-trade-duration", str(max_dur),
            "--gamma", str(gamma),
            "--gae-lambda", str(gae_lambda),
            "--lr", str(lr),
        ]

        if self.override_entropy_var.get() == "on":
            cmd.append("--override-entropy")

        if self.reset_var.get() == "on":
            cmd.append("--reset")

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.training_progress_bar.set(0.0)

        # Reset rollout data tracking
        self.rollout_data = {"steps": [], "returns": [], "win_rates": [], "val_returns": []}
        self.current_rollout_record = {}
        self._render_empty_plots()
        self.canvas.draw_idle()

        # Start dynamic training timer
        self.target_steps = steps
        self.current_steps = 0
        self.current_pct = 0.0
        self.current_eta = None
        self.train_start_time = time.perf_counter()
        self.timer_running = True
        self.stat_timer_val.configure(text="00:00:00", text_color="#38BDF8")
        self.stat_eta_val.configure(text="ETA: Calculating...", text_color="#10B981")
        self._tick_timer()

        self.stat_engine_val.configure(text=f"● OPTIMIZING • {symbol}", text_color="#38BDF8")
        self.status_lbl.configure(text=f"System Status: Training {symbol} on {self.device_option.get().upper()}...", text_color="#38BDF8")

        self.log_to_console("\n-------------------------------------------------------------------------------\n")
        self.log_to_console(f"🚀 Launching Training Subprocess: {symbol} ({steps:,} steps on {self.device_option.get().upper()})\n")
        self.log_to_console(f"Command: {' '.join(cmd)}\n")
        self.log_to_console("-------------------------------------------------------------------------------\n\n")

        env_copy = os.environ.copy()
        env_copy["PYTHONIOENCODING"] = "utf-8"

        self.execution_thread = threading.Thread(
            target=self._execute_subprocess_thread, args=(cmd, env_copy), daemon=True
        )
        self.execution_thread.start()

    def _execute_subprocess_thread(self, cmd, env_copy):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env_copy,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            for line in iter(self.process.stdout.readline, ""):
                if line:
                    self.after(0, self.log_to_console, line)

            self.process.stdout.close()
            return_code = self.process.wait()
            self.after(0, self._on_training_complete, return_code)
        except Exception as e:
            self.after(0, self.log_to_console, f"\n❌ System Execution Exception: {e}\n")
            self.after(0, self._on_training_complete, -1)

    def stop_training(self):
        if self.process and self.process.poll() is None:
            self._stop_timer()
            if self.train_start_time is not None:
                elapsed = time.perf_counter() - self.train_start_time
                h, rem = divmod(int(elapsed), 3600)
                m, s = divmod(rem, 60)
                self.stat_timer_val.configure(text=f"{h:02d}:{m:02d}:{s:02d} (Halted)", text_color="#EF4444")
                self.status_time_lbl.configure(text=f"Halted at {h:02d}:{m:02d}:{s:02d}", text_color="#EF4444")

            self.log_to_console("\n⚠️ Stop signal received. Terminating process tree...\n")
            self.stat_engine_val.configure(text="● STOPPING...", text_color="#EF4444")
            self.status_lbl.configure(text="System Status: Interrupting training process...", text_color="#EF4444")
            self.stop_btn.configure(state="disabled")

            def kill_worker():
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
                    else:
                        self.process.terminate()
                except Exception as e:
                    self.log_to_console(f"⚠️ Warning during process termination: {e}\n")

            threading.Thread(target=kill_worker, daemon=True).start()

    def _on_training_complete(self, return_code):
        self._stop_timer()
        if self.train_start_time is not None:
            elapsed = time.perf_counter() - self.train_start_time
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            final_time = f"{h:02d}:{m:02d}:{s:02d}"
            if return_code == 0:
                self.stat_timer_val.configure(text=final_time, text_color="#10B981")
                self.stat_eta_val.configure(text="ETA: Complete ✅", text_color="#10B981")
                self.status_time_lbl.configure(text=f"Total Time: {final_time}", text_color="#10B981")
                self.training_progress_bar.set(1.0)
            else:
                self.stat_timer_val.configure(text=final_time, text_color="#EF4444")
                self.stat_eta_val.configure(text="ETA: Stopped ⚠️", text_color="#EF4444")
                self.status_time_lbl.configure(text=f"Stopped at {final_time}", text_color="#EF4444")

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

        if return_code == 0:
            self.stat_engine_val.configure(text="● COMPLETED ✅", text_color="#10B981")
            self.status_lbl.configure(text="System Status: Retraining Finished Successfully!", text_color="#10B981")
            self.log_to_console("\n✅ Training pipeline finished successfully.\n")
        else:
            self.stat_engine_val.configure(text="● STOPPED / TERMINATED ⚠️", text_color="#EF4444")
            self.status_lbl.configure(text="System Status: Training Terminated or Failed", text_color="#EF4444")
            self.log_to_console(f"\n⚠️ Process terminated. Exit code: {return_code}\n")

        self.detect_existing_model()

    # ---------------- LOG PARSER & CONSOLE ----------------
    def _parse_log_line(self, line):
        clean = line.strip()

        # Parse live optimization progress bar and ETA
        # Pattern: Optimization: 5,120 / 500,000 steps | [1.0%] | ETA: 0:04:12
        m_prog = re.search(r"Optimization:\s+([\d,]+)\s+/\s+([\d,]+)\s+steps\s+\|\s+\[([\d.]+)%\]\s+\|\s+ETA:\s+([^\s]+)", clean)
        if m_prog:
            try:
                curr_steps = int(m_prog.group(1).replace(",", ""))
                total_steps = int(m_prog.group(2).replace(",", ""))
                pct = float(m_prog.group(3))
                eta_str = m_prog.group(4)

                self.current_steps = curr_steps
                self.target_steps = total_steps
                self.current_pct = pct
                self.current_eta = eta_str
                self.training_progress_bar.set(pct / 100.0)
                self.stat_eta_val.configure(text=f"ETA: {eta_str} • {pct:.1f}%", text_color="#10B981")
            except Exception:
                pass

        # Match ROLLOUT REPORT header
        m_rollout = re.search(r"ROLLOUT #(\d+) REPORT", clean)
        if m_rollout:
            self.current_rollout_record["rollout_num"] = int(m_rollout.group(1))

        # Match Avg Training Return
        m_ret = re.search(r"Avg Training Return:\s+([+\-]?\d+\.?\d*)%", clean)
        if m_ret:
            val = float(m_ret.group(1))
            self.current_rollout_record["train_ret"] = val
            self.rollout_data["returns"].append(val)

        # Match Win Rate
        m_win = re.search(r"Win Rate \(ROI > 0\):\s+(\d+\.?\d*)%", clean)
        if m_win:
            val = float(m_win.group(1))
            self.current_rollout_record["win_rate"] = val
            self.rollout_data["win_rates"].append(val)

        # Match Best Final Capital
        m_cap = re.search(r"Best Final Capital:\s+\$([\d,]+\.?\d*)", clean)
        if m_cap:
            self.current_rollout_record["best_cap"] = float(m_cap.group(1).replace(",", ""))

        # Match Step in Validation Report
        m_step = re.search(r"Step:\s+([\d,]+)", clean)
        if m_step:
            val_step = int(m_step.group(1).replace(",", ""))
            self.current_rollout_record["step"] = val_step
            if val_step not in self.rollout_data["steps"]:
                self.rollout_data["steps"].append(val_step)

        # Match Real Portfolio Return
        m_val_ret = re.search(r"Real Portfolio Return:\s+([+\-]?\d+\.?\d*)%", clean)
        if m_val_ret:
            val = float(m_val_ret.group(1))
            self.current_rollout_record["val_ret"] = val
            self.rollout_data["val_returns"].append(val)

        # Match Model Saved checkpoint
        if "New best model saved" in clean:
            self.current_rollout_record["saved"] = True

        # When validation ends or score printed, commit row and refresh graphs
        if "Validation Score:" in clean or "FastEvalCallback" in clean:
            if "train_ret" in self.current_rollout_record:
                r_num = self.current_rollout_record.get("rollout_num", len(self.rollout_data["steps"]))
                step = self.current_rollout_record.get("step", 0)
                t_ret = self.current_rollout_record.get("train_ret", 0.0)
                w_rate = self.current_rollout_record.get("win_rate", 0.0)
                b_cap = self.current_rollout_record.get("best_cap", 10000.0)
                v_ret = self.current_rollout_record.get("val_ret", 0.0)
                saved = self.current_rollout_record.get("saved", False)

                self.add_table_row(r_num, step, t_ret, w_rate, b_cap, v_ret, saved)
                self.update_live_charts()
                self.current_rollout_record = {}

    def log_to_console(self, text):
        self.log_line_count += 1
        self.line_count_lbl.configure(text=f"{self.log_line_count:,} lines")

        self._parse_log_line(text)

        self.console_text.configure(state="normal")
        self.console_text.insert("end", text)
        if self.auto_scroll_var.get() == "on":
            self.console_text.see("end")
        self.console_text.configure(state="disabled")

    def clear_console(self):
        self.log_line_count = 0
        self.line_count_lbl.configure(text="0 lines")
        self.console_text.configure(state="normal")
        self.console_text.delete("1.0", "end")
        self.console_text.configure(state="disabled")

    def _bind_mousewheel(self, scrollable_frame, speed=60):
        def _custom_mouse_wheel(event):
            try:
                if scrollable_frame.check_if_master_is_canvas(event.widget):
                    if event.delta:
                        units = int(-1 * (event.delta / 120) * speed)
                        scrollable_frame._parent_canvas.yview_scroll(units, "units")
                    elif event.num == 5:
                        scrollable_frame._parent_canvas.yview_scroll(speed, "units")
                    elif event.num == 4:
                        scrollable_frame._parent_canvas.yview_scroll(-speed, "units")
            except Exception:
                pass

        scrollable_frame._mouse_wheel_all = _custom_mouse_wheel

    def _on_window_closing(self):
        if self.process and self.process.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.process.pid)], capture_output=True)
                else:
                    self.process.terminate()
            except Exception:
                pass
        self._stop_timer()
        if self.telemetry_after_id:
            try:
                self.after_cancel(self.telemetry_after_id)
            except Exception:
                pass
        self.destroy()
        sys.exit(0)


if __name__ == "__main__":
    app = ModernRLOptimizerGUI()
    app.mainloop()
