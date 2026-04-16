"""
visualize.py
============
All matplotlib plotting functions for training analysis.

Plots generated:
  1. Reward convergence
  2. Average response time vs NFPA 1710 target
  3. Triage accuracy over training
  4. ICU surge activations
  5. Hospital diversion events
  6. Per-agent Q-value heatmap (critical cardiac scenario)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.environment import encode_state
from src.agents import AGENTS, NUM_ACTIONS


def smooth(data, window=50):
    """Rolling average for noisy training curves."""
    return pd.Series(data).rolling(window, min_periods=1).mean().values


def plot_training_results(
    reward_history,
    response_time_history,
    triage_accuracy_history,
    icu_utilization_history,
    diversion_events,
    Q_tables,
    save_path="outputs/training_results.png",
):
    """
    Generate and save the 6-panel training results figure.

    Parameters
    ----------
    reward_history          : list[float]
    response_time_history   : list[float]
    triage_accuracy_history : list[float]
    icu_utilization_history : list[int]
    diversion_events        : list[int]
    Q_tables                : dict {agent: np.ndarray shape (NUM_STATES, NUM_ACTIONS)}
    save_path               : str — output file path
    """
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Emergency Response MARL — Training Results\n"
        "(State space calibrated to NEMSIS 2023 EMS Statistics)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1 — Reward convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(reward_history, alpha=0.2, color="steelblue")
    ax1.plot(smooth(reward_history), color="steelblue", linewidth=2, label="Smoothed")
    ax1.set_title("Reward Convergence", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend()

    # 2 — Response time vs NFPA 1710 target
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(response_time_history, alpha=0.2, color="tomato")
    ax2.plot(smooth(response_time_history), color="tomato", linewidth=2)
    ax2.axhline(y=8, color="green", linestyle="--", label="NFPA target (8 min)")
    ax2.set_title("Avg Response Time (minutes)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Minutes")
    ax2.legend()

    # 3 — Triage accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(triage_accuracy_history, alpha=0.2, color="mediumorchid")
    ax3.plot(smooth(triage_accuracy_history), color="mediumorchid", linewidth=2)
    ax3.set_title("Triage Accuracy (%)", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Accuracy %")
    ax3.set_ylim(0, 105)

    # 4 — ICU surge activations
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(icu_utilization_history, alpha=0.2, color="darkorange")
    ax4.plot(smooth(icu_utilization_history), color="darkorange", linewidth=2)
    ax4.set_title("ICU Surge Activations", fontweight="bold")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Count per Episode")

    # 5 — Hospital diversions
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(diversion_events, alpha=0.2, color="seagreen")
    ax5.plot(smooth(diversion_events), color="seagreen", linewidth=2)
    ax5.set_title("Hospital Diversion Events", fontweight="bold")
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Diversions per Episode")

    # 6 — Q-value heatmap: critical cardiac, gridlock, busy hospital, afternoon
    ax6 = fig.add_subplot(gs[1, 2])
    sample_state = encode_state(sev=2, call=0, traffic=3, hosp=1, time=2)
    q_matrix = np.array([Q_tables[a][sample_state] for a in AGENTS])
    im = ax6.imshow(q_matrix, cmap="RdYlGn", aspect="auto")
    ax6.set_xticks(range(NUM_ACTIONS))
    ax6.set_xticklabels([f"A{i}" for i in range(NUM_ACTIONS)])
    ax6.set_yticks(range(len(AGENTS)))
    ax6.set_yticklabels(AGENTS, fontsize=8)
    ax6.set_title("Q-values: Critical Cardiac\n(Gridlock, Busy Hospital)", fontweight="bold")
    plt.colorbar(im, ax=ax6)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Plot saved: {save_path}")


def print_policy_summary(Q_tables):
    """
    Print human-readable policies for 5 key clinical scenarios.
    Useful for verifying the agent learned sensible behaviour.
    """
    from src.agents import ACTION_LABELS
    from src.environment import encode_state

    key_scenarios = [
        (2, 0, 3, 2, 1, "Critical Cardiac | Gridlock | Hospital Diversion | Morning"),
        (2, 1, 2, 1, 2, "Critical Trauma  | Heavy Traffic | Hospital Busy | Afternoon"),
        (2, 3, 1, 0, 0, "Critical Stroke  | Moderate Traffic | Normal Load | Night"),
        (1, 2, 2, 1, 3, "Moderate Respiratory | Heavy | Busy | Evening"),
        (0, 4, 0, 0, 2, "Low Severity Other | Clear | Normal | Afternoon"),
    ]

    print("\n" + "═" * 70)
    print("  FINAL LEARNED POLICIES — KEY SCENARIOS")
    print("═" * 70)

    for sev, call, traffic, hosp, time, label in key_scenarios:
        s = encode_state(sev, call, traffic, hosp, time)
        print(f"\n Scenario: {label}")
        for agent in AGENTS:
            best_a = int(np.argmax(Q_tables[agent][s]))
            print(f"   {agent:<15} → {ACTION_LABELS[agent][best_a]}")

    print("\n" + "═" * 70)
