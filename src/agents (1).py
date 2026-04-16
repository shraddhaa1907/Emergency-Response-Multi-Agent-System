"""
agents.py
=========
Q-table initialisation, epsilon-greedy action selection,
and the Q-learning (Bellman) update for each agent.

Algorithm: Decentralised Q-Learning
  Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') − Q(s,a)]

Each agent maintains its own Q-table of shape (NUM_STATES, NUM_ACTIONS).
Agents observe the SAME shared state but act and update independently.
"""

import numpy as np
import random

from src.environment import NUM_STATES

# ── Agent definitions ─────────────────────────────────────────
AGENTS = ["Ambulance", "Hospital", "Traffic", "Dispatcher", "Triage", "ICU_Manager"]
NUM_ACTIONS = 5

# ── Action labels (for human-readable policy output) ─────────
ACTION_LABELS = {
    "Ambulance":   ["Fastest Route",      "Alternate Route",   "Air Transport",
                    "On-Scene Stabilise", "Request Backup"],
    "Hospital":    ["Accept Patient",     "Prepare Trauma Bay","Prepare Cath Lab",
                    "Divert to Nearest",  "Call Specialist"],
    "Traffic":     ["No Action",          "Green Wave",        "Close Intersection",
                    "Alert Highway",      "Reroute Public"],
    "Dispatcher":  ["Single Unit",        "Double Unit",       "Air Ambulance",
                    "Fire Support",       "Police Escort"],
    "Triage":      ["Immediate",          "Delayed",           "Minimal",
                    "Expectant",          "Reassess"],
    "ICU_Manager": ["Hold Bed",           "Create Surge Bed",  "Transfer Patient",
                    "Call Backup Staff",  "Activate Protocol"],
}

# ── Hyperparameters ───────────────────────────────────────────
ALPHA         = 0.1     # learning rate
GAMMA         = 0.95    # discount factor
EPSILON_START = 1.0     # initial exploration rate
EPSILON_DECAY = 0.997   # per-episode decay multiplier
EPSILON_MIN   = 0.05    # minimum exploration floor


def init_q_tables():
    """Initialise one zero Q-table per agent. Returns a dict."""
    return {agent: np.zeros((NUM_STATES, NUM_ACTIONS)) for agent in AGENTS}


def choose_action(q_table, state, epsilon):
    """
    Epsilon-greedy policy:
      - With probability epsilon → random action (explore)
      - Otherwise              → greedy best action (exploit)
    """
    if random.random() < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    return int(np.argmax(q_table[state]))


def update_q(q_table, state, action, reward, next_state):
    """
    Apply the Bellman update to a single (state, action) entry.

    Q(s,a) ← Q(s,a) + α × [r + γ × max Q(s',·) − Q(s,a)]
    """
    old_val  = q_table[state][action]
    next_max = np.max(q_table[next_state])
    q_table[state][action] = old_val + ALPHA * (
        reward + GAMMA * next_max - old_val
    )


def decay_epsilon(epsilon):
    """Apply one decay step and return the new epsilon value."""
    return max(epsilon * EPSILON_DECAY, EPSILON_MIN)


def get_best_action(q_table, state):
    """Return the greedy action for a given state (inference / evaluation)."""
    return int(np.argmax(q_table[state]))
