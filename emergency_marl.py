"""
Emergency Response Multi-Agent Reinforcement Learning System
============================================================
Real-World Data Source: NEMSIS (National EMS Information System) - inspired
state distributions. We generate synthetic data calibrated to NEMSIS 2023
statistics:
  - ~54M EMS activations/year across the US
  - Cardiac arrest, trauma, respiratory, stroke are top call types
  - Average response time: 7 min urban, 14 min rural
  - Hospital diversion (overload) events: ~15% of transports

State Space: 5 variables → 3×5×4×3×5 = 900 discrete states
  - Severity       : 0=low, 1=moderate, 2=critical (cardiac/trauma/stroke)
  - Call Type      : 0=cardiac, 1=trauma, 2=respiratory, 3=stroke, 4=other
  - Traffic        : 0=clear, 1=moderate, 2=heavy, 3=gridlock
  - Hospital Load  : 0=normal, 1=busy, 2=diversion
  - Time of Day    : 0=night(0-6), 1=morning(6-12), 2=afternoon(12-18),
                     3=evening(18-22), 4=late-night(22-24)

6 Agents, each with 5 actions:
  Ambulance      : 0=fastest_route, 1=alternate_route, 2=air_transport,
                   3=on-scene_stabilise, 4=request_backup
  Hospital       : 0=accept, 1=prepare_trauma_bay, 2=prepare_cath_lab,
                   3=divert_to_nearest, 4=call_specialist
  Traffic        : 0=no_action, 1=green_wave, 2=close_intersection,
                   3=alert_highway, 4=reroute_public
  Dispatcher     : 0=single_unit, 1=double_unit, 2=air_ambulance,
                   3=fire_support, 4=police_escort
  Triage         : 0=immediate, 1=delayed, 2=minimal, 3=expectant, 4=reassess
  ICU_Manager    : 0=hold_bed, 1=create_surge_bed, 2=transfer_patient,
                   3=call_backup_staff, 4=activate_protocol
"""

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
import json
import os

# ─────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────
EPISODES       = 2000
MAX_STEPS      = 50
ALPHA          = 0.1        # learning rate
GAMMA          = 0.95       # discount (higher = more future-aware)
EPSILON        = 1.0
EPSILON_DECAY  = 0.997
EPSILON_MIN    = 0.05
SHARED_REWARD_WEIGHT = 0.3  # cooperative component weight

AGENTS = ["Ambulance", "Hospital", "Traffic", "Dispatcher", "Triage", "ICU_Manager"]
NUM_ACTIONS = 5

# State dimensions (NEMSIS-calibrated)
SEV_LEVELS  = 3   # severity
CALL_TYPES  = 5   # call type
TRAF_LEVELS = 4   # traffic
HOSP_LEVELS = 3   # hospital load
TIME_SLOTS  = 5   # time of day
NUM_STATES  = SEV_LEVELS * CALL_TYPES * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS  # 900

# ─────────────────────────────────────────────────────────────────
# NEMSIS-CALIBRATED DATA GENERATOR
# ─────────────────────────────────────────────────────────────────
# Real-world distributions from NEMSIS 2023 Annual Report
# Call type proportions: cardiac 18%, trauma 22%, respiratory 20%,
#                        stroke 10%, other 30%
CALL_TYPE_PROBS   = [0.18, 0.22, 0.20, 0.10, 0.30]
# Severity varies by call type - cardiac/stroke skew critical
SEVERITY_BY_CALL  = {
    0: [0.10, 0.40, 0.50],   # cardiac: mostly critical
    1: [0.20, 0.45, 0.35],   # trauma
    2: [0.30, 0.50, 0.20],   # respiratory
    3: [0.15, 0.35, 0.50],   # stroke: mostly critical
    4: [0.50, 0.35, 0.15],   # other: mostly low
}
# Traffic: urban rush-hour patterns
TRAFFIC_BY_TIME   = {
    0: [0.60, 0.25, 0.10, 0.05],   # night: mostly clear
    1: [0.15, 0.35, 0.35, 0.15],   # morning rush
    2: [0.25, 0.40, 0.25, 0.10],   # afternoon
    3: [0.10, 0.30, 0.40, 0.20],   # evening rush
    4: [0.45, 0.30, 0.15, 0.10],   # late night
}
# Hospital load: ~15% diversion rate (NEMSIS statistic)
HOSP_LOAD_PROBS   = [0.55, 0.30, 0.15]
# Time of day: uniform-ish with peak in afternoon/evening
TIME_PROBS        = [0.12, 0.22, 0.26, 0.24, 0.16]


def generate_realistic_state():
    """Generate state using NEMSIS-calibrated real-world distributions."""
    time     = np.random.choice(TIME_SLOTS, p=TIME_PROBS)
    call     = np.random.choice(CALL_TYPES, p=CALL_TYPE_PROBS)
    sev      = np.random.choice(SEV_LEVELS, p=SEVERITY_BY_CALL[call])
    traffic  = np.random.choice(TRAF_LEVELS, p=TRAFFIC_BY_TIME[time])
    hosp     = np.random.choice(HOSP_LEVELS, p=HOSP_LOAD_PROBS)
    return encode_state(sev, call, traffic, hosp, time)


def encode_state(sev, call, traffic, hosp, time):
    return (sev * CALL_TYPES * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS
            + call * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS
            + traffic * HOSP_LEVELS * TIME_SLOTS
            + hosp * TIME_SLOTS
            + time)


def decode_state(s):
    time    = s % TIME_SLOTS;             s //= TIME_SLOTS
    hosp    = s % HOSP_LEVELS;            s //= HOSP_LEVELS
    traffic = s % TRAF_LEVELS;            s //= TRAF_LEVELS
    call    = s % CALL_TYPES;             s //= CALL_TYPES
    sev     = s
    return sev, call, traffic, hosp, time


# ─────────────────────────────────────────────────────────────────
# REWARD FUNCTIONS  (NEMSIS-inspired clinical outcomes)
# ─────────────────────────────────────────────────────────────────
# Response time targets from NFPA 1710 standard:
#   - BLS on scene: ≤ 4 min (240 sec) in urban
#   - ALS on scene: ≤ 8 min (480 sec)
# Hospital door-to-balloon (cardiac): ≤ 90 min target

def individual_reward(agent, sev, call, traffic, hosp, time, action):
    r = 0

    if agent == "Ambulance":
        # Best action depends on traffic + severity
        if sev == 2 and traffic >= 2 and action == 2:    # air transport for critical + gridlock
            r += 20
        elif sev == 2 and traffic < 2 and action == 0:   # fastest route when clear
            r += 15
        elif traffic >= 2 and action == 1:               # alternate route in heavy traffic
            r += 10
        elif sev <= 1 and action == 3:                   # stabilise on scene if not critical
            r += 8
        elif action == 4 and sev == 2 and hosp == 2:     # backup when critical + hospital full
            r += 12
        else:
            r -= 4

    elif agent == "Hospital":
        if hosp == 2 and action == 3:                    # divert when full → correct
            r += 15
        elif hosp < 2 and action == 0:                   # accept when capacity available
            r += 10
        elif call == 0 and sev == 2 and action == 2:     # cath lab for critical cardiac
            r += 18
        elif call == 1 and sev == 2 and action == 1:     # trauma bay for critical trauma
            r += 18
        elif call == 3 and sev == 2 and action == 4:     # specialist for stroke
            r += 16
        else:
            r -= 3

    elif agent == "Traffic":
        if traffic >= 2 and action == 1:                 # green wave in heavy traffic
            r += 12
        elif traffic == 3 and action == 2:               # close intersection in gridlock
            r += 14
        elif traffic <= 1 and action == 0:               # no action when clear → efficient
            r += 5
        elif time in [1, 3] and traffic >= 2 and action == 3:   # highway alert in rush hour
            r += 10
        else:
            r -= 2

    elif agent == "Dispatcher":
        if sev == 2 and call in [0, 3] and action == 2:  # air ambulance for critical cardiac/stroke
            r += 18
        elif sev == 2 and action == 1:                   # double unit for any critical
            r += 14
        elif call == 1 and action == 3:                  # fire support for trauma
            r += 10
        elif sev == 0 and action == 0:                   # single unit for low severity
            r += 8
        else:
            r -= 3

    elif agent == "Triage":
        if sev == 2 and action == 0:                     # immediate for critical
            r += 18
        elif sev == 1 and action == 1:                   # delayed for moderate
            r += 12
        elif sev == 0 and action == 2:                   # minimal for low
            r += 8
        elif action == 4 and hosp == 1:                  # reassess when hospital busy
            r += 10
        else:
            r -= 5   # wrong triage is costly

    elif agent == "ICU_Manager":
        if hosp == 2 and action == 1:                    # surge bed when at capacity
            r += 16
        elif hosp == 2 and action == 2:                  # transfer patient to free up space
            r += 14
        elif sev == 2 and action == 3:                   # call backup staff for critical
            r += 12
        elif call in [0, 3] and sev == 2 and action == 4:  # activate protocol for cardiac/stroke
            r += 18
        elif hosp < 2 and action == 0:                   # hold bed when normal load
            r += 8
        else:
            r -= 3

    return r


def compute_shared_reward(state_decoded, agent_actions):
    """
    Cooperative reward: bonus when the whole team coordinates well.
    Based on clinical outcome proxy: survival/recovery probability.
    """
    sev, call, traffic, hosp, time = state_decoded
    bonus = 0

    # Dispatcher sent right unit AND Ambulance took right route
    disp_act = agent_actions.get("Dispatcher", -1)
    amb_act  = agent_actions.get("Ambulance", -1)
    if sev == 2 and disp_act == 2 and amb_act == 2:        # both chose air
        bonus += 10
    if sev == 2 and disp_act == 1 and amb_act == 0:        # double unit + fastest route
        bonus += 8

    # Hospital prepared AND Triage flagged correctly
    hosp_act  = agent_actions.get("Hospital", -1)
    triage_act = agent_actions.get("Triage", -1)
    if call == 0 and hosp_act == 2 and triage_act == 0:    # cath lab + immediate
        bonus += 12
    if call == 1 and hosp_act == 1 and triage_act == 0:    # trauma bay + immediate
        bonus += 12

    # Traffic cleared AND response was fast
    traffic_act = agent_actions.get("Traffic", -1)
    if traffic >= 2 and traffic_act in [1, 2] and amb_act in [0, 2]:
        bonus += 6

    return bonus


# ─────────────────────────────────────────────────────────────────
# Q-TABLES + TRAINING
# ─────────────────────────────────────────────────────────────────
Q_tables = {agent: np.zeros((NUM_STATES, NUM_ACTIONS)) for agent in AGENTS}

# Metrics
reward_history          = []
response_time_history   = []
icu_utilization_history = []
diversion_events        = []
triage_accuracy_history = []
episode_logs            = []  # for CSV export

epsilon = EPSILON

SEVERITY_LABELS  = ["Low", "Moderate", "Critical"]
CALL_TYPE_LABELS = ["Cardiac", "Trauma", "Respiratory", "Stroke", "Other"]
TRAFFIC_LABELS   = ["Clear", "Moderate", "Heavy", "Gridlock"]
HOSP_LABELS      = ["Normal", "Busy", "Diversion"]
TIME_LABELS      = ["Night", "Morning", "Afternoon", "Evening", "Late-Night"]

for episode in range(EPISODES):
    state = generate_realistic_state()
    total_reward      = 0
    total_resp_time   = 0
    icu_usage         = 0
    diversions        = 0
    correct_triages   = 0
    total_triages     = 0

    for step in range(MAX_STEPS):
        sev, call, traffic, hosp, time = decode_state(state)
        agent_actions = {}

        # All agents choose action for the SAME shared state
        for agent in AGENTS:
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = int(np.argmax(Q_tables[agent][state]))
            agent_actions[agent] = action

        # Shared reward for coordination
        shared_r = compute_shared_reward((sev, call, traffic, hosp, time), agent_actions)

        # Q-update for each agent
        new_state = generate_realistic_state()
        for agent in AGENTS:
            action = agent_actions[agent]
            ind_r  = individual_reward(agent, sev, call, traffic, hosp, time, action)
            reward = ind_r + SHARED_REWARD_WEIGHT * shared_r
            reward += random.uniform(-1, 1)   # small noise

            old_val   = Q_tables[agent][state][action]
            next_max  = np.max(Q_tables[agent][new_state])
            Q_tables[agent][state][action] = old_val + ALPHA * (
                reward + GAMMA * next_max - old_val
            )
            total_reward += reward

        # Simulate realistic response time (minutes) based on traffic + dispatch
        base_resp = 7 if time in [1, 2, 3] else 10   # urban baseline (NEMSIS avg)
        traffic_penalty = [0, 2, 5, 10][traffic]
        if agent_actions["Ambulance"] == 2:            # air transport
            resp_time = max(base_resp - 3, 4)
        elif agent_actions["Traffic"] in [1, 2]:       # green wave / closure
            resp_time = base_resp + traffic_penalty - 3
        else:
            resp_time = base_resp + traffic_penalty
        total_resp_time += resp_time

        # ICU surge tracking
        if agent_actions["ICU_Manager"] == 1:
            icu_usage += 1

        # Diversion tracking
        if agent_actions["Hospital"] == 3:
            diversions += 1

        # Triage accuracy proxy
        triage_act = agent_actions["Triage"]
        correct = (sev == 2 and triage_act == 0) or \
                  (sev == 1 and triage_act == 1) or \
                  (sev == 0 and triage_act == 2)
        if correct:
            correct_triages += 1
        total_triages += 1

        state = new_state

    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

    avg_resp   = total_resp_time / (MAX_STEPS * len(AGENTS))
    triage_acc = correct_triages / total_triages * 100

    reward_history.append(total_reward)
    response_time_history.append(avg_resp)
    icu_utilization_history.append(icu_usage)
    diversion_events.append(diversions)
    triage_accuracy_history.append(triage_acc)

    episode_logs.append({
        "episode":          episode + 1,
        "total_reward":     round(total_reward, 2),
        "avg_response_min": round(avg_resp, 2),
        "icu_surge_count":  icu_usage,
        "diversions":       diversions,
        "triage_accuracy":  round(triage_acc, 2),
        "epsilon":          round(epsilon, 4),
    })

    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1:4d} | Reward: {total_reward:8.1f} | "
              f"Resp: {avg_resp:.1f}min | TriageAcc: {triage_acc:.1f}%")


# ─────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# 1. Training log CSV
df_log = pd.DataFrame(episode_logs)
df_log.to_csv("outputs/training_log.csv", index=False)

# 2. Final policies CSV
policy_rows = []
for agent in AGENTS:
    for s in range(NUM_STATES):
        sev, call, traffic, hosp, time = decode_state(s)
        best_action = int(np.argmax(Q_tables[agent][s]))
        q_vals      = Q_tables[agent][s].tolist()
        policy_rows.append({
            "agent":        agent,
            "state_id":     s,
            "severity":     SEVERITY_LABELS[sev],
            "call_type":    CALL_TYPE_LABELS[call],
            "traffic":      TRAFFIC_LABELS[traffic],
            "hosp_load":    HOSP_LABELS[hosp],
            "time_of_day":  TIME_LABELS[time],
            "best_action":  best_action,
            "q_values":     str([round(q, 3) for q in q_vals]),
        })
df_policy = pd.DataFrame(policy_rows)
df_policy.to_csv("outputs/final_policies.csv", index=False)

# 3. Q-tables as numpy
for agent in AGENTS:
    np.save(f"outputs/qtable_{agent}.npy", Q_tables[agent])

print("\n✓ Outputs saved: training_log.csv, final_policies.csv, qtable_*.npy")


# ─────────────────────────────────────────────────────────────────
# COMPREHENSIVE PLOTS
# ─────────────────────────────────────────────────────────────────
def smooth(data, window=50):
    return pd.Series(data).rolling(window, min_periods=1).mean().values

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Emergency Response MARL — Training Results\n"
             "(State space calibrated to NEMSIS 2023 EMS Statistics)",
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Plot 1 – Reward convergence
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(reward_history, alpha=0.2, color='steelblue')
ax1.plot(smooth(reward_history), color='steelblue', linewidth=2, label='Smoothed')
ax1.set_title("Reward Convergence", fontweight='bold')
ax1.set_xlabel("Episode"); ax1.set_ylabel("Cumulative Reward")
ax1.legend()

# Plot 2 – Response time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(response_time_history, alpha=0.2, color='tomato')
ax2.plot(smooth(response_time_history), color='tomato', linewidth=2)
ax2.axhline(y=8, color='green', linestyle='--', label='NFPA target (8 min)')
ax2.set_title("Avg Response Time (minutes)", fontweight='bold')
ax2.set_xlabel("Episode"); ax2.set_ylabel("Minutes")
ax2.legend()

# Plot 3 – Triage accuracy
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(triage_accuracy_history, alpha=0.2, color='mediumorchid')
ax3.plot(smooth(triage_accuracy_history), color='mediumorchid', linewidth=2)
ax3.set_title("Triage Accuracy (%)", fontweight='bold')
ax3.set_xlabel("Episode"); ax3.set_ylabel("Accuracy %")
ax3.set_ylim(0, 105)

# Plot 4 – ICU utilization
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(icu_utilization_history, alpha=0.2, color='darkorange')
ax4.plot(smooth(icu_utilization_history), color='darkorange', linewidth=2)
ax4.set_title("ICU Surge Activations", fontweight='bold')
ax4.set_xlabel("Episode"); ax4.set_ylabel("Count per Episode")

# Plot 5 – Hospital diversions
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(diversion_events, alpha=0.2, color='seagreen')
ax5.plot(smooth(diversion_events), color='seagreen', linewidth=2)
ax5.set_title("Hospital Diversion Events", fontweight='bold')
ax5.set_xlabel("Episode"); ax5.set_ylabel("Diversions per Episode")

# Plot 6 – Per-agent Q-value heatmap (critical cardiac state)
ax6 = fig.add_subplot(gs[1, 2])
# Show Q-values for critical cardiac in gridlock with busy hospital at afternoon
sample_state = encode_state(sev=2, call=0, traffic=3, hosp=1, time=2)
q_matrix = np.array([Q_tables[a][sample_state] for a in AGENTS])
im = ax6.imshow(q_matrix, cmap='RdYlGn', aspect='auto')
ax6.set_xticks(range(NUM_ACTIONS))
ax6.set_xticklabels([f"A{i}" for i in range(NUM_ACTIONS)])
ax6.set_yticks(range(len(AGENTS)))
ax6.set_yticklabels(AGENTS, fontsize=8)
ax6.set_title("Q-values: Critical Cardiac\n(Gridlock, Busy Hospital)", fontweight='bold')
plt.colorbar(im, ax=ax6)

plt.savefig("outputs/training_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot saved: outputs/training_results.png")


# ─────────────────────────────────────────────────────────────────
# FINAL POLICY SUMMARY (HUMAN-READABLE)
# ─────────────────────────────────────────────────────────────────
ACTION_LABELS = {
    "Ambulance":   ["Fastest Route", "Alternate Route", "Air Transport",
                    "On-Scene Stabilise", "Request Backup"],
    "Hospital":    ["Accept Patient", "Prepare Trauma Bay", "Prepare Cath Lab",
                    "Divert to Nearest", "Call Specialist"],
    "Traffic":     ["No Action", "Green Wave", "Close Intersection",
                    "Alert Highway", "Reroute Public"],
    "Dispatcher":  ["Single Unit", "Double Unit", "Air Ambulance",
                    "Fire Support", "Police Escort"],
    "Triage":      ["Immediate", "Delayed", "Minimal", "Expectant", "Reassess"],
    "ICU_Manager": ["Hold Bed", "Create Surge Bed", "Transfer Patient",
                    "Call Backup Staff", "Activate Protocol"],
}

print("\n" + "═"*70)
print("  FINAL LEARNED POLICIES — KEY SCENARIOS")
print("═"*70)

key_scenarios = [
    (2, 0, 3, 2, 1, "Critical Cardiac | Gridlock | Hospital Diversion | Morning"),
    (2, 1, 2, 1, 2, "Critical Trauma  | Heavy Traffic | Hospital Busy | Afternoon"),
    (2, 3, 1, 0, 0, "Critical Stroke  | Moderate Traffic | Normal Load | Night"),
    (1, 2, 2, 1, 3, "Moderate Respiratory | Heavy | Busy | Evening"),
    (0, 4, 0, 0, 2, "Low Severity Other | Clear | Normal | Afternoon"),
]

for sev, call, traffic, hosp, time, label in key_scenarios:
    s = encode_state(sev, call, traffic, hosp, time)
    print(f"\n🚨 Scenario: {label}")
    for agent in AGENTS:
        best_a = int(np.argmax(Q_tables[agent][s]))
        print(f"   {agent:<15} → {ACTION_LABELS[agent][best_a]}")

print("\n" + "═"*70)
print(f"Training complete. {EPISODES} episodes | {NUM_STATES} states | {len(AGENTS)} agents")
print("Data calibrated to NEMSIS 2023 (54M+ EMS activations reference)")
