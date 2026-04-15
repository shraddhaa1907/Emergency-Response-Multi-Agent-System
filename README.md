# 🚑 Emergency Response Multi-Agent Reinforcement Learning System

> **Multi-Agent Q-Learning simulation for end-to-end emergency healthcare coordination**  
> State space calibrated to real-world NEMSIS 2023 EMS statistics (54M+ activations reference)

---

## 📌 Project Overview

Emergency healthcare response requires **six different organizations** — ambulances, hospitals, traffic control, dispatchers, triage units, and ICU management — to coordinate their decisions in seconds. Poor coordination leads to preventable deaths.

This system models that coordination problem as a **Multi-Agent Reinforcement Learning (MARL)** problem, where each organization is an independent AI agent learning its own optimal policy through thousands of simulated emergency scenarios.

**Real-world data connection:** State distributions are calibrated to the **NEMSIS 2023 National EMS Dataset** (National Emergency Medical Services Information System, supported by NHTSA), which covers:
- 54,190,579 EMS activations from 14,369 agencies across 54 US states/territories
- Call type proportions: Cardiac 18%, Trauma 22%, Respiratory 20%, Stroke 10%, Other 30%
- Average urban response time: 7 minutes | Rural: 14 minutes
- Hospital diversion rate: ~15% of transports
- Source: [NEMSIS Annual Public Data Report 2023](https://nemsis.org)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHARED ENVIRONMENT STATE                     │
│  severity × call_type × traffic × hospital_load × time_of_day  │
│           3  ×    5    ×    4    ×       3       ×     5        │
│                     = 900 discrete states                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ same state observed by all agents
         ┌───────────────┼───────────────────────────────┐
         ▼               ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │Ambulance │   │ Hospital │   │ Traffic  │   │Dispatcher│
   │ Q-Table  │   │ Q-Table  │   │ Q-Table  │   │ Q-Table  │
   │ 900 × 5  │   │ 900 × 5  │   │ 900 × 5  │   │ 900 × 5  │
   └──────────┘   └──────────┘   └──────────┘   └──────────┘
         ▼               ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐
   │  Triage  │   │ICU_Mgr   │
   │ Q-Table  │   │ Q-Table  │
   │ 900 × 5  │   │ 900 × 5  │
   └──────────┘   └──────────┘
         │               │
         └───────┬────────┘
                 ▼
       Individual Reward + Shared Cooperative Reward
```

---

## 📊 State Space (900 States)

| Variable | Levels | Values | Real-World Basis |
|----------|--------|--------|-----------------|
| **Severity** | 3 | Low / Moderate / Critical | NEMSIS patient acuity codes |
| **Call Type** | 5 | Cardiac / Trauma / Respiratory / Stroke / Other | NEMSIS top dispatch categories |
| **Traffic** | 4 | Clear / Moderate / Heavy / Gridlock | Urban traffic index |
| **Hospital Load** | 3 | Normal / Busy / Diversion | NEMSIS diversion rate ~15% |
| **Time of Day** | 5 | Night / Morning / Afternoon / Evening / Late-Night | EMS peak-hour distributions |

**Encoding:** `state = sev×(5×4×3×5) + call×(4×3×5) + traffic×(3×5) + hosp×5 + time`

---

## 🤖 Agents & Actions

### 1. Ambulance Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | Fastest Route | Critical + clear traffic |
| 1 | Alternate Route | Moderate-heavy traffic |
| 2 | Air Transport | Critical + gridlock, cardiac/stroke |
| 3 | On-Scene Stabilise | Low severity, prepare patient |
| 4 | Request Backup | Critical + hospital in diversion |

### 2. Hospital Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | Accept Patient | Normal/busy load |
| 1 | Prepare Trauma Bay | Critical trauma incoming |
| 2 | Prepare Cath Lab | Critical cardiac incoming |
| 3 | Divert to Nearest | Hospital in diversion |
| 4 | Call Specialist | Stroke cases |

### 3. Traffic Control Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | No Action | Clear traffic |
| 1 | Green Wave | Heavy traffic on route |
| 2 | Close Intersection | Gridlock — create clear path |
| 3 | Alert Highway | Rush hour + heavy traffic |
| 4 | Reroute Public | Reroute civilian traffic |

### 4. Dispatcher Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | Single Unit | Low severity |
| 1 | Double Unit | Any critical case |
| 2 | Air Ambulance | Critical cardiac or stroke |
| 3 | Fire Support | Trauma with entrapment |
| 4 | Police Escort | Urban gridlock scenarios |

### 5. Triage Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | Immediate (Red) | Critical severity |
| 1 | Delayed (Yellow) | Moderate severity |
| 2 | Minimal (Green) | Low severity |
| 3 | Expectant (Black) | Unsurvivable + mass casualty |
| 4 | Reassess | Busy hospital, evolving condition |

### 6. ICU Manager Agent
| Action | Description | Optimal When |
|--------|-------------|--------------|
| 0 | Hold Bed | Normal load |
| 1 | Create Surge Bed | Hospital at/near capacity |
| 2 | Transfer Patient | Free capacity for incoming critical |
| 3 | Call Backup Staff | Any critical case arriving |
| 4 | Activate Protocol | Critical cardiac/stroke protocol |

---

## 🧠 Algorithm: Decentralized Q-Learning with Cooperative Reward

### Q-Update (Bellman Equation)
```
Q(s,a) ← Q(s,a) + α × [r_ind + β×r_shared + γ × max Q(s') − Q(s,a)]
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| α (alpha) | 0.1 | Learning rate — how fast to update beliefs |
| γ (gamma) | 0.95 | Discount factor — weight future rewards heavily |
| ε (epsilon) | 1.0 → 0.05 | Exploration: starts random, becomes strategic |
| ε decay | 0.997/episode | Gradually shift from explore to exploit |
| β (shared weight) | 0.3 | How much team coordination bonus matters |

### Reward Design
Each agent gets:
1. **Individual reward** — based on clinically optimal decisions (e.g., cath lab for cardiac, air transport for gridlock + critical)
2. **Shared cooperative reward** — bonus when agents coordinate (e.g., dispatcher sends air ambulance AND ambulance agent also chose air transport → +10 bonus)

---

## 📈 Metrics Tracked

| Metric | Description | Target |
|--------|-------------|--------|
| Cumulative Reward | Total reward per episode — should converge upward | Increasing trend |
| Avg Response Time | Minutes from dispatch to hospital arrival | ≤ 8 min (NFPA 1710 standard) |
| Triage Accuracy | % of cases correctly triaged by severity | > 90% |
| ICU Surge Activations | Times ICU created surge capacity | Optimise per load |
| Hospital Diversions | Times hospital diverted patients | Minimise unnecessary diversions |

---

## 📁 Output Files

```
outputs/
├── training_log.csv        # Per-episode metrics (reward, response time, accuracy)
├── final_policies.csv      # Best action per agent per state (900 × 6 rows)
├── qtable_Ambulance.npy    # Saved Q-table (900 × 5) for deployment/testing
├── qtable_Hospital.npy
├── qtable_Traffic.npy
├── qtable_Dispatcher.npy
├── qtable_Triage.npy
├── qtable_ICU_Manager.npy
└── training_results.png    # 6-panel training visualization
```

---

## 🚀 How to Run

### Requirements
```bash
pip install numpy pandas matplotlib
```

### Run Training
```bash
python emergency_marl.py
```

### Expected Output
```
Episode  200 | Reward:  12543.2 | Resp: 9.8min | TriageAcc: 61.3%
Episode  400 | Reward:  15821.7 | Resp: 8.9min | TriageAcc: 74.2%
Episode  600 | Reward:  18234.5 | Resp: 8.3min | TriageAcc: 82.1%
Episode  800 | Reward:  19876.3 | Resp: 8.0min | TriageAcc: 88.5%
Episode 1000 | Reward:  20541.1 | Resp: 7.7min | TriageAcc: 92.4%
...
✓ Outputs saved: training_log.csv, final_policies.csv, qtable_*.npy
```

---

## 🔬 Real-World Data Connection

The simulation does **not use a dataset as training input** — this is standard practice in RL. Instead, the environment's **probability distributions are calibrated to real-world data**:

```python
# Call type proportions from NEMSIS 2023 Annual Report
CALL_TYPE_PROBS = [0.18, 0.22, 0.20, 0.10, 0.30]
#                  cardiac trauma resp stroke other

# Hospital diversion rate: ~15% (NEMSIS statistic)
HOSP_LOAD_PROBS = [0.55, 0.30, 0.15]
#                  normal  busy  diversion

# Response time targets from NFPA 1710 standard
# Urban BLS on scene: ≤ 4 min | ALS: ≤ 8 min
```

This is how companies like DeepMind (AlphaFold) and Google (traffic routing) build RL systems — they calibrate simulated environments to real-world statistics.

---

## 🔮 Future Improvements

| Improvement | Impact |
|-------------|--------|
| Use actual NEMSIS CSV data to learn transition probabilities (request at nemsis.org) | Highest realism |
| Deep Q-Network (DQN) — neural network instead of lookup table | Handles continuous states |
| MADDPG (Multi-Agent Deep Deterministic Policy Gradient) | True cooperative MARL |
| Add patient outcome data (survival rates per scenario) | Clinical validation |
| Real-time GPS + hospital EMR integration | Production deployment |
| Multi-city training (transfer learning across city models) | Scalability |

---
