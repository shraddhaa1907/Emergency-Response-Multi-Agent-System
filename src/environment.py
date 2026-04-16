"""
environment.py
==============
State space definition and NEMSIS-calibrated data generator.

State = (severity, call_type, traffic, hospital_load, time_of_day)
        3    x      5        x   4     x      3       x     5      = 900 states

All probability distributions are calibrated to the
NEMSIS 2023 National EMS Dataset (54M+ activations, NHTSA).
"""

import numpy as np

# ── Dimension sizes ────────────────────────────────────────────
SEV_LEVELS  = 3   # 0=low, 1=moderate, 2=critical
CALL_TYPES  = 5   # 0=cardiac, 1=trauma, 2=respiratory, 3=stroke, 4=other
TRAF_LEVELS = 4   # 0=clear, 1=moderate, 2=heavy, 3=gridlock
HOSP_LEVELS = 3   # 0=normal, 1=busy, 2=diversion
TIME_SLOTS  = 5   # 0=night, 1=morning, 2=afternoon, 3=evening, 4=late-night

NUM_STATES  = SEV_LEVELS * CALL_TYPES * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS  # 900

# ── Human-readable labels ──────────────────────────────────────
SEVERITY_LABELS  = ["Low", "Moderate", "Critical"]
CALL_TYPE_LABELS = ["Cardiac", "Trauma", "Respiratory", "Stroke", "Other"]
TRAFFIC_LABELS   = ["Clear", "Moderate", "Heavy", "Gridlock"]
HOSP_LABELS      = ["Normal", "Busy", "Diversion"]
TIME_LABELS      = ["Night", "Morning", "Afternoon", "Evening", "Late-Night"]

# ── NEMSIS 2023 calibrated distributions ──────────────────────
# Call type: cardiac 18%, trauma 22%, respiratory 20%, stroke 10%, other 30%
CALL_TYPE_PROBS = [0.18, 0.22, 0.20, 0.10, 0.30]

# Severity varies by call type (cardiac/stroke skew critical)
SEVERITY_BY_CALL = {
    0: [0.10, 0.40, 0.50],  # cardiac
    1: [0.20, 0.45, 0.35],  # trauma
    2: [0.30, 0.50, 0.20],  # respiratory
    3: [0.15, 0.35, 0.50],  # stroke
    4: [0.50, 0.35, 0.15],  # other
}

# Traffic by time of day (rush hour peaks morning + evening)
TRAFFIC_BY_TIME = {
    0: [0.60, 0.25, 0.10, 0.05],  # night
    1: [0.15, 0.35, 0.35, 0.15],  # morning rush
    2: [0.25, 0.40, 0.25, 0.10],  # afternoon
    3: [0.10, 0.30, 0.40, 0.20],  # evening rush
    4: [0.45, 0.30, 0.15, 0.10],  # late night
}

# Hospital load: ~15% diversion rate (NEMSIS statistic)
HOSP_LOAD_PROBS = [0.55, 0.30, 0.15]

# Time of day: peak in afternoon/evening
TIME_PROBS = [0.12, 0.22, 0.26, 0.24, 0.16]


def encode_state(sev, call, traffic, hosp, time):
    """Convert 5 variables into a single integer state index (0–899)."""
    return (sev  * CALL_TYPES * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS
            + call    * TRAF_LEVELS * HOSP_LEVELS * TIME_SLOTS
            + traffic * HOSP_LEVELS * TIME_SLOTS
            + hosp    * TIME_SLOTS
            + time)


def decode_state(s):
    """Convert integer state index back to (sev, call, traffic, hosp, time)."""
    time    = s % TIME_SLOTS;    s //= TIME_SLOTS
    hosp    = s % HOSP_LEVELS;   s //= HOSP_LEVELS
    traffic = s % TRAF_LEVELS;   s //= TRAF_LEVELS
    call    = s % CALL_TYPES;    s //= CALL_TYPES
    sev     = s
    return sev, call, traffic, hosp, time


def generate_state():
    """
    Sample a realistic emergency scenario using NEMSIS-calibrated distributions.
    Returns a single integer state index.
    """
    time    = np.random.choice(TIME_SLOTS,  p=TIME_PROBS)
    call    = np.random.choice(CALL_TYPES,  p=CALL_TYPE_PROBS)
    sev     = np.random.choice(SEV_LEVELS,  p=SEVERITY_BY_CALL[call])
    traffic = np.random.choice(TRAF_LEVELS, p=TRAFFIC_BY_TIME[time])
    hosp    = np.random.choice(HOSP_LEVELS, p=HOSP_LOAD_PROBS)
    return encode_state(sev, call, traffic, hosp, time)
