"""
rewards.py
==========
Reward functions for each agent, plus a cooperative shared reward.

Individual rewards encode clinically optimal decisions based on:
  - NFPA 1710 response time standards (ALS on-scene ≤ 8 min)
  - Hospital diversion protocols
  - START triage methodology (immediate/delayed/minimal/expectant)
  - Door-to-balloon time for cardiac (≤ 90 min target)

Shared reward gives a bonus when agents coordinate correctly —
e.g. dispatcher sends air ambulance AND ambulance also chose air transport.
"""

import random

# Weight of the cooperative bonus in the total reward signal
SHARED_REWARD_WEIGHT = 0.3


def individual_reward(agent, sev, call, traffic, hosp, time, action):
    """
    Return the reward for a single agent's action in a given state.

    Parameters
    ----------
    agent   : str  — agent name
    sev     : int  — severity (0=low, 1=moderate, 2=critical)
    call    : int  — call type (0=cardiac,1=trauma,2=resp,3=stroke,4=other)
    traffic : int  — traffic level (0=clear … 3=gridlock)
    hosp    : int  — hospital load (0=normal,1=busy,2=diversion)
    time    : int  — time slot (0=night … 4=late-night)
    action  : int  — chosen action (0–4)

    Returns
    -------
    float — reward value (noisy)
    """
    r = 0

    if agent == "Ambulance":
        if sev == 2 and traffic >= 2 and action == 2:   # air transport: critical + gridlock
            r += 20
        elif sev == 2 and traffic < 2 and action == 0:  # fastest route: critical + clear
            r += 15
        elif traffic >= 2 and action == 1:              # alternate route: heavy traffic
            r += 10
        elif sev <= 1 and action == 3:                  # stabilise on-scene: not critical
            r += 8
        elif action == 4 and sev == 2 and hosp == 2:   # request backup: critical + diversion
            r += 12
        else:
            r -= 4

    elif agent == "Hospital":
        if hosp == 2 and action == 3:                   # divert when full
            r += 15
        elif hosp < 2 and action == 0:                  # accept when capacity available
            r += 10
        elif call == 0 and sev == 2 and action == 2:    # cath lab: critical cardiac
            r += 18
        elif call == 1 and sev == 2 and action == 1:    # trauma bay: critical trauma
            r += 18
        elif call == 3 and sev == 2 and action == 4:    # specialist: stroke
            r += 16
        else:
            r -= 3

    elif agent == "Traffic":
        if traffic >= 2 and action == 1:                # green wave: heavy traffic
            r += 12
        elif traffic == 3 and action == 2:              # close intersection: gridlock
            r += 14
        elif traffic <= 1 and action == 0:              # no action: already clear
            r += 5
        elif time in [1, 3] and traffic >= 2 and action == 3:  # highway alert: rush hour
            r += 10
        else:
            r -= 2

    elif agent == "Dispatcher":
        if sev == 2 and call in [0, 3] and action == 2:  # air ambulance: critical cardiac/stroke
            r += 18
        elif sev == 2 and action == 1:                    # double unit: any critical
            r += 14
        elif call == 1 and action == 3:                   # fire support: trauma
            r += 10
        elif sev == 0 and action == 0:                    # single unit: low severity
            r += 8
        else:
            r -= 3

    elif agent == "Triage":
        # Based on START triage methodology
        if sev == 2 and action == 0:                    # immediate (red): critical
            r += 18
        elif sev == 1 and action == 1:                  # delayed (yellow): moderate
            r += 12
        elif sev == 0 and action == 2:                  # minimal (green): low
            r += 8
        elif action == 4 and hosp == 1:                 # reassess: busy hospital
            r += 10
        else:
            r -= 5   # wrong triage is the most penalised mistake

    elif agent == "ICU_Manager":
        if hosp == 2 and action == 1:                   # surge bed: at capacity
            r += 16
        elif hosp == 2 and action == 2:                 # transfer patient: free up space
            r += 14
        elif sev == 2 and action == 3:                  # call backup staff: critical case
            r += 12
        elif call in [0, 3] and sev == 2 and action == 4:  # activate protocol: cardiac/stroke
            r += 18
        elif hosp < 2 and action == 0:                  # hold bed: normal load
            r += 8
        else:
            r -= 3

    # Small noise to prevent Q-table collapse on deterministic rewards
    r += random.uniform(-1, 1)
    return r


def compute_shared_reward(sev, call, traffic, hosp, time, agent_actions):
    """
    Cooperative bonus awarded when agents coordinate well.
    Simulates improved patient outcome from team alignment.

    Parameters
    ----------
    sev, call, traffic, hosp, time : int — decoded state variables
    agent_actions : dict — {agent_name: action_int} for all agents this step

    Returns
    -------
    float — cooperative bonus (added to each agent's individual reward)
    """
    bonus = 0

    disp  = agent_actions.get("Dispatcher", -1)
    amb   = agent_actions.get("Ambulance",  -1)
    hosp_a = agent_actions.get("Hospital",  -1)
    triage = agent_actions.get("Triage",    -1)
    traf   = agent_actions.get("Traffic",   -1)

    # Dispatcher + Ambulance both chose air for critical cardiac/stroke
    if sev == 2 and disp == 2 and amb == 2:
        bonus += 10

    # Double unit dispatched + ambulance took fastest route
    if sev == 2 and disp == 1 and amb == 0:
        bonus += 8

    # Hospital prepared cath lab AND triage flagged as immediate → cardiac protocol
    if call == 0 and hosp_a == 2 and triage == 0:
        bonus += 12

    # Hospital prepared trauma bay AND triage flagged as immediate
    if call == 1 and hosp_a == 1 and triage == 0:
        bonus += 12

    # Traffic cleared the route AND ambulance used it
    if traffic >= 2 and traf in [1, 2] and amb in [0, 2]:
        bonus += 6

    return bonus
