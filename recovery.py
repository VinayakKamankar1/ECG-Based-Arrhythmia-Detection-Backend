"""
Recovery time prediction.

Uses a rule-based formula (no ML averaging bias) that produces
accurate, varied results based on:
  - Arrhythmia type
  - Risk level (Low / Moderate / High)
  - Heart rate deviation from normal
  - % ventricular beats (risk_score)
  - HRV (low HRV = worse prognosis)
"""

import os, pickle
import numpy as np

_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'recovery_model.pkl')
_bundle = None   # kept for backward compat; not used in rule-based path

# ── Base recovery days per arrhythmia type ────────────────────────────────────
# (low_risk_days, moderate_risk_days, high_risk_days)
_BASE = {
    'NSR':      (0,   1,   2),
    'SBR':      (1,   3,   5),
    'STach':    (1,   3,   6),
    'PAC':      (1,   4,   7),
    'Paced':    (0,   2,   4),
    'PVC':      (2,   5,   9),
    'Bigeminy': (3,   7,  12),
    'SVT':      (4,   8,  14),
    'Block':    (5,  10,  16),
    'AFib':     (6,  12,  18),
    'VT':       (10, 16,  22),
}

_RISK_IDX = {'Low': 0, 'Moderate': 1, 'High': 2}


def _category(days: float) -> dict:
    if days < 5:
        return {'label': 'Fast',     'color': '#22c55e',
                'bg': '#f0fdf4', 'icon': 'fa-bolt'}
    if days < 10:
        return {'label': 'Moderate', 'color': '#f59e0b',
                'bg': '#fffbeb', 'icon': 'fa-clock'}
    return     {'label': 'Slow',     'color': '#ef4444',
                'bg': '#fef2f2', 'icon': 'fa-hourglass-half'}


def predict(hrv: dict, beat_counts: dict, total_beats: int,
            risk_score: float, diagnoses: list,
            risk_level: str = 'Moderate') -> dict | None:
    """
    Returns a dict with keys: days, days_rounded, category, primary_type, breakdown
    """
    primary_type = diagnoses[0]['id'] if diagnoses else 'NSR'
    base_row     = _BASE.get(primary_type, _BASE['NSR'])
    risk_idx     = _RISK_IDX.get(risk_level, 1)
    base_days    = float(base_row[risk_idx])

    # ── Modifiers ─────────────────────────────────────────────────────────────

    # 1. Heart rate penalty: >120 or <45 bpm adds days
    hr = hrv.get('hr', 75)
    if hr > 150:
        hr_mod = 3.0
    elif hr > 120:
        hr_mod = 1.5
    elif hr < 45:
        hr_mod = 2.0
    elif hr < 55:
        hr_mod = 0.5
    else:
        hr_mod = 0.0

    # 2. HRV penalty: very low SDNN (<20ms) = poor autonomic function
    sdnn = hrv.get('sdnn', 40)
    if sdnn < 10:
        hrv_mod = 2.0
    elif sdnn < 20:
        hrv_mod = 1.0
    elif sdnn > 80:
        hrv_mod = -0.5   # good HRV = faster recovery
    else:
        hrv_mod = 0.0

    # 3. Ventricular beat burden (risk_score = % V beats)
    if risk_score > 30:
        v_mod = 3.0
    elif risk_score > 20:
        v_mod = 2.0
    elif risk_score > 10:
        v_mod = 1.0
    elif risk_score > 5:
        v_mod = 0.5
    else:
        v_mod = 0.0

    # 4. Multiple arrhythmia types detected = more complex case
    multi_mod = 0.5 * max(0, len(diagnoses) - 1)

    total_days = base_days + hr_mod + hrv_mod + v_mod + multi_mod
    total_days = max(0.0, round(total_days, 1))

    return {
        'days':         total_days,
        'days_rounded': round(total_days),
        'primary_type': primary_type,
        'risk_level':   risk_level,
        'category':     _category(total_days),
        'breakdown': {
            'base':      base_days,
            'hr_mod':    hr_mod,
            'hrv_mod':   hrv_mod,
            'v_mod':     v_mod,
            'multi_mod': multi_mod,
        },
    }
