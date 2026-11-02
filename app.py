import os, io, json, base64, uuid, pickle, tempfile
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import cm

import store
import recovery as rec_model

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ecg-arrhythmia-secret-2024')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

RESULTS_DIR = os.path.join(tempfile.gettempdir(), 'ecg_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_results(data):
    try:
        cutoff = datetime.now().timestamp() - 7200
        for f in os.listdir(RESULTS_DIR):
            fp = os.path.join(RESULTS_DIR, f)
            if os.path.getmtime(fp) < cutoff:
                os.remove(fp)
    except Exception:
        pass
    rid  = str(uuid.uuid4())
    path = os.path.join(RESULTS_DIR, f'{rid}.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    return rid

def load_results(rid):
    if not rid:
        return None
    path = os.path.join(RESULTS_DIR, f'{rid}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_ecg_model.keras')
ARR_CLF_PATH = os.path.join(os.path.dirname(__file__), 'arrhythmia_classifier.pkl')
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), 'sample_ecg.csv')

CLASSES = ['N', 'S', 'V', 'F', 'Q']
CLASS_LABELS = {
    'N': 'Normal Beat',
    'S': 'Supraventricular Ectopic',
    'V': 'Ventricular Ectopic',
    'F': 'Fusion Beat',
    'Q': 'Unknown / Paced'
}
CLASS_COLORS = {
    'N': '#181717', 'S': '#b45309', 'V': '#d04841', 'F': '#6d28d9', 'Q': '#575555'
}
RISK_MAP = {'N': 'Low', 'S': 'Moderate', 'V': 'High', 'F': 'Moderate', 'Q': 'Low'}

WIN_BEATS = 15
SEVERITY_ORDER = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'High': 3}

ARRHYTHMIA_META = {
    'NSR':      {'name': 'Normal Sinus Rhythm',                      'severity': 'Normal',   'color': '#22c55e',
                 'description': 'Heart beats in a normal, regular rhythm from the sinus node.',
                 'advice': 'No arrhythmia detected. Continue routine monitoring as recommended.'},
    'SBR':      {'name': 'Sinus Bradycardia',                        'severity': 'Mild',     'color': '#3b82f6',
                 'description': 'Heart rate below 60 bpm with regular rhythm from the sinus node.',
                 'advice': 'Common in athletes. Consult a cardiologist if accompanied by dizziness or fatigue.'},
    'STach':    {'name': 'Sinus Tachycardia',                        'severity': 'Mild',     'color': '#f59e0b',
                 'description': 'Heart rate above 100 bpm with regular rhythm.',
                 'advice': 'Often caused by stress, fever, or dehydration. Seek evaluation if persistent.'},
    'PAC':      {'name': 'Premature Atrial Contractions (PAC)',      'severity': 'Mild',     'color': '#f59e0b',
                 'description': 'Extra beats originating in the atria before the normal beat.',
                 'advice': 'Usually benign. Reduce caffeine and alcohol. Consult a doctor if frequent.'},
    'PVC':      {'name': 'Premature Ventricular Contractions (PVC)', 'severity': 'Moderate', 'color': '#ef4444',
                 'description': 'Extra beats originating in the ventricles.',
                 'advice': 'Seek cardiology evaluation if PVCs are frequent or cause symptoms.'},
    'Bigeminy': {'name': 'Ventricular Bigeminy',                     'severity': 'Moderate', 'color': '#ef4444',
                 'description': 'Every other beat is a premature ventricular contraction (N-V-N-V pattern).',
                 'advice': 'Cardiology evaluation recommended. May require Holter monitoring.'},
    'SVT':      {'name': 'Supraventricular Tachycardia (SVT)',       'severity': 'Moderate', 'color': '#f59e0b',
                 'description': 'Rapid heart rate originating above the ventricles.',
                 'advice': 'Medical evaluation required. Treatment may include medication or catheter ablation.'},
    'AFib':     {'name': 'Atrial Fibrillation (AFib)',               'severity': 'High',     'color': '#dc2626',
                 'description': 'Irregular, chaotic atrial activity with highly variable RR intervals.',
                 'advice': 'Seek prompt medical attention. AFib increases stroke risk. Anticoagulation may be needed.'},
    'VT':       {'name': 'Ventricular Tachycardia (VT)',             'severity': 'High',     'color': '#dc2626',
                 'description': 'Rapid rhythm originating in the ventricles — potentially life-threatening.',
                 'advice': 'Urgent medical evaluation required. May require ICD or antiarrhythmic therapy.'},
    'Block':    {'name': 'Heart Block / Conduction Defect',          'severity': 'Moderate', 'color': '#8b5cf6',
                 'description': 'Delayed or blocked electrical conduction through the heart.',
                 'advice': 'Cardiology evaluation needed. Pacemaker may be required for high-degree block.'},
    'Paced':    {'name': 'Paced Rhythm',                             'severity': 'Normal',   'color': '#94a3b8',
                 'description': 'Heartbeats are being generated by an implanted pacemaker.',
                 'advice': 'Continue regular pacemaker follow-up with your cardiologist.'},
}

def extract_window_features(beat_seq, rr_seq, fs=360):
    from collections import Counter as C
    n = len(beat_seq); counts = C(beat_seq); total = max(n, 1)
    pct_N = counts.get('N', 0) / total; pct_S = counts.get('S', 0) / total
    pct_V = counts.get('V', 0) / total; pct_F = counts.get('F', 0) / total; pct_Q = counts.get('Q', 0) / total
    rr = np.array(rr_seq, dtype=np.float64); rr_ms = rr / fs * 1000
    mean_rr = float(np.mean(rr_ms)); std_rr = float(np.std(rr_ms))
    rmssd = float(np.sqrt(np.mean(np.diff(rr_ms)**2))) if len(rr_ms) > 1 else 0
    hr = 60000 / mean_rr if mean_rr > 0 else 75; cv_rr = std_rr / mean_rr if mean_rr > 0 else 0
    min_rr = float(np.min(rr_ms)); max_rr = float(np.max(rr_ms)); rr_range = max_rr - min_rr
    rr_diff = np.abs(np.diff(rr_ms)) if len(rr_ms) > 1 else np.array([0])
    pnn50 = float(np.mean(rr_diff > 50)); pnn20 = float(np.mean(rr_diff > 20))
    irr = float(np.mean(rr_diff / (rr_ms[:-1] + 1e-6) > 0.20)) if len(rr_ms) > 1 else 0
    alt = sum(1 for i in range(1, n - 1) if beat_seq[i] != beat_seq[i - 1] and beat_seq[i] == beat_seq[i + 1]) / max(n - 2, 1)
    tri = sum(1 for i in range(2, n) if beat_seq[i] == beat_seq[i - 2] and beat_seq[i] != beat_seq[i - 1]) / max(n - 2, 1)
    if len(rr_ms) > 3:
        rr_norm = (rr_ms - rr_ms.mean()) / (rr_ms.std() + 1e-8)
        hist, _ = np.histogram(rr_norm, bins=5, density=True); hist = hist[hist > 0]
        rr_ent = float(-np.sum(hist * np.log(hist + 1e-10)))
    else:
        rr_ent = 0.0
    mcv = cur = 0
    for b in beat_seq: cur = cur + 1 if b == 'V' else 0; mcv = max(mcv, cur)
    mcs = cur = 0
    for b in beat_seq: cur = cur + 1 if b == 'S' else 0; mcs = max(mcs, cur)
    mcn = cur = 0
    for b in beat_seq: cur = cur + 1 if b == 'N' else 0; mcn = max(mcn, cur)
    trans = sum(1 for i in range(1, n) if beat_seq[i] != beat_seq[i - 1]) / max(n - 1, 1)
    sv_r = pct_S / (pct_V + 1e-6); vs_r = pct_V / (pct_S + 1e-6)
    abn = (counts.get('S', 0) + counts.get('V', 0) + counts.get('F', 0)) / total
    return [pct_N, pct_S, pct_V, pct_F, pct_Q,
            mean_rr, std_rr, rmssd, hr, cv_rr,
            min_rr, max_rr, rr_range, pnn50, pnn20, irr,
            alt, tri, rr_ent,
            mcv / WIN_BEATS, mcs / WIN_BEATS, mcn / WIN_BEATS,
            trans, sv_r, vs_r, abn,
            counts.get('V', 0), counts.get('S', 0), counts.get('N', 0), n]

def diagnose_arrhythmia(beat_preds, peak_positions, input_fs=360):
    from collections import Counter as C
    counts = C(beat_preds)
    total  = max(len(beat_preds), 1)
    pct_q  = counts.get('Q', 0) / total

    def make(cls, conf=0.0):
        meta = ARRHYTHMIA_META.get(cls, ARRHYTHMIA_META['NSR'])
        return [{'id': cls, 'name': meta['name'], 'severity': meta['severity'],
                 'color': meta['color'], 'description': meta['description'],
                 'advice': meta['advice'], 'confidence': conf}]

    if len(peak_positions) >= 2:
        rr_ms   = np.diff(peak_positions) / input_fs * 1000
        mean_rr = float(np.mean(rr_ms))
        std_rr  = float(np.std(rr_ms))
        cv_rr   = std_rr / mean_rr if mean_rr > 0 else 0
        hr      = 60000 / mean_rr if mean_rr > 0 else 75
        rr_diff = np.abs(np.diff(rr_ms)) if len(rr_ms) > 1 else np.array([0.0])
        irreg   = float(np.mean(rr_diff / (rr_ms[:-1] + 1e-6) > 0.20)) if len(rr_ms) > 1 else 0
        pnn50   = float(np.mean(rr_diff > 50)) if len(rr_diff) > 0 else 0
    else:
        mean_rr = 800; std_rr = 0; cv_rr = 0; hr = 75; irreg = 0; pnn50 = 0

    if len(peak_positions) >= 8:
        rr_arr = np.diff(peak_positions) / input_fs * 1000
        if len(rr_arr) >= 6:
            ev = rr_arr[::2]; od = rr_arr[1::2]
            if (len(ev) > 2 and len(od) > 2 and
                    abs(np.mean(ev) - np.mean(od)) > 100 and
                    np.std(ev) < 60 and np.std(od) < 60):
                return make('Bigeminy', 88.0)

    if cv_rr > 0.50 and irreg > 0.70:
        return make('AFib', 90.0)

    if cv_rr > 0.50 and irreg > 0.40:
        return make('VT', 85.0)
    if mean_rr < 400 and cv_rr < 0.15:
        return make('VT', 82.0)

    if cv_rr > 0.20 and irreg > 0.50:
        return make('PVC', 82.0)

    if cv_rr > 0.20 and irreg > 0.20 and pnn50 > 0.50:
        return make('PAC', 75.0)

    if pct_q > 0.30:
        return make('Paced', 85.0)
    if 0.07 < cv_rr < 0.18 and irreg < 0.20 and 55 < hr < 90:
        return make('Paced', 72.0)

    if hr > 85 and cv_rr > 0.08:
        return make('SVT', 72.0)

    if hr < 65 and cv_rr < 0.08 and irreg < 0.05:
        return make('Block', 75.0)

    if hr < 60:
        return make('SBR', 72.0)

    if hr > 100 and cv_rr < 0.10:
        return make('STach', 75.0)

    if 60 <= hr <= 100 and cv_rr < 0.08:
        return make('NSR', 82.0)

    if len(beat_preds) < WIN_BEATS + 1:
        if hr > 100:  return make('STach', 55.0)
        elif hr < 60: return make('SBR', 55.0)
        else:         return make('NSR', 55.0)

    try:
        feats = []
        for i in range(len(beat_preds) - WIN_BEATS):
            rr = np.diff(peak_positions[i:i+WIN_BEATS+1]) * (360.0 / input_fs)
            feats.append(extract_window_features(beat_preds[i:i+WIN_BEATS], rr, fs=360))

        X = np.array(feats, dtype=np.float32)
        if arr_gbm is not None:
            mean_probs = (0.6 * arr_gbm.predict_proba(X) + 0.4 * arr_rf.predict_proba(X)).mean(axis=0)
        elif arr_clf is not None:
            mean_probs = arr_clf.predict_proba(X).mean(axis=0)
        else:
            return make('NSR', 60.0)

        diagnoses = []
        for idx in np.argsort(mean_probs)[::-1]:
            conf = float(mean_probs[idx])
            if conf < 0.03 and len(diagnoses) >= 1:
                break
            cls  = arr_encoder.classes_[idx]
            meta = ARRHYTHMIA_META.get(cls, {})
            diagnoses.append({
                'id': cls, 'name': meta.get('name', cls),
                'severity': meta.get('severity', 'Unknown'),
                'color': meta.get('color', '#94a3b8'),
                'description': meta.get('description', ''),
                'advice': meta.get('advice', ''),
                'confidence': round(conf * 100, 1),
            })
        diagnoses.sort(key=lambda x: (SEVERITY_ORDER.get(x['severity'], 0), x['confidence']), reverse=True)
        return diagnoses if diagnoses else make('NSR', 50.0)
    except Exception as e:
        print(f"[NOTE] Classifier fallback: {e}")
        return make('NSR', 50.0)

BEFORE    = 90
AFTER     = 110
FS        = 360
TARGET_LEN = BEFORE + AFTER

# Beat model load with fallback
model = None
if os.path.exists(MODEL_PATH):
    try:
        import keras
        model = keras.saving.load_model(MODEL_PATH)
        print("[OK] Beat model loaded via Keras")
    except Exception:
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(MODEL_PATH)
            print("[OK] Beat model loaded via TensorFlow")
        except Exception as e:
            print(f"[NOTE] Deep beat model fallback to morphological heuristics ({e})")
            model = None

# Arrhythmia ensemble classifier load
arr_clf = None
arr_gbm = None
arr_rf  = None
arr_encoder = None
if os.path.exists(ARR_CLF_PATH):
    try:
        with open(ARR_CLF_PATH, 'rb') as f:
            _arr = pickle.load(f)
        if 'gbm' in _arr:
            arr_gbm     = _arr['gbm']
            arr_rf      = _arr['rf']
            arr_encoder = _arr['encoder']
            print("[OK] Arrhythmia ensemble classifier loaded")
        else:
            arr_clf     = _arr.get('model')
            arr_encoder = _arr.get('encoder')
            print("[OK] Arrhythmia single classifier loaded")
    except Exception as e:
        print(f"[NOTE] Arrhythmia classifier load deferred: {e}")

def infer_fs(time_arr):
    if len(time_arr) < 2:
        return FS
    diffs = np.diff(time_arr[:min(100, len(time_arr))])
    dt    = np.median(diffs)
    return round(1.0 / dt) if dt > 0 else FS

def detect_peaks(signal, fs=360):
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.ndimage import uniform_filter1d

    sig = signal.astype(np.float64)
    nyq  = fs / 2.0
    b, a = butter(2, [0.5 / nyq, min(40.0 / nyq, 0.99)], btype='band')
    filtered = filtfilt(b, a, sig)

    if np.abs(np.min(filtered)) > np.abs(np.max(filtered)):
        filtered = -filtered

    squared = filtered ** 2
    win   = int(1.5 * fs)
    step  = int(0.15 * fs)
    n     = len(squared)
    local_thresh = np.zeros(n)

    for start in range(0, n, step):
        end = min(start + win, n)
        local_thresh[start:end] = np.maximum(
            local_thresh[start:end],
            0.40 * np.max(squared[start:end])
        )

    local_thresh = uniform_filter1d(local_thresh, size=win)
    min_dist = int(0.2 * fs)
    peaks, _ = find_peaks(squared, distance=min_dist)
    peaks = np.array([p for p in peaks if squared[p] > local_thresh[p]])

    expected_min = max(3, int(len(signal) / fs / 1.5))
    if len(peaks) < expected_min:
        global_thresh = 0.10 * np.max(squared)
        peaks, _ = find_peaks(squared, height=global_thresh, distance=min_dist)

    if len(peaks) > 0:
        heights = squared[peaks]
        median_h = np.median(heights)
        peaks = peaks[heights >= 0.10 * median_h]

    return peaks.tolist(), filtered.astype(np.float32)

def resample_beat(beat, target_len=TARGET_LEN):
    from scipy.signal import resample as sp_resample
    return sp_resample(beat, target_len).astype(np.float32)

def predict_beats(signal_arr, input_fs=360):
    peaks, filtered = detect_peaks(signal_arr, fs=input_fs)
    before = int(BEFORE * input_fs / FS)
    after  = int(AFTER  * input_fs / FS)

    beats, positions = [], []
    for peak in peaks:
        start, end = peak - before, peak + after
        if start < 0 or end > len(filtered):
            continue
        beat = filtered[start:end]
        if len(beat) != TARGET_LEN:
            beat = resample_beat(beat, TARGET_LEN)
        std = beat.std()
        if std < 1e-6:
            continue
        beat = (beat - beat.mean()) / (std + 1e-8)
        beats.append(beat.astype(np.float32))
        positions.append(int(peak))

    if not beats:
        return [], [], [], []

    X = np.array(beats, dtype=np.float32)[..., np.newaxis]
    if model is not None:
        probs = model.predict(X, verbose=0)
    else:
        probs = []
        for b in beats:
            above_half = np.sum(b > 0.5)
            if above_half > 25:
                probs.append([0.10, 0.10, 0.70, 0.05, 0.05])
            else:
                probs.append([0.85, 0.05, 0.05, 0.02, 0.03])
        probs = np.array(probs, dtype=np.float32)

    preds = []
    confidences = []
    for i, p in enumerate(probs):
        cls  = CLASSES[np.argmax(p)]
        conf = float(np.max(p))
        beat = beats[i]
        above_half = np.sum(beat > 0.5)
        t_area = np.sum(beat[120:180])
        r_area = np.sum(beat[70:110])
        t_inverted = (t_area < 0) and (r_area > 0)
        wide_qrs   = above_half > 25

        if conf < 0.50 and wide_qrs and t_inverted:
            cls  = 'V'
            conf = 0.55
        elif conf < 0.50 and wide_qrs:
            cls  = 'V'
            conf = 0.45

        preds.append(cls)
        confidences.append(conf)

    return positions, preds, confidences, probs.tolist()

def compute_hrv(peaks, fs=360):
    if len(peaks) < 2:
        return {'mean_rr': 0, 'sdnn': 0, 'rmssd': 0, 'hr': 0}
    rr = np.diff(peaks) / fs * 1000
    return {
        'mean_rr': round(float(np.mean(rr)), 1),
        'sdnn':    round(float(np.std(rr)), 1),
        'rmssd':   round(float(np.sqrt(np.mean(np.diff(rr)**2))), 1) if len(rr) > 1 else 0,
        'hr':      round(60000 / float(np.mean(rr)), 1) if np.mean(rr) > 0 else 0
    }

def make_ecg_plot(time, signal, peaks, preds):
    fig, ax = plt.subplots(figsize=(14, 3.5), dpi=90)
    ax.plot(time, signal, color='#5a82de', linewidth=1.0, alpha=0.95)
    for i, peak in enumerate(peaks):
        if peak < len(time):
            c = CLASS_COLORS.get(preds[i], '#94a3b8')
            ax.axvline(x=time[peak], color=c, alpha=0.4, linewidth=1)
            ax.scatter(time[peak], signal[peak], color=c, s=30, zorder=5)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('Amplitude (mV)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_pie_chart(beat_counts):
    labels = [f"{k} — {CLASS_LABELS[k]}" for k in beat_counts]
    sizes  = list(beat_counts.values())
    clrs   = [CLASS_COLORS[k] for k in beat_counts]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=90)
    wedges, _, autotexts = ax.pie(sizes, labels=None, colors=clrs,
                                   autopct='%1.1f%%', startangle=90,
                                   pctdistance=0.75,
                                   wedgeprops=dict(width=0.55))
    for at in autotexts:
        at.set_fontsize(9)
    ax.legend(wedges, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=1)
    ax.set_title('Beat Distribution', fontsize=11, fontweight='bold', pad=10)
    fig.patch.set_facecolor('#ffffff')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_rr_plot(peaks, fs=360):
    if len(peaks) < 3:
        return None
    rr = np.diff(peaks) / fs * 1000
    fig, ax = plt.subplots(figsize=(7, 3), dpi=90)
    ax.plot(rr, color='#181717', linewidth=1.2, marker='o', markersize=3)
    ax.axhline(np.mean(rr), color='#d04841', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(rr):.0f} ms')
    ax.fill_between(range(len(rr)), rr, alpha=0.10, color='#5a82de')
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlabel('Beat Index', fontsize=9)
    ax.set_ylabel('RR Interval (ms)', fontsize=9)
    ax.set_title('RR Interval Tachogram', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def make_beat_heatmap(signal_arr, peaks, preds, input_fs=360):
    from scipy.signal import resample as sp_resample
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    BEFORE_S = int(90 * input_fs / 360)
    AFTER_S  = int(110 * input_fs / 360)
    TARGET   = 200

    beat_matrix, valid_preds = [], []
    for i, peak in enumerate(peaks):
        start, end = peak - BEFORE_S, peak + AFTER_S
        if start < 0 or end > len(signal_arr):
            continue
        beat = signal_arr[start:end].astype(np.float64)
        if len(beat) != TARGET:
            beat = sp_resample(beat, TARGET)
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        beat_matrix.append(beat)
        valid_preds.append(preds[i])

    if len(beat_matrix) < 2:
        return None

    beat_matrix = np.array(beat_matrix)
    n_beats     = len(beat_matrix)
    x_ms        = np.linspace(-250, 305, TARGET)

    cls_colors  = {'N':'#22c55e','S':'#f59e0b','V':'#ef4444','F':'#8b5cf6','Q':'#94a3b8'}
    cls_alpha   = {'N': 0.25,    'S': 0.85,    'V': 0.90,    'F': 0.85,   'Q': 0.60}

    n_mask = np.array([p == 'N' for p in valid_preds])
    normal_tpl = beat_matrix[n_mask].mean(axis=0) if n_mask.sum() > 0 else beat_matrix.mean(axis=0)
    anomaly = np.sqrt(((beat_matrix - normal_tpl) ** 2).mean(axis=1))

    fig = plt.figure(figsize=(14, 11), dpi=90)
    fig.patch.set_facecolor('#ffffff')
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2.5, 3, 1.5],
                            width_ratios=[20, 1],
                            hspace=0.45, wspace=0.04)

    ax1  = fig.add_subplot(gs[0, 0])
    ax2  = fig.add_subplot(gs[1, 0])
    cax  = fig.add_subplot(gs[1, 1])
    ax3  = fig.add_subplot(gs[2, 0])

    for cls_order in ['N', 'Q', 'F', 'S', 'V']:
        for i, pred in enumerate(valid_preds):
            if pred != cls_order:
                continue
            lw  = 0.6 if pred == 'N' else 1.4
            ax1.plot(x_ms, beat_matrix[i],
                     color=cls_colors[pred],
                     alpha=cls_alpha[pred],
                     linewidth=lw)

    ax1.plot(x_ms, normal_tpl, color='#1e293b', linewidth=2.0,
             linestyle='--', label='Normal template', zorder=10)

    for xv, lbl in [(-60,'Q'), (0,'R'), (55,'S'), (155,'T')]:
        ax1.axvline(xv, color='#cbd5e1', linewidth=0.8, linestyle=':', zorder=0)
        ax1.text(xv, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 3,
                 lbl, ha='center', va='bottom', fontsize=7.5, color='#94a3b8')

    ax1.set_xlim(x_ms[0], x_ms[-1])
    ax1.set_xlabel('Time relative to R-peak (ms)', fontsize=9)
    ax1.set_ylabel('Amplitude (σ)', fontsize=9)
    ax1.set_title('Beat Morphology Overlay — All beats superimposed by class',
                  fontsize=10, fontweight='bold')
    ax1.set_facecolor('#f8fafc')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.25)

    legend_els = [Line2D([0],[0], color='#1e293b', lw=2, linestyle='--', label='Normal template')]
    legend_els += [Patch(facecolor=cls_colors[k], label=f'{k} — {CLASS_LABELS[k]}')
                   for k in ['N','S','V','F','Q'] if k in set(valid_preds)]
    ax1.legend(handles=legend_els, fontsize=7.5, loc='upper right', framealpha=0.9, ncol=min(len(legend_els), 4))

    vmax = float(np.percentile(np.abs(beat_matrix), 97))
    im   = ax2.imshow(beat_matrix, aspect='auto', cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax,
                      extent=[x_ms[0], x_ms[-1], n_beats - 0.5, -0.5],
                      interpolation='bilinear')

    for i, pred in enumerate(valid_preds):
        if pred != 'N':
            c = cls_colors.get(pred, '#94a3b8')
            ax2.add_patch(plt.Rectangle(
                (x_ms[0], i - 0.5), x_ms[-1] - x_ms[0], 1,
                linewidth=1.5, edgecolor=c, facecolor='none', zorder=3))

    for i, pred in enumerate(valid_preds):
        c = cls_colors.get(pred, '#94a3b8')
        ax2.text(x_ms[-1] + 8, i, pred, ha='left', va='center',
                 fontsize=6.5, fontweight='bold', color=c, clip_on=False)

    for xv, lbl in [(-60,'Q'), (0,'R'), (55,'S'), (155,'T')]:
        ax2.axvline(xv, color='white', linewidth=0.7, linestyle=':', alpha=0.6)

    ax2.set_xlim(x_ms[0], x_ms[-1])
    ax2.set_xlabel('Time relative to R-peak (ms)', fontsize=9)
    ax2.set_ylabel('Beat Index', fontsize=9)
    ax2.set_title('Beat Amplitude Heatmap — Red = high, Blue = low amplitude',
                  fontsize=10, fontweight='bold')
    ax2.set_facecolor('#f8fafc')

    cb = plt.colorbar(im, cax=cax)
    cb.set_label('Amplitude (σ)', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    bar_colors = [cls_colors.get(p, '#94a3b8') for p in valid_preds]
    bars = ax3.bar(range(n_beats), anomaly, color=bar_colors, width=0.8, zorder=3)

    if n_mask.sum() > 1:
        thr = anomaly[n_mask].mean() + 1.5 * anomaly[n_mask].std()
        ax3.axhline(thr, color='#ef4444', linewidth=1.2, linestyle='--',
                    label=f'Anomaly threshold (μ+1.5σ = {thr:.2f})', zorder=4)
        ax3.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
        ax3.fill_between([-0.5, n_beats - 0.5], thr,
                         max(anomaly.max() * 1.1, thr * 1.2),
                         color='#fef2f2', alpha=0.5, zorder=0)

    ax3.set_xlim(-0.5, n_beats - 0.5)
    ax3.set_xlabel('Beat Index', fontsize=9)
    ax3.set_ylabel('Anomaly Score', fontsize=9)
    ax3.set_title('Per-Beat Anomaly Score — Distance from normal template',
                  fontsize=10, fontweight='bold')
    ax3.set_facecolor('#f8fafc')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, axis='y', alpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ─── API ROUTES ──────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'CardioAI Arrhythmia Detection REST API',
        'version': '2.0.0',
        'models': {
            'beat_model': model is not None,
            'arrhythmia_classifier': (arr_gbm is not None) or (arr_clf is not None),
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/sample_ecg.csv')
def serve_sample():
    if os.path.exists(SAMPLE_CSV):
        return send_file(SAMPLE_CSV, as_attachment=True, download_name='sample_ecg.csv', mimetype='text/csv')
    return jsonify({'status': 'error', 'message': 'Sample CSV not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files and request.files['file'].filename:
            f        = request.files['file']
            df       = pd.read_csv(io.StringIO(f.read().decode('utf-8')))
            filename = f.filename
        else:
            if not os.path.exists(SAMPLE_CSV):
                return jsonify({'status': 'error', 'message': 'No input file provided and sample_ecg.csv is missing'}), 400
            df       = pd.read_csv(SAMPLE_CSV)
            filename = 'sample_ecg.csv'

        signal_col = None
        for c in ['signal', 'Signal', 'SIGNAL', 'amplitude', 'Amplitude',
                  'value', 'Value', 'ecg', 'ECG', 'mlii', 'MLII', 'v5', 'V5']:
            if c in df.columns:
                signal_col = c
                break
        if signal_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(f"No numeric signal column found. Columns: {list(df.columns)}")
            signal_col = numeric_cols[-1]

        time_col = None
        for c in ['time', 'Time', 'TIME', 't', 'timestamp', 'Timestamp', 'seconds']:
            if c in df.columns:
                time_col = c
                break

        signal   = df[signal_col].dropna().values.astype(np.float32)
        time_arr = df[time_col].values if time_col else np.arange(len(signal)) / FS
        input_fs = infer_fs(time_arr) if time_col else FS
        time     = time_arr.tolist()

        recording_secs = len(signal) / input_fs
        short_warning = recording_secs < 10.0

        positions, preds, confidences, all_probs = predict_beats(signal, input_fs=input_fs)

        beat_counts = {c: preds.count(c) for c in CLASSES if preds.count(c) > 0}
        dominant    = max(beat_counts, key=beat_counts.get) if beat_counts else 'N'
        hrv         = compute_hrv(positions, fs=input_fs)
        duration    = round(float(time[-1]) - float(time[0]), 2) if len(time) > 1 else 0

        ecg_img  = make_ecg_plot(time, signal.tolist(), positions, preds)
        pie_img  = make_pie_chart(beat_counts) if beat_counts else None
        rr_img   = make_rr_plot(positions, fs=input_fs)
        heat_img = make_beat_heatmap(signal, positions, preds, input_fs=input_fs)

        risk_score = sum(1 for p in preds if p == 'V') / max(len(preds), 1) * 100
        diagnoses  = diagnose_arrhythmia(preds, positions, input_fs=input_fs)
        primary    = diagnoses[0] if diagnoses else {}

        _sev_map = {'High': 'High', 'Moderate': 'Moderate', 'Mild': 'Low', 'Normal': 'Low', 'Unknown': 'Low'}
        if primary:
            risk_level = _sev_map.get(primary.get('severity', 'Normal'), 'Low')
        else:
            risk_level = 'High' if risk_score > 20 else ('Moderate' if risk_score > 5 else 'Low')

        results = {
            'filename':       filename,
            'timestamp':      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration':       duration,
            'input_fs':       input_fs,
            'total_beats':    len(preds),
            'dominant':       dominant,
            'dominant_label': CLASS_LABELS.get(dominant, ''),
            'beat_counts':    beat_counts,
            'risk_level':     risk_level,
            'risk_score':     round(risk_score, 1),
            'short_warning':  short_warning,
            'recording_secs': round(recording_secs, 1),
            'hrv':            hrv,
            'diagnoses':      diagnoses,
            'primary':        primary,
            'peaks':          positions,
            'predictions':    preds,
            'confidences':    [round(c * 100, 1) for c in confidences],
            'all_probs':      [[round(p * 100, 1) for p in row] for row in all_probs],
            'ecg_img':        ecg_img,
            'pie_img':        pie_img,
            'rr_img':         rr_img,
            'heat_img':       heat_img,
            'time':           time,
            'signal':         signal.tolist(),
            'classes':        CLASSES,
            'class_labels':   CLASS_LABELS,
            'class_colors':   CLASS_COLORS,
        }

        try:
            recovery = rec_model.predict(
                hrv=hrv, beat_counts=beat_counts,
                total_beats=len(preds), risk_score=risk_score,
                diagnoses=diagnoses, risk_level=risk_level,
            )
            results['recovery'] = recovery
        except Exception as _re:
            print(f"[WARNING] Recovery prediction failed: {_re}")
            results['recovery'] = None

        rid = save_results(results)
        session['result_id'] = rid

        patient_id = (request.form.get('patient_id', '') or (request.json.get('patient_id', '') if request.is_json else '')).strip()
        if patient_id:
            store.add_ecg_record(
                patient_id     = patient_id,
                filename       = filename,
                total_beats    = len(preds),
                dominant_rhythm= dominant,
                risk_level     = risk_level,
                risk_score     = risk_score,
                diagnoses      = diagnoses,
                hrv            = hrv,
                beat_counts    = beat_counts,
                recovery_days  = results['recovery']['days'] if results.get('recovery') is not None else None,
            )

        return jsonify({
            'status': 'ok',
            'result_id': rid,
            'data': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/results/<result_id>')
def get_result(result_id):
    data = load_results(result_id)
    if not data:
        return jsonify({'status': 'error', 'message': 'Result not found or expired'}), 404
    return jsonify({'status': 'ok', 'result_id': result_id, 'data': data})

@app.route('/api/patients', methods=['GET', 'POST'])
def api_patients():
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        pid = store.add_patient(
            doctor_id       = 'default',
            full_name       = data.get('full_name', '').strip(),
            dob             = data.get('dob', ''),
            gender          = data.get('gender', ''),
            address         = data.get('address', ''),
            mobile_no       = data.get('mobile_no', ''),
            email           = data.get('email', ''),
            disease         = data.get('disease', ''),
            medical_history = data.get('medical_history', ''),
        )
        return jsonify({'status': 'ok', 'patient_id': pid})

    patients = store.get_patients()
    cutoff   = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    return jsonify({
        'status': 'ok',
        'patients': patients,
        'recent_count': sum(1 for p in patients if p.get('created_at', '')[:10] >= cutoff),
        'high_risk_count': sum(1 for p in patients if p.get('ecg_records') and p['ecg_records'][0].get('risk_level') == 'High'),
        'total_records': sum(len(p.get('ecg_records', [])) for p in patients)
    })

@app.route('/api/patients/<patient_id>')
def api_patient_detail(patient_id):
    patient = store.get_patient(patient_id)
    if not patient:
        return jsonify({'status': 'error', 'message': 'Patient not found'}), 404
    return jsonify({
        'status': 'ok',
        'patient': patient,
        'improvements': store.get_improvements(patient),
        'latest_record': patient['ecg_records'][0] if patient.get('ecg_records') else None
    })

@app.route('/download_report')
def download_report():
    rid  = request.args.get('result_id') or session.get('result_id')
    data = load_results(rid)
    if not data:
        return "No results found for report generation", 404

    pt = {
        'name':    request.args.get('pt_name', '').strip(),
        'phone':   request.args.get('pt_phone', '').strip(),
        'address': request.args.get('pt_address', '').strip(),
        'history': request.args.get('pt_history', '').strip(),
    }

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    doc = SimpleDocTemplate(tmp.name, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    normal_style = styles['Normal']
    h2_style = ParagraphStyle('h2', fontSize=12, fontName='Helvetica-Bold',
                               textColor=colors.HexColor('#1e293b'), spaceAfter=8, spaceBefore=12)

    header_data = [[
        Paragraph('<b><font size=16 color="#181717">CardioAI</font></b><br/>'
                  '<font size=8 color="#94a3b8">ECG Arrhythmia Detection System · MIT-BIH · AAMI Standard</font>',
                  normal_style),
        Paragraph(f'<font size=8 color="#94a3b8">REPORT GENERATED</font><br/>'
                  f'<b>{data["timestamp"]}</b><br/>'
                  f'<font size=8 color="#94a3b8">File: {data["filename"]}</font>',
                  ParagraphStyle('right', alignment=2, fontSize=9))
    ]]
    ht = Table(header_data, colWidths=[10*cm, 6*cm])
    ht.setStyle(TableStyle([
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW',   (0,0), (-1,0),  2, colors.HexColor('#d04841')),
        ('BOTTOMPADDING',(0,0),(-1,0),  12),
    ]))
    story.append(ht)
    story.append(Spacer(1, 0.4*cm))

    if pt['name'] or pt['phone'] or pt['address']:
        story.append(Paragraph("PATIENT INFORMATION",
                                ParagraphStyle('sec', fontSize=8, fontName='Helvetica-Bold',
                                               textColor=colors.HexColor('#94a3b8'),
                                               spaceBefore=4, spaceAfter=6)))
        pt_rows = [
            ['Full Name',  pt['name']  or '—', 'Mobile', pt['phone']  or '—'],
            ['Address',    pt['address'] or '—', '', ''],
        ]
        if pt['history']:
            pt_rows.append(['Medical History', pt['history'], '', ''])

        pt_table = Table(pt_rows, colWidths=[3*cm, 7*cm, 2.5*cm, 3.5*cm])
        pt_table.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('FONTNAME',    (0,0), (0,-1),  'Helvetica-Bold'),
            ('FONTNAME',    (2,0), (2,-1),  'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 9),
            ('TEXTCOLOR',   (0,0), (0,-1),  colors.HexColor('#94a3b8')),
            ('TEXTCOLOR',   (2,0), (2,-1),  colors.HexColor('#94a3b8')),
            ('TEXTCOLOR',   (1,0), (1,-1),  colors.HexColor('#1e293b')),
            ('TEXTCOLOR',   (3,0), (3,-1),  colors.HexColor('#1e293b')),
            ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING',     (0,0), (-1,-1), 8),
            ('SPAN',        (1,1), (3,1)),
        ]))
        if pt['history']:
            pt_table.setStyle(TableStyle([('SPAN', (1,2), (3,2))]))
        story.append(pt_table)
        story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("ECG Arrhythmia Analysis Report",
                            ParagraphStyle('title', fontSize=14, fontName='Helvetica-Bold',
                                           textColor=colors.HexColor('#1e293b'), spaceAfter=12)))

    if data.get('diagnoses'):
        story.append(Paragraph("Arrhythmia Diagnosis", h2_style))
        for d in data['diagnoses']:
            sev_colors = {'High':'#fef2f2','Moderate':'#fffbeb','Mild':'#eff6ff','Normal':'#f0fdf4'}
            bg = sev_colors.get(d['severity'], '#f8fafc')
            diag_rows = [
                [Paragraph(f'<b>{d["name"]}</b>', normal_style),
                 Paragraph(f'<b>{d["severity"]}</b>  {d.get("confidence","")}{"%" if d.get("confidence") else ""}',
                           ParagraphStyle('sev', fontSize=9, textColor=colors.HexColor(d['color'])))],
                [Paragraph(d['description'], ParagraphStyle('desc', fontSize=9, textColor=colors.HexColor('#475569'))), ''],
                [Paragraph(f'<b>Advice:</b> {d["advice"]}',
                           ParagraphStyle('adv', fontSize=9, textColor=colors.HexColor('#334155'))), ''],
            ]
            dt = Table(diag_rows, colWidths=[12*cm, 4*cm])
            dt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg)),
                ('LINEAFTER',  (0,0), (0,-1),  2, colors.HexColor(d['color'])),
                ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#e2e8f0')),
                ('PADDING',    (0,0), (-1,-1), 8),
                ('SPAN',       (0,1), (1,1)),
                ('SPAN',       (0,2), (1,2)),
            ]))
            story.append(dt)
            story.append(Spacer(1, 0.25*cm))
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Clinical Summary", h2_style))
    summary_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['Total Beats',      str(data['total_beats']),   'Heart Rate',    f"{data['hrv'].get('hr','N/A')} bpm"],
        ['Duration',         f"{data['duration']} s",    'Mean RR',       f"{data['hrv'].get('mean_rr','N/A')} ms"],
        ['Dominant Rhythm',  f"{data['dominant']} — {data['dominant_label']}", 'SDNN', f"{data['hrv'].get('sdnn','N/A')} ms"],
        ['Risk Level',       data['risk_level'],          'RMSSD',         f"{data['hrv'].get('rmssd','N/A')} ms"],
        ['Sampling Rate',    f"{data.get('input_fs',360)} Hz", '', ''],
    ]
    st = Table(summary_data, colWidths=[4*cm, 4.5*cm, 4*cm, 3.5*cm])
    st.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#d04841')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',      (2,1), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,1), (0,-1), colors.HexColor('#64748b')),
        ('TEXTCOLOR',     (2,1), (2,-1), colors.HexColor('#64748b')),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',       (0,0), (-1,-1), 8),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Beat Classification Summary", h2_style))
    beat_data = [['Class', 'Label', 'Count', 'Percentage', 'Risk']]
    total = data['total_beats']
    for cls, cnt in data['beat_counts'].items():
        pct  = round(cnt / total * 100, 1)
        risk = {'N':'Low','S':'Moderate','V':'High','F':'Moderate','Q':'Low'}.get(cls,'')
        beat_data.append([cls, CLASS_LABELS.get(cls,''), str(cnt), f"{pct}%", risk])
    bt = Table(beat_data, colWidths=[2*cm, 6.5*cm, 2.5*cm, 3*cm, 2*cm])
    bt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f1f5f9'), colors.white]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('PADDING',       (0,0), (-1,-1), 7),
    ]))
    story.append(bt)
    story.append(Spacer(1, 0.4*cm))

    if data.get('ecg_img'):
        story.append(Paragraph("ECG Signal with Beat Annotations", h2_style))
        img = Image(io.BytesIO(base64.b64decode(data['ecg_img'])), width=15*cm, height=3.5*cm)
        story.append(img)
        story.append(Spacer(1, 0.4*cm))

    row_imgs = []
    if data.get('pie_img'):
        row_imgs.append(Image(io.BytesIO(base64.b64decode(data['pie_img'])), width=7*cm, height=5.5*cm))
    if data.get('rr_img'):
        row_imgs.append(Image(io.BytesIO(base64.b64decode(data['rr_img'])), width=7*cm, height=5.5*cm))
    if row_imgs:
        story.append(Table([row_imgs], colWidths=[7.5*cm]*len(row_imgs)))
        story.append(Spacer(1, 0.4*cm))

    if data.get('heat_img'):
        story.append(Paragraph("Beat Morphology Heatmap", h2_style))
        story.append(Paragraph(
            "Each row represents one beat. Color shows deviation from the normal template "
            "(red = above normal, blue = below normal). Abnormal beats are marked with dashed lines.",
            ParagraphStyle('sub', fontSize=8, textColor=colors.HexColor('#64748b'), spaceAfter=6)))
        hmap = Image(io.BytesIO(base64.b64decode(data['heat_img'])), width=15*cm, height=7*cm)
        story.append(hmap)
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(
        "Disclaimer: This report is generated by an AI model for research and educational purposes only. "
        "It is not a substitute for professional medical diagnosis or clinical evaluation. "
        "Always consult a qualified cardiologist for medical decisions.",
        ParagraphStyle('disc', fontSize=8, textColor=colors.HexColor('#94a3b8'),
                       borderPadding=8, backColor=colors.HexColor('#f8fafc'))))

    doc.build(story)
    return send_file(tmp.name, as_attachment=True,
                     download_name=f"ECG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                     mimetype='application/pdf')

_corrections = []

@app.route('/api/correct', methods=['POST'])
def api_correct():
    import re as _re
    body     = request.get_json(silent=True) or {}
    ai_pred  = _re.sub(r'<[^>]+>', '', str(body.get('ai_prediction',     '')).strip())[:120]
    doc_corr = _re.sub(r'<[^>]+>', '', str(body.get('doctor_correction', '')).strip())[:80]
    rid      = body.get('result_id') or session.get('result_id')

    if not doc_corr:
        return jsonify({'status': 'error', 'message': 'Correction cannot be empty.'}), 400

    entry = {
        'ai_prediction':     ai_pred,
        'doctor_correction': doc_corr,
        'doctor_id':         'default',
        'result_id':         rid,
        'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    _corrections.append(entry)

    if rid:
        try:
            saved = load_results(rid)
            if saved:
                saved['doctor_correction']    = doc_corr
                saved['doctor_correction_ts'] = entry['timestamp']
                path = os.path.join(RESULTS_DIR, f'{rid}.json')
                with open(path, 'w') as _f:
                    json.dump(saved, _f)
        except Exception:
            pass

    return jsonify({
        'status':            'ok',
        'ai_prediction':     ai_pred,
        'doctor_correction': doc_corr,
        'timestamp':         entry['timestamp'],
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
