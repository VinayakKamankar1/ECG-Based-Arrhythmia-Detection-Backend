"""
JSON-file based storage — no database required.
All data lives in ecg-dashboard/data/*.json
"""

import os, json, hashlib, uuid
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DOCTORS_FILE  = os.path.join(DATA_DIR, 'doctors.json')
PATIENTS_FILE = os.path.join(DATA_DIR, 'patients.json')

# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return json.load(f)

def _save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def _now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ── doctors ──────────────────────────────────────────────────────────────────

def create_doctor(username, email, password, full_name,
                  specialization='', license_number='', phone=''):
    doctors = _load(DOCTORS_FILE)
    if any(d['username'] == username or d['email'] == email for d in doctors):
        return None
    doc = {
        'id':              str(uuid.uuid4()),
        'username':        username,
        'email':           email,
        'password_hash':   _hash(password),
        'full_name':       full_name,
        'specialization':  specialization,
        'license_number':  license_number,
        'phone':           phone,
        'created_at':      _now(),
    }
    doctors.append(doc)
    _save(DOCTORS_FILE, doctors)
    return doc['id']

def authenticate_doctor(username, password):
    for d in _load(DOCTORS_FILE):
        if d['username'] == username and d['password_hash'] == _hash(password):
            return d
    return None

def get_doctor(doctor_id):
    for d in _load(DOCTORS_FILE):
        if d['id'] == doctor_id:
            return d
    return None

# ── patients ─────────────────────────────────────────────────────────────────

def add_patient(doctor_id='default', full_name='', dob='', gender='', address='',
                mobile_no='', email='', disease='', medical_history=''):
    patients = _load(PATIENTS_FILE)
    p = {
        'id':              str(uuid.uuid4()),
        'doctor_id':       doctor_id,
        'full_name':       full_name,
        'dob':             dob,
        'gender':          gender,
        'address':         address,
        'mobile_no':       mobile_no,
        'email':           email,
        'disease':         disease,
        'medical_history': medical_history,
        'created_at':      _now(),
        'appointments':    [],   # list of {date, notes, status}
        'ecg_records':     [],   # list of {date, filename, beats, rhythm, risk, diagnoses, hrv}
    }
    patients.append(p)
    _save(PATIENTS_FILE, patients)
    return p['id']

def get_patients(doctor_id=None):
    if not doctor_id:
        return _load(PATIENTS_FILE)
    pts = [p for p in _load(PATIENTS_FILE) if p.get('doctor_id') == doctor_id]
    return pts if pts else _load(PATIENTS_FILE)

def get_patient(patient_id):
    for p in _load(PATIENTS_FILE):
        if p['id'] == patient_id:
            return p
    return None

def update_patient(patient_id, **kwargs):
    patients = _load(PATIENTS_FILE)
    for p in patients:
        if p['id'] == patient_id:
            for k, v in kwargs.items():
                if k in p:
                    p[k] = v
            _save(PATIENTS_FILE, patients)
            return True
    return False

def delete_patient(patient_id):
    patients = _load(PATIENTS_FILE)
    patients = [p for p in patients if p['id'] != patient_id]
    _save(PATIENTS_FILE, patients)

def add_appointment(patient_id, date, notes='', status='Completed'):
    patients = _load(PATIENTS_FILE)
    for p in patients:
        if p['id'] == patient_id:
            p['appointments'].insert(0, {
                'id':     str(uuid.uuid4()),
                'date':   date,
                'notes':  notes,
                'status': status,
            })
            _save(PATIENTS_FILE, patients)
            return True
    return False

def add_ecg_record(patient_id, filename, total_beats, dominant_rhythm,
                   risk_level, risk_score, diagnoses, hrv, beat_counts,
                   recovery_days=None):
    patients = _load(PATIENTS_FILE)
    for p in patients:
        if p['id'] == patient_id:
            p['ecg_records'].insert(0, {
                'id':              str(uuid.uuid4()),
                'date':            _now(),
                'filename':        filename,
                'total_beats':     total_beats,
                'dominant_rhythm': dominant_rhythm,
                'risk_level':      risk_level,
                'risk_score':      round(float(risk_score), 1),
                'diagnoses':       diagnoses,
                'hrv':             hrv,
                'beat_counts':     beat_counts,
                'recovery_days':   round(float(recovery_days), 1) if recovery_days is not None else None,
            })
            _save(PATIENTS_FILE, patients)
            return True
    return False

def get_improvements(patient):
    """Compare latest two ECG records."""
    recs = patient.get('ecg_records', [])
    if len(recs) < 2:
        return None
    latest, prev = recs[0], recs[1]
    risk_order = {'Low': 1, 'Moderate': 2, 'High': 3}
    return {
        'total_records': len(recs),
        'latest_date':   latest['date'],
        'prev_date':     prev['date'],
        'risk': {
            'from':     prev['risk_level'],
            'to':       latest['risk_level'],
            'improved': risk_order.get(latest['risk_level'], 2) <= risk_order.get(prev['risk_level'], 2),
        },
        'hrv': {
            'sdnn':  {'from': prev['hrv'].get('sdnn', 0),  'to': latest['hrv'].get('sdnn', 0)},
            'rmssd': {'from': prev['hrv'].get('rmssd', 0), 'to': latest['hrv'].get('rmssd', 0)},
        },
    }
