# ECG Arrhythmia Detection - REST Backend

A production-ready Flask REST API service for AI-powered ECG arrhythmia detection and patient recovery prediction. Designed for seamless deployment on [Render](https://render.com).

## 🚀 Features
- **Signal Processing & R-Peak Detection:** Local-adaptive thresholding bandpass filter (0.5–40 Hz) and squaring.
- **Beat & Arrhythmia Classification:** Stage-1 beat detection with Stage-2 ensemble classification trained on the MIT-BIH Arrhythmia Database.
- **11 Clinical Arrhythmia Diagnoses:** NSR, SBR, STach, PAC, PVC, Bigeminy, SVT, AFib, VT, Block, Paced.
- **Recovery Time Prediction:** Clinically tuned model assessing arrhythmia severity, HRV, ventricular burden, and heart rate.
- **PDF Report Generation:** ReportLab-powered clinical reports with beat annotations, tachogram, and morphology heatmaps.
- **Patient Records & Improvements:** In-memory & JSON-backed tracking of longitudinal patient ECG history.

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` or `/health` | Healthcheck and service status |
| `POST` | `/predict` | Upload an ECG CSV file (`multipart/form-data`) or use sample fallback; returns full JSON analysis & charts |
| `GET` | `/api/results/<result_id>` | Retrieve a previously computed ECG result by ID |
| `GET` | `/api/patients` | Retrieve list of registered patients and summary stats |
| `POST` | `/api/patients` | Register a new patient profile |
| `GET` | `/api/patients/<patient_id>` | Retrieve patient history and longitudinal improvement metrics |
| `GET` | `/download_report?result_id=<id>` | Download a generated clinical PDF report |
| `POST` | `/api/correct` | Human-in-the-loop diagnostic override / clinician correction |
| `GET` | `/sample_ecg.csv` | Download sample ECG CSV file for testing |

---

## 🛠️ Deploy to Render

1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository:
   `https://github.com/VinayakKamankar1/ECG-Based-Arrhythmia-Detection-Backend.git`
4. Configure the service:
   - **Name:** `ecg-arrhythmia-backend` (or your preferred name)
   - **Environment:** `Python 3`
   - **Branch:** `main`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Plan:** Free
5. Click **Create Web Service**.
6. Once deployed, copy your Render service URL (e.g., `https://ecg-arrhythmia-backend.onrender.com`) to configure your frontend.

---

## 💻 Local Development

```bash
# Clone the repository
git clone https://github.com/VinayakKamankar1/ECG-Based-Arrhythmia-Detection-Backend.git
cd ECG-Based-Arrhythmia-Detection-Backend

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```
The server will start at `http://127.0.0.1:5000`.
