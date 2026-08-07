# Final Year Project-Vanshika Tangari-DeepFakeShield

## 🛡️ DeepFakeShield AI - Multimodal Deepfake Detection Platform

**Detect manipulated media using AI-powered analysis of video, audio, and lip-sync patterns.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)

---

## 👤 Student Information

| Field              | Details                              |
| ------------------ | ------------------------------------ |
| **Student Name**   | Vanshika Tangri                      |
| **Student Number** | 2315843                              |
| **Project**        | Final Year Project                   |
| **Project Video**  | https://youtu.be/O3Fyx1679Cw         |

---

## 📖 Project Overview

DeepFakeShield AI is a comprehensive web-based platform designed to detect AI-generated deepfake media. The platform uses a multimodal approach combining video forensics, audio spoof detection, and lip-sync verification to provide accurate authenticity assessments.

### Key Capabilities

- **Real-time Analysis**: Upload videos, images, or audio files for instant deepfake detection
- **Multimodal Detection**: Combines video, audio, and lip-sync analysis for robust results
- **Detailed Forensic Reports**: Generates comprehensive PDF reports with technical findings
- **Modern Web Interface**: Beautiful glassmorphism UI with real-time progress tracking
- **Evidence Timeline**: Visual representation of suspicious segments in media

---

## ✨ Features

| Feature                       | Description                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| 🎬 **Video Forensics**        | ViT-based frame analysis for face manipulation, boundary artifacts, temporal inconsistencies |
| 🔊 **Audio Spoof Detection**  | MFCC analysis, voice cloning detection, spectral pattern recognition                         |
| 👄 **Lip-Sync Verification**  | Audio-visual synchronization analysis, phoneme accuracy, viseme matching                     |
| 🔀 **Multimodal Fusion**      | Calibrated scoring across all modalities using ensemble methods                              |
| 📊 **Evidence Visualization** | Heatmaps, spectrograms, timelines showing detected anomalies                                 |
| 📄 **PDF Reports**            | Detailed forensic reports with technical summaries                                           |

---

## 🚀 Setup Instructions

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
- **PostgreSQL** - [Download PostgreSQL](https://www.postgresql.org/download/)
- **Node.js** (optional, for frontend development)
- **Git** - For cloning the repository

### Option A: Docker Compose (recommended — starts DB, Redis, API, and both Celery workers)

```bash
git clone https://github.com/vtangri/DeepFakeShield.git
cd DeepFakeShield
docker compose up --build
```

This starts Postgres, Redis, the FastAPI backend, and the preprocess/inference Celery workers.
`docker-compose.yml` does **not** include a frontend container, so serve the static frontend
separately in a second terminal:

```bash
cd frontend
python3 -m http.server 8080
```

Then open:
- 🌐 **Frontend**: http://localhost:8080
- 📚 **API Documentation**: http://localhost:8000/docs

### Option B: Manual local setup

#### Step 1: Clone the Repository

```bash
git clone https://github.com/vtangri/DeepFakeShield.git
cd DeepFakeShield
```

#### Step 2: Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
```

#### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

#### Step 4: Configure Environment Variables

```bash
cp .env.example backend/.env
```

Edit `backend/.env` — the variable actually read by `backend/app/core/config.py` is
**`SECRET_KEY`**, not `JWT_SECRET_KEY`. See `.env.example` for the full list (database URL,
Redis URL, storage path, ML model version tags).

#### Step 5: Set Up the Database

The default `DATABASE_URL` in `.env.example` is SQLite (`sqlite+aiosqlite:///backend/prod.db`),
so no separate database server is required for local development. To use PostgreSQL instead,
set `DATABASE_URL=postgresql://user:password@localhost:5432/deepfakeshield` and create the DB
first (`createdb deepfakeshield`). Either way, apply migrations:

```bash
cd backend
alembic upgrade head
```

#### Step 6: Start Redis (required for Celery)

```bash
redis-server
```

#### Step 7: Start the Backend, Celery Workers, and Frontend

Each in its own terminal:

```bash
# Backend API
cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Celery preprocess worker
cd backend && celery -A app.core.celery_app worker --loglevel=info -Q preprocess

# Celery inference worker
cd backend && celery -A app.core.celery_app worker --loglevel=info -Q inference

# Frontend (static file server)
cd frontend && python3 -m http.server 8080
```

#### Step 8: Access the Application

- 🌐 **Frontend**: http://localhost:8080
- 📚 **API Documentation**: http://localhost:8000/docs
- 🔧 **API ReDoc**: http://localhost:8000/redoc

> Analysis jobs require the backend, Redis, **and both** Celery workers running — the API alone
> will accept uploads but jobs will never leave "queued" without the workers.

---

## 📁 Project Structure

```
deepfakeshield/
├── backend/                    # Backend API server
│   ├── app/
│   │   ├── api/routes/         # REST API endpoints
│   │   │   ├── analysis.py     # Analysis job management
│   │   │   ├── auth.py         # Authentication (JWT)
│   │   │   ├── media.py        # File upload handling
│   │   │   └── reports.py      # PDF report generation
│   │   ├── core/               # Core configuration
│   │   │   ├── config.py       # Application settings
│   │   │   ├── security.py     # JWT & password hashing
│   │   │   └── celery_app.py   # Task queue config
│   │   ├── models/             # SQLAlchemy ORM models
│   │   ├── schemas/            # Pydantic validation schemas
│   │   ├── services/           # Business logic services
│   │   │   └── pdf_service.py  # PDF generation
│   │   ├── train/               # Training helper utilities used by the API layer
│   │   ├── workers/             # Celery tasks that call into ml/inference/
│   │   └── main.py             # FastAPI application
│   ├── alembic/                # Database migrations
│   ├── requirements.txt        # Python dependencies
│   └── tests/                  # Unit tests
├── frontend/                   # Web frontend
│   ├── index.html              # Single-page application
│   ├── css/
│   │   └── styles.css          # Glassmorphism theme
│   └── js/
│       └── app.js              # API client & UI logic
├── ml/                         # Machine learning models
│   ├── inference/              # Model inference code (video_forensics, audio_spoof, lipsync, fusion)
│   ├── training/                # Model training scripts (run on Kaggle GPU)
│   ├── datasets/                # Dataset loaders
│   └── models/                  # Trained checkpoints (.pt) — gitignored, downloaded from Kaggle
├── docker-compose.yml          # Docker configuration (backend + workers + DB + Redis; no frontend container)
└── README.md                   # This file
```

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint                | Description               |
| ------ | ----------------------- | ------------------------- |
| POST   | `/api/v1/auth/register` | Register new user account |
| POST   | `/api/v1/auth/login`    | Login and get JWT tokens  |
| GET    | `/api/v1/auth/me`       | Get current user profile  |

### Media Management

| Method | Endpoint               | Description                   |
| ------ | ---------------------- | ----------------------------- |
| POST   | `/api/v1/media/upload` | Upload video/audio/image file |
| GET    | `/api/v1/media/{id}`   | Get media item details        |

### Analysis

| Method | Endpoint                       | Description                 |
| ------ | ------------------------------ | --------------------------- |
| POST   | `/api/v1/analysis/start`       | Start deepfake analysis job |
| GET    | `/api/v1/analysis/{id}/status` | Get job progress status     |
| GET    | `/api/v1/analysis/{id}/result` | Get full analysis results   |

### Reports

| Method | Endpoint                           | Description                  |
| ------ | ---------------------------------- | ---------------------------- |
| GET    | `/api/v1/reports`                  | List all analysis reports    |
| GET    | `/api/v1/analysis/{id}/report.pdf` | Download forensic PDF report |

---

## 🧠 ML Models & Detection Methods

Each service in `ml/inference/` has two operating modes: **TRAINED** (a fine-tuned checkpoint
found in `ml/models/`) and a **fallback mode** based on real, non-learned signal processing —
never a random or keyword-based score. See `ml/inference/*.py` docstrings for exact algorithms.

### Video Analysis (`ml/inference/video_forensics.py`)

| Aspect            | Details                                                                     |
| ----------------- | ---------------------------------------------------------------------------- |
| **Model**         | ViT-B/16 (Vision Transformer) backbone + custom binary classification head  |
| **Training Data** | 140k Real and Fake Faces (Kaggle: xhlulu/140k-real-and-fake-faces)          |
| **Fallback mode** | Inter-frame ViT feature-variance / cosine-distance anomaly scoring (no weights needed) |
| **Detection**     | Facial region inconsistency across frames                                   |
| **Output**        | Per-frame manipulation probability, suspicious frame identification         |

### Audio Analysis (`ml/inference/audio_spoof.py`)

| Aspect            | Details                                                                     |
| ----------------- | ---------------------------------------------------------------------------- |
| **Model**         | Custom 4-block CNN over an 80-bin Mel spectrogram                           |
| **Training Data** | ASVspoof 2019 Logical Access (Kaggle: awsaf49/asvpoof-2019-dataset)         |
| **Fallback mode** | MFCC variance, spectral flatness, harmonic-to-noise ratio, zero-crossing rate (Sahidullah et al., 2015) |
| **Detection**     | Synthetic speech / voice cloning spectral artifacts                         |
| **Output**        | Spoof probability, spectral feature breakdown                               |

### Lip-Sync Analysis (`ml/inference/lipsync.py`)

| Aspect        | Details                                                                       |
| ------------- | -------------------------------------------------------------------------------|
| **Method**    | Cross-correlation between mouth-openness signal and audio RMS energy envelope (Chung & Zisserman, 2016) — no trained network required |
| **Detection** | Sync offset above 80ms between mouth movement and audio                      |
| **Output**    | Sync offset (ms), mismatch score. Returns `null`/`NOT_APPLICABLE` when there is no audio track or too few detected faces — it does not fabricate a score for silent or still-image input. |

---

## 🛠️ Tech Stack

| Layer              | Technologies                               |
| ------------------ | ------------------------------------------ |
| **Backend**        | Python 3.9+, FastAPI, SQLAlchemy, Pydantic |
| **Database**       | PostgreSQL, Redis (caching)                |
| **ML**             | PyTorch, torchaudio, OpenCV, Transformers  |
| **Frontend**       | Vanilla JavaScript, CSS3 (Glassmorphism)   |
| **Authentication** | JWT (JSON Web Tokens)                      |
| **Task Queue**     | Celery (optional, for async processing)    |
| **Deployment**     | Docker, Nginx, Gunicorn                    |

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v --cov=app
```

---

## 📊 Analysis Output Details

The platform provides comprehensive forensic analysis including:

### Video Forensics Output

- Frames analyzed count
- Faces detected
- Manipulation type (face_swap, face_reenactment, lip_sync_manipulation)
- Manipulation method (DeepFaceLab, FaceSwap, FSGAN, etc.)
- Blending score
- Artifact detection (boundary, temporal, color histogram)

### Audio Forensics Output

- Voice cloning detection
- Cloning method identification
- MFCC anomaly score
- Formant consistency
- Speaker embedding distance
- Naturalness score

### Lip-Sync Forensics Output

- Sync offset (milliseconds)
- Correlation score
- Phoneme accuracy
- Viseme match rate

### Technical Metadata

- Models used
- Inference time
- Media resolution, FPS, codec
- File hash (SHA-256)

---

## 🔒 Security Features

- **JWT Authentication** - Secure token-based authentication
- **Password Hashing** - bcrypt with salt
- **CORS Protection** - Configurable allowed origins
- **Input Validation** - Pydantic schema validation
- **File Validation** - MIME type and size verification

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments & References

- Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* — ViT-B/16 backbone used in video forensics.
- Chung, J. S., & Zisserman, A. (2016). *Out of Time: Automated Lip Sync in the Wild.* — cross-correlation approach used in lip-sync verification.
- Sahidullah, M., Kinnunen, T., & Cser, R. (2015). *A Comparison of Features for Synthetic Speech Detection.* — spectral features used in the audio spectral-mode fallback.
- 140k Real and Fake Faces (Kaggle: xhlulu/140k-real-and-fake-faces) — video model training data.
- ASVspoof 2019 (Kaggle: awsaf49/asvpoof-2019-dataset) — audio model training data.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization.* — optimizer (AdamW) used for both training runs.
- FastAPI for the web framework; PyTorch/torchvision/torchaudio for deep learning infrastructure.

---

**Built with ❤️ for media authenticity**

_DeepFakeShield AI - Protecting Truth in the Age of Synthetic Media_
