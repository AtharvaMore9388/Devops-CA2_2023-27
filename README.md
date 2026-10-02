<div align="center">

# 🎭 Deepfake Detection using Deep Learning
### DevOps CA2 — Complete DevOps Pipeline Implementation

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![Django](https://img.shields.io/badge/Django-5.0-green?style=for-the-badge&logo=django)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?style=for-the-badge&logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestrated-326CE5?style=for-the-badge&logo=kubernetes)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions)

</div>

---

## 👥 Group Details

| Field | Details |
|---|---|
| **Subject** | DevOps |
| **Assignment** | CA2 |
| **Group No.** | **13** |
| **Batch** | 2023–27 |

### 👨‍💻 Group Members

| PRN | Name | Role |
|---|---|---|
| 23070122175 | Deepak Rathod | DevOps & Deployment |
| 23070122189 | Samartha Shrestha | Containerization & Orchestration |
| 24070122508 | Atharva More | ML Pipeline & Django App |
| 23070122266 | Sonali Gupta | Monitoring & Documentation |

---

## 📌 Project Overview

This project implements a **Deepfake Face Detection** web application using Deep Learning, wrapped with a complete **DevOps pipeline** covering CI/CD, Configuration Management, Containerization, Orchestration, and Monitoring.

### 🧠 ML Architecture
- **ResNeXt-50 + LSTM** — Classifies real vs. fake faces from images/video sequences
- **MAE (Masked Autoencoder)** — Identity-preserving face reconstruction for FAKE detections
- **CAM Heatmaps** — Visually highlights manipulated face regions
- **face_recognition (dlib)** — Accurate face crop and detection

### 🌐 Web Stack
- **Django 5.0** — Web framework
- **Gunicorn** — WSGI server
- **Nginx** — Reverse proxy
- **SQLite** — Database

---

## 🗂️ Repository Structure

```
Devops-CA2_2023-27/
│
├── 📁 devops/                              ← All DevOps CA2 deliverables
│   ├── 📁 step1-github-actions/
│   │   └── .github/workflows/
│   │       └── ci-cd.yml                  ← Step 1: CI/CD Pipeline
│   │
│   ├── 📁 step2-ansible/
│   │   ├── playbook.yml                   ← Step 2: Ansible Playbook
│   │   └── inventory.ini                  ← Ansible Inventory
│   │
│   ├── 📁 step3-docker-k8s/
│   │   ├── Dockerfile.prod                ← Step 3: Multi-stage Docker build
│   │   ├── k8s-namespace.yaml             ← K8s Namespace
│   │   ├── k8s-deployment.yaml            ← K8s Deployment (rolling update)
│   │   ├── k8s-service.yaml               ← K8s Service (LoadBalancer)
│   │   ├── k8s-pvcs.yaml                  ← Persistent Volume Claims
│   │   └── k8s-hpa.yaml                   ← Horizontal Pod Autoscaler
│   │
│   ├── 📁 step4-monitoring/
│   │   ├── docker-compose.monitoring.yml  ← Step 4: Monitoring stack
│   │   ├── prometheus.yml                 ← Prometheus scrape config
│   │   ├── grafana-dashboards/
│   │   │   └── deepfake-dashboard.json    ← Grafana dashboard (8 panels)
│   │   └── grafana-provisioning/          ← Auto-provisioning configs
│   │
│   └── 📁 step5-report/
│       └── REPORT.md                      ← Step 5: Reflection & Report
│
├── 📁 reconstruction/                      ← MAE face reconstruction module
│   ├── __init__.py
│   └── mae_modules.py                     ← DFREC architecture (arXiv:2412.07260)
│
└── 📁 deepfake_vid/
    └── Deepfake_detection_using_deep_learning-master/
        └── Django Application/            ← Main Django project
            ├── manage.py
            ├── requirements.txt
            ├── Dockerfile
            ├── ml_app/                    ← Core Django app
            │   ├── views.py               ← Prediction logic
            │   ├── models.py
            │   ├── forms.py
            │   └── urls.py
            ├── project_settings/          ← Django settings
            ├── templates/                 ← HTML templates
            ├── static/                    ← CSS, JS, images
            ├── models/                    ← ML model weights (.pt, .pth)
            ├── uploaded_images/           ← User uploaded images
            └── uploaded_videos/           ← User uploaded videos
```

---

## ⚙️ Step 1 — Deployment Strategy (GitHub Actions CI/CD)

**Tool:** GitHub Actions  
**File:** [`devops/step1-github-actions/.github/workflows/ci-cd.yml`](devops/step1-github-actions/.github/workflows/ci-cd.yml)

### Pipeline Flow

```
Push to main
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  JOB 1      │────▶│  JOB 2           │────▶│  JOB 3              │
│  Lint & Test│     │  Build & Push    │     │  Deploy to K8s      │
│             │     │  Docker → GHCR   │     │  Rolling Update     │
│ • flake8    │     │                  │     │  ↳ Rollback on fail │
│ • manage.py │     │ • Multi-stage    │     │                     │
│   check     │     │ • SHA-tagged     │     │                     │
└─────────────┘     └──────────────────┘     └─────────────────────┘
```

| Stage | Tool | Description |
|---|---|---|
| Trigger | GitHub Actions | On push/PR to main branch |
| Test | flake8 + Django check | Lint Python code, validate Django config |
| Build | Docker Buildx | Multi-stage build, push to GHCR with SHA tag |
| Deploy | kubectl set image | Zero-downtime rolling update |
| Verify | kubectl rollout status | Waits for all pods healthy (120s timeout) |
| Rollback | kubectl rollout undo | Auto-rollback if deployment fails |

**Why GitHub Actions?** Free, native GitHub integration, YAML-based, secrets management built-in.

---

## 🔧 Step 2 — Configuration Management & IaC (Ansible)

**Tool:** Ansible  
**Files:** [`devops/step2-ansible/playbook.yml`](devops/step2-ansible/playbook.yml) | [`devops/step2-ansible/inventory.ini`](devops/step2-ansible/inventory.ini)

### What the Playbook Configures

| Task | Description |
|---|---|
| System Packages | Installs Python 3.10, cmake, OpenCV, Nginx, Supervisor |
| User Management | Creates `deepfake` user/group with restricted permissions |
| Directory Setup | `/opt/deepfake-detection/{images, videos, models, logs}` |
| Python venv | Isolated virtualenv, installs all pip dependencies |
| Environment Config | Writes `.env` file (`DEBUG=False`, `SECRET_KEY`, etc.) |
| Nginx | Reverse proxy with 300s timeout for ML inference |
| Supervisor | Gunicorn daemon with auto-restart on crash |
| Health Check | Verifies app returns HTTP 200 after setup |

### Run the Playbook
```bash
# Install Ansible
pip install ansible

# Run playbook
ansible-playbook -i devops/step2-ansible/inventory.ini devops/step2-ansible/playbook.yml
```

---

## 🐳 Step 3 — Containerization & Orchestration

### Docker (Multi-Stage Build)
**File:** [`devops/step3-docker-k8s/Dockerfile.prod`](devops/step3-docker-k8s/Dockerfile.prod)

```dockerfile
# Stage 1: Builder — installs all Python deps (torch, dlib, OpenCV)
FROM python:3.10-slim AS builder
...

# Stage 2: Runtime — lean final image with only installed packages
FROM python:3.10-slim AS runtime
# Runs as non-root user (UID 1000) for security
```

```bash
# Build the image
docker build -f devops/step3-docker-k8s/Dockerfile.prod -t deepfake-detection:latest .

# Run locally
docker run -p 8000:8000 deepfake-detection:latest
```

### Kubernetes Manifests

| File | Purpose |
|---|---|
| [`k8s-namespace.yaml`](devops/step3-docker-k8s/k8s-namespace.yaml) | Isolates workload in `deepfake-ns` |
| [`k8s-deployment.yaml`](devops/step3-docker-k8s/k8s-deployment.yaml) | 2 replicas, rolling update, liveness+readiness probes |
| [`k8s-service.yaml`](devops/step3-docker-k8s/k8s-service.yaml) | LoadBalancer port 80 → 8000 |
| [`k8s-pvcs.yaml`](devops/step3-docker-k8s/k8s-pvcs.yaml) | PVCs: images (5Gi), models (10Gi) |
| [`k8s-hpa.yaml`](devops/step3-docker-k8s/k8s-hpa.yaml) | Auto-scales 2→6 pods at CPU 70% |

### Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f devops/step3-docker-k8s/

# Rolling update (zero downtime — maxUnavailable: 0)
kubectl set image deployment/deepfake-detection \
  deepfake-app=ghcr.io/your-username/deepfake-detection-app:sha-abc123 \
  --namespace=deepfake-ns

# Watch rollout
kubectl rollout status deployment/deepfake-detection -n deepfake-ns

# Rollback if needed
kubectl rollout undo deployment/deepfake-detection -n deepfake-ns
```

---

## 📊 Step 4 — Monitoring & Logging (Prometheus + Grafana)

**Files:** [`devops/step4-monitoring/`](devops/step4-monitoring/)

### Monitoring Stack

| Tool | Role | Port |
|---|---|---|
| **Prometheus** | Metrics scraping & storage | 9090 |
| **Grafana** | Dashboards & alerting | 3000 |
| **Node Exporter** | Host CPU/RAM/disk metrics | 9100 |
| **cAdvisor** | Container resource metrics | 8080 |
| **django-prometheus** | App `/metrics` endpoint | 8000 |

### Start Monitoring Stack
```bash
cd devops/step4-monitoring
docker compose -f docker-compose.monitoring.yml up -d

# Access Grafana
# URL:      http://localhost:3000
# Username: admin
# Password: deepfake123
```

### Grafana Dashboard Panels
1. 🟢 **App Uptime** — UP/DOWN status (green/red)
2. 📈 **HTTP Request Rate** — requests/second by method
3. ⏱️ **P95 Latency** — 95th percentile response time
4. 🔴 **Error Rate** — 4xx and 5xx responses per second
5. 💻 **CPU Usage** — Host utilization gauge (0–100%)
6. 🧠 **Memory Usage** — Host memory gauge (0–100%)
7. 🔢 **Total Predictions** — Business metric counter
8. 🗄️ **DB Query Rate** — Django database operations/s

---

## 📝 Step 5 — Reflection & Report

**File:** [`devops/step5-report/REPORT.md`](devops/step5-report/REPORT.md)

The report covers:
- 🏗️ **Architecture** — System design and DevOps infrastructure diagram
- 🔄 **Pipeline Flow** — GitHub Actions 3-job CI/CD workflow
- 🔧 **Configuration** — Ansible task breakdown and idempotency
- 📦 **Containerization** — Docker multi-stage + Kubernetes zero-downtime deploys
- 📊 **Monitoring** — Prometheus metrics + Grafana dashboard setup
- ⚠️ **Challenges** — Problems faced and how they were solved
- 💡 **Lessons Learned** — Key takeaways from the DevOps implementation

---

## 🚀 Quick Start — Run Locally

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/AtharvaMore9388/Devops-CA2_2023-27.git
cd Devops-CA2_2023-27

# Navigate to Django app
cd "deepfake_vid/Deepfake_detection_using_deep_learning-master/Django Application"

# Install dependencies
pip install -r requirements.txt
pip install django-prometheus

# Run migrations
python manage.py migrate

# Start the server
python manage.py runserver 8000
```

Visit **http://127.0.0.1:8000** 🎉

> **Note:** Place `.pt` model weight files in the `models/` directory before running predictions. Model files are not included in the repo due to their large size (200MB+ each).

---

## 🧪 Challenges & Solutions

| Challenge | Solution |
|---|---|
| `torch` not installed in venv | Used system Python; documented correct environment |
| ML inference takes 30–60 seconds | Set Nginx/Gunicorn timeout to 300s |
| Model weight files too large for GitHub | Used Kubernetes PersistentVolumeClaims — models mounted, not baked in image |
| `face_recognition` (dlib) slow to compile | Pre-built in Docker builder stage with cmake |
| MAE weights missing warning | Graceful fallback to OpenCV inpainting |
| Rolling update caused brief downtime | Fixed with `maxUnavailable: 0` in K8s deployment strategy |
| Prometheus not scraping Django app | Installed `django-prometheus`, added `/metrics` URL and middleware |

---

## 🔬 References

1. DFREC: *DeepFake Identity Recovery Based on Identity-aware Masked Autoencoder* — [arXiv:2412.07260v2](https://arxiv.org/abs/2412.07260), Mar 2025
2. FRG2D with Residual Outlook Attention — ACM Transactions, 2025
3. [Django Documentation](https://docs.djangoproject.com/)
4. [Prometheus Documentation](https://prometheus.io/docs/)
5. [Kubernetes Documentation](https://kubernetes.io/docs/)

---

## 📅 Submission Details

| Item | Detail |
|---|---|
| **GitHub Repo** | https://github.com/AtharvaMore9388/Devops-CA2_2023-27 |
| **Submission Deadline** | 5th October 2026 |
| **Google Sheet Filled** | ✅ Yes |
| **Group Number** | 13 |

---

<div align="center">

**DevOps CA2 | Group 13 | Batch 2023–27**

Made with ❤️ by Deepak Rathod · Samartha Shrestha · Atharva More · Sonali Gupta

</div>
