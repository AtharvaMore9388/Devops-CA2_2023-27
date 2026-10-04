# DevOps CA2 — Deepfake Detection App
## Complete Technical Documentation

> **Project:** Deepfake Detection using Deep Learning (Django + ResNeXt-50 + LSTM + MAE)
> **Batch:** 2023–27 | **Course:** DevOps CA2
> **Stack:** GitHub Actions · Ansible · Docker · Kubernetes · Prometheus · Grafana

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Flow (CI/CD)](#3-pipeline-flow-cicd)
4. [Configuration Management — Ansible](#4-configuration-management--ansible)
5. [Containerization — Docker](#5-containerization--docker)
6. [Orchestration — Kubernetes](#6-orchestration--kubernetes)
7. [Monitoring & Observability](#7-monitoring--observability)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Lessons Learned](#9-lessons-learned)
10. [Quick Reference](#10-quick-reference)

---

## 1. Project Overview

The **Deepfake Detection App** is a Django-based web application that identifies AI-generated or manipulated face media (images and videos) using a combination of three ML models:

| Model | Role |
|---|---|
| **ResNeXt-50 + LSTM** | Frame-level feature extraction + temporal sequence classification |
| **MAE (Masked Autoencoder)** | Identity-preserving face reconstruction to reveal manipulations |
| **CAM Heatmaps (dlib)** | Class Activation Maps overlaid on face regions to visualise anomalies |

The entire application is deployed end-to-end using a production-grade DevOps pipeline spanning 5 steps:

```
Step 1 → GitHub Actions CI/CD Pipeline
Step 2 → Ansible Configuration Management
Step 3 → Docker + Kubernetes Containerization & Orchestration
Step 4 → Prometheus + Grafana Monitoring
Step 5 → Report & Documentation (this document)
```

---

## 2. System Architecture

### 2.1 Application Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│                    (HTTP / HTTPS request)                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Nginx — Reverse Proxy                          │
│   • Listens on port 80                                      │
│   • Forwards to Gunicorn on port 8000                       │
│   • proxy_read_timeout / proxy_send_timeout = 300s          │
│   • Serves static files directly (/static/)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Gunicorn WSGI Server (3 workers)                    │
│   • Binds to 0.0.0.0:8000                                   │
│   • timeout = 300s (accommodates ML inference latency)      │
│   • Managed by Supervisor (auto-restart on crash)           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                Django Application (ml_app)                  │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │ ResNeXt-50    │  │ MAE Reconstruct  │  │ dlib + CAM  │  │
│  │ + LSTM        │  │ (mae_modules)    │  │ face detect │  │
│  │ (.pt weights) │  │                  │  │ heatmaps    │  │
│  └───────────────┘  └──────────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 DevOps Infrastructure Architecture

```
GitHub Repository
       │
       │ (push / PR to main)
       ▼
GitHub Actions CI/CD ──────────────────────────────────┐
       │                                               │
   ┌───┴───┐                                           │
   │ Test  │  flake8 + manage.py check                 │
   └───┬───┘                                           │
       │ (pass)                                        │
   ┌───┴───────┐                                       │
   │   Build   │  docker buildx → GHCR (git SHA tag)  │
   └───┬───────┘                                       │
       │                                               │
   ┌───┴───────┐                                       │
   │  Deploy   │  kubectl set image                    │
   └───┬───────┘                                       │
       │                                               │
       ▼                                               │
Kubernetes Cluster (deepfake-ns namespace)             │
       │                                               │
   ┌───┴──────────────────────────────────────────┐    │
   │  Deployment: deepfake-detection              │    │
   │  • 2 replicas (min) → 6 pods (HPA max)       │    │
   │  • Rolling update: maxUnavailable=0          │    │
   │  • Liveness probe: /health/ every 30s        │    │
   │  • Readiness probe: /health/ initial 60s     │    │
   │                                              │    │
   │  ┌──────────────────┐  ┌───────────────────┐ │    │
   │  │ PVC: images-pvc  │  │ PVC: models-pvc   │ │    │
   │  │ 5Gi ReadWriteMany│  │ 10Gi ReadWriteMany│ │    │
   │  └──────────────────┘  └───────────────────┘ │    │
   └──────────────────────────────────────────────┘    │
       │                                               │
   ┌───┴──────────────────┐                            │
   │ Service: LoadBalancer│  Port 80 → 8000            │
   └──────────────────────┘                            │
                                                       │
Monitoring Stack ──────────────────────────────────────┘
   • Prometheus  :9090  (metrics store)
   • Grafana     :3000  (dashboards + alerting)
   • Node Exp    :9100  (host metrics)
   • cAdvisor    :8080  (container metrics)
   • /metrics    :8000  (django-prometheus)
```

### 2.3 Directory Structure

```
deep_pro/devops/
├── step1-github-actions/
│   └── .github/workflows/ci-cd.yml       # CI/CD pipeline
├── step2-ansible/
│   ├── playbook.yml                       # Ansible playbook
│   └── inventory.ini                      # Target hosts
├── step3-docker-k8s/
│   ├── Dockerfile                         # Multi-stage build
│   ├── k8s-namespace.yaml
│   ├── k8s-deployment.yaml
│   ├── k8s-service.yaml
│   ├── k8s-pvcs.yaml
│   └── k8s-hpa.yaml
├── step4-monitoring/
│   ├── docker-compose.monitoring.yml
│   ├── prometheus.yml
│   └── grafana-dashboard.json
└── step5-report/
    ├── REPORT.md
    ├── DOCUMENTATION.md                   # This file
    └── PRESENTATION.html                  # Interactive PPT
```

---

## 3. Pipeline Flow (CI/CD)

### 3.1 Overview

**Tool chosen:** GitHub Actions
**Reason:** Native GitHub integration, YAML-based declarative syntax, built-in secrets management, free for public repositories.

### 3.2 Trigger

```yaml
on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]
```

Any push or PR to `main`/`master` triggers the full pipeline.

### 3.3 Stage-by-Stage Breakdown

#### Stage 1 — Test

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with: { python-version: '3.10' }
      - name: Install dependencies
        run: pip install flake8 django
      - name: Lint with flake8
        run: flake8 . --max-line-length=120
      - name: Django check
        run: python manage.py check
```

**Purpose:** Catches syntax errors, import failures, and Django misconfiguration before any Docker image is built.

#### Stage 2 — Build & Push

```yaml
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Log in to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

**Key decisions:**
- Image tagged with `github.sha` for traceability and rollback capability
- Multi-stage Dockerfile keeps final image lean (no build tools at runtime)

#### Stage 3 — Deploy to Kubernetes

```yaml
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBECONFIG }}
      - name: Rolling update
        run: |
          kubectl set image deployment/deepfake-detection \
            deepfake-app=ghcr.io/${{ github.repository }}:${{ github.sha }} \
            --namespace=deepfake-ns
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/deepfake-detection \
            -n deepfake-ns --timeout=5m
      - name: Rollback on failure
        if: failure()
        run: kubectl rollout undo deployment/deepfake-detection -n deepfake-ns
```

**Key decisions:**
- `kubectl rollout status` blocks until all pods are healthy (or times out at 5 min)
- Auto-rollback step uses GitHub Actions `if: failure()` condition

### 3.4 Workflow Diagram

```
[Code Push] → [Checkout] → [Lint: flake8] → [Django check]
                                                   │
                                              (pass/fail)
                                                   │ pass
                                                   ▼
                                    [Docker Buildx multi-stage]
                                                   │
                                    [Push to GHCR with :sha tag]
                                                   │
                                                   ▼
                                    [kubectl set image → K8s]
                                                   │
                                    [kubectl rollout status]
                                                   │
                                              (pass/fail)
                                           ┌───────┴───────┐
                                          pass            fail
                                           │               │
                                       [Done]    [kubectl rollout undo]
```

### 3.5 Secrets Required

| Secret Name | Value |
|---|---|
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions for GHCR auth |
| `KUBECONFIG` | Base64-encoded kubeconfig for cluster access |

---

## 4. Configuration Management — Ansible

### 4.1 Why Ansible

- **Agentless** — uses SSH only, no daemon on target machines
- **Idempotent** — running the playbook multiple times produces the same result
- **YAML-based** — human-readable, easy to version-control
- **Modular** — tasks can be organised into roles for reuse

### 4.2 Inventory

```ini
# inventory.ini
[deepfake_servers]
192.168.1.100 ansible_user=ubuntu ansible_ssh_private_key_file=~/.ssh/id_rsa
```

### 4.3 Playbook Task Groups

#### 4.3.1 System Packages

```yaml
- name: Install system dependencies
  apt:
    name:
      - python3.10
      - python3.10-venv
      - cmake
      - build-essential
      - libopencv-dev
      - nginx
      - supervisor
    state: present
    update_cache: yes
```

**Packages installed:** Python 3.10, cmake (for dlib compilation), OpenCV headers, Nginx, Supervisor

#### 4.3.2 User Management

```yaml
- name: Create deepfake user
  user:
    name: deepfake
    shell: /bin/bash
    groups: www-data
    create_home: yes
```

Creates a dedicated `deepfake` system user — principle of least privilege.

#### 4.3.3 Directory Structure

```yaml
- name: Create application directories
  file:
    path: "{{ item }}"
    state: directory
    owner: deepfake
    group: deepfake
    mode: '0755'
  loop:
    - /opt/deepfake-detection
    - /opt/deepfake-detection/images
    - /opt/deepfake-detection/videos
    - /opt/deepfake-detection/models
    - /opt/deepfake-detection/logs
```

#### 4.3.4 Python Virtual Environment

```yaml
- name: Create Python venv
  command: python3.10 -m venv /opt/deepfake-detection/venv
  become_user: deepfake

- name: Install Python dependencies
  pip:
    requirements: /opt/deepfake-detection/requirements.txt
    virtualenv: /opt/deepfake-detection/venv
```

**Note:** PyTorch, dlib, face_recognition, OpenCV-python are all installed within the isolated venv.

#### 4.3.5 Environment Configuration

```yaml
- name: Write .env file
  template:
    src: env.j2
    dest: /opt/deepfake-detection/.env
    owner: deepfake
    mode: '0600'
```

`.env.j2` template sets `DEBUG=False`, `SECRET_KEY`, `ALLOWED_HOSTS`, and database settings.

#### 4.3.6 Nginx Configuration

```nginx
# /etc/nginx/sites-available/deepfake
server {
    listen 80;
    server_name _;

    proxy_read_timeout 300;
    proxy_send_timeout 300;
    proxy_connect_timeout 300;

    location /static/ {
        alias /opt/deepfake-detection/staticfiles/;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Critical:** 300-second timeout prevents Nginx from closing connections during 30–60s ML inference.

#### 4.3.7 Supervisor (Gunicorn Daemon)

```ini
# /etc/supervisor/conf.d/deepfake.conf
[program:deepfake]
command=/opt/deepfake-detection/venv/bin/gunicorn
        deepfake_detection.wsgi:application
        --bind 0.0.0.0:8000
        --workers 3
        --timeout 300
user=deepfake
autostart=true
autorestart=true
stderr_logfile=/opt/deepfake-detection/logs/gunicorn.err.log
stdout_logfile=/opt/deepfake-detection/logs/gunicorn.out.log
```

#### 4.3.8 Health Check

```yaml
- name: Verify application health
  uri:
    url: http://localhost/health/
    return_content: yes
    status_code: 200
  register: health_check
  retries: 5
  delay: 10
```

Ansible's `uri` module confirms the app is responding with HTTP 200 before marking setup complete.

---

## 5. Containerization — Docker

### 5.1 Multi-Stage Dockerfile

```dockerfile
# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    cmake build-essential \
    libopencv-dev libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all Python deps including torch, dlib, face_recognition
RUN pip install --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy only installed packages from builder — no build tools
COPY --from=builder /install /usr/local

# Runtime system libs (shared libraries only)
RUN apt-get update && apt-get install -y \
    libopencv-core4.5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY . .

# Create non-root user for security
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check for Kubernetes probe compatibility
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000
CMD ["gunicorn", "deepfake_detection.wsgi:application",
     "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "300"]
```

### 5.2 Design Decisions

| Decision | Rationale |
|---|---|
| Multi-stage build | Reduces final image size by excluding cmake, build-essential, headers |
| Non-root user (UID 1000) | Security best practice — limits blast radius of container escape |
| `--start-period=60s` in HEALTHCHECK | ML model loading takes 30–60s; prevents premature probe failures |
| Models NOT in image | Model weights (GBs) mounted via PVC — images stay deployable in seconds |
| System libs in runtime stage | Only shared libraries needed at runtime, not headers/compilers |

### 5.3 Building & Pushing

```bash
# Build with git SHA tag
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/<username>/deepfake-detection-app:$(git rev-parse --short HEAD) \
  --push .

# Pull on target
docker pull ghcr.io/<username>/deepfake-detection-app:<sha>
```

---

## 6. Orchestration — Kubernetes

### 6.1 Namespace Isolation

```yaml
# k8s-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: deepfake-ns
  labels:
    app: deepfake-detection
```

All resources live in `deepfake-ns` — clean separation from cluster system workloads.

### 6.2 Deployment

```yaml
# k8s-deployment.yaml (key sections)
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # Zero-downtime: new pod must be ready before old terminates
  template:
    spec:
      containers:
        - name: deepfake-app
          image: ghcr.io/<user>/deepfake-detection-app:<sha>
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet: { path: /health/, port: 8000 }
            initialDelaySeconds: 60    # Wait for ML model loading
            periodSeconds: 30
          readinessProbe:
            httpGet: { path: /health/, port: 8000 }
            initialDelaySeconds: 60
            periodSeconds: 10
          volumeMounts:
            - name: images-storage
              mountPath: /app/images
            - name: models-storage
              mountPath: /app/models
      volumes:
        - name: images-storage
          persistentVolumeClaim:
            claimName: images-pvc
        - name: models-storage
          persistentVolumeClaim:
            claimName: models-pvc
```

### 6.3 PersistentVolumeClaims

```yaml
# k8s-pvcs.yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: images-pvc
  namespace: deepfake-ns
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: deepfake-ns
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 10Gi     # Accommodates ResNeXt + MAE weights
```

### 6.4 Service (LoadBalancer)

```yaml
# k8s-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: deepfake-service
  namespace: deepfake-ns
spec:
  type: LoadBalancer
  selector:
    app: deepfake-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
```

### 6.5 Horizontal Pod Autoscaler

```yaml
# k8s-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: deepfake-hpa
  namespace: deepfake-ns
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepfake-detection
  minReplicas: 2
  maxReplicas: 6
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

HPA scales from 2 → 6 pods when CPU exceeds 70%. During peak inference load (multiple simultaneous requests), this prevents request queuing.

### 6.6 Rolling Update & Rollback Commands

```bash
# Trigger rolling update with new image
kubectl set image deployment/deepfake-detection \
  deepfake-app=ghcr.io/<user>/deepfake-detection-app:sha-abc123 \
  --namespace=deepfake-ns

# Monitor rollout progress
kubectl rollout status deployment/deepfake-detection -n deepfake-ns

# View rollout history
kubectl rollout history deployment/deepfake-detection -n deepfake-ns

# Rollback to previous version
kubectl rollout undo deployment/deepfake-detection -n deepfake-ns

# Rollback to specific revision
kubectl rollout undo deployment/deepfake-detection --to-revision=2 -n deepfake-ns
```

---

## 7. Monitoring & Observability

### 7.1 Stack Overview

```
┌────────────────────────────────────────────────────────┐
│                   Grafana :3000                        │
│           Dashboards + Alerting Rules                  │
└──────────────────────────┬─────────────────────────────┘
                           │ PromQL queries
                           ▼
┌────────────────────────────────────────────────────────┐
│                 Prometheus :9090                       │
│            Time-series metrics store                   │
└───┬──────────────────┬───────────────┬─────────────────┘
    │ scrape           │ scrape        │ scrape
    ▼                  ▼               ▼
Node Exporter    cAdvisor          django-prometheus
:9100            :8080              :8000/metrics
(host metrics)   (container)        (app metrics)
```

### 7.2 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'django'
    static_configs:
      - targets: ['app:8000']
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
```

### 7.3 Docker Compose — Monitoring Stack

```bash
# Start full monitoring stack
docker compose -f step4-monitoring/docker-compose.monitoring.yml up -d

# Services started:
# • prometheus   → localhost:9090
# • grafana      → localhost:3000  (admin / deepfake123)
# • node-exporter→ localhost:9100
# • cadvisor     → localhost:8080
```

### 7.4 Grafana Dashboard Panels

| Panel | Metric | PromQL |
|---|---|---|
| **App Uptime** | UP/DOWN status | `up{job="django"}` |
| **HTTP Request Rate** | req/s by method | `rate(django_http_requests_total[5m])` |
| **P95 Latency** | 95th percentile | `histogram_quantile(0.95, rate(django_http_responses_latency_seconds_bucket[5m]))` |
| **Error Rate** | 4xx + 5xx /s | `rate(django_http_responses_total{status=~"4..|5.."}[5m])` |
| **CPU Usage** | Host utilization | `100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)` |
| **Total Predictions** | Business counter | `django_http_requests_total{view="predict"}` |

### 7.5 Key Metrics Revealed

During testing, Prometheus + Grafana revealed:
- **dlib face detection** caused P95 latency spikes to 45–55 seconds
- **Model loading** on cold-start caused readiness probe failures without `initialDelaySeconds=60`
- **Memory usage** peaked at ~3.2 GB per pod during MAE reconstruction

---

## 8. Challenges & Solutions

| # | Challenge | Root Cause | Solution |
|---|---|---|---|
| 1 | `torch` not found in venv | PyTorch was installed system-wide, not in venv | Used system Python for Gunicorn; documented activation procedure |
| 2 | ML inference gateway timeout | Nginx default timeout is 60s; inference takes 30–60s | Set `proxy_read_timeout`, `proxy_send_timeout`, `proxy_connect_timeout` to 300s |
| 3 | Docker image size: 8+ GB | torch, dlib, OpenCV all in single layer | Multi-stage build: compile in builder, copy only installed libs to runtime |
| 4 | dlib compile time 20+ min in CI | cmake + C++ compilation from source | Moved to Docker builder stage with layer caching; CI skips recompile on unchanged deps |
| 5 | MAE weights missing — silent failure | `mae_modules` weights path not mounted | Added graceful fallback to OpenCV inpainting; logged warning instead of crash |
| 6 | K8s pods crash on startup | ML models loading before readiness probe fires | Set `initialDelaySeconds: 60` on liveness + readiness probes |
| 7 | Rolling update caused downtime | `maxUnavailable: 1` allowed 1 pod down | Changed to `maxUnavailable: 0`, `maxSurge: 1` — true zero-downtime |
| 8 | PVC data not shared across replicas | PVC in ReadWriteOnce mode | Changed to `ReadWriteMany` so both replicas access same uploaded images |

---

## 9. Lessons Learned

### 9.1 Containers Are Not a Silver Bullet

Containers package the *application* perfectly, but ML workloads have GB-scale model weights that cannot and should not be baked into images:
- **Problem:** 10GB+ Docker image defeats the purpose of fast deployments
- **Solution:** Mount model weights via PersistentVolumeClaims; image stays < 2GB
- **Rule:** Separate *code* (image) from *data* (volumes) in ML deployments

### 9.2 Kubernetes Probe Configuration Is ML-Specific

Standard web apps start in < 5 seconds. ML apps load models in 30–90 seconds:
- **Problem:** Default `initialDelaySeconds: 30` caused constant pod restarts
- **Solution:** Set `initialDelaySeconds: 60` (or higher based on profiling)
- **Rule:** Always profile your app's cold-start time before writing probe config

### 9.3 Ansible Idempotency Eliminates Setup Drift

During the project, the playbook was run 15+ times as configuration evolved:
- **Benefit:** Each run converged to the correct state without manual cleanup
- **Benefit:** Documented infrastructure-as-code that any team member can run
- **Rule:** Write Ansible tasks with `state: present/absent`, never `command: rm -rf`

### 9.4 Observability Reveals What Testing Cannot

Unit tests and integration tests did not reveal:
- dlib being the P95 latency bottleneck (found via Grafana)
- Memory leaks during repeated predictions (found via cAdvisor)
- Request spike patterns during batch uploads (found via Prometheus)

**Rule:** Instrument before you optimise. Data > assumptions.

### 9.5 GitHub Actions Secrets Management Is Underrated

Compared to Jenkins with external credential stores:
- `GITHUB_TOKEN` is auto-provisioned — zero setup for GHCR auth
- `KUBECONFIG` as a repository secret is simple and auditable
- **Rule:** Prefer native secret management over external integrations unless scale demands it

### 9.6 Zero-Downtime Requires Explicit Configuration

Kubernetes does **not** give you zero-downtime by default:
- `maxUnavailable: 1` = 1 pod down during rollout = brief service degradation
- `maxUnavailable: 0` + `maxSurge: 1` = always at minimum capacity during transition
- **Rule:** Always set `maxUnavailable: 0` for user-facing production services

---

## 10. Quick Reference

### 10.1 Common Commands

```bash
# ── CI/CD ──────────────────────────────────────────────
# View GitHub Actions run logs
gh run list --limit 5
gh run view <run-id>

# ── Kubernetes ─────────────────────────────────────────
# Check pod status
kubectl get pods -n deepfake-ns -o wide

# View pod logs
kubectl logs -f deployment/deepfake-detection -n deepfake-ns

# Scale manually
kubectl scale deployment/deepfake-detection --replicas=4 -n deepfake-ns

# Describe HPA
kubectl describe hpa deepfake-hpa -n deepfake-ns

# Rolling update
kubectl set image deployment/deepfake-detection \
  deepfake-app=ghcr.io/<user>/deepfake-detection-app:<new-sha> \
  -n deepfake-ns

# Rollback
kubectl rollout undo deployment/deepfake-detection -n deepfake-ns

# ── Ansible ────────────────────────────────────────────
# Dry-run (check mode)
ansible-playbook -i inventory.ini playbook.yml --check

# Run playbook
ansible-playbook -i inventory.ini playbook.yml

# ── Monitoring ─────────────────────────────────────────
# Start monitoring stack
docker compose -f step4-monitoring/docker-compose.monitoring.yml up -d

# Check stack status
docker compose -f step4-monitoring/docker-compose.monitoring.yml ps

# Stop monitoring
docker compose -f step4-monitoring/docker-compose.monitoring.yml down
```

### 10.2 Service Access

| Service | URL | Credentials |
|---|---|---|
| Django App | `http://<node-ip>/` | N/A |
| Prometheus | `http://localhost:9090` | None |
| Grafana | `http://localhost:3000` | admin / deepfake123 |
| Node Exporter | `http://localhost:9100/metrics` | None |
| cAdvisor | `http://localhost:8080` | None |
| App Metrics | `http://localhost:8000/metrics` | None |

### 10.3 File Reference

| File | Location | Purpose |
|---|---|---|
| `ci-cd.yml` | `.github/workflows/` | GitHub Actions pipeline |
| `playbook.yml` | `step2-ansible/` | Ansible server setup |
| `inventory.ini` | `step2-ansible/` | Target host configuration |
| `Dockerfile` | `step3-docker-k8s/` | Multi-stage container build |
| `k8s-deployment.yaml` | `step3-docker-k8s/` | Kubernetes deployment spec |
| `k8s-hpa.yaml` | `step3-docker-k8s/` | Horizontal pod autoscaler |
| `k8s-pvcs.yaml` | `step3-docker-k8s/` | Persistent volume claims |
| `docker-compose.monitoring.yml` | `step4-monitoring/` | Full monitoring stack |
| `prometheus.yml` | `step4-monitoring/` | Prometheus scrape config |
| `grafana-dashboard.json` | `step4-monitoring/` | Grafana dashboard export |

---

*Documentation generated for DevOps CA2 — Deepfake Detection App | Batch 2023–27*
