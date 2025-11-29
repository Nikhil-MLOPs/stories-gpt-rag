# 📚 Stories GPT RAG – End-to-End AI Chatbot (Production Ready)

Stories GPT RAG is a **Retrieval-Augmented Generation (RAG)** based AI chatbot that allows users to upload story files (`.txt`, `.pdf`, `.doc`, `.docx`) or paste text, and ask context-based questions. It is built fully from scratch up to cloud deployment and monitoring using:

🧠 FastAPI · OpenAI · ChromaDB  
📊 MLflow · DVC  
🐳 Docker · GitHub Actions CI/CD  
☸ Kubernetes (k3s) on AWS EC2  
📈 Prometheus · Grafana Monitoring  
🎨 Pure HTML & CSS (No JavaScript)

---

## 🚀 Features

✔ Upload txt/pdf/doc/docx or paste text  
✔ Answer context-based questions using RAG pipeline  
✔ OpenAI Embeddings + ChromaDB for Retrieval  
✔ Track 5 experiments using MLflow & DVC  
✔ Dockerized & deployed on AWS EC2  
✔ Kubernetes (k3s) orchestration  
✔ GitHub Actions CI/CD automation  
✔ Monitoring using `/metrics`, Prometheus, Grafana  
✔ HTML-only UI (No JavaScript)

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | FastAPI, Python 3.12.7 |
| LLM/Embedding | OpenAI API |
| Vector Store | ChromaDB |
| Experiment Tracking | MLflow, DVC |
| Deployment | Docker, AWS EC2, k3s |
| CI/CD | GitHub Actions, AWS ECR |
| Monitoring | Prometheus, Grafana |
| UI | HTML & CSS (No JS) |

---

## ▶️ Setup and Run Locally

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
source .venv/bin/activate     # Linux/Mac

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run FastAPI App

uvicorn app.main:app --reload

### 4️⃣ Local Access URLs

Feature	      URL

Upload UI	  http://127.0.0.1:8000/

Chat UI	      http://127.0.0.1:8000/chat-ui

API Docs	  http://127.0.0.1:8000/docs

Metrics	      http://127.0.0.1:8000/metrics

### 🐳 Docker Usage

docker build -t stories-gpt-rag:latest .

docker run -p 8000:8000 stories-gpt-rag:latest

Access at → http://localhost:8000

### ⚙️ CI/CD (GitHub Actions)

Runs automatically on git push to main:

✔ Lint (ruff)

✔ Test (pytest)

✔ Build Docker image

✔ Push to AWS ECR

✔ SSH to EC2 & Restart Kubernetes deployment

File: .github/workflows/ci-cd.yml

### ☸ Kubernetes Deployment (k3s on AWS EC2)
cd ~/stories-gpt-rag

git pull origin main

sudo docker build -t stories-gpt-rag:latest .

sudo docker save stories-gpt-rag.tar

sudo k3s ctr images import stories-gpt-rag.tar

cd k8s

sudo kubectl apply -k .

sudo kubectl rollout restart deployment stories-gpt-rag

sudo kubectl get pods

🔗 Application Live URL: http://<EC2_PUBLIC_IP>:30235/

### 📈 Monitoring & Observability
Component	        Access

Metrics Endpoint	/metrics

Grafana Dashboard	http://<EC2_IP>:32000

Grafana Login	    admin / admin123

Prometheus          Scraping	Enabled via ServiceMonitor

Metrics Tracked:

- Request count

- Latency per endpoint

- CPU/Memory usage

- OpenAI API call latency