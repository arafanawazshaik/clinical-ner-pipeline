# Clinical NER Pipeline

An ML-based system that extracts structured medical entities from free-text clinical notes. Combines a fine-tuned BioBERT NER model with rule-based extraction and negation detection.

## Entities Extracted

- **DIAGNOSIS** — type 2 diabetes mellitus, CHF, COPD
- **MEDICATION** — metformin, lisinopril, aspirin
- **PROCEDURE** — coronary artery bypass graft, MRI brain
- **DOSAGE** — 500mg twice daily, 10mg PO QD
- **DATE** — 03/15/2022, January 2021

## Architecture
```
Clinical Note → Preprocessing → BioBERT NER + Rule Engine → Post-processing → Validated JSON
```

**Key components:**
- **Preprocessing** — text cleaning, abbreviation expansion, PHI masking, section detection
- **BioBERT NER** — fine-tuned token classification (F1: 0.9988 on synthetic data)
- **Rule Engine** — regex patterns + dictionary lookup for medications, dosages, dates
- **Negation Detection** — NegEx-style pre/post negation triggers
- **Post-processing** — entity merging, deduplication, confidence routing
- **FastAPI Service** — REST API with `/extract`, `/batch`, `/health` endpoints

## Quick Start
```bash
# Setup
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_synthetic_data.py

# Train model
python scripts/train.py --epochs 2 --batch-size 4

# Start API
uvicorn api.app:app --port 8000

# Run tests
pytest tests/ -v
```

## API Usage
```bash
# Health check
curl http://localhost:8000/health

# Extract entities
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient has type 2 diabetes on metformin 500mg daily.", "document_id": "note_001"}'
```

## Example Output
```json
{
  "document_id": "note_001",
  "entities": [
    {
      "text": "type 2 diabetes",
      "label": "DIAGNOSIS",
      "confidence": 0.998,
      "negated": false,
      "source": "model"
    },
    {
      "text": "metformin",
      "label": "MEDICATION",
      "confidence": 0.999,
      "negated": false,
      "source": "model"
    }
  ]
}
```

## Project Structure
```
clinical-ner-pipeline/
├── api/                    # FastAPI service
├── config/                 # Settings and label schema
├── src/
│   ├── preprocessing/      # Text cleaning, PHI masking, section detection
│   ├── labeling/           # Synthetic data generation, BIO conversion
│   ├── training/           # BioBERT fine-tuning, class balancing
│   ├── rules/              # Rule-based extraction, negation detection
│   ├── inference/          # Prediction, post-processing, schema formatting
│   ├── evaluation/         # Metrics (seqeval entity-level F1)
│   └── utils/              # Structured logging
├── scripts/                # CLI entry points
├── tests/                  # Pytest test suite (20 tests)
└── data/                   # Synthetic/labeled data (not in repo)
```

## Tech Stack

- **NER Model** — BioBERT (dmis-lab/biobert-base-cased-v1.2) via HuggingFace Transformers
- **Framework** — PyTorch, FastAPI
- **Validation** — Pydantic v2
- **Evaluation** — seqeval (entity-level P/R/F1)
- **Testing** — Pytest
- **Experiment Tracking** — MLflow

## Training Results

| Epoch | Train Loss | Val Loss | Val F1 |
|-------|-----------|----------|--------|
| 1     | 0.3968    | 0.0052   | 0.9939 |
| 2     | 0.0026    | 0.0009   | 0.9988 |

*Trained on 500 synthetic clinical notes across 5 note types.*

## Infrastructure

### Docker
```bash
# Build image
docker build -f docker/Dockerfile.inference -t clinical-ner-pipeline .

# Run container
docker run -p 8000:8000 -v ./models:/app/models:ro clinical-ner-pipeline

# Full stack with monitoring
docker-compose up -d
```

- **API** → http://localhost:8000
- **Prometheus** → http://localhost:9090
- **Grafana** → http://localhost:3000 (admin/admin)

### Kubernetes
```bash
# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods -n clinical-ner
kubectl get hpa -n clinical-ner
```

**K8s features:**
- 2 replicas with auto-scaling (2→8 pods based on CPU/memory)
- Liveness and readiness probes via `/health`
- Resource limits (2Gi memory, 2 CPU per pod)
- Prometheus scraping annotations

### Monitoring

Prometheus metrics available at `/metrics`:
- `ner_requests_total` — request count by endpoint and status
- `ner_request_latency_seconds` — latency histogram
- `ner_entities_extracted_total` — entities extracted by type