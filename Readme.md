# OptiMIR

OptiMIR is a **production-grade Retrieval-Augmented Generation (RAG) system** for accurate, grounded question-answering over financial documents. It supports multi-modal ingestion, hybrid retrieval, strict grounding, and observability through LangSmith.

---

## 🧠 Overview

OptiMIR addresses the challenges of applying large language models to financial documents, including long reports, tables, scanned pages, and domain-specific identifiers. The system prioritizes **accuracy, reliability, and scalability** for real-world usage in compliance-sensitive environments.

---

## 🔍 Core Features

- **Multi-Modal Document Ingestion**
  - Page-wise PDF processing
  - Conditional vision processing for images, tables, and diagrams
  - Structured extraction and chunk embedding

- **Hybrid Retrieval**
  - Semantic vector search
  - BM25 keyword search
  - Score fusion and metadata filtering for high recall

- **Grounded Generation**
  - Constraints to ensure responses are based solely on retrieved content
  - Explicit refusal when evidence is insufficient
  - Model routing for precision vs. latency tradeoffs

- **Observability & Evaluation**
  - Integration with **LangSmith** for execution traces and insights
  - RAG performance metrics for relevance, recall, precision, and faithfulness

- **Production Deployment**
  - Containerized backend with FastAPI
  - Infrastructure as Code using Terraform
  - Serverless deployment (e.g., Google Cloud Run)

---

## 🧩 Architecture

```

User Upload → Classification → Ingestion
→ Hybrid Retrieval → Prompt Construction
→ Grounded Response → Evaluation/Tracing

````

1. **Document Classification**  
   Rule-based filtering ensures only supported financial documents enter the pipeline.

2. **Ingestion**
   - Text extraction on a per-page basis
   - Conditional vision processing for images and tables
   - Chunking with semantic overlap and embeddings stored in a vector database

3. **Retrieval**
   - Combined semantic and keyword search with weighted reranking
   - Metadata filters for refined retrieval

4. **Grounded Responses**
   - Template-based prompt construction
   - Streaming answers via Server-Sent Events (SSE)
   - Refusal if no grounded context is available

5. **Evaluation & Tracing**
   - LangSmith captures execution traces across the pipeline
   - Model call insights and quality metrics facilitate debugging and improvement

---

## 🛠️ Tech Stack

**Backend**
- Python 3.11
- FastAPI (async APIs + SSE)
- Vector store (ChromaDB)
- Embeddings and retrieval tools

**LLM Providers**
- OpenAI (GPT-4 series, GPT-4o-mini)
- Anthropic (Claude models)

**Frontend**
- Next.js with a streaming UI
- Model selector and query controls

**Infra & Deployment**
- Docker containers
- Terraform for Infrastructure as Code
- Serverless runtime (e.g., Google Cloud Run)
- Secret management (e.g., Google Secret Manager)

**Observability**
- LangSmith for traces, metrics, and evaluation dashboards

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deepmehta27/OptiMIR-Optimized-Multi-Modal-Intelligent-Retrieval.git
   cd backend
   
2. **Set up a Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Examples:

   ```bash
   export OPENAI_API_KEY="…"
   export ANTHROPIC_API_KEY="…"
   export LANGSMITH_API_KEY="…"
   export LANGSMITH_PROJECT="OptiMIR"
   export CHROMADB_PATH="…"
   ```

---

## 🚀 Deployment

OptiMIR uses Terraform to provision cloud infrastructure:

```bash
cd infra
terraform init
terraform plan
terraform apply
```

Before running `apply`, ensure all required cloud provider credentials and secret values are configured securely in your provider’s secret manager.

---

## 📄 Usage

1. **Start the backend**

   ```bash
   uvicorn backend.app:app --reload
   ```

2. **Start the frontend**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Open your browser**
   Visit `http://localhost:3000` and upload a financial document to begin.

---

## 📁 Repository Structure

```
.
├── backend/              # FastAPI backend APIs
├── frontend/             # Next.js frontend
├── infra/                # Terraform IaC
├── ingestion/            # Document processing
├── retrieval/            # Search & reranking logic
├── models/               # Prompt templates & routing
├── eval/                 # Evaluation & LangSmith configs
└── README.md
```

## ✉️ Contact

Developed by Deep Mehta — AI Engineering & RAG Systems
For questions, feedback, or collaboration, please open an issue or reach out via GitHub.

````

