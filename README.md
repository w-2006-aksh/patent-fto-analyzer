# Autonomous Patent FTO Analysis System

A full-stack Freedom-to-Operate (FTO) patent analysis tool. This system takes an initial invention description, queries the European Patent Office (EPO), scores relevant claims using vector embeddings, and halts for human review before drafting a final infringement risk report.

## System Architecture & Features

The core engine is built on an 11-stage **LangGraph** pipeline with the following technical highlights:

* **Parallelized Retrieval & Rate Limiting:** An LLM decomposes the user's invention into 6 targeted search phrases. To handle the EPO OPS API efficiently, search and claim hydration are parallelized across a 3-worker thread pool. A custom mutex-lock rate limiter ensures dispatching stays safely below the EPO's ~2.5 req/s ceiling, cutting retrieval latency by 3x compared to a sequential loop.

* **Tiered Relevance Filtering:** Search results are merged into a core-first shortlist of <20 candidates. These are batch-scored for relevance, and the top 6 are embedded claim-by-claim into a local ChromaDB instance.

* **Element-Mapping Risk Assessment:** The system performs claim-level infringement assessment via element-mapping analysis. It utilizes per-patent top-2 vector retrieval to ensure no single massive patent monopolizes the LLM context window. Patents are then classified into 'significant risk' or 'cleared' sets.

* **Human-in-the-Loop (HITL) Checkpointing:** The LangGraph pipeline pauses at a checkpoint, presenting preliminary risk cards to the user for approval before spending API tokens on final markdown report generation.

* **Token Pacing & Fault Tolerance:** Built with rolling TPM/RPM pacing to keep multi-call LLM workloads safely within Groq's free-tier limits, alongside quality-gated autonomous retries for JSON parse failures.

## Tech Stack

* **Backend:** Python, FastAPI, LangGraph, LangChain
* **AI & Vector DB:** Groq (Llama 3.3 70B), ChromaDB (Local)
* **Frontend:** React 19, Vite, Tailwind CSS
* **External APIs:** EPO OPS REST API

## Local Setup

**Prerequisites:** Python 3.12+, Node.js, and API credentials for EPO OPS and Groq.

### 1. Backend Configuration

Clone the repository and set up the Python environment:

```bash
git clone https://github.com/w-2006-aksh/patent-agent-system.git
cd patent-agent-system

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```env
EPO_API_KEY=your_epo_key
EPO_API_SECRET=your_epo_secret
GROQ_API_KEY=your_groq_key
```

Start the FastAPI backend:

```bash
python api.py
```

### 2. Frontend Configuration

In a separate terminal, navigate to the frontend directory and start the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

The application will be available at:

```text
http://localhost:5173
```