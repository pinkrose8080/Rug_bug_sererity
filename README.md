# 🐛 Lightweight RAG-Based Bug Severity Classification Using LLMs

> **Empirical Benchmark on Mozilla Defect Corpus (Mozilla Bugzilla / BugsRepo)**  
> Benchmarking Zero-shot, Fixed Few-shot, and Lightweight RAG (gpt-4o-mini) against an optimized character n-gram Support Vector Classifier.

---

## 📌 Overview & Research Aim
Bug severity assignment is a critical operational bottleneck in open-source defect management. General-purpose large language models (LLMs) can infer defect severity, but lacking project-specific historical memory, they reason from generic priors and frequently misalign with community triage conventions.

This repository implements a **serverless, in-memory Lightweight Retrieval-Augmented Generation (RAG)** framework:
1. **Vector Indexing (Offline):** 2,306 historical Bugzilla defect reports are embedded into a dense 1,536-dimensional space using OpenAI `text-embedding-3-small` and stored in an in-memory NumPy matrix.
2. **Precedent Retrieval (Online):** For each unseen bug report, the Top-3 nearest historical precedents are retrieved via Cosine Similarity in RAM (~1.0 ms).
3. **Deterministic Classification:** Precedents, similarity scores, and historical ground-truth labels are dynamically assembled into an in-context prompt evaluated by `gpt-4o-mini` ($T = 0.0$, enforced JSON schema) to generate a binary severity label (`Severe` vs. `Non-Severe`) and an auditable technical rationale.

---

## 📊 Benchmark Results (Holdout Test Partition: N = 576)

Evaluated under strict zero-leakage conditions (0 shared bug IDs between historical memory and holdout test set: 278 Severe, 298 Non-Severe):

| # | Approach | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Correct / Errors |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Zero-shot** (`gpt-4o-mini`) | 65.10% | 64.10% | 62.95% | 63.52% | 375 / 201 |
| 2 | **Fixed Few-shot** (`gpt-4o-mini`, 3 static examples) | 67.36% | 66.30% | 65.83% | 66.06% | 388 / 188 |
| 3 | **Lightweight RAG** (Proposed, Top-3 Retrieved) | 82.81% | **84.82%** | 78.42% | 81.50% | 477 / 99 |
| 4 | **TF-IDF + LinearSVC Baseline** (Char n-grams 2-5, $C=0.3$) | **88.19%** | 84.31% | **92.81%** | **88.36%** | **508 / 68** |

### 🔍 Key Findings & Statistical Significance
* **RAG vs. Prompting Baselines:** Lightweight RAG provides an accuracy gain of **+17.71 pp** over Zero-shot and **+15.45 pp** over Fixed Few-shot without model fine-tuning.
* **Error Reduction:** RAG eliminates **50.7%** of Zero-shot errors, primarily by suppressing false alarms (False Positives dropped from 98 in Zero-shot to 39 in RAG; False Alarm Rate = 13.1%).
* **McNemar's Test:** Comparing paired predictions on all 576 test samples ($b = 45$, $c = 76$) yields $\chi^2 = 7.4380$ ($p = 0.0064 < 0.01$). The lexical baseline's advantage is statistically significant due to surface technical terminology regularities (`segfault`, `crash`, `leak`).
* **Engineering Trade-off:** TF-IDF is an opaque statistical black box; RAG uniquely provides natural-language explainability and instant zero-retraining memory adaptation at under **$0.0002** and **~0.6s** per triage.

---

## 🏛️ System Architecture

```
 OFFLINE INDEXING                               ONLINE TRIAGING
 ----------------                               ---------------
 Mozilla Bugzilla (BugsRepo)                     Unseen Defect Summary
        │                                               │
  Curation + Dedup + Normal Filter                      Text Preprocessing
        │                                               │
  2,306 Historical Reports                              text-embedding-3-small (1536-d)
        │                                               │
  text-embedding-3-small                         Cosine Similarity Search (In-Memory)
        │                                               │
  Vector Index (2306 × 1536) ──────────────────────────► Top-3 Precedents + Similarity
                                                        │
                                                 Dynamic Prompt Assembler
                                                        │
                                                 gpt-4o-mini (T = 0.0, JSON Mode)
                                                        │
                                                 {"severity": "Severe" | "Non-Severe",
                                                  "reason": "..."}
```

---

## 📁 Repository Structure

```
├── data/
│   ├── historical_data_clean.csv       # 2,306 curated historical defect reports
│   ├── test_data_clean.csv             # 576 clean holdout test reports (zero leakage)
│   ├── historical_embeddings_clean.npy # Precomputed 1,536-d embeddings for historical memory
│   └── test_embeddings_clean.npy       # Precomputed 1,536-d embeddings for test reports
├── results/
│   ├── Lightweight RAG-Based Bug Severity Classification Using LLMs - Final Report.docx # Academic Report (12 pages)
│   ├── figure1_architecture.png        # Architecture workflow diagram
│   ├── code_snippet_rag.png            # Source code walkthrough figure
│   ├── figure3_retrieval_evidence.png  # Execution terminal log (Top-3 precedents)
│   ├── figure4_model_output_json.png   # JSON model response output
│   ├── figure5_metrics_comparison.png  # 4-approach metrics comparison barplot
│   ├── figure6_confusion_matrices.png  # 4-approach confusion matrix heatmaps
│   ├── predictions_detailed.csv        # Detailed model predictions on holdout set
│   ├── summary_results.csv             # Summary metrics table
│   └── manual_review_20_samples.csv    # 20-sample qualitative audit across archetypes
├── fetch_data.py                       # Standalone data verification and acquisition script
├── prepare_data.py                     # Data integrity and leakage validation script
├── main.py                             # All-in-One Benchmark & RAG Engine
├── requirements.txt                    # Project dependencies
├── project_plan.md                     # Roadmap and verified experimental benchmarks
└── README.md                           # Project documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Requirements & Environment
```bash
git clone https://github.com/your-username/Lightweight-RAG-Bug-Severity-Classification.git
cd Lightweight-RAG-Bug-Severity-Classification
pip install -r requirements.txt
```

### 2. Data Verification (Optional Standalone Step)
To independently verify and inspect the dataset partitions:
```bash
python fetch_data.py
```
> **Note:** `main.py` runs directly using the clean datasets already provided in `data/` and will **NOT** trigger `fetch_data.py` automatically.

### 3. Verify Dataset Integrity
Verify that the holdout test set has zero overlap with the historical memory:
```bash
python prepare_data.py
```

### 4. Run Benchmark Engine
Set your OpenAI API key and execute the all-in-one engine:
```bash
export OPENAI_API_KEY="sk-..."    # On Linux/macOS
set OPENAI_API_KEY=sk-...         # On Windows CMD
$env:OPENAI_API_KEY="sk-..."      # On Windows PowerShell

python main.py --full
```

---

## 📑 Citation & Academic Attribution
If you use this benchmark or codebase in your research, please refer to the final report:
* **Report:** *Lightweight RAG-Based Bug Severity Classification Using Large Language Models: An Empirical Benchmark on Mozilla Defect Corpus*, October 2026.
* **Corpus:** BugsRepo: A curated multi-project dataset of Bugzilla defect histories (Zenodo: 15004067).
