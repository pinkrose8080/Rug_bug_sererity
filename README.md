🐛 Lightweight RAG-Based Bug Severity Classification Using LLMs
> **Empirical Benchmark on Mozilla Defect Corpus (Mozilla Bugzilla / BugsRepo)**  
> Benchmarking Zero-shot, Fixed Few-shot, and Lightweight RAG (gpt-4o-mini) against a character n-gram TF-IDF + LinearSVC baseline.
---
📌 Overview & Research Aim
Bug severity assignment is a critical operational bottleneck in open-source defect management. General-purpose large language models (LLMs) can infer defect severity, but lacking project-specific historical memory, they reason from generic priors and frequently misalign with community triage conventions.
This repository implements a serverless, in-memory Lightweight Retrieval-Augmented Generation (RAG) framework:
Vector Indexing (Offline): 2,306 historical Bugzilla defect reports are embedded into a dense 1,536-dimensional space using OpenAI `text-embedding-3-small` and stored in an in-memory NumPy matrix.
Precedent Retrieval (Online): For each unseen bug report, the Top-3 nearest historical precedents are retrieved via Cosine Similarity in RAM.
Deterministic Classification: Precedents, similarity scores, and historical ground-truth labels are dynamically assembled into an in-context prompt evaluated by `gpt-4o-mini` ($T = 0.0$, enforced JSON schema) to generate a binary severity label (`Severe` vs. `Non-Severe`) and an auditable technical rationale.
---
📊 Benchmark Results (Holdout Test Partition: N = 576)
Evaluated under strict zero-leakage conditions (0 shared bug IDs between historical memory and holdout test set: 278 Severe, 298 Non-Severe):
#	Approach	Accuracy (%)	Precision (%)	Recall (%)	F1-Score (%)	Correct / Errors
1	Zero-shot (`gpt-4o-mini`)	65.10%	64.10%	62.95%	63.52%	375 / 201
2	Fixed Few-shot (`gpt-4o-mini`, 3 static examples)	67.36%	66.30%	65.83%	66.06%	388 / 188
3	Lightweight RAG (Proposed, Top-3 Retrieved)	82.81%	84.82%	78.42%	81.50%	477 / 99
4	TF-IDF + LinearSVC Baseline (Char n-grams 2-5, $C=0.3$)	88.19%	84.31%	92.81%	88.36%	508 / 68
🔍 Key Findings
RAG vs. Prompting Baselines: Lightweight RAG provides an accuracy gain of +17.71 pp over Zero-shot and +15.45 pp over Fixed Few-shot without model fine-tuning.
Error Reduction: RAG eliminates 50.7% of Zero-shot errors, primarily by suppressing false alarms (False Positives dropped from 98 in Zero-shot to 39 in RAG; False Alarm Rate = 13.1%).
Engineering Trade-off: TF-IDF + LinearSVC is more accurate and runs offline. RAG gives a short written reason for each decision. It can also add new reports to its memory without retraining.
---
🏛️ System Architecture
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
📁 Repository Structure
```
├── data/
│   ├── historical_data_clean.csv       # 2,306 historical bug reports
│   ├── test_data_clean.csv             # 576 test reports (no shared bug IDs)
│   ├── historical_embeddings_clean.npy # 1,536-d embeddings of the historical reports
│   └── test_embeddings_clean.npy       # 1,536-d embeddings of the test reports
├── prepare_data.py                     # Checks the data (no overlap between the two sets)
├── main.py                             # Runs the four approaches and saves the results
├── requirements.txt                    # Python packages
└── README.md                           # Project documentation
```
When you run `main.py`, it creates a `results/` folder with these files:
`predictions_detailed.csv`: the predictions of all approaches for each test report
`summary_results.csv`: accuracy, precision, recall, and F1-score of each approach
`metrics_comparison.png` (copy: `figure5_metrics_comparison.png`): bar chart of the metrics
`confusion_matrices.png` (copy: `figure6_confusion_matrices.png`): confusion matrix heatmaps
`manual_review_20_samples.csv`: 20 test reports with the RAG prediction and reason, for manual review
---
🚀 Quickstart & Reproduction
1. Requirements & Environment
```bash
git clone https://github.com/pinkrose8080/Rug_bug_sererity.git
cd Rug_bug_sererity
pip install -r requirements.txt
```
2. Verify Dataset Integrity
Verify that the holdout test set has zero overlap with the historical memory:
```bash
python prepare_data.py
```
3. Run Benchmark Engine
Set your OpenAI API key and execute the all-in-one engine:
```bash
export OPENAI_API_KEY="sk-..."    # On Linux/macOS
set OPENAI_API_KEY=sk-...         # On Windows CMD
$env:OPENAI_API_KEY="sk-..."      # On Windows PowerShell

python main.py --full
```
> **Note:** `--full` runs all 576 test reports. Without it, the script asks you to choose: 50 reports for a quick test, or all 576 reports.
---
📑 Citation & Academic Attribution
If you use this benchmark or codebase in your research, please refer to the final report:
Report: Lightweight RAG-Based Bug Severity Classification Using Large Language Models: An Empirical Benchmark on Mozilla Defect Corpus, Fall 2026.
Corpus: BugsRepo: A comprehensive curated dataset of bug reports, comments and contributors information from Bugzilla (Zenodo: 15004067).
