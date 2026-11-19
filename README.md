# Bug Severity Classification Using Lightweight RAG

This project classifies Mozilla bug reports as **Severe** or **Non-Severe** using `gpt-4o-mini`. It compares zero-shot, few-shot and a lightweight RAG approach with a TF-IDF + LinearSVC baseline.

## How RAG works

1. Historical bug reports are converted to embeddings and stored in memory.
2. For each new report, the 3 most similar historical reports are retrieved.
3. These reports and their labels are added to the prompt, and the model returns the label with a short reason.

## Results

Test set: 576 reports, with no overlap with the historical data.

| Approach | Accuracy | Precision | Recall | F1-Score |
|---|:-:|:-:|:-:|:-:|
| Zero-shot | 65.10% | 64.10% | 62.95% | 63.52% |
| Few-shot | 67.36% | 66.30% | 65.83% | 66.06% |
| **Lightweight RAG** | **82.81%** | **84.82%** | **78.42%** | **81.50%** |
| TF-IDF + LinearSVC | 88.19% | 84.31% | 92.81% | 88.36% |

RAG improved accuracy by about 17 points over zero-shot. TF-IDF + LinearSVC was the most accurate, but RAG explains each decision.

## Dataset

BugsRepo – Mozilla Bugzilla bug reports (Zenodo: 15004067).
