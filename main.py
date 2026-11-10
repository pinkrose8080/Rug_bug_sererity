# -*- coding: utf-8 -*-
"""
===============================================================================
🐛 Lightweight RAG Bug Severity Classification (All-in-One Engine)
===============================================================================
Benchmarks four defect severity classification approaches on Mozilla Bugzilla:
1. Zero-shot Prompting (gpt-4o-mini)
2. Fixed Few-shot Prompting (gpt-4o-mini, 3 static exemplars)
3. Lightweight RAG (text-embedding-3-small + Top-3 Cosine Retrieval + gpt-4o-mini)
4. Tuned Character TF-IDF + LinearSVC Baseline (char n-grams 2-5, C=0.3)
"""
import os
import sys
import io
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

HISTORICAL_CSV = os.path.join(DATA_DIR, 'historical_data_clean.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_data_clean.csv')
EMBEDDINGS_CACHE = os.path.join(DATA_DIR, 'historical_embeddings_clean.npy')
TEST_EMBEDDINGS_CACHE = os.path.join(DATA_DIR, 'test_embeddings_clean.npy')

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K_SIMILAR = 3

FIXED_FEW_SHOT_EXAMPLES = [
    {
        "bug_id": "1739372",
        "summary": "Crash in [@ VisitDocAccessibleParentDescendantsAtTopLevelInContentProcess<T>]",
        "severity": "Severe",
        "reason": "Causes fatal application termination and memory corruption."
    },
    {
        "bug_id": "1700486",
        "summary": "Library column order affect the column resize behavior in user interface",
        "severity": "Non-Severe",
        "reason": "Minor layout behavior that does not impede essential system functionality."
    },
    {
        "bug_id": "1797139",
        "summary": "input element stopped working after change type from date to datetime-local",
        "severity": "Severe",
        "reason": "Directly breaks user input submission in core interface."
    }
]

SYSTEM_PROMPT = """You are an expert software engineering assistant specialized in defect triaging and bug severity classification.
Classify incoming bug reports into exactly one of two categories:
- "Severe": Critical bugs causing system crashes, memory corruption, data loss, security vulnerabilities, or blocking essential core workflows.
- "Non-Severe": Minor bugs such as cosmetic glitches, minor UI alignment, formatting errors, or low-impact test issues.

Respond strictly in valid JSON format:
{
  "severity": "Severe" or "Non-Severe",
  "reason": "A concise one-sentence technical explanation justifying your classification."
}
Do not write anything outside the JSON object."""


def get_client():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        print("\n" + "=" * 80)
        print("⚠️ Set the OPENAI_API_KEY environment variable before running, or enter it below:")
        print("=" * 80)
        key = input("API Key: ").strip()
        if not key:
            print("❌ Cancelled. Exiting.")
            sys.exit(1)

    print("⏳ Validating API key with OpenAI...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        client.models.list()
        print("✅ OpenAI client authenticated successfully!")
        return client
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        sys.exit(1)


def get_embeddings(client, texts, model=EMBEDDING_MODEL, batch_size=128):
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [t if str(t).strip() else "empty bug report" for t in texts[i:i + batch_size]]
        try:
            resp = client.embeddings.create(input=batch, model=model)
            all_vectors.extend([item.embedding for item in resp.data])
        except Exception:
            time.sleep(2)
            resp = client.embeddings.create(input=batch, model=model)
            all_vectors.extend([item.embedding for item in resp.data])
    return np.array(all_vectors, dtype=np.float32)


def call_gpt(client, prompt, model=LLM_MODEL, max_retries=3):
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            data = json.loads(res.choices[0].message.content.strip())
            sev = "Severe" if "severe" in str(data.get("severity", "")).lower() and "non" not in str(data.get("severity", "")).lower() else "Non-Severe"
            reason = str(data.get("reason", "No reason provided."))
            return sev, reason
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return "Severe", f"API fallback: {str(e)[:60]}"
    return "Severe", "API retry limit exceeded"


def compute_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, pos_label="Severe", zero_division=0)
    r = recall_score(y_true, y_pred, pos_label="Severe", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="Severe", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["Severe", "Non-Severe"])
    return {
        "Approach": name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision (%)": round(p * 100, 2),
        "Recall (%)": round(r * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "Confusion_Matrix": cm
    }


def main():
    print("=" * 80)
    print("🚀 Lightweight RAG Bug Severity Classification Benchmark Engine")
    print("=" * 80)

    # 1. Load Data
    hist_df = pd.read_csv(HISTORICAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Integrity assertions (zero leakage)
    hist_ids = set(hist_df['bug_id'].astype(str))
    test_ids = set(test_df['bug_id'].astype(str))
    if hist_ids & test_ids:
        raise ValueError(f"Data leakage detected! Shared bug IDs: {len(hist_ids & test_ids)}")

    print(f" Loaded historical defect memory: {len(hist_df):,} samples")
    print(f" Loaded holdout test partition: {len(test_df):,} samples ({test_df['severity_label'].value_counts().to_dict()})")

    # 2. Select evaluation size
    if len(sys.argv) > 1 and sys.argv[1] in ["--full", "-f", "2"]:
        choice = "2"
    elif not sys.stdin.isatty():
        choice = "2"
    else:
        try:
            print("\nEvaluation Options:")
            print("1) Quick test on 50 samples")
            print("2) Full holdout test on all 576 samples (Recommended)")
            choice = input("Choice (1 or 2) [default: 2]: ").strip() or "2"
        except (EOFError, io.UnsupportedOperation):
            choice = "2"

    eval_size = 50 if choice == "1" else len(test_df)
    eval_df = test_df.head(eval_size).copy()
    print(f"\n📊 Running benchmark on {len(eval_df)} test samples...")

    # 3. Train TF-IDF + LinearSVC Baseline
    print("\n⚡ Training character TF-IDF + LinearSVC Baseline (char 2-5, C=0.3)...")
    vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True)
    X_train = vec.fit_transform(hist_df['summary'])
    X_eval = vec.transform(eval_df['summary'])

    clf = LinearSVC(C=0.3, random_state=42)
    clf.fit(X_train, hist_df['severity_label'])
    tfidf_preds = clf.predict(X_eval)
    eval_df['TFIDF_CharSVM_Pred'] = tfidf_preds

    # 4. Embeddings & LLM Setup
    client = get_client()

    if os.path.exists(EMBEDDINGS_CACHE):
        print(f"📂 Loading cached historical embeddings from disk...")
        hist_embeddings = np.load(EMBEDDINGS_CACHE)
    else:
        print(f"⚡ Generating historical embeddings with ({EMBEDDING_MODEL})...")
        hist_embeddings = get_embeddings(client, hist_df['summary'].tolist())
        np.save(EMBEDDINGS_CACHE, hist_embeddings)

    if len(eval_df) == len(test_df) and os.path.exists(TEST_EMBEDDINGS_CACHE):
        print(f"📂 Loading cached test embeddings from disk...")
        test_embeddings = np.load(TEST_EMBEDDINGS_CACHE)
    else:
        print(f"⚡ Generating embeddings for {len(eval_df)} test reports...")
        test_embeddings = get_embeddings(client, eval_df['summary'].tolist())
        if len(eval_df) == len(test_df):
            np.save(TEST_EMBEDDINGS_CACHE, test_embeddings)

    # 5. Execute LLM Pipelines
    zero_preds, zero_reasons = [], []
    few_preds, few_reasons = [], []
    rag_preds, rag_reasons = [], []
    rag_retrieved_logs = []

    print(f"\n🤖 Running LLM evaluation (Zero-shot, Few-shot, RAG) with ({LLM_MODEL})...")
    for idx in tqdm(range(len(eval_df)), desc="Classifying bugs"):
        row = eval_df.iloc[idx]
        summary = row['summary']
        q_vec = test_embeddings[idx]

        # 5.1 Zero-shot
        p_zero = f"Classify severity of this bug report:\n\"{summary}\"\n\nRespond in JSON: {{\"severity\": \"Severe\"|\"Non-Severe\", \"reason\": \"...\"}}"
        pz, rz = call_gpt(client, p_zero)
        zero_preds.append(pz)
        zero_reasons.append(rz)

        # 5.2 Fixed Few-shot
        ex_str = "\n".join([f"Example: \"{e['summary']}\" -> {e['severity']}\nReason: {e['reason']}" for e in FIXED_FEW_SHOT_EXAMPLES])
        p_few = f"Reference Examples:\n{ex_str}\n\nNow classify this new bug:\n\"{summary}\"\n\nRespond in JSON: {{\"severity\": \"Severe\"|\"Non-Severe\", \"reason\": \"...\"}}"
        pf, rf = call_gpt(client, p_few)
        few_preds.append(pf)
        few_reasons.append(rf)

        # 5.3 Lightweight RAG
        sims = cosine_similarity(q_vec.reshape(1, -1), hist_embeddings)[0]
        eligible = np.array([str(bug_id) not in {str(row['bug_id'])} for bug_id in hist_df['bug_id']])
        top_k_idx = np.argsort(np.where(eligible, sims, -np.inf))[::-1][:TOP_K_SIMILAR]
        top_cases = hist_df.iloc[top_k_idx].to_dict('records')
        for i, c_idx in enumerate(top_k_idx):
            top_cases[i]['similarity'] = float(sims[c_idx])

        rag_str = "\n".join([f"Similar Historical Bug #{c['bug_id']} (Sim: {c['similarity']:.2f}): \"{c['summary']}\" -> Severity: {c['severity_label']}" for c in top_cases])
        p_rag = (
            f"Similar Historical Bug Reports Retrieved (Ranked by semantic similarity):\n{rag_str}\n\n"
            f"Target Bug to Classify:\n\"{summary}\"\n\n"
            f"Respond in JSON: {{\"severity\": \"Severe\"|\"Non-Severe\", \"reason\": \"...\"}}"
        )
        pr, rr = call_gpt(client, p_rag)
        rag_preds.append(pr)
        rag_reasons.append(rr)

        top_log = ", ".join([f"#{c['bug_id']}({c['severity_label']})" for c in top_cases])
        rag_retrieved_logs.append(top_log)

    eval_df['ZeroShot_Pred'] = zero_preds
    eval_df['ZeroShot_Reason'] = zero_reasons
    eval_df['FewShot_Pred'] = few_preds
    eval_df['FewShot_Reason'] = few_reasons
    eval_df['RAG_Pred'] = rag_preds
    eval_df['RAG_Reason'] = rag_reasons
    eval_df['RAG_Retrieved_Examples'] = rag_retrieved_logs

    eval_df.to_csv(os.path.join(RESULTS_DIR, 'predictions_detailed.csv'), index=False, encoding='utf-8')

    # 6. Metrics Computation
    y_true = eval_df['severity_label'].tolist()
    m_zero = compute_metrics(y_true, zero_preds, "Zero-shot")
    m_few = compute_metrics(y_true, few_preds, "Fixed Few-shot")
    m_rag = compute_metrics(y_true, rag_preds, "Lightweight RAG")
    m_tfidf = compute_metrics(y_true, tfidf_preds, "TF-IDF + LinearSVC")

    all_metrics = [m_zero, m_few, m_rag, m_tfidf]
    summary_df = pd.DataFrame([{k: v for k, v in m.items() if k != "Confusion_Matrix"} for m in all_metrics])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'summary_results.csv'), index=False, encoding='utf-8')

    # 7. Visualization: Metrics Comparison (Figure 5)
    plt.figure(figsize=(10, 5), dpi=300)
    df_melt = summary_df.melt(id_vars="Approach", value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"], var_name="Metric", value_name="Score")
    palette = ["#3b82f6", "#f59e0b", "#10b981", "#6366f1"]
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Approach", palette=palette)
    plt.title("Bug Severity Classification Performance Benchmark Across Evaluated Approaches", fontsize=12, fontweight='bold', pad=12)
    plt.ylim(0, 105)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1.2), ha='center', va='bottom', color='#1e293b', fontweight='bold', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'metrics_comparison.png'), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, 'figure5_metrics_comparison.png'), dpi=300)
    plt.close()

    # 8. Visualization: Confusion Matrices (Figure 6)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)
    labels = ["Severe", "Non-Sev"]
    for i, m in enumerate(all_metrics):
        sns.heatmap(m["Confusion_Matrix"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels if i == 0 else False,
                    ax=axes[i], annot_kws={"size": 12, "weight": "bold", "color": "#0f172a"})
        axes[i].set_title(f"{m['Approach']}\n(Acc: {m['Accuracy (%)']}%, F1: {m['F1-Score (%)']}%)", fontweight='bold', fontsize=10.5, pad=8)
        axes[i].set_xlabel("Predicted Label", fontsize=9.5, fontweight='bold')
        if i == 0:
            axes[i].set_ylabel("True Ground Truth", fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=300)
    plt.savefig(os.path.join(RESULTS_DIR, 'figure6_confusion_matrices.png'), dpi=300)
    plt.close()

    # 9. Qualitative Samples
    sample_20 = eval_df[['bug_id', 'summary', 'severity_label', 'RAG_Pred', 'RAG_Reason', 'RAG_Retrieved_Examples']].head(20).copy()
    sample_20['Correct'] = sample_20['severity_label'] == sample_20['RAG_Pred']
    sample_20.to_csv(os.path.join(RESULTS_DIR, 'manual_review_20_samples.csv'), index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print("🏆 FINAL BENCHMARK RESULTS SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"🎉 Results, tables, and figures saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
