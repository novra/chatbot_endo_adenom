"""
Run RAGAS evaluation for the active adenomyosis/endometriosis chatbot.

Usage:
    python evaluation.py --testset ragas_testset.csv --limit 5

Required columns in the test set:
    question, ground_truth

Outputs:
    outputs/ragas_eval_dataset.csv
    outputs/ragas_scores.csv
    outputs/ragas_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


DEFAULT_TESTSET = "ragas_testset.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_EVALUATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct:fastest"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def get_hf_token() -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError(
            "HF_TOKEN atau HUGGINGFACE_API_KEY belum ditemukan. "
            "Tambahkan salah satunya di .env sebelum menjalankan evaluasi."
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def load_testset(path: str, limit: int | None = None) -> list[dict[str, str]]:
    testset_path = Path(path)
    if not testset_path.exists():
        raise FileNotFoundError(f"Test set tidak ditemukan: {testset_path}")

    df = pd.read_csv(testset_path)
    required_columns = {"question", "ground_truth"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Test set harus memiliki kolom {sorted(required_columns)}. "
            f"Kolom yang belum ada: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["question", "ground_truth"])
    if limit:
        df = df.head(limit)

    rows = df[["question", "ground_truth"]].to_dict(orient="records")
    if not rows:
        raise ValueError("Test set kosong setelah validasi.")
    return rows


def build_evaluator(token: str, model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_name,
        api_key=token,
        base_url=HF_ROUTER_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
        timeout=180,
        max_retries=3,
    )
    embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
    return llm, embeddings


def verify_hf_router(token: str) -> None:
    from openai import OpenAI

    client = OpenAI(
        api_key=token,
        base_url=HF_ROUTER_BASE_URL,
        timeout=20,
        max_retries=1,
    )
    try:
        client.models.list()
    except Exception as error:
        raise RuntimeError(
            "Tidak bisa terhubung ke Hugging Face router "
            f"({HF_ROUTER_BASE_URL}). Periksa DNS/koneksi internet, firewall, "
            "proxy, dan izin token untuk Inference Providers. "
            f"Detail: {type(error).__name__}: {error}"
        ) from error


def collect_rag_outputs(chatbot: Any, testset: list[dict[str, str]]) -> pd.DataFrame:
    records = []

    for index, item in enumerate(testset, start=1):
        question = item["question"]
        print(f"\n[{index}/{len(testset)}] Menjalankan chatbot")
        print(f"Pertanyaan: {question}")

        response = chatbot.ask(question)
        retrieval_query = chatbot._build_retrieval_query(question)
        retrieved_docs = chatbot.source_retriever_chain.invoke(retrieval_query)

        records.append(
            {
                "question": question,
                "answer": response.get("answer", ""),
                "contexts": [doc.page_content for doc in retrieved_docs],
                "ground_truth": item["ground_truth"],
                "sources": "; ".join(
                    sorted(
                        {
                            str(doc.metadata.get("source", "Unknown"))
                            for doc in retrieved_docs
                        }
                    )
                ),
            }
        )

    return pd.DataFrame(records)


def run_ragas(dataset_df: pd.DataFrame, llm: Any, embeddings: Any):
    from datasets import Dataset, Features, Sequence, Value
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(
        {
            "question": dataset_df["question"].astype(str).tolist(),
            "answer": dataset_df["answer"].astype(str).tolist(),
            "contexts": dataset_df["contexts"].tolist(),
            "ground_truth": dataset_df["ground_truth"].astype(str).tolist(),
        },
        features=Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "contexts": Sequence(Value("string")),
                "ground_truth": Value("string"),
            }
        ),
    )
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    print("\nMemulai evaluasi RAGAS...")
    return evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)


def save_results(result: Any, dataset_df: pd.DataFrame, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_file = output_path / "ragas_eval_dataset.csv"
    dataset_df.to_csv(dataset_file, index=False)

    try:
        score_df = result.to_pandas()
    except AttributeError:
        score_df = pd.DataFrame([dict(result)])

    score_file = output_path / "ragas_scores.csv"
    score_df.to_csv(score_file, index=False)

    numeric_columns = score_df.select_dtypes(include="number").columns
    summary = {column: float(score_df[column].mean()) for column in numeric_columns}
    summary_file = output_path / "ragas_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nRingkasan skor rata-rata:")
    for metric, value in summary.items():
        print(f"- {metric}: {value:.4f}")

    print(f"\nDataset evaluasi: {dataset_file}")
    print(f"Skor per pertanyaan: {score_file}")
    print(f"Ringkasan JSON: {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi chatbot dengan RAGAS.")
    parser.add_argument("--testset", default=DEFAULT_TESTSET, help="Path CSV test set.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah pertanyaan.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder output.")
    parser.add_argument(
        "--evaluator-model",
        default=os.getenv("RAGAS_EVALUATOR_MODEL", DEFAULT_EVALUATOR_MODEL),
        help="Model HuggingFace untuk LLM evaluator RAGAS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = get_hf_token()
    os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", ".ragas_chroma_db")
    verify_hf_router(token)

    from chatbot import ChatBot

    print("Menginisialisasi ChatBot aktif...")
    chatbot = ChatBot()
    if chatbot.rag_chain is None:
        raise RuntimeError("ChatBot gagal diinisialisasi. RAG chain belum tersedia.")

    testset = load_testset(args.testset, args.limit)
    llm, embeddings = build_evaluator(token, args.evaluator_model)
    dataset_df = collect_rag_outputs(chatbot, testset)
    result = run_ragas(dataset_df, llm, embeddings)
    save_results(result, dataset_df, args.output_dir)


if __name__ == "__main__":
    main()
