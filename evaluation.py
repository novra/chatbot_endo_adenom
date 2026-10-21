"""
Run RAGAS evaluation for the active adenomyosis/endometriosis chatbot.

Usage:
    python evaluation.py --testset ragas_testset.csv --limit 5
    python evaluation.py --testset dataset.json ragas_supplement_testset.csv

Required columns in the test set:
    question, ground_truth

Optional column:
    conversation_id

Rows with the same ``conversation_id`` are evaluated sequentially with the
same chat history, just like the Streamlit application. Without that column,
every row is treated as an independent conversation.

Outputs:
    outputs/ragas_eval_dataset.csv
    outputs/ragas_scores.csv
    outputs/ragas_summary.json
    outputs/ragas_summary_by_category.csv
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
DEFAULT_EVALUATOR_MODEL = "Qwen/Qwen3-4B-Instruct-2507:nscale"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
METRIC_COLUMNS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


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


def infer_category(question: str) -> str:
    normalized = question.lower()
    herbal_terms = (
        "herbal",
        "jamu",
        "kunyit",
        "kurkumin",
        "mahkota dewa",
        "komplementer",
    )
    if any(term in normalized for term in herbal_terms):
        return "herbal"
    has_adenomyosis = "adenomiosis" in normalized or "adenomyosis" in normalized
    has_endometriosis = "endometriosis" in normalized or "endometrioma" in normalized
    if has_adenomyosis and has_endometriosis:
        return "keduanya"
    if has_adenomyosis:
        return "adenomiosis"
    if has_endometriosis:
        return "endometriosis"
    return "guardrail_safety"


def load_testset(path: str) -> list[dict[str, Any]]:
    testset_path = Path(path)
    if not testset_path.exists():
        raise FileNotFoundError(f"Test set tidak ditemukan: {testset_path}")

    if testset_path.suffix.lower() == ".json":
        df = pd.read_json(testset_path)
        df = df.rename(columns={"user_input": "question", "reference": "ground_truth"})
    elif testset_path.suffix.lower() == ".csv":
        df = pd.read_csv(testset_path)
    else:
        raise ValueError(f"Format test set tidak didukung: {testset_path.suffix}")

    required_columns = {"question", "ground_truth"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Test set harus memiliki kolom {sorted(required_columns)}. "
            f"Kolom yang belum ada: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["question", "ground_truth"])
    selected_columns = ["question", "ground_truth"]
    for optional_column in ("conversation_id", "category", "question_type", "id"):
        if optional_column in df.columns:
            selected_columns.append(optional_column)

    rows = df[selected_columns].to_dict(orient="records")
    for row in rows:
        if not row.get("category") or pd.isna(row.get("category")):
            row["category"] = infer_category(str(row["question"]))
        if not row.get("question_type") or pd.isna(row.get("question_type")):
            row["question_type"] = "unspecified"
    if not rows:
        raise ValueError("Test set kosong setelah validasi.")
    return rows


def load_testsets(paths: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    """Combine CSV/JSON test sets and remove duplicate questions."""
    combined: list[dict[str, str]] = []
    seen_questions: set[str] = set()

    for path in paths:
        for row in load_testset(path):
            question_key = " ".join(str(row["question"]).lower().split())
            if question_key in seen_questions:
                continue
            seen_questions.add(question_key)
            combined.append(row)

    if limit is not None:
        combined = combined[:limit]
    if not combined:
        raise ValueError("Tidak ada pertanyaan unik setelah test set digabungkan.")
    return combined


def build_evaluator(token: str, model_name: str, max_tokens: int = 4096):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI

    extra_body: dict[str, Any] = {}
    if model_name.lower().startswith("qwen/"):
        # Qwen 3.x enables thinking by default. Thinking blocks waste tokens
        # and are not valid input for RAGAS' Pydantic output parser.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    llm = ChatOpenAI(
        model=model_name,
        api_key=token,
        base_url=HF_ROUTER_BASE_URL,
        # RAGAS requires strict structured output. Deterministic decoding and
        # a larger output budget reduce malformed/truncated JSON responses.
        temperature=0,
        max_tokens=max_tokens,
        timeout=300,
        max_retries=4,
        # Complement RAGAS' Pydantic prompts with provider-level JSON mode so
        # the response contains no prose or Markdown fences.
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body=extra_body,
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


def ask_with_retrieval_trace(
    chatbot: Any,
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Run the public chatbot flow and capture the documents it actually uses.

    ``ChatBot.ask`` performs guardrail/contextual-question processing before its
    RAG chain retrieves documents. Calling the retriever again from this script
    with the raw question can therefore evaluate a different context. The small
    wrapper below observes the first retrieval made by the normal ``ask`` flow
    without changing that flow or issuing a separate retrieval request.
    """
    original_retrieve = chatbot._retrieve_relevant_docs
    retrieval_calls: list[list[Any]] = []

    def traced_retrieve(retrieval_question: str, limit: int = 3):
        documents = original_retrieve(retrieval_question, limit=limit)
        retrieval_calls.append(list(documents))
        return documents

    chatbot._retrieve_relevant_docs = traced_retrieve
    try:
        response = chatbot.ask(question, chat_history=chat_history)
    finally:
        chatbot._retrieve_relevant_docs = original_retrieve

    # The first call is the retrieval used to construct the LLM context. A
    # guardrail rejection performs no retrieval and correctly yields [].
    used_documents = retrieval_calls[0] if retrieval_calls else []
    return response, used_documents


def is_failed_chatbot_response(response: dict[str, Any]) -> bool:
    """Identify transport/system fallbacks that must not be scored as answers."""
    answer = str(response.get("answer", "")).strip().lower()
    failure_markers = (
        "gangguan teknis",
        "error sistem",
        "sistem belum siap",
        "server ai sedang sibuk",
        "error autentikasi",
        "model sedang loading",
    )
    return not answer or any(marker in answer for marker in failure_markers)


def collect_rag_outputs(
    chatbot: Any,
    testset: list[dict[str, Any]],
    output_dir: str,
    resume: bool = False,
    answer_retries: int = 2,
) -> pd.DataFrame:
    records = []
    conversation_histories: dict[str, list[dict[str, Any]]] = {}
    checkpoint_path = Path(output_dir) / "ragas_eval_dataset.csv"
    existing_by_question: dict[str, dict[str, Any]] = {}
    if resume and checkpoint_path.exists():
        existing_df = pd.read_csv(checkpoint_path)
        existing_by_question = {
            str(row["question"]): row.to_dict() for _, row in existing_df.iterrows()
        }
        print(f"Melanjutkan dari checkpoint: {len(existing_by_question)} jawaban tersedia.")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(testset, start=1):
        question = str(item["question"])
        conversation_id = item.get("conversation_id")
        has_conversation = pd.notna(conversation_id)
        history_key = str(conversation_id) if has_conversation else ""
        chat_history = conversation_histories.setdefault(history_key, []) if has_conversation else []

        print(f"\n[{index}/{len(testset)}] Menjalankan chatbot")
        print(f"Pertanyaan: {question}")

        cached_answer_failed = (
            question in existing_by_question
            and is_failed_chatbot_response(existing_by_question[question])
        )
        if question in existing_by_question and not cached_answer_failed:
            cached_record = existing_by_question[question]
            cached_record["contexts"] = json.loads(str(cached_record["contexts"]))
            cached_record.setdefault("category", item.get("category", infer_category(question)))
            cached_record.setdefault("question_type", item.get("question_type", "unspecified"))
            if has_conversation:
                # Rebuild the conversation state from cached turns so later
                # uncached follow-ups still receive the correct history.
                chat_history.extend(
                    [
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": str(cached_record.get("answer", "")),
                        },
                    ]
                )
            records.append(cached_record)
            print("Menggunakan jawaban dari checkpoint.")
            continue

        response: dict[str, Any] = {}
        retrieved_docs: list[Any] = []
        for attempt in range(1, answer_retries + 2):
            response, retrieved_docs = ask_with_retrieval_trace(
                chatbot,
                question,
                chat_history=chat_history[-20:],
            )
            if not is_failed_chatbot_response(response):
                break
            print(f"Jawaban chatbot gagal pada percobaan {attempt}; mencoba ulang.")
        if is_failed_chatbot_response(response):
            raise RuntimeError(
                f"Chatbot tetap gagal setelah {answer_retries + 1} percobaan: {question}"
            )

        if has_conversation:
            chat_history.extend(
                [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": response.get("answer", ""),
                        "sources": response.get("sources", []),
                        "metadata": response.get("metadata", {}),
                    },
                ]
            )

        response_metadata = response.get("metadata", {})
        guardrail = response_metadata.get("guardrail", {})
        generation = response_metadata.get("_generation", {})

        records.append(
            {
                "question": question,
                "answer": response.get("answer", ""),
                "contexts": [doc.page_content for doc in retrieved_docs],
                "ground_truth": item["ground_truth"],
                "category": item.get("category", infer_category(question)),
                "question_type": item.get("question_type", "unspecified"),
                "conversation_id": history_key if has_conversation else "",
                "guardrail_blocked": bool(guardrail.get("blocked", False)),
                "generator_model": generation.get("model", ""),
                "generator_primary_model": generation.get("primary_model", ""),
                "generator_fallback_rank": generation.get("fallback_rank", ""),
                "generator_fallback_used": generation.get("fallback_used", False),
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

        checkpoint_df = pd.DataFrame(records).copy()
        checkpoint_df["contexts"] = checkpoint_df["contexts"].apply(json.dumps)
        checkpoint_df.to_csv(checkpoint_path, index=False)

    return pd.DataFrame(records)


def run_ragas_batch(
    dataset_df: pd.DataFrame,
    llm: Any,
    embeddings: Any,
    timeout: int,
    max_retries: int,
    max_workers: int,
) -> pd.DataFrame:
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": str(row["question"]),
                "response": str(row["answer"]),
                "retrieved_contexts": list(row["contexts"]),
                "reference": str(row["ground_truth"]),
            }
            for _, row in dataset_df.iterrows()
        ]
    )
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(
            name="context_precision",
            llm=evaluator_llm,
        ),
        LLMContextRecall(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=20,
            max_workers=max_workers,
            seed=42,
        ),
        raise_exceptions=False,
    )
    return result.to_pandas()


def run_ragas(
    dataset_df: pd.DataFrame,
    llm: Any,
    embeddings: Any,
    output_dir: str,
    batch_size: int,
    timeout: int,
    max_retries: int,
    max_workers: int,
    resume: bool = False,
) -> pd.DataFrame:
    """Evaluate in bounded batches and checkpoint after every batch."""
    score_file = Path(output_dir) / "ragas_scores.csv"
    completed_questions: set[str] = set()
    score_frames: list[pd.DataFrame] = []

    if resume and score_file.exists():
        existing_scores = pd.read_csv(score_file)
        if "category" not in existing_scores.columns:
            existing_scores["category"] = existing_scores["question"].map(infer_category)
        if "question_type" not in existing_scores.columns:
            existing_scores["question_type"] = "unspecified"
        for metric in METRIC_COLUMNS:
            if metric not in existing_scores.columns:
                existing_scores[metric] = float("nan")

        # A row is complete only when every requested metric parsed
        # successfully. Partial rows are intentionally retried on resume.
        complete_mask = existing_scores[list(METRIC_COLUMNS)].notna().all(axis=1)
        complete_scores = existing_scores.loc[complete_mask].copy()
        completed_questions = set(complete_scores["question"].astype(str))
        if not complete_scores.empty:
            score_frames.append(complete_scores)
        print(
            f"Melanjutkan {len(completed_questions)} skor lengkap dari checkpoint; "
            f"{int((~complete_mask).sum())} skor parsial/NaN akan diulang."
        )

    pending_df = dataset_df[~dataset_df["question"].isin(completed_questions)].reset_index(drop=True)
    total_batches = (len(pending_df) + batch_size - 1) // batch_size
    print(
        f"\nMemulai evaluasi RAGAS: {len(pending_df)} pertanyaan, "
        f"{total_batches} batch, maksimal {max_workers} worker."
    )

    for start in range(0, len(pending_df), batch_size):
        batch_number = start // batch_size + 1
        batch_df = pending_df.iloc[start:start + batch_size].reset_index(drop=True)
        print(f"\nBatch RAGAS {batch_number}/{total_batches} ({len(batch_df)} pertanyaan)")
        batch_scores = run_ragas_batch(
            batch_df,
            llm,
            embeddings,
            timeout,
            max_retries,
            max_workers,
        )
        for metadata_column in ("category", "question_type"):
            batch_scores[metadata_column] = batch_df[metadata_column].tolist()
        score_frames.append(batch_scores)
        combined_scores = pd.concat(score_frames, ignore_index=True)
        combined_scores.to_csv(score_file, index=False)

    if not score_frames:
        raise RuntimeError("Tidak ada hasil RAGAS yang dihasilkan.")
    return pd.concat(score_frames, ignore_index=True)


def save_results(score_df: pd.DataFrame, dataset_df: pd.DataFrame, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_file = output_path / "ragas_eval_dataset.csv"
    persisted_dataset = dataset_df.copy()
    persisted_dataset["contexts"] = persisted_dataset["contexts"].apply(json.dumps)
    persisted_dataset.to_csv(dataset_file, index=False)

    score_file = output_path / "ragas_scores.csv"
    score_df.to_csv(score_file, index=False)

    summary: dict[str, Any] = {
        "questions": int(len(score_df)),
        "metrics": {},
    }
    for column in METRIC_COLUMNS:
        valid_count = int(score_df[column].notna().sum())
        summary["metrics"][column] = {
            "mean": None if valid_count == 0 else float(score_df[column].mean()),
            "valid": valid_count,
            "missing": int(len(score_df) - valid_count),
            "coverage": valid_count / len(score_df) if len(score_df) else 0.0,
        }

    if all(item["valid"] == 0 for item in summary["metrics"].values()):
        raise RuntimeError(
            "Semua metrik RAGAS menghasilkan NaN. Periksa dukungan model evaluator "
            "pada Hugging Face Inference Providers dan log exception di atas."
        )
    summary_file = output_path / "ragas_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    category_summary = score_df.groupby("category")[list(METRIC_COLUMNS)].agg(
        ["mean", "count"]
    )
    category_file = output_path / "ragas_summary_by_category.csv"
    category_summary.to_csv(category_file)

    print("\nRingkasan skor rata-rata:")
    for metric, values in summary["metrics"].items():
        mean_display = "N/A" if values["mean"] is None else f"{values['mean']:.4f}"
        print(
            f"- {metric}: {mean_display} "
            f"(coverage {values['valid']}/{summary['questions']})"
        )

    print(f"\nDataset evaluasi: {dataset_file}")
    print(f"Skor per pertanyaan: {score_file}")
    print(f"Ringkasan JSON: {summary_file}")
    print(f"Ringkasan per kategori: {category_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi chatbot dengan RAGAS.")
    parser.add_argument(
        "--testset",
        nargs="+",
        default=[DEFAULT_TESTSET],
        help="Satu atau beberapa path test set CSV/JSON.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah pertanyaan.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder output.")
    parser.add_argument("--batch-size", type=int, default=5, help="Pertanyaan per batch RAGAS.")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout tiap pekerjaan RAGAS.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry tiap pekerjaan RAGAS.")
    parser.add_argument("--max-workers", type=int, default=2, help="Jumlah worker paralel RAGAS.")
    parser.add_argument(
        "--evaluator-max-tokens",
        type=int,
        default=4096,
        help="Batas token keluaran evaluator agar JSON RAGAS tidak terpotong.",
    )
    parser.add_argument("--resume", action="store_true", help="Lanjutkan dari checkpoint output.")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Kumpulkan jawaban chatbot dan metadata generator tanpa menjalankan RAGAS.",
    )
    parser.add_argument(
        "--answer-retries",
        type=int,
        default=2,
        help="Jumlah percobaan ulang jika jawaban chatbot berupa error teknis.",
    )
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

    testset = load_testsets(args.testset, args.limit)
    llm, embeddings = build_evaluator(
        token,
        args.evaluator_model,
        args.evaluator_max_tokens,
    )
    dataset_df = collect_rag_outputs(
        chatbot,
        testset,
        args.output_dir,
        args.resume,
        args.answer_retries,
    )
    if args.collect_only:
        dataset_file = Path(args.output_dir) / "ragas_eval_dataset.csv"
        print(f"\nPengumpulan jawaban selesai: {dataset_file}")
        return
    score_df = run_ragas(
        dataset_df,
        llm,
        embeddings,
        args.output_dir,
        args.batch_size,
        args.timeout,
        args.max_retries,
        args.max_workers,
        args.resume,
    )
    save_results(score_df, dataset_df, args.output_dir)


if __name__ == "__main__":
    main()
