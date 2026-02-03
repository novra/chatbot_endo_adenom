# ==============================================================================
# evaluation.py - Final Script v2 (dengan Retry Logic & Peningkatan Stabilitas)
# ==============================================================================

import os
import pandas as pd
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from tqdm import tqdm
from langdetect import detect, LangDetectException

# Impor dari LangChain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEmbeddings
)

# Impor dari RAGAS dan Datasets
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

# Impor kelas ChatBot dari file lokal
from chatbot_adenom.chatbot_old import ChatBot


# === 1. KONFIGURASI DAN INISIALISASI GLOBAL ===
print("Mempersiapkan konfigurasi global...")
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_KEY tidak ditemukan di file .env")

# Inisialisasi LLM untuk Evaluasi
# Inisialisasi LLM untuk Evaluasi
llm_evaluator_endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.2,
    max_new_tokens=2048,
    client_kwargs={"timeout": 180}
)
# <-- SOLUSI FINAL: Atur retry langsung di dalam constructor, bukan dengan .with_retry()
# Ini adalah cara yang lebih modern dan kompatibel.
chat_evaluator_llm = ChatHuggingFace(
    llm=llm_evaluator_endpoint,
    max_retries=3  # Coba lagi maksimal 3 kali jika gagal
)

# Inisialisasi Model Embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# === 2. DEFINISI RANTAI (CHAINS) UNTUK PEMBUATAN DATA ===

# Rantai penerjemahan (dengan retry)
translation_prompt = PromptTemplate.from_template(
    "Anda adalah penerjemah ahli. Terjemahkan teks berikut ke dalam Bahasa Indonesia yang alami dan akurat. "
    "Jangan menambahkan komentar atau penjelasan apa pun, berikan hanya hasil terjemahannya saja.\n\nTEKS:\n{text}"
)
translation_chain = translation_prompt | chat_evaluator_llm | StrOutputParser() # chat_evaluator_llm sudah punya retry

# Definisi struktur output Q&A
class QAPair(BaseModel):
    question: str = Field(description="Pertanyaan yang dibuat dari konteks dalam Bahasa Indonesia.")
    ground_truth: str = Field(description="Jawaban ringkas dan akurat untuk pertanyaan, HANYA dari konteks yang diberikan, dalam Bahasa Indonesia.")

# Rantai pembuatan Q&A (dengan retry)
json_parser = JsonOutputParser(pydantic_object=QAPair)
qa_generation_prompt = PromptTemplate(
    template="""
Anda adalah seorang ahli medis yang bertugas membuat set data tanya-jawab.
Berdasarkan KONTEKS di bawah ini, buatlah satu pasangan pertanyaan dan jawaban yang relevan.

PERATURAN:
- Pertanyaan harus spesifik dan hanya bisa dijawab menggunakan informasi dari KONTEKS.
- Jawaban (`ground_truth`) harus ringkas, akurat, dan diekstrak langsung dari KONTEKS.
- Jangan menambahkan informasi di luar KONTEKS.
- PENTING: Pertanyaan dan jawaban HARUS dibuat dalam Bahasa Indonesia.

KONTEKS:
{context}

{format_instructions}
""",
    input_variables=["context"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)
# <-- SOLUSI MASALAH 1: Rantai Q&A juga menggunakan LLM yang sudah memiliki logika retry
qa_generation_chain = qa_generation_prompt | chat_evaluator_llm | json_parser


# === 3. FUNGSI-FUNGSI PEMBANTU (Tidak ada perubahan di sini) ===

def load_and_chunk_single_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
            if not text.strip(): return []
        doc_obj = Document(page_content=text, metadata={"source": os.path.basename(pdf_path)})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        return text_splitter.split_documents([doc_obj])
    except Exception as e:
        print(f"  Gagal memproses file {os.path.basename(pdf_path)}: {e}")
        return []

# Ganti fungsi lama dengan yang ini di evaluation.py

def generate_qa_test_set_for_chunks(chunks, num_questions, cache_path):
    """Membuat atau memuat dataset Q&A dari cache, dengan validasi data yang ketat."""
    if os.path.exists(cache_path):
        print(f"  Memuat test set dari cache: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            # Lakukan validasi juga pada data yang dimuat dari cache
            cached_data = json.load(f)
            validated_data = [
                item for item in cached_data 
                if isinstance(item, dict) and "question" in item and "ground_truth" in item
            ]
            if len(validated_data) < len(cached_data):
                print("  Peringatan: Beberapa data dari cache tidak valid dan akan dilewati.")
            return validated_data

    print(f"  Membuat test set baru ({num_questions} pertanyaan)...")
    qa_pairs = []
    
    num_to_gen = len(chunks) if len(chunks) < num_questions else num_questions
    if num_to_gen < num_questions:
        print(f"  Peringatan: Chunks ({len(chunks)}) < pertanyaan diminta ({num_questions}). Hanya membuat {num_to_gen} pertanyaan.")

    for chunk in tqdm(chunks[:num_to_gen], desc="  Membuat Q&A"):
        try:
            original_text = chunk.page_content
            processed_text = original_text
            try:
                lang = detect(original_text)
            except LangDetectException:
                lang = 'id'
            
            if lang != 'id':
                print(f"\n    Mendeteksi bahasa '{lang}', menerjemahkan...")
                processed_text = translation_chain.invoke({"text": original_text})
            
            # Panggil LLM untuk membuat Q&A
            qa_pair = qa_generation_chain.invoke({"context": processed_text})
            
            # --- BLOK VALIDASI DATA (SOLUSI) ---
            # Periksa apakah output adalah dictionary dan memiliki semua kunci yang diperlukan.
            if isinstance(qa_pair, dict) and "question" in qa_pair and "ground_truth" in qa_pair:
                # Hanya jika valid, tambahkan ke daftar
                qa_pairs.append(qa_pair)
            else:
                # Jika tidak valid, cetak pesan dan lewati
                print(f"\n    Gagal validasi output LLM (kunci 'question'/'ground_truth' hilang): {qa_pair}")
            # --- AKHIR BLOK VALIDASI ---

        except Exception as e:
            print(f"\n    Gagal memproses chunk (exception): {e}")
            
    # Simpan hanya data yang sudah tervalidasi ke cache
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=4, ensure_ascii=False)
    return qa_pairs

def run_evaluation_for_qa_set(chatbot, qa_set):
    rag_data = []
    for qa_pair in tqdm(qa_set, desc="  Menjalankan Chatbot"):
        response = chatbot.ask(qa_pair["question"])
        retrieved_docs = chatbot.source_retriever_chain.invoke(qa_pair["question"])
        rag_data.append({
            "question": qa_pair["question"], "answer": response.get("answer", ""),
            "contexts": [doc.page_content for doc in retrieved_docs], "ground_truth": qa_pair["ground_truth"]
        })
    dataset = Dataset.from_pandas(pd.DataFrame(rag_data))
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    print("  Memulai evaluasi RAGAS...")
    results = evaluate(dataset, metrics=metrics, llm=chat_evaluator_llm, embeddings=embeddings_model)
    print("  Evaluasi RAGAS selesai.")
    return results


# === 4. PROSES UTAMA EKSEKUSI ===

if __name__ == "__main__":
    DATA_FOLDER, CACHE_FOLDER, MIN_QUESTIONS_PER_DOC = "./data_adenomyosis", "./qa_cache", 10
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    print("\n>>> Langkah 1: Inisialisasi ChatBot...")
    chatbot_instance = ChatBot()
    if chatbot_instance.rag_chain is None:
        print("Inisialisasi ChatBot gagal. Program berhenti."); exit()

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.pdf')]
    all_results_summary = []
    print(f"\n>>> Langkah 2: Memulai proses evaluasi untuk {len(pdf_files)} dokumen...")

    for i, filename in enumerate(pdf_files):
        print(f"\n{'='*60}\nPROSES DOKUMEN {i+1}/{len(pdf_files)}: {filename}\n{'='*60}")
        pdf_path = os.path.join(DATA_FOLDER, filename)
        doc_chunks = load_and_chunk_single_pdf(pdf_path)
        if not doc_chunks: print("  Tidak ada konten, lanjut."); continue
        print(f"  Dokumen dipecah menjadi {len(doc_chunks)} chunks.")
        
        cache_file = os.path.join(CACHE_FOLDER, f"{filename}.json")
        qa_dataset = generate_qa_test_set_for_chunks(doc_chunks, MIN_QUESTIONS_PER_DOC, cache_file)
        if not qa_dataset: print("  Gagal membuat Q&A, lanjut."); continue
        print(f"  Berhasil membuat/memuat {len(qa_dataset)} pertanyaan.")

        evaluation_results = run_evaluation_for_qa_set(chatbot_instance, qa_dataset)
        
        # <-- SOLUSI MASALAH 3: Penanganan hasil yang lebih aman
        if isinstance(evaluation_results, dict):
            # Versi RAGAS baru mengembalikan dictionary. Kita ubah ke DataFrame.
            scores_df = pd.DataFrame.from_records([evaluation_results])
            mean_scores = scores_df.iloc[0].to_dict() # Ambil baris pertama sebagai dictionary
            mean_scores['document'] = filename
            all_results_summary.append(mean_scores)
            
            print("\n  --- Skor Rata-rata untuk Dokumen Ini ---")
            for metric, score in mean_scores.items():
                if metric != 'document': print(f"  {metric:<20}: {score:.4f}")
            print("  --------------------------------------")
        else:
            print("  Gagal mendapatkan hasil evaluasi dalam format yang diharapkan.")
            
    if all_results_summary:
        print(f"\n{'='*60}\n          LAPORAN RINGKASAN EVALUASI FINAL\n{'='*60}")
        summary_df = pd.DataFrame(all_results_summary)
        summary_df = summary_df[['document'] + [col for col in summary_df.columns if col != 'document']]
        print(summary_df.to_string())
        output_path = "evaluation_summary_per_document.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nLaporan ringkasan lengkap telah disimpan di: {output_path}")
    else:
        print("\nTidak ada hasil evaluasi yang berhasil dikumpulkan.")