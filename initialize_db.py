#!/usr/bin/env python
"""
Script sederhana untuk inisialisasi ChromaDB dari PDF files
Tanpa UI Streamlit - hanya fokus ke database building
"""

import os
import sys
from pathlib import Path

# ============================================
# FIX: Disable ChromaDB Telemetry
# ============================================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["POSTHOG_DISABLED"] = "1"

try:
    import fitz  # PyMuPDF
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    import re
    from datetime import datetime
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

def _resolve_persist_directory():
    """Pilih direktori Chroma yang writable."""
    candidates = [
        Path("./chroma_db_adenomyosis"),
        Path("/tmp/chroma_db_adenomyosis"),
    ]

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            with open(test_file, "w", encoding="utf-8") as handle:
                handle.write("ok")
            test_file.unlink(missing_ok=True)
            print(f"✅ Using persist directory: {path}")
            return str(path)
        except Exception as e:
            print(f"❌ Cannot use {path}: {e}")
            continue

    print("⚠️ Falling back to default directory")
    return "./chroma_db_adenomyosis"

def _extract_metadata_from_filename(filename):
    """Ekstraksi metadata dari nama file PDF."""
    metadata = {
        "source": filename,
        "source_type": "Unknown",
        "validity_level": "medium",
        "year": 0,
        "category": "general"
    }
    
    # Extract year from filename
    year_match = re.search(r'(19|20)\d{2}', filename)
    if year_match:
        metadata["year"] = int(year_match.group())
    
    # Kategorisasi berdasarkan nama file
    filename_lower = filename.lower()
    
    # Source type detection
    if any(keyword in filename_lower for keyword in ['jurnal', 'journal', 'paper', 'research']):
        metadata["source_type"] = "Jurnal Ilmiah"
        metadata["validity_level"] = "high"
    elif any(keyword in filename_lower for keyword in ['guideline', 'pedoman', 'clinical']):
        metadata["source_type"] = "Guideline Klinis"
        metadata["validity_level"] = "very_high"
    elif any(keyword in filename_lower for keyword in ['textbook', 'buku', 'book']):
        metadata["source_type"] = "Buku Teks"
        metadata["validity_level"] = "high"
    elif any(keyword in filename_lower for keyword in ['review', 'tinjauan']):
        metadata["source_type"] = "Review Article"
        metadata["validity_level"] = "high"
    
    # Category detection
    if any(keyword in filename_lower for keyword in ['adenomyosis', 'adenom']):
        metadata["category"] = "adenomyosis"
    elif any(keyword in filename_lower for keyword in ['endometriosis', 'endo']):
        metadata["category"] = "endometriosis"
    elif any(keyword in filename_lower for keyword in ['diagnosis', 'diagnosa', 'diagnostic']):
        metadata["category"] = "diagnosis"
    elif any(keyword in filename_lower for keyword in ['treatment', 'pengobatan', 'terapi', 'therapy']):
        metadata["category"] = "treatment"
    
    return metadata

def _load_pdfs_from_folder(folder_path):
    """Load PDFs dengan metadata enrichment."""
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' tidak ditemukan.")
        return []
    
    docs = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            print(f"  🔄 Processing: {pdf_file.name}")
            
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_file)
            text = ""
            
            for page_num, page in enumerate(doc):
                text += page.get_text()
            
            if text.strip():
                # Extract metadata from filename
                metadata = _extract_metadata_from_filename(pdf_file.name)
                metadata["page_count"] = len(doc)
                metadata["processed_date"] = datetime.now().isoformat()
                
                # Create Document
                lang_doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                docs.append(lang_doc)
                print(f"    ✅ Loaded: {len(text)} characters, {len(doc)} pages")
            
            doc.close()
            
        except Exception as e:
            print(f"    ❌ Error loading {pdf_file.name}: {e}")
            continue
    
    return docs

def initialize_database():
    """Inisialisasi ChromaDB dari PDF files."""
    persist_directory = _resolve_persist_directory()
    
    # Check if database already exists
    db_exists = os.path.exists(persist_directory) and \
                len(os.listdir(persist_directory)) > 0
    
    if db_exists:
        print(f"\n📊 Database sudah ada di {persist_directory}")
        print("Skip initialization.")
        return
    
    print(f"\n🚀 Starting Database Initialization...")
    print(f"📁 Persist directory: {persist_directory}")
    
    # Initialize embeddings model
    print("\n🔄 Initializing embeddings model...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ Embeddings model initialized")
    
    # Load PDFs
    print("\n📖 Loading PDF documents...")
    docs = _load_pdfs_from_folder('./data_adenomyosis')
    
    if not docs:
        print("❌ Tidak ada dokumen yang berhasil dimuat!")
        return
    
    print(f"\n✅ Loaded {len(docs)} documents")
    
    # Split documents
    print("\n✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(split_docs)} chunks")
    
    # Add chunk_id metadata
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = idx
    
    # Create ChromaDB
    print("\n🗃️ Creating ChromaDB...")
    try:
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ ChromaDB created successfully!")
        print(f"✅ Total documents: {len(split_docs)} chunks from {len(docs)} PDFs")
        print(f"✅ Location: {persist_directory}")
        
    except Exception as e:
        print(f"❌ Error creating ChromaDB: {e}")
        raise

if __name__ == "__main__":
    try:
        initialize_database()
        print("\n" + "="*50)
        print("✅ Database initialization complete!")
        print("="*50)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
