import re
import pandas as pd
import streamlit as st
from functools import lru_cache

# Topik umum adenomiosis dengan jawaban cepat
COMMON_TOPICS = {
    "Apa itu adenomyosis?": {
        "answer": "Adenomyosis adalah kondisi dimana jaringan yang biasanya melapisi rahim (endometrium) tumbuh ke dalam otot rahim. Ini dapat menyebabkan perdarahan menstruasi yang berat dan nyeri.",
        "icon": "🔍"
    },
    "Apa perbedaan adenomyosis dan endometriosis?": {
        "answer": "Adenomyosis terjadi ketika jaringan endometrium tumbuh ke dalam otot rahim, sedangkan endometriosis terjadi ketika jaringan endometrium tumbuh di luar rahim, seperti pada ovarium atau saluran tuba.",
        "icon": "⚖️"
    },
    "Apa gejala adenomyosis?": {
        "answer": "Gejala utama adenomyosis meliputi: perdarahan menstruasi yang berat, nyeri menstruasi yang parah, nyeri panggul kronis, dan ketidaknyamanan selama berhubungan seksual.",
        "icon": "🩸"
    },
    "Bagaimana cara mengobati adenomyosis?": {
        "answer": "Pengobatan adenomyosis meliputi: obat anti-inflamasi, terapi hormon, prosedur minimal invasif, dan dalam kasus parah mungkin dibutuhkan histerektomi (pengangkatan rahim).",
        "icon": "💊"
    },
    "Apakah adenomyosis bisa sembuh?": {
        "answer": "Adenomyosis biasanya mereda setelah menopause. Untuk wanita yang belum menopause, pengobatan dapat mengelola gejala tetapi belum ada penyembuhan total selain histerektomi.",
        "icon": "🔄"
    }
}

@lru_cache(maxsize=32)
def clean_response(response):
    """Membersihkan dan memformat respons chatbot"""
    if "Answer:" in response:
        response = response.split("Answer:")[1].strip()
    
    # Remove bold markdown **...**
    response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
    
    # Remove token instruction markers
    response = re.sub(r'\[/?INST\]', '', response)
    
    # Remove numeric citations like [1], [2]
    response = re.sub(r'\[\d+\]', '', response)
    
    # Replace 3 or more newlines with max 2
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in response.split('\n')]
    
    return '\n'.join(lines).strip()


def get_icon_for_topic(topic):
    """Mendapatkan ikon yang sesuai untuk topik"""
    icons = {
        "gejala": "🩸",
        "pengobatan": "💊",
        "diagnosis": "🔬",
        "penyebab": "🧬",
        "pencegahan": "🛡️",
        "komplikasi": "⚠️",
        "operasi": "🩺",
        "kesuburan": "👶",
        "rahim": "♀️",
        "menstruasi": "📅",
        "nyeri": "😣",
        "hormon": "⚗️",
        "dokter": "👩‍⚕️"
    }
    
    for key, icon in icons.items():
        if key in topic.lower():
            return icon
    return "💬"  # Default icon

def format_sources(sources):
    """Format sumber dengan gaya APA"""
    if not sources:
        return "Tidak ada sumber yang tersedia"
        
    if isinstance(sources, str):
        sources = [sources]
        
    formatted = []
    seen_sources = set()  # Untuk mencegah duplikasi
    current_year = 2025  # Tahun saat ini
    
    for idx, source in enumerate(sources):
        if source in seen_sources:
            continue
            
        # Membersihkan nama sumber
        source_name = source.replace(".pdf", "").replace("_", " ")
        
        # Ekstrak informasi untuk format APA
        # Coba ekstrak tahun dari nama file jika tersedia
        year_match = re.search(r'20\d{2}', source_name)
        year = year_match.group(0) if year_match else str(current_year)
        
        # Coba ekstrak penulis dari nama file
        author = "Anonim"  # Default jika penulis tidak ditemukan
        
        # Membersihkan nama dokumen untuk judul
        title = source_name
        title = re.sub(r'^\d+\s*-?\s*', '', title)  # Hapus format nomor di awal (misal: "123-")
        title = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '', title)  # Hapus format tanggal
        
        # Cek beberapa pola nama untuk ekstrak penulis
        if "article" in title.lower():
            # Format jurnal, mungkin tidak ada penulis di nama file
            title = re.sub(r'article\s+text', '', title, flags=re.IGNORECASE).strip()
        elif "case report" in title.lower():
            # Format laporan kasus
            title = re.sub(r'case\s+report', '', title, flags=re.IGNORECASE).strip()
        elif "kti" in title.lower():
            # Karya Tulis Ilmiah, mungkin ada nama penulis
            author_match = re.search(r'kti\s+([a-zA-Z\s]+)', title, flags=re.IGNORECASE)
            if author_match:
                author = author_match.group(1).strip()
                title = re.sub(r'kti\s+[a-zA-Z\s]+', '', title, flags=re.IGNORECASE).strip()
        elif "arifin" in title.lower():
            # Nama penulis terdeteksi
            author = "Arifin"
        
        # Singkat judul yang terlalu panjang
        if len(title) > 50:
            title = title[:47] + "..."
            
        # Menjadikan judul dan penulis lebih mudah dibaca
        title = title.strip().title()
        author = author.strip().title()
        
        # Format APA: Penulis (Tahun). Judul.
        apa_reference = f"[{idx+1}] {author} ({year}). *{title}*."
        formatted.append(apa_reference)
        seen_sources.add(source)
        
    if not formatted:
        return "Informasi dari pengetahuan umum"
    
    formatted.insert(0, "### Daftar Pustaka:")    
    return "\n\n".join(formatted)
    
def initialize_session_state():
    """Menginisialisasi variabel status sesi"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Selamat datang! Saya asisten dokter yang akan membantu menjawab pertanyaan Anda tentang Adenomyosis. Apa yang ingin Anda ketahui?"}
        ]
    
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
        
    if "sources" not in st.session_state:
        st.session_state.sources = []
        
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"