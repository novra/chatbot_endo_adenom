import os

# Workaround: avoid Streamlit file watcher scanning torch.classes
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from dotenv import load_dotenv

# Konfigurasi halaman (Wajib di baris pertama)
st.set_page_config(
    page_title="Asisten Adenomyosis",
    page_icon="🩺",
    layout="wide"
)

load_dotenv()

from chatbot import ChatBot
from utils import initialize_session_state
from visualizations import show_adenomyosis_diagram, show_symptoms_chart, show_treatment_options, show_age_distribution, show_adenomyosis_vs_endometriosis
from common_questions import COMMON_QUESTIONS

# Inisialisasi ChatBot dengan caching
@st.cache_resource
def get_chatbot():
    """Menginisialisasi dan mengembalikan instance ChatBot."""
    try:
        bot = ChatBot()
        return bot
    except ValueError as e:
        st.error(f"Error Inisialisasi: {e}. Mohon periksa file .env Anda.")
        return None

bot = get_chatbot()

# Inisialisasi session state
initialize_session_state()

# --- Fungsi Render UI ---

def render_header():
    """Menampilkan header aplikasi."""
    st.title("🩺 Asisten Ahli: Adenomyosis & Endometriosis")
    st.markdown(
        "Aplikasi ini menggunakan teknologi **RAG (Retrieval-Augmented Generation)** dengan model **Gemma 2** "
        "dan database vektor lokal **ChromaDB** dengan **metadata filtering** untuk memberikan jawaban berbasis bukti ilmiah."
    )
    st.markdown(
        "<div style='text-align: right; opacity: 0.7; font-size: 0.9em; margin-top: 1em;'>"
        "Dibuat oleh: <strong>Nuraisa Novia Hidayati</strong> (Riset Prototipe)"
        "</div>",
        unsafe_allow_html=True
    )
    st.divider()

def render_sidebar():
    """Menampilkan sidebar navigasi."""
    with st.sidebar:
        st.header("Navigasi")
        st.session_state.current_page = st.radio(
            "Pilih Halaman:",
            ["💬 Chat Konsultasi", "📊 Visualisasi Data", "📚 Informasi Umum"],
            key="nav_radio"
        )
        st.divider()
        
        # Info tentang improvement
        st.success("**✨ Perbaikan Terbaru:**")
        st.markdown("""
        - 🎯 Semantic Chunking (context preservation)
        - 🏷️ Metadata Filtering (validitas sumber)
        - 🧠 Chain-of-Thought Reasoning
        """)
        
        st.divider()
        st.info(
            "**Disclaimer:** Informasi ini dikurasi dari database riset terbatas. "
            "Selalu konsultasikan keputusan medis dengan dokter spesialis (Obgyn)."
        )

def render_chat_page():
    """Menampilkan halaman chat dengan metadata enrichment."""
    st.header("💬 Chat dengan Asisten AI")

    # Tampilkan riwayat pesan
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Tampilkan sources dengan metadata jika ada
            if "sources" in message and message["sources"]:
                st.markdown("---")
                st.markdown("**📚 Sumber Referensi:**")
                
                # Jika ada metadata, tampilkan dengan detail
                if "metadata" in message and message["metadata"]:
                    for source in message["sources"]:
                        source_clean = source.replace(".pdf", "")
                        meta = message["metadata"].get(source, {})
                        
                        # Badge berdasarkan validity
                        validity = meta.get("validity", "unknown")
                        if validity == "high":
                            badge = "🟢 High"
                        elif validity == "medium":
                            badge = "🟡 Medium"
                        else:
                            badge = "⚪ Low"
                        
                        source_type = meta.get("type", "Unknown")
                        year = meta.get("year", "N/A")
                        
                        st.caption(f"• **{source_clean}** | {source_type} | Validitas: {badge} | Tahun: {year}")
                else:
                    # Fallback jika metadata tidak ada
                    sources_clean = [s.replace(".pdf", "") for s in message["sources"]]
                    st.caption(f"📄 {', '.join(sources_clean)}")

    # Input User
    if user_input := st.chat_input("Contoh: Apa bedanya adenomiosis dengan endometriosis?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Menganalisis dokumen medis dengan metadata filtering..."):
                response = bot.ask(user_input)
                st.markdown(response["answer"])
                
                # Tampilkan sumber dengan metadata
                if response["sources"]:
                    st.markdown("---")
                    st.markdown("**📚 Sumber Referensi:**")
                    
                    for source in response["sources"]:
                        source_clean = source.replace(".pdf", "")
                        meta = response["metadata"].get(source, {})
                        
                        # Badge berdasarkan validity
                        validity = meta.get("validity", "unknown")
                        if validity == "high":
                            badge = "🟢 High"
                        elif validity == "medium":
                            badge = "🟡 Medium"
                        else:
                            badge = "⚪ Low"
                        
                        source_type = meta.get("type", "Unknown")
                        year = meta.get("year", "N/A")
                        
                        st.caption(f"• **{source_clean}** | {source_type} | Validitas: {badge} | Tahun: {year}")
                else:
                    st.caption("ℹ️ Sumber: Pengetahuan Umum Model (Tidak ada dokumen spesifik ditemukan)")
                
                # Simpan ke session state dengan metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"],
                    "metadata": response["metadata"]
                })

def render_visualization_page():
    """Menampilkan halaman visualisasi."""
    st.header("📊 Visualisasi Data Riset")
    st.write("Grafik ini merepresentasikan data umum mengenai Adenomiosis.")
    
    viz_tabs = st.tabs(["Anatomi", "Gejala", "Pengobatan", "Umur", "Perbandingan"])
    with viz_tabs[0]: show_adenomyosis_diagram()
    with viz_tabs[1]: show_symptoms_chart()
    with viz_tabs[2]: show_treatment_options()
    with viz_tabs[3]: show_age_distribution()
    with viz_tabs[4]: show_adenomyosis_vs_endometriosis()

def render_info_page():
    """Menampilkan FAQ."""
    st.header("📚 FAQ (Frequently Asked Questions)")
    for q_data in COMMON_QUESTIONS.values():
        with st.expander(q_data["question"]):
            st.markdown(q_data["answer"])

# --- Main App ---
def main():
    if not bot:
        st.error("Gagal memuat sistem AI.")
        return

    render_header()
    render_sidebar()

    if st.session_state.current_page == "💬 Chat Konsultasi":
        render_chat_page()
    elif st.session_state.current_page == "📊 Visualisasi Data":
        render_visualization_page()
    else:
        render_info_page()

if __name__ == "__main__":
    main()
