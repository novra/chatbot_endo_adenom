import streamlit as st
from dotenv import load_dotenv

# Konfigurasi halaman harus menjadi perintah Streamlit pertama
st.set_page_config(
    page_title="Asisten Adenomyosis",
    page_icon="\U0001F9EA",
    layout="wide",
)

# Muat environment variables dari file .env
load_dotenv()

from chatbot import ChatBot
from utils import initialize_session_state  # Kita masih butuh ini
from visualizations import (
    show_adenomyosis_diagram,
    show_symptoms_chart,
    show_treatment_options,
    show_age_distribution,
    show_adenomyosis_vs_endometriosis,
)
from common_questions import COMMON_QUESTIONS, QUICK_TOPICS

# Inisialisasi ChatBot dengan caching agar tidak loading ulang
@st.cache_resource
def get_chatbot():
    """Menginisialisasi dan mengembalikan instance ChatBot yang di-cache."""
    try:
        bot = ChatBot()
        return bot
    except ValueError as e:
        st.error(f"Error Inisialisasi: {e}. Mohon periksa file .env Anda.")
        return None


bot = get_chatbot()

# Inisialisasi session state
initialize_session_state()


# --- Fungsi-fungsi untuk Render UI ---

def render_header():
    """Menampilkan header aplikasi."""
    st.title("\U0001F9EA Asisten Ahli: Adenomyosis & Endometriosis")
    st.markdown(
        "Aplikasi ini menggunakan RAG (Retrieval-Augmented Generation) dengan model **Mistral** dan database **Pinecone** "
        "untuk menjawab pertanyaan berdasarkan dokumen medis."
    )
    st.markdown(
        "<div style='text-align: right; opacity: 0.7; font-size: 0.9em; margin-top: 1em;'>"
        "Dibuat oleh: <strong>Nuraisa Novia Hidayati</strong> (Versi Mistral)"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()


def render_sidebar():
    """Menampilkan sidebar navigasi."""
    with st.sidebar:
        st.header("Navigasi")
        st.session_state.current_page = st.radio(
            "Pilih Halaman:",
            ["\U0001F4AC Chat", "\U0001F4CA Visualisasi", "\U0001F4DA Informasi Umum"],
            key="nav_radio",
            captions=["Tanya jawab dengan AI", "Grafik & data visual", "Pertanyaan umum"],
        )
        st.divider()
        st.info(
            "**Perhatian:** Informasi dari chatbot ini bersifat edukatif dan tidak menggantikan "
            "konsultasi medis profesional."
        )


def render_chat_page():
    """Menampilkan halaman utama untuk chat."""
    st.header("\U0001F4AC Chat dengan Asisten AI")

    # Menampilkan riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Tampilkan sumber jika ada
            if "sources" in message and message["sources"]:
                sources_text = "Sumber: " + ", ".join(message["sources"])
                st.caption(sources_text)

    # Input dari pengguna
    if user_input := st.chat_input("Ketik pertanyaan Anda tentang adenomyosis..."):
        # Tambah pesan pengguna ke riwayat
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate dan tampilkan respons dari chatbot
        with st.chat_message("assistant"):
            with st.spinner("Asisten sedang berpikir..."):
                response = bot.ask(user_input)
                st.markdown(response["answer"])

                # Tampilkan sumber di bawah jawaban
                sources_text = (
                    "Sumber: " + ", ".join(response["sources"])
                    if response["sources"]
                    else "Sumber: Tidak ditemukan dokumen spesifik."
                )
                st.caption(sources_text)

                # Tambah respons AI ke riwayat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )


def render_visualization_page():
    """Menampilkan halaman visualisasi data."""
    st.header("\U0001F4CA Visualisasi Data")
    st.write("Grafik dan diagram untuk membantu memahami adenomyosis secara visual.")

    viz_tabs = st.tabs(
        ["Anatomi", "Gejala Umum", "Opsi Pengobatan", "Distribusi Usia", "Perbandingan"]
    )
    with viz_tabs[0]:
        show_adenomyosis_diagram()
    with viz_tabs[1]:
        show_symptoms_chart()
    with viz_tabs[2]:
        show_treatment_options()
    with viz_tabs[3]:
        show_age_distribution()
    with viz_tabs[4]:
        show_adenomyosis_vs_endometriosis()


def render_info_page():
    """Menampilkan halaman FAQ."""
    st.header("\U0001F4DA Informasi & Pertanyaan Umum")
    st.write("Berikut adalah jawaban untuk beberapa pertanyaan yang sering diajukan.")

    for q_data in COMMON_QUESTIONS.values():
        with st.expander(q_data["question"]):
            st.markdown(q_data["answer"])


# --- Main App ---
def main():
    """Fungsi utama untuk menjalankan aplikasi Streamlit."""
    if not bot:
        st.error("Gagal memuat chatbot. Aplikasi tidak dapat dijalankan.")
        return

    render_header()
    render_sidebar()

    # Navigasi halaman
    if st.session_state.current_page == "\U0001F4AC Chat":
        render_chat_page()
    elif st.session_state.current_page == "\U0001F4CA Visualisasi":
        render_visualization_page()
    else:
        render_info_page()


if __name__ == "__main__":
    main()
