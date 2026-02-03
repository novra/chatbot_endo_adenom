import streamlit as st
import pandas as pd
import altair as alt

def show_adenomyosis_diagram():
    """Tampilkan diagram adenomyosis"""
    st.image("assets/endo_adenom.png", caption="Diagram Adenomyosis", width=700)

def show_symptoms_chart():
    """Tampilkan bagan gejala adenomyosis"""
    # Data gejala adenomyosis dan persentase penderita
    data = pd.DataFrame({
        'Gejala': ['Nyeri menstruasi', 'Perdarahan berat', 'Nyeri panggul', 'Nyeri saat berhubungan', 'Kembung', 'Kelelahan'],
        'Persentase': [65, 60, 50, 30, 20, 15]
    })
    
    # Membuat chart
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Persentase:Q', title='Persentase Penderita (%)'),
        y=alt.Y('Gejala:N', title='Gejala', sort='-x'),
        color=alt.Color('Persentase:Q', scale=alt.Scale(scheme='reds'), legend=None),
        tooltip=['Gejala', 'Persentase']
    ).properties(
        title='Gejala Umum Adenomyosis',
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
def show_treatment_options():
    """Tampilkan opsi pengobatan untuk adenomyosis"""
    # Data pilihan pengobatan
    data = pd.DataFrame({
        'Pengobatan': ['Obat antiinflamasi', 'Terapi hormon', 'Prosedur minimal invasif', 'Histerektomi'],
        'Efektivitas': [3, 4, 4, 5],
        'Kategori': ['Non-invasif', 'Non-invasif', 'Invasif', 'Invasif']
    })
    
    # Membuat chart
    chart = alt.Chart(data).mark_circle(size=200).encode(
        x=alt.X('Pengobatan:N', title='Pilihan Pengobatan'),
        y=alt.Y('Efektivitas:Q', title='Tingkat Efektivitas (1-5)', scale=alt.Scale(domain=[0, 6])),
        color=alt.Color('Kategori:N', scale=alt.Scale(domain=['Non-invasif', 'Invasif'],
                                                  range=['#5cb85c', '#d9534f'])),
        tooltip=['Pengobatan', 'Efektivitas', 'Kategori']
    ).properties(
        title='Pilihan Pengobatan Adenomyosis',
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)

def show_age_distribution():
    """Tampilkan distribusi umur pasien adenomyosis"""
    # Data distribusi umur
    data = pd.DataFrame({
        'Kelompok Umur': ['20-29', '30-39', '40-49', '50+'],
        'Persentase': [10, 30, 45, 15]
    })
    
    # Membuat chart
    chart = alt.Chart(data).mark_area(
        opacity=0.7,
        interpolate='monotone'
    ).encode(
        x=alt.X('Kelompok Umur:N', title='Kelompok Umur (Tahun)'),
        y=alt.Y('Persentase:Q', title='Persentase Kasus (%)', scale=alt.Scale(domain=[0, 50])),
        color=alt.value('#ff6b6b')
    ).properties(
        title='Distribusi Umur Pasien Adenomyosis',
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)

def show_adenomyosis_vs_endometriosis():
    """Tampilkan perbandingan adenomyosis dan endometriosis"""
    
    # Data perbandingan
    data = pd.DataFrame({
        'Fitur': ['Lokasi', 'Diagnosis', 'Perdarahan', 'Infertilitas', 'Pengobatan Hormonal'],
        'Adenomyosis': [5, 3, 5, 3, 4],
        'Endometriosis': [2, 5, 3, 5, 4]
    })
    
    # Restruktur data untuk Altair
    data_melted = pd.melt(
        data, 
        id_vars=['Fitur'], 
        value_vars=['Adenomyosis', 'Endometriosis'],
        var_name='Kondisi', 
        value_name='Nilai'
    )
    
    # Membuat chart radar
    chart = alt.Chart(data_melted).mark_line(point=True).encode(
        x=alt.X('Fitur:N', title=None),
        y=alt.Y('Nilai:Q', title='Tingkat Keparahan (1-5)'),
        color=alt.Color('Kondisi:N', scale=alt.Scale(domain=['Adenomyosis', 'Endometriosis'],
                                                range=['#FF4B4B', '#1E88E5'])),
        tooltip=['Fitur', 'Kondisi', 'Nilai']
    ).properties(
        title='Perbandingan Adenomyosis vs Endometriosis',
        width=600,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Keterangan tambahan
    st.markdown("""
    **Keterangan:**
    - **Lokasi**: Adenomyosis (Dalam rahim) vs Endometriosis (Luar rahim)
    - **Diagnosis**: Adenomyosis (Lebih sulit) vs Endometriosis (Gold standard melalui laparoskopi)
    - **Perdarahan**: Adenomyosis sering menyebabkan perdarahan berat
    - **Infertilitas**: Endometriosis lebih sering dikaitkan dengan infertilitas
    - **Pengobatan Hormonal**: Keduanya merespons terapi hormonal
    """)