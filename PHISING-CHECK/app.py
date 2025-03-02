import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Phishing URL",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #1E88E5, #5E35B1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .input-area {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .safe-result {
        background-color: #DCEDC8;
        color: #33691E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #33691E;
        margin: 10px 0;
    }
    
    .phishing-result {
        background-color: #FFEBEE;
        color: #B71C1C;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #B71C1C;
        margin: 10px 0;
    }
    
    .feature-box {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-bottom: 10px;
    }
    
    .footer {
        text-align: center;
        color: #777;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
    
    /* Membuat tampilan tombol lebih menarik */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Daftar domain terpercaya
trusted_domains = [
    'google.com', 'www.google.com', 'facebook.com', 'www.facebook.com',
    'youtube.com', 'www.youtube.com', 'amazon.com', 'www.amazon.com',
    'twitter.com', 'www.twitter.com', 'instagram.com', 'www.instagram.com',
    'linkedin.com', 'www.linkedin.com', 'microsoft.com', 'www.microsoft.com',
    'apple.com', 'www.apple.com', 'netflix.com', 'www.netflix.com',
    'github.com', 'www.github.com', 'wikipedia.org', 'www.wikipedia.org',
    'yahoo.com', 'www.yahoo.com', 'paypal.com', 'www.paypal.com',
    'whatsapp.com', 'www.whatsapp.com', 'gmail.com', 'mail.google.com',
    'outlook.com', 'www.outlook.com',
]

#--------- FUNGSI MEMUAT MODEL ---------#
@st.cache_resource
def load_model():
    try:
        if os.path.exists('models/random_forest_model.pkl'):
            model = joblib.load('models/random_forest_model.pkl')
            model_name = "Random Forest"
        elif os.path.exists('models/gradient_boosting_model.pkl'):
            model = joblib.load('models/gradient_boosting_model.pkl')
            model_name = "Gradient Boosting"
        else:
            st.error("Model tidak ditemukan di folder 'models/'")
            return None, None
        
        return model, model_name
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None, None

#--------- FUNGSI UTILITAS ---------#
def similar_to_trusted(domain):
    domain = domain.lower()
    for trusted in ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'paypal', 'netflix', 'twitter']:
        if trusted in domain and domain not in trusted_domains:
            return 1
    return 0

def extract_features(url):
    try:
        # Parsing URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # Fitur dasar yang penting
        features = {
            'length_url': len(url),
            'length_hostname': len(domain),
            'ip': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
            'nb_dots': url.count('.'),
            'nb_hyphens': url.count('-'),
            'nb_at': url.count('@'),
            'nb_qm': url.count('?'),
            'nb_and': url.count('&'),
            'nb_eq': url.count('='),
            'nb_slash': url.count('/'),
            'nb_www': 1 if 'www' in url else 0,
            'nb_com': 1 if '.com' in url else 0,
            'https_token': 1 if 'https' in url else 0,
            'ratio_digits_url': sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
            'ratio_digits_host': sum(c.isdigit() for c in domain) / len(domain) if len(domain) > 0 else 0,
            'phish_hints': 1 if any(hint in url.lower() for hint in ['secure', 'account', 'webscr', 'login', 'signin']) else 0,
            'google_index': 0,  # Default value
            'is_trusted_domain': 1 if domain.lower() in trusted_domains else 0,
            'similar_to_trusted': similar_to_trusted(domain),
            'has_https': 1 if url.startswith('https') else 0,
            'unusual_chars': 1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0,
            'subdomain_count': len(domain.split('.')) - 1
        }
        
        return features, domain
    except Exception as e:
        st.error(f"Error mengekstrak fitur: {str(e)}")
        return None, None

#--------- FUNGSI DETEKSI PHISHING ---------#
def detect_phishing(url, model):
    # Ekstrak fitur
    features, domain = extract_features(url)
    if not features:
        return None
    
    # Buat DataFrame
    features_df = pd.DataFrame([features])
    
    # Logika keputusan sederhana
    # 1. Jika domain terpercaya, langsung klasifikasikan sebagai aman
    if domain.lower() in trusted_domains:
        return {
            'url': url,
            'domain': domain,
            'is_phishing': False,
            'probability': 0.01,
            'reason': 'Domain terpercaya',
            'features': features
        }
    
    # 2. Jika menggunakan alamat IP, kemungkinan besar phishing
    if features['ip'] == 1:
        try:
            prob = model.predict_proba(features_df)[0][1]
            adjusted_prob = min(1.0, prob * 1.2)  # Tingkatkan probabilitas
        except:
            adjusted_prob = 0.9  # Default tinggi untuk IP
            
        return {
            'url': url,
            'domain': domain,
            'is_phishing': True,
            'probability': adjusted_prob,
            'reason': 'Menggunakan alamat IP sebagai domain',
            'features': features
        }
    
    # 3. Jika mirip domain terpercaya, tingkatkan kecurigaan
    if features['similar_to_trusted'] == 1:
        try:
            prob = model.predict_proba(features_df)[0][1]
            adjusted_prob = min(1.0, prob * 1.15)  # Tingkatkan probabilitas
        except:
            adjusted_prob = 0.7  # Default
            
        return {
            'url': url,
            'domain': domain,
            'is_phishing': adjusted_prob > 0.5,
            'probability': adjusted_prob,
            'reason': 'Menyerupai domain terpercaya',
            'features': features
        }
    
    # 4. Gunakan model untuk kasus lainnya
    try:
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
    except Exception as e:
        # Fallback ke heuristik sederhana
        heuristic_score = 0
        if features['length_url'] > 100: heuristic_score += 0.3
        if features['nb_dots'] > 4: heuristic_score += 0.2
        if features['has_https'] == 0: heuristic_score += 0.2
        if features['unusual_chars'] == 1: heuristic_score += 0.3
        
        prediction = heuristic_score > 0.5
        probability = heuristic_score
    
    return {
        'url': url,
        'domain': domain,
        'is_phishing': bool(prediction),
        'probability': probability,
        'reason': 'Berdasarkan analisis machine learning',
        'features': features
    }

#--------- FUNGSI VISUALISASI ---------#
def create_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 0.5, 'tickcolor': "#636363"},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 30], 'color': "#81C784"},
                {'range': [30, 60], 'color': "#FFD54F"},
                {'range': [60, 80], 'color': "#FF8A65"},
                {'range': [80, 100], 'color': "#E57373"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.8,
                'value': value * 100
            }
        }
    ))
    
    fig.update_layout(
        height=200, 
        margin=dict(l=10, r=10, t=30, b=10),
        font={'size': 14}
    )
    return fig

def main():
    # Load model
    model, model_name = load_model()
    
    if not model:
        st.warning("⚠️ Model tidak tersedia. Aplikasi berjalan dalam mode demo.")
        model_name = "Demo Mode"
    
    # Header
    st.markdown("<h1 class='main-title'>Sistem Deteksi URL Phishing</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtitle'>Menggunakan Machine Learning: {model_name}</p>", unsafe_allow_html=True)
    
    # Input area
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    url_input = st.text_input("", placeholder="Masukkan URL untuk dianalisis... (contoh: https://example.com)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        analyze_button = st.button("🔍 Analisis URL", type="primary", key="analyze")
    with col2:
        clear_button = st.button("🗑️ Reset", key="clear")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Reset input
    if clear_button:
        url_input = ""
        st.rerun()
    
    # Analisis URL
    if analyze_button and url_input:
        with st.spinner("Menganalisis URL..."):
            # Validasi URL sederhana
            if not (url_input.startswith('http://') or url_input.startswith('https://')):
                url_input = 'http://' + url_input
            
            # Deteksi phishing
            result = detect_phishing(url_input, model)
            
            if result:
                # Tampilkan hasil
                prob_phishing = result['probability']
                
                # Gauge phishing probability
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Hasil analisis
                    if result['is_phishing']:
                        st.markdown(f"""
                        <div class='phishing-result'>
                            <h2>⚠️ URL MENCURIGAKAN</h2>
                            <p>URL: <b>{result['url']}</b></p>
                            <p>Domain: <b>{result['domain']}</b></p>
                            <p>Probabilitas Phishing: <b>{prob_phishing:.2%}</b></p>
                            <p>Alasan: <b>{result['reason']}</b></p>
                            <hr>
                            <p>⚠️ <b>Peringatan:</b> Jangan memasukkan informasi sensitif atau data pribadi di URL ini!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='safe-result'>
                            <h2>✅ URL AMAN</h2>
                            <p>URL: <b>{result['url']}</b></p>
                            <p>Domain: <b>{result['domain']}</b></p>
                            <p>Probabilitas Aman: <b>{(1-prob_phishing):.2%}</b></p>
                            <p>Alasan: <b>{result['reason']}</b></p>
                            <hr>
                            <p>✓ <b>URL ini terlihat aman.</b> Namun tetap waspadai informasi sensitif yang Anda berikan.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.plotly_chart(create_gauge(prob_phishing), use_container_width=True)
                
                # Fitur-fitur utama
                st.subheader("Fitur URL yang Dianalisis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.markdown("**Karakteristik URL**")
                    st.markdown(f"📏 Panjang URL: **{result['features']['length_url']}**")
                    st.markdown(f"🔢 Digit dalam URL: **{result['features']['ratio_digits_url']:.2%}**")
                    st.markdown(f"📍 Menggunakan IP: **{'Ya' if result['features']['ip'] else 'Tidak'}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.markdown("**Karakter Khusus**")
                    st.markdown(f"🔹 Jumlah Titik: **{result['features']['nb_dots']}**")
                    st.markdown(f"🔹 Jumlah Hyphen: **{result['features']['nb_hyphens']}**")
                    st.markdown(f"🔹 Subdomain: **{result['features']['subdomain_count']}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='feature-box'>", unsafe_allow_html=True)
                    st.markdown("**Keamanan Domain**")
                    st.markdown(f"🔒 HTTPS: **{'Ya' if result['features']['has_https'] else 'Tidak'}**")
                    st.markdown(f"✓ Domain Terpercaya: **{'Ya' if result['features']['is_trusted_domain'] else 'Tidak'}**")
                    st.markdown(f"⚠️ Mirip Domain Terpercaya: **{'Ya' if result['features']['similar_to_trusted'] else 'Tidak'}**")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Tips keamanan berdasarkan hasil
                if result['is_phishing']:
                    st.markdown("""
                    ### 🛡️ Tips Keamanan untuk URL Mencurigakan:
                    
                    * ⛔ **Jangan** memasukkan informasi pribadi seperti password, nomor kartu kredit, dll.
                    * 🔍 Selalu teliti URL sebelum mengklik dan memasukkan data
                    * 🚫 Hindari membuka lampiran atau mengklik tautan dari sumber tidak dikenal
                    * 🔐 Aktifkan otentikasi dua faktor (2FA) untuk akun penting 
                    """)
                else:
                    st.markdown("""
                    ### 🛡️ Tips Keamanan Umum:
                    
                    * ✅ Meskipun URL terdeteksi aman, tetap waspada saat memasukkan data sensitif
                    * 🔍 Perhatikan apakah koneksi menggunakan HTTPS (gembok hijau di browser)
                    * 🔄 Gunakan password yang kuat dan berbeda untuk setiap situs
                    * 🛑 Berhati-hati dengan permintaan informasi yang tidak biasa meskipun dari situs yang terlihat legitim
                    """)
            else:
                st.error("Gagal menganalisis URL. Pastikan format URL valid.")
    
    

if __name__ == "__main__":
    main()