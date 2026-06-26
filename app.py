import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan tools preprocessing yang sudah disimpan sebelumnya
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
num_cols = joblib.load('num_cols.pkl')
cat_cols = joblib.load('cat_cols.pkl')

st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi Data Science untuk memprediksi potensi loyalitas pelanggan.")

st.sidebar.header("Input Fitur Pelanggan")

# Dictionary untuk menampung input data
input_data = {}

# Grup 1: Profil Dasar Pelanggan
with st.sidebar.expander("👤 Profil Dasar Pelanggan", expanded=True):
    input_data['age'] = st.number_input("Usia Pelanggan (age)", min_value=1, max_value=100, value=30)
    input_data['gender'] = st.selectbox("Jenis Kelamin (gender)", ['Male', 'Female'])
    input_data['country'] = st.selectbox("Negara (country)", ['United States', 'Indonesia', 'Other'])
    input_data['city'] = st.text_input("Kota (city)", "Semarang")
    input_data['device_type'] = st.selectbox("Jenis Perangkat", ['Mobile', 'Desktop'])

# Grup 2: Aktivitas Aplikasi & Marketing
with st.sidebar.expander("📊 Aktivitas & Interaksi App", expanded=False):
    input_data['total visits'] = st.number_input("Total Kunjungan", min_value=0, value=5)
    input_data['avg_session_time'] = st.number_input("Rata-rata Waktu Sesi (menit)", min_value=0.0, value=15.5)
    input_data['pages_per_session'] = st.number_input("Halaman per Sesi", min_value=0.0, value=3.0)
    input_data['email_open_rate'] = st.slider("Email Open Rate", 0.0, 100.0, 50.0)
    input_data['email click rate'] = st.slider("Email Click Rate", 0.0, 100.0, 10.0)
    input_data['acquisition_channel'] = st.sidebar.selectbox("Saluran Akuisisi", ['Email', 'Ads', 'Organic'])

# Grup 3: Transaksi & Finansial
with st.sidebar.expander("💳 Riwayat Transaksi & Layanan", expanded=False):
    input_data['subscription_type'] = st.selectbox("Tipe Langganan", ['Basic', 'Premium'])
    input_data['is_premium_user'] = st.selectbox("User Premium? (0=Tidak, 1=Ya)", [0, 1])
    input_data['total_spent'] = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=100.0)
    input_data['avg_order_value'] = st.number_input("Rata-rata Nilai Order ($)", min_value=0.0, value=20.0)
    input_data['discount used'] = st.selectbox("Pernah Pakai Diskon? (0=Tidak, 1=Ya)", [0, 1])
    input_data['last_3_month_purchase'] = st.number_input("Frekuensi Pembelian 3 Bulan Terakhir", min_value=0, value=2)
    input_data['lifetime value'] = st.number_input("Lifetime Value ($)", min_value=0.0, value=500.0)
    input_data['payment_method'] = st.selectbox("Metode Pembayaran", ['Credit Card', 'Bank Transfer', 'E-Wallet'])

# Grup 4: Kepuasan & Komplain
with st.sidebar.expander("⚠️ Kepuasan & Dukungan", expanded=False):
    input_data['support_tickets'] = st.number_input("Jumlah Tiket Dukungan (Komplain)", min_value=0, value=0)
    input_data['refund_requested'] = st.selectbox("Pernah Minta Refund? (0=Tidak, 1=Ya)", [0, 1])
    input_data['delivery_delay_days'] = st.number_input("Keterlambatan Pengiriman (Hari)", min_value=0, value=0)
    input_data['satisfaction score'] = st.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 4.0)
    input_data['nps_score'] = st.slider("Net Promoter Score (1-10)", 1, 10, 8)
    input_data['marketing_spend_per_user'] = st.number_input("Biaya Marketing per User ($)", min_value=0.0, value=10.0)

# Mengubah input menjadi DataFrame tunggal
input_df = pd.DataFrame([input_data])

if st.button("Prediksi Status Churn"):
    # 1. Membuat DataFrame dengan kolom yang SAMA PERSIS dengan num_cols + cat_cols yang diharapkan model
    full_features = num_cols + cat_cols
    
    # Buat DataFrame kosong dengan struktur kolom yang benar
    prepared_df = pd.DataFrame(columns=full_features)
    
    # Isi DataFrame kosong tersebut dengan nilai dari input_data berdasarkan kecocokan nama fitur
    for col in full_features:
        standardized_col = col.lower().replace(" ", "").replace("_", "")
        matched_value = None
        
        for input_key, input_val in input_data.items():
            if input_key.lower().replace(" ", "").replace("_", "") == standardized_col:
                matched_value = input_val
                break
                
        # Jika ketemu pasangannya masukkan nilainya, jika tidak beri nilai default
        if matched_value is not None:
            prepared_df.loc[0, col] = matched_value
        else:
            prepared_df.loc[0, col] = 0 if col in num_cols else ""

    # 2. Terapkan tipe data agar sesuai dengan numpy/pandas sebelum transform
    for col in num_cols:
        prepared_df[col] = pd.to_numeric(prepared_df[col])
    for col in cat_cols:
        prepared_df[col] = prepared_df[col].astype(str)

    # 3. Lakukan Preprocessing menggunakan nama kolom yang sudah dijamin sinkron
    prepared_df[num_cols] = num_imputer.transform(prepared_df[num_cols])
    prepared_df[cat_cols] = cat_imputer.transform(prepared_df[cat_cols])
    
    input_num_scaled = scaler.transform(prepared_df[num_cols])
    input_cat_encoded = encoder.transform(prepared_df[cat_cols])
    
    input_final = np.hstack((input_num_scaled, input_cat_encoded))
    
    # 4. Melakukan Prediksi
    prediction = model.predict(input_final)
    prediction_proba = model.predict_proba(input_final)[0][1]
    
    st.subheader("Hasil Prediksi:")
    if prediction[0] == 1:
        st.error(f"⚠️ Pelanggan Berpotensi CHURN (Berhenti Berlangganan) dengan probabilitas {prediction_proba*100:.2f}%")
    else:
        st.success(f"✅ Pelanggan Berpotensi TETAP AKTIF (Loyal) dengan probabilitas {(1-prediction_proba)*100:.2f}%")
