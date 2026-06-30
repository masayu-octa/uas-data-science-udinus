import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan tools preprocessing
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
num_cols = joblib.load('num_cols.pkl')
cat_cols = joblib.load('cat_cols.pkl')

st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
st.write("Aplikasi ringkas untuk memprediksi potensi loyalitas pelanggan berdasarkan fitur utama.")

st.sidebar.header("Input Fitur Utama Pelanggan")

input_data = {}

# 1. Fitur Demografi & Profil
input_data['age'] = st.sidebar.number_input("Usia Pelanggan", min_value=1, max_value=100, value=30)
input_data['subscription_type'] = st.sidebar.selectbox("Tipe Langganan", ['Basic', 'Premium'])

# 2. Fitur Finansial / Transaksi
input_data['total_spent'] = st.sidebar.number_input("Total Pengeluaran ($)", min_value=0.0, value=150.0)
input_data['lifetime value'] = st.sidebar.number_input("Lifetime Value ($)", min_value=0.0, value=500.0)

# 3. Fitur Kepuasan & Interaksi (Paling Menentukan Churn)
input_data['support_tickets'] = st.sidebar.number_input("Jumlah Komplain (Support Tickets)", min_value=0, value=1)
input_data['satisfaction score'] = st.sidebar.slider("Skor Kepuasan (1-5)", 1.0, 5.0, 4.0)
input_data['total visits'] = st.sidebar.number_input("Total Kunjungan ke Aplikasi", min_value=0, value=10)

fitur_sisa_num = {
    'avg_session_time': 10.0,        # netral (rentang data ~0-20)
    'pages_per_session': 4.0,        # netral
    'email_open_rate': 0.4,          # netral, sebelumnya 50 (skala 0-1 di data asli)
    'email_click_rate': 0.2,         # netral
    'avg_order_value': 60.0,         # netral, mendekati median
    'marketing_spend_per_user': 17.0,# netral, mendekati median
    'delivery_delay_days': 3,        # netral, mendekati median, sebelumnya 0 (terlalu ideal)
    'is_premium_user': 0,
    'discount used': 0,
    'refund_requested': 0,
    'last_3_month_purchase_freq': 5, 
    'nps_score': 5                   # netral, sebelumnya 7 (terlalu bagus -> menahan ke arah loyal)
}
fitur_sisa_cat = {
    'gender': 'Male', 'country': 'United States', 'city': 'Unknown',
    'acquisition_channel': 'Organic', 'device_type': 'Mobile', 'payment_method': 'Credit Card'
}

# Gabungkan input user dengan data default
for k, v in fitur_sisa_num.items(): input_data[k] = v
for k, v in fitur_sisa_cat.items(): input_data[k] = v

# Mengubah input menjadi DataFrame tunggal
input_df = pd.DataFrame([input_data])

THRESHOLD_CHURN = 0.40

if st.button("Prediksi Status Churn"):
    full_features = num_cols + cat_cols
    prepared_df = pd.DataFrame(columns=full_features)

    for col in full_features:
        standardized_col = col.lower().replace(" ", "").replace("_", "")
        matched_value = None
        for input_key, input_val in input_data.items():
            if input_key.lower().replace(" ", "").replace("_", "") == standardized_col:
                matched_value = input_val
                break
        if matched_value is not None:
            prepared_df.loc[0, col] = matched_value
        else:
            prepared_df.loc[0, col] = 0 if col in num_cols else ""

    for col in num_cols: prepared_df[col] = pd.to_numeric(prepared_df[col])
    for col in cat_cols: prepared_df[col] = prepared_df[col].astype(str)

    prepared_df[num_cols] = num_imputer.transform(prepared_df[num_cols])
    prepared_df[cat_cols] = cat_imputer.transform(prepared_df[cat_cols])

    input_num_scaled = scaler.transform(prepared_df[num_cols])
    input_cat_encoded = encoder.transform(prepared_df[cat_cols])
    input_final = np.hstack((input_num_scaled, input_cat_encoded))

    prediction_proba = model.predict_proba(input_final)[0][1]  # probabilitas churn (kelas 1)

    st.subheader("Hasil Prediksi:")

    if prediction_proba >= THRESHOLD_CHURN:
        st.error(f"⚠️ Pelanggan Berpotensi CHURN (Berhenti Berlangganan) dengan probabilitas {prediction_proba*100:.2f}%")
    else:
        st.success(f"✅ Pelanggan Berpotensi TETAP AKTIF (Loyal) dengan probabilitas {(1-prediction_proba)*100:.2f}%")

    st.caption(f"Probabilitas churn mentah dari model: {prediction_proba*100:.2f}% | Threshold keputusan: {THRESHOLD_CHURN*100:.0f}%")
