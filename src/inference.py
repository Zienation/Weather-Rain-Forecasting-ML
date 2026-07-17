# ======================================================================
# INFERENCE SCRIPT - RAIN PREDICTION SYSTEM
# ======================================================================

import pandas as pd
import joblib
import os
import numpy as np

def load_inference_artifacts(model_path):
    """Memuat model, scaler, fitur, dan threshold dari file pkl."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
    
    artifacts = joblib.load(model_path)
    return artifacts['model'], artifacts['scaler'], artifacts['features'], artifacts['optimal_threshold']

def preprocess_data(df, features):
    """Melakukan preprocessing dan feature engineering pada raw data agar sesuai dengan format training."""
    df = df.copy()
    
    # 1. Konversi format tipe data tanggal dan ekstraksi temporal
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    # 2. Imputasi nilai kosong (numeric dengan median, categorical dengan mode)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if 'Location' in df.columns:
            df[col] = df.groupby('Location')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())
        
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if len(df[col].mode()) > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # 3. Pembuatan Fitur Interaksi Perubahan Cuaca Harian
    df['Pressure_Diff'] = df['Pressure3pm'] - df['Pressure9am']
    df['Humidity_Diff'] = df['Humidity3pm'] - df['Humidity9am']
    df['Temp_Diff'] = df['Temp3pm'] - df['Temp9am']
    df['WindSpeed_Diff'] = df['WindSpeed3pm'] - df['WindSpeed9am']
    df['Durnal_Temp_Range'] = df['MaxTemp'] - df['MinTemp']
    
    # 4. Mapping Kategori Biner ke Numerik
    if 'RainToday' in df.columns:
        df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
        
    # 5. Frequency Encoding untuk Location
    if 'Location' in df.columns:
        loc_freq = df['Location'].value_counts() / len(df)
        df['Location_Freq'] = df['Location'].map(loc_freq)
        df.drop(columns=['Location'], inplace=True)
        
    df.drop(columns=['Date'], inplace=True, errors='ignore')
    
    # 6. Encoding Fitur Kategorikal Multi-Kelas
    categorical_cols_to_encode = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    existing_cat_cols = [col for col in categorical_cols_to_encode if col in df.columns]
    df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True, dtype=int)
    
    # 7. Penyelarasan kolom agar sesuai persis dengan target `features` training
    for col in features:
        if col not in df.columns:
            df[col] = 0
            
    return df

def predict_rain(input_data, model, scaler, features, threshold):
    """
    Melakukan prediksi hujan berdasarkan input data baru.
    input_data: DataFrame yang berisi fitur cuaca.
    """
    # Memastikan urutan fitur sesuai saat training
    input_data = input_data[features]
    
    # Scaling data
    input_data_scaled = scaler.transform(input_data)
    
    # Prediksi probabilitas
    prob = model.predict_proba(input_data_scaled)[:, 1]
    
    # Klasifikasi berdasarkan threshold optimal
    prediction = (prob >= threshold).astype(int)
    
    return prob, prediction

def run_simulation(data_path=None):
    """Simulasi inferensi sederhana."""
    # Mendapatkan base directory proyek (satu tingkat di atas folder src)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path model dan dataset default
    model_path = os.path.join(base_dir, 'models', 'smart_farming_lgbm_model.pkl')
    if data_path is None:
        data_path = os.path.join(base_dir, 'Data', 'weatherAUS.csv')
        
    model, scaler, features, threshold = load_inference_artifacts(model_path)
    
    # Memuat data untuk simulasi
    df = pd.read_csv(data_path)
    
    # Lakukan preprocessing pada seluruh data agar frekuensi & encoding terbentuk sempurna
    df_preprocessed = preprocess_data(df, features)
    sample_data = df_preprocessed.head(5) # Ambil 5 hari pertama sebagai contoh simulasi
    
    # Proses prediksi
    probs, preds = predict_rain(sample_data, model, scaler, features, threshold)
    
    # Format output
    output = pd.DataFrame({
        'Probabilitas': [f"{p*100:.2f}%" for p in probs],
        'Aksi': ['Tunda Pemupukan & Siapkan Pelindung' if k == 1 else 'Aman untuk Tanam' for k in preds]
    })
    
    print("\n[+] Hasil Inferensi Model:")
    print(output)

if __name__ == "__main__":
    # Pastikan file model ada di direktori kerja
    run_simulation()