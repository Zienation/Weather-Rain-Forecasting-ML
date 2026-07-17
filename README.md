# Rain Prediction System (Sistem Prediksi Hujan - Smart Farming)

Proyek ini bertujuan untuk memprediksi probabilitas terjadinya hujan menggunakan teknik pembelajaran mesin (Machine Learning) berbasis model ensemble (Stacking Classifier). Hasil prediksi digunakan sebagai sistem pendukung keputusan *Smart Farming*, misalnya memberikan rekomendasi tindakan seperti menunda pemupukan atau menyiapkan pelindung tanaman.

---

## ⚙️ Dependencies (Persyaratan Sistem)

Proyek ini memerlukan Python versi **3.10 ke atas** (kompatibel hingga versi **3.14** menggunakan prebuilt wheels). Library utama yang digunakan meliputi:

* **Pandas & NumPy**: Untuk manipulasi struktur data tabular dan operasi matriks.
* **Joblib**: Untuk memuat model terkompresi `.pkl`.
* **Scikit-Learn (v1.7.2+)**: Untuk scaling data (`RobustScaler`), model baseline (`LogisticRegression`), dan kerangka Stacking (`StackingClassifier`).
* **XGBoost (v2.1.4+)**: Algoritma boosting pendukung.
* **LightGBM (v4.6.0+)**: Classifier utama untuk prediksi probabilitas hujan.
* **CatBoost**: Algoritma boosting pendukung berbasis kategori.

### Cara Instalasi Dependencies:
Buka terminal Anda di folder root proyek, lalu jalankan:
```bash
pip install -r requirements.txt
```

---

## 🚀 Cara Menjalankan Secara Lokal

Script inferensi berada di dalam direktori `src/`. Anda dapat mensimulasikan hasil prediksi pada 5 hari pertama dari dataset dengan menjalankan perintah berikut dari folder root proyek:

```bash
python src/inference.py
```

### Penjelasan Output:
Script akan memuat dataset mentah secara dinamis, melakukan preprocessing dan rekayasa fitur (imputasi, pembuatan kolom diferensial, dummy encoding), melakukan penskalaan data, lalu menghasilkan output berupa:
* **Probabilitas**: Tingkat kemungkinan terjadinya hujan dalam persentase.
* **Aksi Rekomendasi**: 
  * `Tunda Pemupukan & Siapkan Pelindung` (jika probabilitas melebihi batas threshold optimal).
  * `Aman untuk Tanam` (jika probabilitas di bawah batas threshold optimal).

---

## 📓 Menjalankan di Google Colab

Anda juga dapat menjalankan proses EDA, rekayasa fitur, pelatihan model, dan evaluasi langsung di Google Colab.

* **Link Notebook**: [Google Colab - Weather Rain Forecasting ML](https://colab.research.google.com/drive/1g0ZBjQARQp243VQ8EUoDuxhG0iBLm0q4?usp=sharing)

### Langkah-langkah Menjalankan di Google Colab:
1. Klik link di atas untuk membuka notebook di Google Colab.
2. Buat salinan notebook tersebut ke Google Drive Anda dengan mengklik menu **File > Save a copy in Drive** (Simpan salinan di Drive).
3. Sambungkan ke runtime Google Colab dengan mengklik tombol **Connect** (Sambungkan) di pojok kanan atas.
4. Jalankan sel instalasi library eksternal (langkah pertama dalam notebook):
   ```python
   !pip install category_encoders eli5 catboost optuna shap -q
   ```
5. Unggah dataset `weatherAUS.csv` ke dalam direktori kerja Colab atau hubungkan ke Google Drive untuk membaca dataset.
6. Jalankan setiap sel kode secara berurutan dengan menekan `Shift + Enter` atau tombol Play di sebelah kiri sel.

---

## 💡 Key Insights (Wawasan Proyek)

Berdasarkan analisis data (EDA) dan proses modeling pada notebook, berikut adalah beberapa insight penting yang didapatkan:

1. **Pentingnya Rekayasa Fitur Perubahan Cuaca (Daily Interaction Features)**:
   Perbedaan cuaca antara pagi (9 AM) dan sore (3 PM) sangat berkorelasi dengan potensi hujan. Fitur-fitur bentukan seperti:
   * `Pressure_Diff` (Selisih tekanan udara)
   * `Humidity_Diff` (Selisih kelembapan udara)
   * `Temp_Diff` (Selisih suhu udara)
   * `Durnal_Temp_Range` (Rentang suhu harian dari Min ke Max)
   
   Menjadi indikator kuat bagi model dalam memprediksi perubahan cuaca dibandingkan hanya menggunakan fitur mentah.

2. **Pendekatan Stacking Ensemble**:
   Prediksi cuaca merupakan tantangan dengan pola non-linear yang kompleks. Dengan menggunakan **StackingClassifier** yang menggabungkan kekuatan **XGBoost**, **LightGBM**, dan **CatBoost**, model mampu mengurangi variansi error dan meningkatkan skor F1 secara signifikan dibandingkan dengan model tunggal.

3. **Optimasi Threshold pada Imbalanced Data**:
   Dataset cuaca umumnya memiliki ketidakseimbangan kelas (hari hujan jauh lebih sedikit dibanding hari cerah). Penggunaan threshold default ($0.5$) sering kali tidak akurat untuk memberikan peringatan dini bagi petani. Menghitung dan menerapkan **Threshold Optimal** khusus membantu meminimalkan risiko gagal memprediksi hujan (False Negative) yang bisa berdampak buruk bagi pertanian.

4. **Kesiapan Produksi (Production Readiness)**:
   Pada tahap inferensi, sangat penting untuk menjaga konsistensi prapemrosesan data (seperti penyelarasan kolom dummy encoding hasil dari `pd.get_dummies` dan frekuensi encoding lokasi) agar model tidak mengalami kegagalan akibat input data baru yang strukturnya berbeda dengan data training.
