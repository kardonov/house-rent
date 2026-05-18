# 🏠 Rent Price Predictor

Aplikasi web prediksi harga sewa rumah berbasis Machine Learning menggunakan **Streamlit** dan **Random Forest Regressor**.

---

## 📋 Deskripsi

Aplikasi ini memprediksi estimasi harga sewa properti berdasarkan berbagai fitur seperti lokasi, ukuran, jumlah kamar, dan status furnitur. Dibangun dengan pipeline preprocessing sklearn yang bersih dan visualisasi interaktif menggunakan Plotly.

---

## 🚀 Fitur Utama

- **🔮 Prediksi Harga** — Input detail properti dan dapatkan estimasi harga sewa secara real-time beserta rentang estimasi ±10%
- **📊 Eksplorasi Data** — Visualisasi interaktif distribusi harga per kota, BHK, furnitur, dan korelasi antar fitur
- **🤖 Performa Model** — Evaluasi model dengan scatter plot aktual vs prediksi, distribusi residual, dan feature importance

---

## 🛠️ Tech Stack

| Komponen | Library |
|---|---|
| Web Framework | `streamlit` |
| Machine Learning | `scikit-learn` |
| Data Processing | `pandas`, `numpy` |
| Visualisasi | `plotly` |
| Model | `RandomForestRegressor` + `TransformedTargetRegressor` |
| Preprocessing | `ColumnTransformer`, `OrdinalEncoder`, `StandardScaler` |

---

## 📦 Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/rent-price-predictor.git
cd rent-price-predictor
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`

---

## 📁 Struktur Proyek

```
rent-price-predictor/
│
├── app.py                  # File utama aplikasi Streamlit
├── requirements.txt        # Daftar dependensi
└── README.md               # Dokumentasi proyek
```

---

## 📄 requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
plotly>=5.20.0
```

---

## 🧠 Detail Model

### Fitur Input

| Fitur | Tipe | Keterangan |
|---|---|---|
| BHK | Numerik | Jumlah kamar tidur |
| Size | Numerik | Luas properti (sq ft) |
| Bathroom | Numerik | Jumlah kamar mandi |
| Floor | Numerik | Lantai unit |
| Total Floors | Numerik | Total lantai gedung |
| month_sin / month_cos | Numerik | Encoding musiman bulan iklan |
| Area Type | Kategorikal | Carpet Area / Super Area |
| Area Locality | Kategorikal | Nama lokasi/kelurahan |
| City | Kategorikal | Kota properti |
| Furnishing Status | Kategorikal | Unfurnished / Semi-Furnished / Furnished |
| Tenant Preferred | Kategorikal | Bachelors / Family / Bachelors/Family |
| Point of Contact | Kategorikal | Contact Owner / Contact Agent |

### Pipeline Preprocessing

```
Input DataFrame
      │
      ▼
ColumnTransformer
  ├── StandardScaler      → fitur numerik (7 kolom)
  └── OrdinalEncoder      → fitur kategorikal (6 kolom)
      │
      ▼
TransformedTargetRegressor
  ├── func: log1p(y)      → transformasi target saat training
  ├── RandomForestRegressor (n_estimators=150)
  └── inverse_func: expm1 → balik transformasi saat prediksi
```

### Metrik Evaluasi

| Metrik | Keterangan |
|---|---|
| R² Score | Proporsi variansi yang dijelaskan model |
| RMSE | Root Mean Squared Error dalam satuan ₹ |

---

## 🖥️ Tampilan Aplikasi

### Halaman Prediksi
Form input properti di panel kiri, hasil prediksi dengan gauge chart di panel kanan.

### Halaman Eksplorasi Data
Empat tab: analisis per kota, BHK & ukuran, furnitur & penyewa, dan distribusi + heatmap korelasi.

### Halaman Performa Model
Scatter plot aktual vs prediksi, histogram residual, dan bar chart feature importance.

---

## ⚙️ Konfigurasi

Ubah parameter model di fungsi `train_model()` pada `app.py`:

```python
# Ganti jumlah pohon
rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)

# Ganti proporsi data uji
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

## 📂 Menggunakan Dataset Asli

Secara default aplikasi menggunakan data sintetik. Untuk menggunakan dataset asli `House_Rent_Dataset.csv`, ganti fungsi `load_and_preprocess()`:

```python
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("House_Rent_Dataset.csv")

    # Ekstrak nomor lantai dari kolom "Floor" (contoh: "2 out of 5")
    df["Total Floors"] = df["Floor"].str.extract(r"out of (\d+)").astype(int)
    df["Floor"] = df["Floor"].str.extract(r"^(\d+)").fillna(0).astype(int)

    # Tambah fitur musiman dari kolom tanggal jika tersedia
    # df["Posted On"] = pd.to_datetime(df["Posted On"])
    # months = df["Posted On"].dt.month
    # df["month_sin"] = np.sin(2 * np.pi * months / 12)
    # df["month_cos"] = np.cos(2 * np.pi * months / 12)

    return df
```

Dataset tersedia di: [Kaggle — House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

---

## 🤝 Kontribusi

Pull request sangat disambut. Untuk perubahan besar, buka issue terlebih dahulu untuk mendiskusikan yang ingin diubah.

---

## 📝 Lisensi

[MIT](LICENSE)
