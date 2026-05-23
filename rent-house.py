import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rent Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: blue; }
    .stApp { background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%); }
    .metric-card {
        background: black;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .predict-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        color: blue;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .predict-price {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #667eea44;
    }
    div[data-testid="stSidebar"] {
        background: white;
        box-shadow: 2px 0 12px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ── Load & Preprocess Data ────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(csv_path: str = "House_Rent_Dataset_Indonesia_Rupiah.csv"):
    """
    Load dataset asli jika CSV tersedia, fallback ke data sintetik.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Ekstrak Floor & Total Floors dari kolom string (contoh: "2 out of 5")
        df["Total Floors"] = (
            df["Floor"]
            .str.extract(r"out of (\d+)", expand=False)
            .fillna(1)
            .astype(int)
        )
        df["Floor"] = (
            df["Floor"]
            .str.extract(r"^(\d+)", expand=False)
            .fillna(0)
            .astype(int)
        )

        # Fitur musiman dari kolom tanggal
        df["Posted On"] = pd.to_datetime(["Posted On"], errors="coerce")
        months = ["Posted On"].dt.month.fillna(6).astype(int)
        df["month_sin"] = np.sin(2 * np.pi * months / 12)
        df["month_cos"] = np.cos(2 * np.pi * months / 12)

        # Hapus baris dengan nilai penting yang kosong
        required_cols = ["BHK", "Rent", "Size", "City", "Furnishing Status",
                         "Tenant Preferred", "Bathroom", "Point of Contact",
                         "Area Type", "Area Locality", "Floor", "Total Floors"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        st.sidebar.success("✅ Dataset asli dimuat")

    else:
        st.sidebar.info("ℹ️ CSV tidak ditemukan")
    return df


@st.cache_resource
def train_model(df):
    cat_cols = ["Area Type", "Area Locality", "City",
                "Furnishing Status", "Tenant Preferred", "Point of Contact"]
    num_cols = ["BHK", "Size", "Bathroom", "Floor", "Total Floors",
                 "month_sin", "month_cos"]
    feat_cols = num_cols + cat_cols 

    X = df[feat_cols].copy()
    y = df["Rent"].copy()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])

    X_scaled = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model = TransformedTargetRegressor(regressor=rf, func=np.log1p, inverse_func=np.expm1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    importances = model.regressor_.feature_importances_
    feat_imp = pd.DataFrame({"Feature": feat_cols, "Importance": importances})
    feat_imp = feat_imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, preprocessor, feat_cols, cat_cols, rmse, r2, y_test, y_pred, feat_imp


# ── Load data & train ─────────────────────────────────────────────────────────
with st.spinner("🔄 Memuat data & melatih model…"):
    df = load_and_preprocess()
    model, preprocessor, feat_cols, cat_cols, rmse, r2, y_test, y_pred, feat_imp = train_model(df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=64)
    st.title("🏠 Rent Predictor")
    st.caption("Machine Learning · Random Forest")
    st.divider()
    page = st.radio("Navigasi", ["🔮 Prediksi Harga", "📊 Eksplorasi Data", "🤖 Performa Model"])
    st.divider()
    st.markdown("**Dataset Info**")
    st.metric("Total Data", f"{len(df):,}")
    st.metric("Fitur Model", len(feat_cols))
    st.metric("Model", "Random Forest")


# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDIKSI
# ══════════════════════════════════════════════════════════════════
if page == "🔮 Prediksi Harga":
    st.title("🔮 Prediksi Harga Sewa Rumah")
    st.markdown("Isi detail properti di bawah untuk mendapatkan estimasi harga sewa.")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-title">📋 Detail Properti</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            bhk = st.selectbox("Jumlah BHK", [1, 2, 3, 4, 5], index=1)
            size = st.number_input("Ukuran (sq ft)", min_value=100, max_value=8000, value=800, step=50)
            bathroom = st.selectbox("Kamar Mandi", [1, 2, 3, 4, 5, 6], index=1)
        with c2:
            city = st.selectbox("Kota", df["City"].unique())
            area_type = st.selectbox("Tipe Area", df["Area Type"].unique())
            furnishing = st.selectbox("Furnitur", df["Furnishing Status"].unique())

        c3, c4 = st.columns(2)
        with c3:
            locality_opts = df[df["City"] == city]["Area Locality"].unique()
            locality = st.selectbox("Lokasi", sorted(locality_opts))
            tenant = st.selectbox("Preferensi Penyewa", df["Tenant Preferred"].unique())
        with c4:
            total_floors = st.slider("Total Lantai Gedung", 1, 30, 5)
            floor = st.slider("Lantai Unit", 0, total_floors, 1)
            contact = st.selectbox("Kontak Via", df["Point of Contact"].unique())

        month_num = st.slider("Bulan Iklan Ditayangkan", 1, 12, 6,
                              format="%d",
                              help="1=Januari … 12=Desember")
        month_sin_val = np.sin(2 * np.pi * month_num / 12)
        month_cos_val = np.cos(2 * np.pi * month_num / 12)

        predict_btn = st.button("🔮 Prediksi Harga Sewa", use_container_width=True, type="primary")

    with col_right:
        if predict_btn:
            input_data = pd.DataFrame([{
                "BHK": bhk, "Size": size, "Bathroom": bathroom,
                "Floor": floor, "Total Floors": total_floors,
                "month_sin": month_sin_val, "month_cos": month_cos_val,
                "Area Type": area_type, "Area Locality": locality,
                "City": city, "Furnishing Status": furnishing,
                "Tenant Preferred": tenant, "Point of Contact": contact,
            }])
            input_data = input_data[feat_cols]
            input_scaled = preprocessor.transform(input_data)
            pred = model.predict(input_scaled)[0]

            st.markdown(f"""
            <div class="predict-card">
                <div style="font-size:1rem;opacity:.85;margin-bottom:4px">💡 Estimasi Harga Sewa(Dalam Juta)</div>
                <div class="predict-price">Rp{pred:,.0f}</div>
                <div style="opacity:.8;margin-top:6px">per bulan</div>
                <hr style="border-color:rgba(255,255,255,0.3);margin:16px 0">
                <div style="display:flex;justify-content:space-around;font-size:.85rem">
                    <div><b>{bhk} BHK</b><br>Tipe</div>
                    <div><b>{size:,} sqft</b><br>Luas</div>
                    <div><b>{city}</b><br>Kota</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rentang estimasi ±10%
            st.markdown("#### 📉 Rentang Estimasi")
            low, high = pred * 0.9, pred * 1.1
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number={"prefix": "Rp", "valueformat": ",.2f"},
                gauge={
                    "axis": {"range": [low * 0.8, high * 1.2]},
                    "bar": {"color": "#667eea"},
                    "steps": [
                        {"range": [low * 0.8, low], "color": "#e8eaf6"},
                        {"range": [low, high], "color": "#c5cae9"},
                        {"range": [high, high * 1.2], "color": "#e8eaf6"},
                    ],
                    "threshold": {
                        "line": {"color": "#764ba2", "width": 4},
                        "thickness": 0.8,
                        "value": pred,
                    },
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(f"Kisaran wajar: Rp{low:,.0f} – Rp{high:,.0f}")

        else:
            st.info("⬅️ Isi detail properti di sebelah kiri, lalu klik **Prediksi**.")
            # Preview stat
            st.markdown("#### 📊 Statistik Harga Sewa Dataset")
            fig_box = px.box(df, x="City", y="Rent", color="City",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_box.update_layout(showlegend=False, height=320,
                                  margin=dict(t=10, b=10), yaxis_title="Rent (Rp)")
            st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 2 — EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Data":
    st.title("📊 Eksplorasi Data")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rata-rata Rent", f"Rp{df['Rent'].mean():,.0f}")
    k2.metric("Median Rent", f"Rp{df['Rent'].median():,.0f}")
    k3.metric("Maks Rent", f"Rp{df['Rent'].max():,.0f}")
    k4.metric("Min Rent", f"Rp{df['Rent'].min():,.0f}")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["🏙️ Kota", "🛏️ BHK & Ukuran", "🪑 Furnitur & Penyewa", "🔢 Distribusi"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df.groupby("City")["Rent"].mean().reset_index().sort_values("Rent"),
                         x="Rent", y="City", orientation="h", color="Rent",
                         color_continuous_scale="Purples",
                         title="Rata-rata Harga Sewa per Kota")
            fig.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            city_cnt = df["City"].value_counts().reset_index()
            city_cnt.columns = ["City", "Count"]
            fig2 = px.pie(city_cnt, values="Count", names="City",
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          title="Distribusi Iklan per Kota", hole=0.4)
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.violin(df, x="City", y="Rent", box=True, color="City",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Distribusi Rent per Kota")
        fig3.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            avg_bhk = df.groupby("BHK")["Rent"].mean().reset_index()
            fig = px.bar(avg_bhk, x="BHK", y="Rent", color="BHK",
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         title="Rata-rata Rent per BHK")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.scatter(df.sample(1000), x="Size", y="Rent",
                              color="BHK", opacity=0.6,
                              color_continuous_scale="Viridis",
                              title="Ukuran vs Rent (1000 sampel)")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="BHK", y="Rent", color="City",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Rent per BHK dipecah per Kota")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            avg_furn = df.groupby("Furnishing Status")["Rent"].mean().reset_index()
            fig = px.bar(avg_furn, x="Furnishing Status", y="Rent",
                         color="Furnishing Status",
                         color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
                         title="Rata-rata Rent per Status Furnitur")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            avg_ten = df.groupby("Tenant Preferred")["Rent"].mean().reset_index()
            fig2 = px.bar(avg_ten, x="Tenant Preferred", y="Rent",
                          color="Tenant Preferred",
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          title="Rata-rata Rent per Preferensi Penyewa")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.box(df, x="Furnishing Status", y="Rent", color="City",
                      color_discrete_sequence=px.colors.qualitative.Pastel,
                      title="Rent per Furnitur & Kota")
        fig3.update_layout(height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Rent", nbins=60,
                               color_discrete_sequence=["#667eea"],
                               title="Distribusi Harga Sewa")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="Size", nbins=60,
                                color_discrete_sequence=["#764ba2"],
                                title="Distribusi Ukuran Properti (sq ft)")
            st.plotly_chart(fig2, use_container_width=True)

        # Correlation heatmap (numeric)
        num_corr = df[["Rent", "BHK", "Size", "Bathroom", "Floor", "Total Floors"]].corr()
        fig3 = px.imshow(num_corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Korelasi Pearson Fitur Numerik",
                         aspect="auto")
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PAGE 3 — PERFORMA MODEL
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Performa Model":
    st.title("🤖 Performa Model")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² Score", f"{r2:.4f}")
    k2.metric("RMSE", f"{rmse:,.0f}")
    k3.metric("Model", "Random Forest")
    k4.metric("n_estimators", "150")

    st.divider()
    tab1, tab2 = st.tabs(["📈 Aktual vs Prediksi", "🎯 Feature Importance"])

    with tab1:
        # Actual vs Predicted scatter
        y_test_arr = np.array(y_test)
        fig = px.scatter(x=y_test_arr, y=y_pred,
                         labels={"x": "Harga Aktual (Rp)", "y": "Harga Prediksi (Rp)"},
                         title="Aktual vs Prediksi",
                         opacity=0.5, color_discrete_sequence=["#667eea"])
        lim = [min(y_test_arr.min(), y_pred.min()), max(y_test_arr.max(), y_pred.max())]
        fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                      line=dict(color="#764ba2", width=2, dash="dash"))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Residual distribution
        residuals = y_pred - y_test_arr
        fig2 = px.histogram(x=residuals, nbins=60,
                            color_discrete_sequence=["#764ba2"],
                            labels={"x": "Residual (Rp)"},
                            title="Distribusi Residual (Prediksi − Aktual)")
        fig2.add_vline(x=0, line_dash="dash", line_color="red")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        fig = px.bar(feat_imp.head(14), x="Importance", y="Feature",
                     orientation="h",
                     color="Importance", color_continuous_scale="Purples",
                     title="Feature Importance (Top 14)")
        fig.update_layout(height=480, coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(feat_imp.style.background_gradient(
            subset=["Importance"], cmap="Purples"), use_container_width=True)
