import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# ---------- Pengaturan halaman ----------
st.set_page_config(page_title="Dashboard Data Panel Fiskal", page_icon="📊", layout="wide")
st.title("📊 Dashboard Data Panel Fiskal Kota di Indonesia")
st.caption("Sumber: dataset_fiskal_97_hipoFull.csv | 97 kota, 2014-2024")

# ---------- Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("dataset_fiskal_97_hipoFull.csv")

df = load_data()

fiskal = ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark",
          "DAK_Fisik", "DAK_Nonfisik", "PAD", "ICT"]
belanja = [c for c in df.columns if c.startswith("bm_")]
outcome = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]

# ---------- Sidebar (filter) ----------
st.sidebar.header("⚙️ Filter")
mode = st.sidebar.radio("Cakupan data", ["Satu Kota", "Semua Kota"])
tahun = st.sidebar.slider("Rentang tahun", int(df.Tahun.min()), int(df.Tahun.max()),
                          (int(df.Tahun.min()), int(df.Tahun.max())))

if mode == "Satu Kota":
    kota = st.sidebar.selectbox("Pilih kota", sorted(df.Kota.unique()))
    data = df[(df.Kota == kota) & (df.Tahun.between(*tahun))]
    judul = kota
else:
    data = df[df.Tahun.between(*tahun)]
    judul = "Rata-rata 97 Kota"

st.subheader(f"📍 {judul} ({tahun[0]}-{tahun[1]})")

# ---------- Tab ----------
tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Tren", "Korelasi", "Regresi"])

# === Tab 1: Ringkasan ===
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("IPM", f"{data.IPM.mean():.2f}")
    c2.metric("Kemiskinan (%)", f"{data.KEMISKINAN.mean():.2f}")
    c3.metric("Pengangguran (%)", f"{data.PENGANGGURAN.mean():.2f}")
    c4.metric("Ketimpangan", f"{data.KETIMPANGAN.mean():.3f}")
    c5.metric("Pertumbuhan Ekonomi (%)", f"{data.PE.mean():.2f}")

    st.markdown("**Data**")
    st.dataframe(data, width="stretch")
    st.markdown("**Statistik deskriptif**")
    st.dataframe(data[fiskal + outcome].describe().T, width="stretch")

# === Tab 2: Tren ===
with tab2:
    var = st.selectbox("Pilih variabel", outcome + fiskal + belanja)
    tren = data.groupby("Tahun", as_index=False)[var].mean()
    fig = px.line(tren, x="Tahun", y=var, markers=True, title=f"Tren {var} - {judul}")
    st.plotly_chart(fig, width="stretch")

    if mode == "Semua Kota":
        top = (data.groupby("Kota", as_index=False)[var].mean()
                   .sort_values(var, ascending=False).head(10))
        fig2 = px.bar(top, x=var, y="Kota", orientation="h", title=f"10 Kota Tertinggi - {var}")
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, width="stretch")

# === Tab 3: Korelasi ===
with tab3:
    pilih = st.multiselect("Variabel untuk korelasi", fiskal + belanja + outcome,
                           default=fiskal + outcome)
    if len(pilih) >= 2:
        corr = data[pilih].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pilih minimal 2 variabel ya.")

# === Tab 4: Regresi ===
with tab4:
    y_var = st.selectbox("Variabel Y (dependen)", outcome, index=3)
    x_vars = st.multiselect("Variabel X (independen)", fiskal + belanja,
                            default=["PAD", "DAU_Grant", "DAK_Fisik"])
    fe = False
    if mode == "Semua Kota":
        fe = st.checkbox("Pakai Fixed Effect Kota (LSDV)")

    if not x_vars:
        st.info("Pilih minimal 1 variabel X.")
    elif len(data) <= len(x_vars) + 1:
        st.warning("Data terlalu sedikit untuk jumlah X ini. Kurangi variabel X atau pilih 'Semua Kota'.")
    else:
        X = data[x_vars].copy()
        if fe:
            X = pd.concat([X, pd.get_dummies(data.Kota, drop_first=True, dtype=float)], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(data[y_var], X).fit()

        m1, m2, m3 = st.columns(3)
        m1.metric("R-squared", f"{model.rsquared:.3f}")
        m2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
        m3.metric("Jumlah observasi", int(model.nobs))

        hasil = pd.DataFrame({
            "Koefisien": model.params,
            "Std. Error": model.bse,
            "t-stat": model.tvalues,
            "p-value": model.pvalues,
        }).loc[["const"] + x_vars].round(4)
        hasil["Signifikan (5%)"] = hasil["p-value"].apply(lambda p: "Ya" if p < 0.05 else "Tidak")
        st.markdown("**Hasil regresi**")
        st.dataframe(hasil, width="stretch")

        with st.expander("Lihat summary lengkap"):
            st.text(model.summary().as_text())

        fig = px.scatter(x=model.fittedvalues, y=data[y_var],
                         labels={"x": "Nilai prediksi", "y": f"{y_var} aktual"},
                         title="Aktual vs Prediksi")
        fig.add_shape(type="line", x0=data[y_var].min(), y0=data[y_var].min(),
                      x1=data[y_var].max(), y1=data[y_var].max(), line=dict(dash="dash"))
        st.plotly_chart(fig, width="stretch")