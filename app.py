from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#6C5CE7", "#FF7AA2", "#00B8A9", "#FFB703", "#A855F7"]
px.defaults.color_continuous_scale = ["#F4F1FF", "#A855F7", "#6C5CE7"]

# ---------- Pengaturan halaman ----------
st.set_page_config(
    page_title="Dashboard Data Panel Fiskal",
    page_icon=str(Path(__file__).parent / "favicon.png"),
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

h1, h2, h3, h4, [data-testid="stMetricValue"] { font-family: 'Poppins', sans-serif; }
h3, h4 { color: #6C5CE7; }

/* banner utama */
.hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #A855F7 50%, #FF7AA2 100%);
    padding: 2rem 2.2rem; border-radius: 22px; color: white;
    box-shadow: 0 10px 28px rgba(108, 92, 231, 0.30); margin-bottom: 1.2rem;
}
.hero h1 { color: white; margin: 0; font-size: 2.1rem; }
.hero p { margin: 0.4rem 0 0; opacity: 0.92; font-size: 1.02rem; }

/* kartu metrik */
[data-testid="stMetric"] {
    background: white; border-radius: 16px; padding: 14px 18px;
    border-left: 6px solid #6C5CE7; box-shadow: 0 4px 14px rgba(108, 92, 231, 0.15);
}
[data-testid="stMetricValue"] { color: #6C5CE7; font-weight: 700; }

/* tab */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #F4F1FF; border-radius: 12px; padding: 8px 18px;
}
.stTabs [aria-selected="true"] { background: #6C5CE7; }
.stTabs [aria-selected="true"] p { color: white; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #F4F1FF 0%, #FFE9F1 100%); }
</style>

<div class="hero">
    <h1>Dashboard Data Panel Fiskal</h1>
    <p>97 kota di Indonesia (2014-2024): tren, korelasi, regresi, machine learning, dan visualisasi 3D</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv("dataset_fiskal_97_hipoFull.csv")

df = load_data()

fiskal = ["DBH_PAJAK", "DBH_SDA", "DBH_Lainnya", "DAU_Grant", "DAU_Earmark",
          "DAK_Fisik", "DAK_Nonfisik", "PAD", "ICT"]
belanja = [c for c in df.columns if c.startswith("bm_")]
outcome = ["KEMISKINAN", "PENGANGGURAN", "KETIMPANGAN", "IPM", "PE"]

def kekuatan(r):
    a = abs(r)
    if a >= 0.7: return "sangat kuat"
    if a >= 0.5: return "kuat"
    if a >= 0.3: return "sedang"
    if a >= 0.1: return "lemah"
    return "sangat lemah"

def tafsir_r2(r2):
    if r2 >= 0.5: return "cukup baik"
    if r2 >= 0.2: return "terbatas"
    return "sangat rendah"

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    ":material/home: Ringkasan",
    ":material/show_chart: Tren",
    ":material/grid_on: Korelasi",
    ":material/functions: Regresi",
    ":material/psychology: Machine Learning",
    ":material/view_in_ar: Regresi 3D",
])
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

    st.markdown("#### 📝 Interpretasi")
    rata_all = df[df.Tahun.between(*tahun)][outcome].mean()
    arti = {"IPM": "semakin tinggi semakin baik",
            "KEMISKINAN": "semakin rendah semakin baik",
            "PENGANGGURAN": "semakin rendah semakin baik",
            "KETIMPANGAN": "semakin rendah (mendekati 0) semakin merata",
            "PE": "semakin tinggi semakin baik"}
    if mode == "Satu Kota":
        for v in outcome:
            nilai = data[v].mean()
            banding = "di atas" if nilai > rata_all[v] else "di bawah"
            st.write(f"- **{v}** {judul} rata-rata **{nilai:.2f}**, {banding} rata-rata 97 kota "
                     f"({rata_all[v]:.2f}). _({arti[v]})_")
    else:
        for v in outcome:
            pk = data.groupby("Kota")[v].mean()
            st.write(f"- **{v}** rata-rata **{data[v].mean():.2f}**. Tertinggi: {pk.idxmax()} "
                     f"({pk.max():.2f}), terendah: {pk.idxmin()} ({pk.min():.2f}). _({arti[v]})_")

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

    if len(tren) >= 2:
        awal, akhir = tren[var].iloc[0], tren[var].iloc[-1]
        th_max = int(tren.loc[tren[var].idxmax(), "Tahun"])
        th_min = int(tren.loc[tren[var].idxmin(), "Tahun"])
        arah = "meningkat" if akhir > awal else "menurun"
        st.markdown("#### 📝 Interpretasi")
        st.write(f"Secara keseluruhan, {var} di {judul} **{arah}** dari {awal:.2f} "
                 f"({int(tren.Tahun.iloc[0])}) menjadi {akhir:.2f} ({int(tren.Tahun.iloc[-1])}), "
                 f"selisih {akhir - awal:+.2f}. Nilai tertinggi terjadi pada {th_max} dan "
                 f"terendah pada {th_min}. Lihat juga pola grafik di atas, karena perubahan "
                 f"antartahun bisa naik-turun dan tidak selalu searah dengan perubahan total.")

# === Tab 3: Korelasi ===
with tab3:
    pilih = st.multiselect("Variabel untuk korelasi", fiskal + belanja + outcome,
                           default=fiskal + outcome)
    if len(pilih) >= 2:
        corr = data[pilih].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(fig, width="stretch")
        pasangan = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
        kuat = pasangan.abs().idxmax()
        r = pasangan[kuat]
        n_sedang = int((pasangan.abs() >= 0.3).sum())
        arah = "positif" if r > 0 else "negatif"
        st.markdown("#### 📝 Interpretasi")
        st.write(f"Korelasi terkuat ada pada **{kuat[0]}** dan **{kuat[1]}** dengan r = **{r:.2f}** "
                 f"(hubungan {arah}, kekuatan {kekuatan(r)}). Dari {len(pasangan)} pasangan variabel, "
                 f"**{n_sedang}** pasangan memiliki korelasi sedang atau lebih (|r| ≥ 0.3). "
                 f"Perlu diingat, korelasi hanya menunjukkan keterkaitan, bukan sebab-akibat.")
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

        sig = hasil[(hasil["p-value"] < 0.05) & (hasil.index != "const")]
        st.markdown("#### 📝 Interpretasi")
        st.write(f"Nilai R² sebesar **{model.rsquared:.3f}**, artinya variabel X yang dipilih "
                 f"menjelaskan sekitar **{model.rsquared * 100:.1f}%** variasi {y_var}; sisanya "
                 f"dipengaruhi faktor lain di luar model. Daya jelas model tergolong "
                 f"**{tafsir_r2(model.rsquared)}**.")
        if model.f_pvalue < 0.05:
            st.write(f"Uji F signifikan (p = {model.f_pvalue:.4f}): secara bersama-sama variabel X "
                     f"berpengaruh terhadap {y_var}.")
        else:
            st.write(f"Uji F tidak signifikan (p = {model.f_pvalue:.4f}): secara bersama-sama "
                     f"variabel X belum terbukti berpengaruh terhadap {y_var}.")
        if len(sig) == 0:
            st.write("Secara parsial, tidak ada variabel X yang signifikan pada taraf 5% (p-value > 0.05).")
        else:
            for v, row in sig.iterrows():
                arah = "naik" if row["Koefisien"] > 0 else "turun"
                st.write(f"- **{v}** signifikan (p = {row['p-value']:.4f}): setiap kenaikan 1 satuan {v}, "
                         f"{y_var} rata-rata {arah} {abs(row['Koefisien']):.4f} satuan, dengan variabel lain tetap.")

        # === Tab 5: Machine Learning ===
with tab5:
    st.markdown("### 🤖 Prediksi dengan Random Forest")

    if mode == "Satu Kota":
        st.warning("Data satu kota cuma sekitar 11 baris, terlalu sedikit untuk machine learning. "
                   "Ganti ke **Semua Kota** di sidebar.")
    else:
        y_ml = st.selectbox("Variabel yang diprediksi (Y)", outcome, index=3, key="y_ml")
        x_ml = st.multiselect("Fitur (X)", fiskal + belanja, default=fiskal, key="x_ml")

        c1, c2 = st.columns(2)
        test_size = c1.slider("Porsi data uji (%)", 10, 40, 20) / 100
        n_tree = c2.slider("Jumlah pohon", 50, 500, 200, step=50)

        if len(x_ml) < 1:
            st.info("Pilih minimal 1 fitur.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[x_ml], data[y_ml], test_size=test_size, random_state=42)

            rf = RandomForestRegressor(n_estimators=n_tree, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)

            lr = LinearRegression().fit(X_train, y_train)
            pred_lr = lr.predict(X_test)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Random Forest", f"{r2_score(y_test, pred):.3f}")
            m2.metric("R² Regresi Linear", f"{r2_score(y_test, pred_lr):.3f}")
            m3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, pred)):.3f}")
            m4.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")

            imp = (pd.DataFrame({"Fitur": x_ml, "Importance": rf.feature_importances_})
                   .sort_values("Importance"))
            fig = px.bar(imp, x="Importance", y="Fitur", orientation="h",
                         title="Feature Importance (fitur paling berpengaruh)")
            st.plotly_chart(fig, width="stretch")

            fig2 = px.scatter(x=y_test, y=pred,
                              labels={"x": f"{y_ml} aktual", "y": f"{y_ml} prediksi"},
                              title="Aktual vs Prediksi (data uji)")
            fig2.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                           x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
            st.plotly_chart(fig2, width="stretch")

            st.caption("R² mendekati 1 = prediksi bagus. R² mendekati 0 atau negatif = "
                       "model tidak lebih baik dari menebak rata-rata.")

            r2_rf = r2_score(y_test, pred)
            r2_lr = r2_score(y_test, pred_lr)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            top = imp.iloc[-1]
            lebih = "lebih baik" if r2_rf > r2_lr else "tidak lebih baik"
            st.markdown("#### 📝 Interpretasi")
            st.write(f"Random Forest menghasilkan R² data uji sebesar **{r2_rf:.3f}**, {lebih} "
                     f"dibanding regresi linear (R² = {r2_lr:.3f}). Rata-rata kesalahan prediksi (RMSE) "
                     f"**{rmse:.2f}**, atau sekitar **{rmse / data[y_ml].mean() * 100:.1f}%** dari rata-rata {y_ml}.")
            st.write(f"Fitur dengan importance tertinggi adalah **{top['Fitur']}** ({top['Importance'] * 100:.1f}%). "
                     f"Importance hanya menunjukkan seberapa sering fitur dipakai model, bukan arah pengaruhnya.")
            if r2_rf < 0.2:
                st.write(f"Karena R² masih rendah, model belum mampu memprediksi {y_ml} dengan baik. "
                         f"Artinya variabel yang dipilih belum cukup menjelaskan {y_ml} pada data ini.")

            # === Tab 6: Regresi Linear Berganda 3D ===
with tab6:
    st.markdown("### 🧊 Regresi Linear Berganda 3D")
    st.caption("Y = b0 + b1·X1 + b2·X2. Titik = data aktual, bidang = hasil regresi.")

    pilih_x = fiskal + belanja
    c1, c2, c3 = st.columns(3)
    x1 = c1.selectbox("Variabel X1", pilih_x, index=pilih_x.index("PAD"), key="x1_3d")
    x2 = c2.selectbox("Variabel X2", pilih_x, index=pilih_x.index("DAU_Grant"), key="x2_3d")
    y3 = c3.selectbox("Variabel Y", outcome, index=3, key="y_3d")

    if x1 == x2:
        st.warning("X1 dan X2 harus beda ya.")
    elif len(data) < 5:
        st.warning("Data terlalu sedikit. Perluas rentang tahun atau pilih 'Semua Kota'.")
    else:
        X3 = sm.add_constant(data[[x1, x2]])
        m3 = sm.OLS(data[y3], X3).fit()
        b0, b1, b2 = m3.params["const"], m3.params[x1], m3.params[x2]

        k1, k2, k3 = st.columns(3)
        k1.metric("R-squared", f"{m3.rsquared:.3f}")
        k2.metric("Adj. R-squared", f"{m3.rsquared_adj:.3f}")
        k3.metric("Prob (F-stat)", f"{m3.f_pvalue:.4f}")

        st.info(f"**Persamaan:** {y3} = {b0:.3f} + ({b1:.4f} × {x1}) + ({b2:.4f} × {x2})")

        # bidang regresi
        g1 = np.linspace(data[x1].min(), data[x1].max(), 20)
        g2 = np.linspace(data[x2].min(), data[x2].max(), 20)
        xx, yy = np.meshgrid(g1, g2)
        zz = b0 + b1 * xx + b2 * yy

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=data[x1], y=data[x2], z=data[y3], mode="markers",
            marker=dict(size=3, opacity=0.7, color="#6C5CE7"), name="Data aktual"))
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz, opacity=0.5,
            colorscale=[[0, "#C4B5FD"], [1, "#FF7AA2"]], showscale=False,
            name="Bidang regresi"))
        fig.update_layout(
            height=650, margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(xaxis_title=x1, yaxis_title=x2, zaxis_title=y3))
        st.plotly_chart(fig, width="stretch")

        hasil3 = pd.DataFrame({
            "Koefisien": m3.params, "Std. Error": m3.bse,
            "t-stat": m3.tvalues, "p-value": m3.pvalues}).round(4)
        hasil3["Signifikan (5%)"] = hasil3["p-value"].apply(lambda p: "Ya" if p < 0.05 else "Tidak")
        st.dataframe(hasil3, width="stretch")

        st.markdown("#### 📝 Interpretasi")
        st.write(f"Model menjelaskan **{m3.rsquared * 100:.1f}%** variasi {y3} (R² = {m3.rsquared:.3f}), "
                 f"tergolong **{tafsir_r2(m3.rsquared)}**. Uji F "
                 f"{'signifikan' if m3.f_pvalue < 0.05 else 'tidak signifikan'} (p = {m3.f_pvalue:.4f}).")
        for v, b in [(x1, b1), (x2, b2)]:
            p = m3.pvalues[v]
            arah = "naik" if b > 0 else "turun"
            ket = "signifikan" if p < 0.05 else "tidak signifikan"
            st.write(f"- **{v}**: koefisien {b:.4f} (p = {p:.4f}, {ket} pada 5%). Setiap kenaikan 1 satuan {v}, "
                     f"{y3} diperkirakan {arah} {abs(b):.4f} satuan, dengan {x2 if v == x1 else x1} tetap.")
        if abs(b1) + abs(b2) < 0.05:
            st.write("Bidang regresi hampir datar, menandakan pengaruh kedua variabel terhadap Y sangat kecil.")