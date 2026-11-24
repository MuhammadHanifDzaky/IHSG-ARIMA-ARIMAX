import io
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="ARIMA vs ARIMAX IHSG", page_icon="📈", layout="wide")

VARIABEL = ["IHSG", "USDIDR", "SP500", "VIX"]
EXOG = ["USDIDR_lag1", "SP500_lag1", "VIX_lag1"]
KOLOM_ANALISIS = ["IHSG"] + EXOG
DEFAULT_CSV = "dataset_ihsg_eksogen.csv"
KANDIDAT = [(0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0), (0, 1, 2),
            (1, 1, 2), (2, 1, 1), (2, 1, 2), (3, 1, 0), (0, 1, 3)]


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    missing = [c for c in ["Date"] + VARIABEL if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    return df


@st.cache_data(show_spinner=False)
def siapkan_data(df: pd.DataFrame, rasio_latih: float):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.sort_values("Date").reset_index(drop=True)
    n_awal = len(d)
    d = d[d["IHSG"].notna()].reset_index(drop=True)  # kalender bursa IHSG
    n_ihsg = len(d)
    for c in ["USDIDR", "SP500", "VIX"]:
        d[f"{c}_lag1"] = d[c].shift(1)  # hindari kebocoran informasi
    final = d[["Date", "IHSG"] + EXOG].dropna().reset_index(drop=True)
    n_train = int(len(final) * rasio_latih)
    return d, final, n_train, n_awal, n_ihsg


@st.cache_resource(show_spinner="Mengestimasi model...")
def fit_model(train_ihsg: pd.Series, train_exog, orde, pakai_exog: bool):
    if pakai_exog:
        return SARIMAX(train_ihsg, exog=train_exog, order=orde).fit(disp=False)
    return SARIMAX(train_ihsg, order=orde).fit(disp=False)


def uji_stasioner(series):
    adf = adfuller(series, autolag="AIC")
    kp = kpss(series, regression="c", nlags="auto")
    return adf[0], adf[1], kp[0], kp[1]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Pengaturan")
upload = st.sidebar.file_uploader("Upload dataset CSV", type="csv",
                                  help="Kolom wajib: Date, IHSG, USDIDR, SP500, VIX")
if upload is not None:
    sumber = upload
elif os.path.exists(DEFAULT_CSV):
    sumber = DEFAULT_CSV
    st.sidebar.info(f"Memakai `{DEFAULT_CSV}` dari repo.")
else:
    sumber = None

rasio = st.sidebar.slider("Proporsi data latih", 0.5, 0.95, 0.80, 0.05)
orde_pilih = st.sidebar.selectbox("Orde model untuk evaluasi", KANDIDAT,
                                  index=KANDIDAT.index((0, 1, 3)))

st.title("📈 Peramalan IHSG: ARIMA vs ARIMAX")
st.caption("Variabel eksogen: USD/IDR, S&P 500, VIX (lag 1 sesi untuk menghindari kebocoran informasi masa depan).")

if sumber is None:
    st.warning("Upload `dataset_ihsg_eksogen.csv` lewat sidebar untuk memulai "
               "(kolom: Date, IHSG, USDIDR, SP500, VIX).")
    st.stop()

try:
    df = load_csv(sumber)
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

df_clean, final, n_train, n_awal, n_ihsg = siapkan_data(df, rasio)
train, test = final.iloc[:n_train].reset_index(drop=True), final.iloc[n_train:].reset_index(drop=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1 · Data", "2 · Stasioneritas", "3 · Pemilihan Model", "4 · Evaluasi", "5 · Ramalan Besok"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris awal", n_awal)
    c2.metric("Setelah filter hari IHSG", n_ihsg)
    c3.metric("Siap analisis", len(final))
    c4.metric("Latih / Uji", f"{len(train)} / {len(test)}")
    st.caption(f"Latih: {train['Date'].min().date()} – {train['Date'].max().date()} · "
               f"Uji: {test['Date'].min().date()} – {test['Date'].max().date()}")

    desk = df[VARIABEL].describe().T
    desk["median"] = df[VARIABEL].median()
    desk = desk[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
    st.subheader("Statistik deskriptif")
    st.dataframe(desk.round(4), width="stretch")
    st.subheader("Korelasi Pearson")
    st.dataframe(df[VARIABEL].corr().round(4), width="stretch")

    st.subheader("Grafik deret waktu")
    dp = df.copy()
    dp["Date"] = pd.to_datetime(dp["Date"])
    dp = dp.sort_values("Date")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, col, w in zip(axes, VARIABEL, ["tab:blue", "tab:green", "tab:orange", "tab:red"]):
        ax.plot(dp["Date"], dp[col], color=w, linewidth=1)
        ax.set_ylabel(col)
        ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Box-Cox (data latih)")
    st.dataframe(pd.DataFrame(
        {"Variabel": KOLOM_ANALISIS,
         "Lambda": [round(stats.boxcox(train[c])[1], 4) for c in KOLOM_ANALISIS]}
    ).set_index("Variabel"), width="content")

    st.subheader("ADF & KPSS")
    baris = []
    for c in KOLOM_ANALISIS:
        a, pa, k, pk = uji_stasioner(train[c])
        a1, pa1, k1, pk1 = uji_stasioner(train[c].diff().dropna())
        baris.append({"Variabel": c, "ADF level": a, "p ADF level": pa, "KPSS level": k, "p KPSS level": pk,
                      "ADF diff1": a1, "p ADF diff1": pa1, "KPSS diff1": k1, "p KPSS diff1": pk1})
    st.dataframe(pd.DataFrame(baris).set_index("Variabel").round(4), width="stretch")

    st.subheader("IHSG setelah differencing orde 1")
    d1 = train["IHSG"].diff().dropna()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(train["Date"].iloc[1:], d1, linewidth=1)
    ax.axhline(0, color="red", linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    plot_acf(d1, lags=20, ax=axes[0])
    plot_pacf(d1, lags=20, ax=axes[1], method="ywm")
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.caption("Estimasi 11 kandidat × 2 jenis model (ARIMA & ARIMAX). Klik tombol untuk menjalankan.")
    if st.button("Jalankan perbandingan AIC/BIC"):
        hasil = []
        prog = st.progress(0.0)
        total = len(KANDIDAT) * 2
        i = 0
        for tipe in ["ARIMA", "ARIMAX"]:
            for o in KANDIDAT:
                m = fit_model(train["IHSG"], train[EXOG] if tipe == "ARIMAX" else None, o, tipe == "ARIMAX")
                lb = acorr_ljungbox(m.resid.iloc[1:], lags=[10], return_df=True)["lb_pvalue"].values[0]
                hasil.append({"Jenis": tipe, "Model": f"{tipe}{o}", "k": len(m.params),
                              "AIC": m.aic, "BIC": m.bic, "LjungBox_p10": lb})
                i += 1
                prog.progress(i / total)
        prog.empty()
        tk = pd.DataFrame(hasil)
        tk["Selisih_AIC"] = tk.groupby("Jenis")["AIC"].transform(lambda s: s - s.min())
        tk["Selisih_BIC"] = tk.groupby("Jenis")["BIC"].transform(lambda s: s - s.min())
        st.session_state["tabel_kandidat"] = tk.round(4)
    if "tabel_kandidat" in st.session_state:
        tk = st.session_state["tabel_kandidat"]
        for tipe in ["ARIMA", "ARIMAX"]:
            st.subheader(tipe)
            st.dataframe(tk[tk["Jenis"] == tipe].drop(columns="Jenis"), width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 4
m_arima = fit_model(train["IHSG"], None, orde_pilih, False)
m_arimax = fit_model(train["IHSG"], train[EXOG], orde_pilih, True)

with tab4:
    st.subheader(f"Signifikansi parameter · orde {orde_pilih}")
    cA, cB = st.columns(2)
    for col, nama, m in [(cA, "ARIMA", m_arima), (cB, "ARIMAX", m_arimax)]:
        col.markdown(f"**{nama}**")
        col.dataframe(pd.DataFrame({"Koef": m.params, "SE": m.bse, "z": m.zvalues, "p": m.pvalues}).round(4))

    st.subheader("Ljung-Box residual")
    lags = [5, 10, 15, 20]
    lbA = acorr_ljungbox(m_arima.resid.iloc[1:], lags=lags, return_df=True)
    lbX = acorr_ljungbox(m_arimax.resid.iloc[1:], lags=lags, return_df=True)
    st.dataframe(pd.DataFrame({"LB ARIMA": lbA["lb_stat"], "p ARIMA": lbA["lb_pvalue"],
                               "LB ARIMAX": lbX["lb_stat"], "p ARIMAX": lbX["lb_pvalue"]}).round(4))

    # one-step-ahead pada data uji (parameter dari data latih)
    pred_arima = m_arima.apply(final["IHSG"]).fittedvalues.iloc[n_train:].reset_index(drop=True)
    pred_arimax = m_arimax.apply(final["IHSG"], exog=final[EXOG]).fittedvalues.iloc[n_train:].reset_index(drop=True)
    aktual = test["IHSG"].reset_index(drop=True)

    def metrik(y, p, nama):
        mse = mean_squared_error(y, p)
        return {"Model": nama, "MSE": mse, "RMSE": np.sqrt(mse),
                "MAPE (%)": mean_absolute_percentage_error(y, p) * 100}

    tabel_eval = pd.DataFrame([metrik(aktual, pred_arima, f"ARIMA{orde_pilih}"),
                               metrik(aktual, pred_arimax, f"ARIMAX{orde_pilih}")]).round(4)
    st.subheader("Akurasi peramalan data uji (one-step-ahead)")
    st.dataframe(tabel_eval, hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(test["Date"], aktual, label="Aktual", color="black", linewidth=1.3)
    ax.plot(test["Date"], pred_arima, label="ARIMA", color="tab:blue", linestyle="--")
    ax.plot(test["Date"], pred_arimax, label="ARIMAX", color="tab:red", linestyle=":")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    tabel_ramalan = pd.DataFrame({
        "Tanggal": test["Date"].dt.strftime("%Y-%m-%d"), "IHSG_Aktual": aktual,
        "Ramalan_ARIMA": pred_arima, "Galat_ARIMA": aktual - pred_arima,
        "Ramalan_ARIMAX": pred_arimax, "Galat_ARIMAX": aktual - pred_arimax}).round(4)
    with st.expander("Tabel ramalan lengkap"):
        st.dataframe(tabel_ramalan, hide_index=True, width="stretch")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        desk.round(4).to_excel(w, sheet_name="1_Deskriptif")
        df[VARIABEL].corr().round(4).to_excel(w, sheet_name="2_Korelasi")
        tabel_eval.to_excel(w, sheet_name="8_Evaluasi_Metrik", index=False)
        tabel_ramalan.to_excel(w, sheet_name="9_Ramalan_Data_Uji", index=False)
    st.download_button("⬇️ Unduh hasil (Excel)", buf.getvalue(), "hasil_arima_arimax.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------- tab 5
with tab5:
    st.subheader("Ramalan IHSG sesi perdagangan berikutnya")
    st.caption("Nilai eksogen untuk besok = nilai USDIDR/SP500/VIX terakhir yang sudah teramati "
               "(sesuai definisi lag 1), jadi ramalan 1 langkah valid tanpa asumsi tambahan.")
    terakhir = df_clean.iloc[-1]
    exog_next = pd.DataFrame([[terakhir["USDIDR"], terakhir["SP500"], terakhir["VIX"]]], columns=EXOG)
    f_arima = m_arima.apply(final["IHSG"]).forecast(1).iloc[0]
    f_arimax = m_arimax.apply(final["IHSG"], exog=final[EXOG]).forecast(1, exog=exog_next).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("IHSG terakhir", f"{terakhir['IHSG']:,.2f}", help=str(terakhir["Date"].date()))
    c2.metric(f"ARIMA{orde_pilih}", f"{f_arima:,.2f}", f"{f_arima - terakhir['IHSG']:+,.2f}")
    c3.metric(f"ARIMAX{orde_pilih}", f"{f_arimax:,.2f}", f"{f_arimax - terakhir['IHSG']:+,.2f}")
    st.info("Ini alat analisis akademik, bukan saran investasi.")
