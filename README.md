# Peramalan IHSG: ARIMA vs ARIMAX

Aplikasi Streamlit untuk analisis & peramalan IHSG dengan eksogen USDIDR, S&P 500, VIX (lag 1).

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
Taruh `dataset_ihsg_eksogen.csv` (kolom: Date, IHSG, USDIDR, SP500, VIX) di folder ini, atau upload lewat sidebar.
