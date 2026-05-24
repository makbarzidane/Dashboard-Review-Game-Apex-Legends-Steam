# Dashboard Analisis Sentimen Apex Legends - Revisi

Dashboard ini dibuat untuk menjalankan model revisi analisis sentimen ulasan berbahasa Inggris game Apex Legends pada Steam.

## Isi file
- `app.py` : aplikasi dashboard Streamlit.
- `requirements.txt` : library yang dibutuhkan.
- `runtime.txt` : rekomendasi Python untuk Streamlit Cloud.
- `apex_sentiment_model_final_revisi.pkl` : paket model final berisi model dan vectorizer.
- `model_nb_apex_revisi.pkl` : file model Naive Bayes cadangan.
- `tfidf_vectorizer_revisi.pkl` : file TF-IDF vectorizer cadangan.
- `dataset_final_bersih_tanpa_kosong.csv` : dataset final setelah preprocessing.
- `scraping_data_2026.csv` : dataset hasil scraping mentah.

## Cara menjalankan di lokal
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Cara deploy ke Streamlit Cloud
1. Upload semua file dalam folder ini ke repository GitHub.
2. Pastikan `app.py`, `requirements.txt`, model `.pkl`, dan dataset `.csv` berada di root repository.
3. Buka Streamlit Cloud.
4. Pilih repository.
5. Pada bagian Main file path, isi:
   `app.py`
6. Klik Deploy.

## Catatan penting
- Dashboard tidak melakukan training ulang.
- Dataset dibaca dengan `keep_default_na=False` agar token `nan` pada `final_text` tidak dianggap sebagai data kosong oleh pandas.
- Pemeriksaan model pada tab dashboard hanya membaca model tersimpan dan memprediksi dataset aktif, bukan menggantikan evaluasi train-test pada notebook penelitian.
