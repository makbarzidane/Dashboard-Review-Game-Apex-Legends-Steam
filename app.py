import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from wordcloud import WordCloud


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen Apex Legends",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.25rem;
        }
        .subtitle {
            font-size: 1rem;
            color: #CBD5E1;
            margin-bottom: 1.2rem;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            margin-right: 0.35rem;
            border-radius: 999px;
            background: rgba(245, 158, 11, 0.16);
            border: 1px solid rgba(245, 158, 11, 0.35);
            color: #FDE68A;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .note-box {
            padding: 0.95rem 1rem;
            border-radius: 1rem;
            background: rgba(15, 23, 42, 0.70);
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: #E2E8F0;
        }
        div[data-testid="stMetric"] {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 0.85rem 1rem;
            border-radius: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            background: rgba(148, 163, 184, 0.12);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

LABEL_MAPPING = {0: "Negative", 1: "Positive"}

# Hasil evaluasi utama dari notebook penelitian.
# Nilai ini berasal dari data uji setelah proses balancing dan train-test split,
# sehingga berbeda dari pemeriksaan terhadap seluruh dataset aktif di dashboard.
RESEARCH_ACCURACY = 0.813237
RESEARCH_CM = np.array([[467, 85], [121, 430]])
RESEARCH_REPORT = pd.DataFrame(
    [
        {"Metrik": "Negative", "precision": 0.794218, "recall": 0.846014, "f1-score": 0.819298, "support": 552},
        {"Metrik": "Positive", "precision": 0.834951, "recall": 0.780399, "f1-score": 0.806754, "support": 551},
        {"Metrik": "accuracy", "precision": 0.813237, "recall": 0.813237, "f1-score": 0.813237, "support": 0.813237},
        {"Metrik": "macro avg", "precision": 0.814585, "recall": 0.813207, "f1-score": 0.813026, "support": 1103},
        {"Metrik": "weighted avg", "precision": 0.814566, "recall": 0.813237, "f1-score": 0.813032, "support": 1103},
    ]
)
LABEL_TO_CODE = {"Negative": 0, "Positive": 1}
STEMMER = PorterStemmer()


# =========================================================
# LOAD MODEL DAN DATA
# =========================================================
@st.cache_resource(show_spinner="Memuat model dan vectorizer...")
def load_model_bundle():
    """
    Prioritas load:
    1. apex_sentiment_model_final_revisi.pkl, berisi dict model + vectorizer.
    2. model_nb_apex_revisi.pkl + tfidf_vectorizer_revisi.pkl, sebagai fallback.
    """
    bundle_path = "apex_sentiment_model_final_revisi.pkl"
    model_path = "model_nb_apex_revisi.pkl"
    vectorizer_path = "tfidf_vectorizer_revisi.pkl"

    if os.path.exists(bundle_path):
        bundle = joblib.load(bundle_path)
        if isinstance(bundle, dict) and "model" in bundle and "vectorizer" in bundle:
            return bundle

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
            "config": {},
            "protected_tokens": [],
            "important_words": [],
        }

    st.error(
        "File model belum ditemukan. Pastikan file `apex_sentiment_model_final_revisi.pkl` "
        "atau pasangan `model_nb_apex_revisi.pkl` dan `tfidf_vectorizer_revisi.pkl` berada satu folder dengan app.py."
    )
    st.stop()


@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    final_path = "dataset_final_bersih_tanpa_kosong.csv"
    raw_path = "scraping_data_2026.csv"

    final_df = pd.DataFrame()
    raw_df = pd.DataFrame()

    if os.path.exists(final_path):
        # keep_default_na=False penting agar token final_text "nan" tidak dibaca sebagai nilai kosong.
        final_df = pd.read_csv(final_path, keep_default_na=False)
    else:
        st.warning("File `dataset_final_bersih_tanpa_kosong.csv` belum ditemukan.")

    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path, keep_default_na=False)
    else:
        st.warning("File `scraping_data_2026.csv` belum ditemukan.")

    return final_df, raw_df


def ensure_label_columns(df):
    df = df.copy()

    if "label" not in df.columns and "label_encoded" in df.columns:
        df["label"] = df["label_encoded"].map(LABEL_MAPPING)

    if "label_encoded" not in df.columns and "label" in df.columns:
        df["label_encoded"] = df["label"].map(LABEL_TO_CODE)

    if "final_text" in df.columns:
        df["final_text"] = df["final_text"].astype(str).fillna("").str.strip()

    if "review" in df.columns:
        df["review"] = df["review"].astype(str).fillna("").str.strip()

    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    elif "timestamp_created" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
    else:
        df["date_parsed"] = pd.NaT

    return df


def build_balanced_data(df, random_state=42):
    if df.empty or "label_encoded" not in df.columns:
        return df.copy()

    temp = df.dropna(subset=["label_encoded"]).copy()
    temp["label_encoded"] = temp["label_encoded"].astype(int)

    df_pos = temp[temp["label_encoded"] == 1]
    df_neg = temp[temp["label_encoded"] == 0]

    if len(df_pos) == 0 or len(df_neg) == 0:
        return temp

    n_min = min(len(df_pos), len(df_neg))

    if len(df_pos) > n_min:
        df_pos = resample(df_pos, replace=False, n_samples=n_min, random_state=random_state)

    if len(df_neg) > n_min:
        df_neg = resample(df_neg, replace=False, n_samples=n_min, random_state=random_state)

    return (
        pd.concat([df_pos, df_neg], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


# =========================================================
# PREPROCESSING UNTUK PREDIKSI MANUAL
# =========================================================
def get_stop_words(bundle):
    protected_tokens = set(bundle.get("protected_tokens", []))
    important_words = set(bundle.get("important_words", []))

    negation_words = {
        "no", "not", "nor", "never", "none", "neither", "without", "cannot",
        "dont", "don't", "didnt", "didn't", "cant", "can't", "wont", "won't",
        "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    }

    keep_words = protected_tokens | important_words | negation_words
    return set(ENGLISH_STOP_WORDS) - keep_words


def replace_protected_phrases(text, protected_tokens):
    text = str(text)
    # Frasa yang disimpan memakai underscore, misalnya "not good" -> "not_good".
    phrase_pairs = []
    for token in protected_tokens:
        if "_" in token:
            phrase_pairs.append((token.replace("_", " "), token))

    # Frasa lebih panjang diproses lebih dulu.
    phrase_pairs = sorted(phrase_pairs, key=lambda item: len(item[0].split()), reverse=True)

    for phrase, token in phrase_pairs:
        text = re.sub(rf"\b{re.escape(phrase)}\b", token, text)

    return text


def preprocess_text(text, bundle):
    stop_words = get_stop_words(bundle)
    protected_tokens = set(bundle.get("protected_tokens", []))

    step_case = str(text).lower()
    step_case = re.sub(r"can['’]?t|cannot", "can not", step_case)
    step_case = re.sub(r"won['’]?t", "will not", step_case)
    step_case = re.sub(r"n['’]?t\b", " not", step_case)

    step_clean = re.sub(r"http\S+|www\.\S+", " ", step_case)
    step_clean = re.sub(r"[@#]\S+", " ", step_clean)
    step_clean = re.sub(r"[^a-zA-Z_\s]", " ", step_clean)
    step_clean = re.sub(r"\s+", " ", step_clean).strip()

    step_norm = re.sub(r"(.)\1{2,}", r"\1\1", step_clean)
    step_norm = replace_protected_phrases(step_norm, protected_tokens)
    step_norm = re.sub(r"\s+", " ", step_norm).strip()

    tokens = step_norm.split()
    filtered = [word for word in tokens if word not in stop_words]

    stemmed = []
    for word in filtered:
        if word in protected_tokens:
            stemmed.append(word)
        else:
            stemmed.append(STEMMER.stem(word))

    final_text = " ".join(stemmed)

    return {
        "case_folding": step_case,
        "cleansing": step_clean,
        "normalization": step_norm,
        "tokenizing": tokens,
        "filtering": filtered,
        "stemming": stemmed,
        "final_text": final_text,
    }


# =========================================================
# HELPER DASHBOARD
# =========================================================
def apply_filters(df, selected_labels, keyword, date_range):
    filtered = df.copy()

    if selected_labels and "label" in filtered.columns:
        filtered = filtered[filtered["label"].isin(selected_labels)]

    if date_range and "date_parsed" in filtered.columns:
        start_date, end_date = date_range
        if start_date and end_date:
            date_only = filtered["date_parsed"].dt.date
            filtered = filtered[(date_only >= start_date) & (date_only <= end_date)]

    if keyword:
        keyword = keyword.strip()
        if keyword:
            text_columns = [col for col in ["review", "final_text", "cleansing", "normalization"] if col in filtered.columns]
            if text_columns:
                mask = pd.Series(False, index=filtered.index)
                for col in text_columns:
                    mask = mask | filtered[col].astype(str).str.contains(keyword, case=False, na=False, regex=False)
                filtered = filtered[mask]

    return filtered


def label_counts(df):
    if df.empty or "label" not in df.columns:
        return pd.DataFrame({"Sentimen": ["Positive", "Negative"], "Jumlah": [0, 0]})

    counts = df["label"].value_counts().reindex(["Positive", "Negative"], fill_value=0).reset_index()
    counts.columns = ["Sentimen", "Jumlah"]
    return counts


def make_wordcloud(df):
    if df.empty or "final_text" not in df.columns:
        return None

    text = " ".join(df["final_text"].astype(str).tolist()).strip()
    if not text:
        return None

    wc = WordCloud(
        width=1200,
        height=520,
        background_color="white",
        max_words=140,
        collocations=False,
        regexp=r"\w[\w_]+",
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def top_tfidf_features(df, vectorizer, top_n=20):
    if df.empty or "final_text" not in df.columns:
        return pd.DataFrame(columns=["Fitur", "Skor TF-IDF"])

    texts = df["final_text"].astype(str).fillna("")
    X = vectorizer.transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(X.sum(axis=0)).ravel()

    if scores.size == 0:
        return pd.DataFrame(columns=["Fitur", "Skor TF-IDF"])

    top_idx = scores.argsort()[::-1][:top_n]
    return pd.DataFrame(
        {
            "Fitur": feature_names[top_idx],
            "Skor TF-IDF": scores[top_idx],
        }
    )


def top_ngram_features(df, n=2, top_n=15):
    if df.empty or "final_text" not in df.columns:
        return pd.DataFrame(columns=["Fitur", "Jumlah"])

    counter = Counter()
    for text in df["final_text"].astype(str):
        tokens = text.split()
        if len(tokens) < n:
            continue
        grams = zip(*[tokens[i:] for i in range(n)])
        counter.update([" ".join(gram) for gram in grams])

    return pd.DataFrame(counter.most_common(top_n), columns=["Fitur", "Jumlah"])


def model_feature_direction(model, vectorizer, top_n=15):
    if not hasattr(model, "feature_log_prob_") or len(getattr(model, "classes_", [])) < 2:
        return pd.DataFrame(columns=["Fitur", "Arah Sentimen", "Skor"])

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = list(model.classes_)

    try:
        neg_idx = classes.index(0)
        pos_idx = classes.index(1)
    except ValueError:
        return pd.DataFrame(columns=["Fitur", "Arah Sentimen", "Skor"])

    diff = model.feature_log_prob_[pos_idx] - model.feature_log_prob_[neg_idx]
    pos_features = pd.DataFrame(
        {
            "Fitur": feature_names[diff.argsort()[::-1][:top_n]],
            "Arah Sentimen": "Cenderung Positive",
            "Skor": diff[diff.argsort()[::-1][:top_n]],
        }
    )
    neg_features = pd.DataFrame(
        {
            "Fitur": feature_names[diff.argsort()[:top_n]],
            "Arah Sentimen": "Cenderung Negative",
            "Skor": diff[diff.argsort()[:top_n]],
        }
    )
    return pd.concat([pos_features, neg_features], ignore_index=True)


def evaluate_on_dataset(df, model, vectorizer):
    required = {"final_text", "label_encoded"}
    if df.empty or not required.issubset(df.columns):
        return None

    eval_df = df.copy()
    eval_df = eval_df[eval_df["final_text"].astype(str).str.strip() != ""]
    eval_df = eval_df.dropna(subset=["label_encoded"])

    if eval_df.empty:
        return None

    y_true = eval_df["label_encoded"].astype(int)
    X = vectorizer.transform(eval_df["final_text"].astype(str))
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pred_df = eval_df[["review", "final_text", "label", "label_encoded"]].copy() if "review" in eval_df.columns else eval_df[["final_text", "label", "label_encoded"]].copy()
    pred_df["prediksi_model"] = [LABEL_MAPPING.get(int(p), str(p)) for p in y_pred]
    pred_df["status_prediksi"] = np.where(pred_df["label"] == pred_df["prediksi_model"], "Sesuai", "Berbeda")

    return {
        "accuracy": acc,
        "report": pd.DataFrame(report).T.reset_index().rename(columns={"index": "Metrik"}),
        "cm": cm,
        "pred_df": pred_df,
    }


def predict_review(text, bundle):
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    processed = preprocess_text(text, bundle)
    X = vectorizer.transform([processed["final_text"]])
    prediction = model.predict(X)[0]
    label = LABEL_MAPPING.get(int(prediction), str(prediction))

    prob_df = pd.DataFrame()
    confidence = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        rows = []
        for idx, cls in enumerate(classes):
            rows.append(
                {
                    "Sentimen": LABEL_MAPPING.get(int(cls), str(cls)),
                    "Probabilitas": float(probs[idx]),
                }
            )
        prob_df = pd.DataFrame(rows)
        confidence = float(prob_df.loc[prob_df["Sentimen"] == label, "Probabilitas"].iloc[0])

    return label, confidence, prob_df, processed


def format_percent(value):
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


# =========================================================
# MAIN
# =========================================================
bundle = load_model_bundle()
model = bundle["model"]
vectorizer = bundle["vectorizer"]
config = bundle.get("config", {})

final_df, raw_df = load_data()
final_df = ensure_label_columns(final_df)
raw_df = ensure_label_columns(raw_df)
balanced_df = build_balanced_data(final_df, random_state=int(config.get("random_state", 42)))

st.markdown('<div class="main-title">Dashboard Analisis Sentimen Ulasan Apex Legends</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Implementasi model Multinomial Naive Bayes dengan fitur TF-IDF N-Gram untuk data ulasan Steam.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <span class="pill">TF-IDF N-Gram</span>
    <span class="pill">Multinomial Naive Bayes</span>
    <span class="pill">Apex Legends Steam Reviews</span>
    <span class="pill">Streamlit Dashboard</span>
    """,
    unsafe_allow_html=True,
)

if final_df.empty:
    st.error("Dataset final belum tersedia, sehingga dashboard tidak bisa menampilkan visualisasi utama.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Pengaturan Dashboard")

    data_choice = st.radio(
        "Data yang ditampilkan",
        ["Dataset Final Bersih", "Dataset Setelah Balancing", "Dataset Scraping Mentah"],
        index=0,
        help="Dataset Final Bersih dipakai untuk analisis utama. Dataset Balancing mengikuti jumlah kelas minoritas.",
    )

    selected_base = {
        "Dataset Final Bersih": final_df,
        "Dataset Setelah Balancing": balanced_df,
        "Dataset Scraping Mentah": raw_df if not raw_df.empty else final_df,
    }[data_choice]

    label_options = [label for label in ["Positive", "Negative"] if label in selected_base.get("label", pd.Series()).unique()]
    selected_labels = st.multiselect("Filter label", options=label_options, default=label_options)

    date_range = None
    date_series = selected_base["date_parsed"].dropna() if "date_parsed" in selected_base.columns else pd.Series(dtype="datetime64[ns]")
    if not date_series.empty:
        min_date = date_series.min().date()
        max_date = date_series.max().date()
        date_range = st.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            date_range = date_range
        else:
            date_range = None

    keyword = st.text_input(
        "Cari kata pada review/final_text",
        placeholder="contoh: server, cheater, not_good, bug",
    )

    st.divider()
    st.caption("File aktif")
    st.code("apex_sentiment_model_final_revisi.pkl")
    st.code("dataset_final_bersih_tanpa_kosong.csv")

    st.divider()
    st.caption("Konfigurasi model")
    st.write(f"N-Gram: `{config.get('ngram_range', getattr(vectorizer, 'ngram_range', '-'))}`")
    st.write(f"Max Features: `{config.get('max_features', getattr(vectorizer, 'max_features', '-'))}`")
    st.write(f"Alpha NB: `{config.get('alpha', getattr(model, 'alpha', '-'))}`")

active_df = apply_filters(selected_base, selected_labels, keyword, date_range)
eval_result = evaluate_on_dataset(active_df, model, vectorizer) if data_choice != "Dataset Scraping Mentah" else None

# KPI
total_scraping = len(raw_df) if not raw_df.empty else 0
total_final = len(final_df)
total_balanced = len(balanced_df)
positive_final = int((final_df["label"] == "Positive").sum()) if "label" in final_df.columns else 0
negative_final = int((final_df["label"] == "Negative").sum()) if "label" in final_df.columns else 0
active_count = len(active_df)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Data Scraping", f"{total_scraping:,}")
col2.metric("Data Final Bersih", f"{total_final:,}")
col3.metric("Positive Final", f"{positive_final:,}")
col4.metric("Negative Final", f"{negative_final:,}")
col5.metric("Data Balanced", f"{total_balanced:,}")
col6.metric("Data Aktif", f"{active_count:,}")

st.markdown(
    """
    """,
    unsafe_allow_html=True,
)

tab_ringkasan, tab_preprocessing, tab_fitur, tab_evaluasi, tab_wordcloud, tab_prediksi, tab_data = st.tabs(
    [
        "Ringkasan Data",
        "Tahapan Preprocessing",
        "Fitur TF-IDF & N-Gram",
        "Pemeriksaan Model",
        "WordCloud",
        "Prediksi Manual",
        "Tabel Dataset",
    ]
)

with tab_ringkasan:
    st.subheader("Ringkasan Distribusi Sentimen")

    if active_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        counts = label_counts(active_df)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_bar = px.bar(
                counts,
                x="Sentimen",
                y="Jumlah",
                text="Jumlah",
                title=f"Distribusi Sentimen - {data_choice}",
                color="Sentimen",
                color_discrete_map={"Positive": "#22C55E", "Negative": "#EF4444"},
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, yaxis_title="Jumlah Ulasan")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_pie = px.pie(
                counts,
                names="Sentimen",
                values="Jumlah",
                hole=0.42,
                title="Proporsi Sentimen",
                color="Sentimen",
                color_discrete_map={"Positive": "#22C55E", "Negative": "#EF4444"},
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Perbandingan Dataset Final dan Dataset Balanced")
        compare_final = label_counts(final_df)
        compare_final["Dataset"] = "Dataset Final Bersih"
        compare_balanced = label_counts(balanced_df)
        compare_balanced["Dataset"] = "Dataset Setelah Balancing"
        compare_df = pd.concat([compare_final, compare_balanced], ignore_index=True)

        fig_compare = px.bar(
            compare_df,
            x="Sentimen",
            y="Jumlah",
            color="Dataset",
            barmode="group",
            text="Jumlah",
            title="Perbandingan Distribusi Sentimen Sebelum dan Sesudah Balancing",
        )
        fig_compare.update_traces(textposition="outside")
        st.plotly_chart(fig_compare, use_container_width=True)

        if "date_parsed" in active_df.columns and active_df["date_parsed"].notna().any():
            st.subheader("Tren Jumlah Ulasan Berdasarkan Tanggal")
            trend_df = active_df.dropna(subset=["date_parsed"]).copy()
            trend_df["tanggal"] = trend_df["date_parsed"].dt.date
            trend_df = trend_df.groupby(["tanggal", "label"]).size().reset_index(name="Jumlah")

            fig_trend = px.line(
                trend_df,
                x="tanggal",
                y="Jumlah",
                color="label",
                markers=True,
                title="Tren Ulasan Harian Berdasarkan Sentimen",
                color_discrete_map={"Positive": "#22C55E", "Negative": "#EF4444"},
            )
            fig_trend.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Ulasan")
            st.plotly_chart(fig_trend, use_container_width=True)

with tab_preprocessing:
    st.subheader("Contoh Tahapan Preprocessing pada Dataset Final")
    st.caption("Bagian ini menampilkan perubahan teks dari review asli sampai final_text.")

    if active_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        sample_cols = [
            col
            for col in [
                "review",
                "case_folding",
                "cleansing",
                "normalization",
                "tokenizing",
                "filtering",
                "stemming",
                "final_text",
                "label",
            ]
            if col in active_df.columns
        ]

        st.dataframe(active_df[sample_cols].head(12), use_container_width=True, height=420)

        st.markdown("#### Jumlah data pada setiap tahap")
        stage_cols = [col for col in ["review", "case_folding", "cleansing", "normalization", "tokenizing", "filtering", "stemming", "final_text"] if col in final_df.columns]
        stage_rows = []
        for col in stage_cols:
            non_empty = int((final_df[col].astype(str).str.strip() != "").sum())
            stage_rows.append({"Tahap": col, "Data Tidak Kosong": non_empty, "Total Data": len(final_df)})
        stage_df = pd.DataFrame(stage_rows)

        fig_stage = px.bar(
            stage_df,
            x="Tahap",
            y="Data Tidak Kosong",
            text="Data Tidak Kosong",
            title="Ketersediaan Data pada Setiap Tahap Preprocessing",
        )
        fig_stage.update_traces(textposition="outside")
        fig_stage.update_layout(yaxis_title="Jumlah Data")
        st.plotly_chart(fig_stage, use_container_width=True)

with tab_fitur:
    st.subheader("Top Fitur TF-IDF dan N-Gram")

    if active_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        top_n = st.slider("Jumlah fitur yang ditampilkan", min_value=10, max_value=50, value=20, step=5)

        tfidf_df = top_tfidf_features(active_df, vectorizer, top_n=top_n)
        col_a, col_b = st.columns([1.1, 0.9])

        with col_a:
            fig_tfidf = px.bar(
                tfidf_df.sort_values("Skor TF-IDF"),
                x="Skor TF-IDF",
                y="Fitur",
                orientation="h",
                text="Skor TF-IDF",
                title=f"Top {top_n} Fitur Berdasarkan Bobot TF-IDF",
            )
            fig_tfidf.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_tfidf, use_container_width=True)

        with col_b:
            st.dataframe(tfidf_df, use_container_width=True, height=520)

        st.markdown("#### Top Bigram pada Data Aktif")
        bigram_df = top_ngram_features(active_df, n=2, top_n=20)
        fig_bigram = px.bar(
            bigram_df.sort_values("Jumlah"),
            x="Jumlah",
            y="Fitur",
            orientation="h",
            text="Jumlah",
            title="Top 20 Bigram Berdasarkan Frekuensi",
        )
        fig_bigram.update_traces(textposition="outside")
        st.plotly_chart(fig_bigram, use_container_width=True)

        st.markdown("#### Fitur yang Cenderung Mengarah ke Sentimen Positive dan Negative")
        direction_df = model_feature_direction(model, vectorizer, top_n=15)
        if direction_df.empty:
            st.info("Arah fitur tidak dapat dihitung dari model yang dimuat.")
        else:
            fig_direction = px.bar(
                direction_df.sort_values("Skor"),
                x="Skor",
                y="Fitur",
                color="Arah Sentimen",
                orientation="h",
                title="Indikasi Arah Fitur Berdasarkan Log Probability Naive Bayes",
                color_discrete_map={"Cenderung Positive": "#22C55E", "Cenderung Negative": "#EF4444"},
            )
            st.plotly_chart(fig_direction, use_container_width=True)
            st.dataframe(direction_df, use_container_width=True)

with tab_evaluasi:
    st.subheader("Evaluasi Model pada Data Uji Penelitian")
    st.info(
        "Hasil ini merupakan evaluasi utama model berdasarkan data uji "
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Akurasi Penelitian", f"{RESEARCH_ACCURACY:.4f}")
    metric_cols[1].metric("Total Data Uji", "1,103")
    metric_cols[2].metric("Data Uji Negative", "552")
    metric_cols[3].metric("Data Uji Positive", "551")

    col_r1, col_r2 = st.columns([0.9, 1.1])
    with col_r1:
        research_cm_df = pd.DataFrame(
            RESEARCH_CM,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        fig_research_cm = px.imshow(
            research_cm_df,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix Data Uji Penelitian",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_research_cm, use_container_width=True)

    with col_r2:
        st.write("Classification Report Data Uji Penelitian")
        research_report_df = RESEARCH_REPORT.copy()
        numeric_cols = research_report_df.select_dtypes(include=[np.number]).columns
        research_report_df[numeric_cols] = research_report_df[numeric_cols].round(6)
        st.dataframe(research_report_df, use_container_width=True, height=330)

    st.divider()
    st.subheader("Pemeriksaan Tambahan terhadap Dataset Aktif")
    st.warning(
        "Pemeriksaan ini hanya digunakan untuk mengecek prediksi model pada dataset aktif dashboard, bukan sebagai evaluasi utama penelitian. "
    )

    if data_choice == "Dataset Scraping Mentah":
        st.warning("Dataset scraping mentah belum memiliki final_text, sehingga pemeriksaan model tidak dijalankan pada tab ini.")
    elif eval_result is None:
        st.warning("Pemeriksaan model tidak dapat dilakukan karena data aktif kosong atau kolom yang dibutuhkan belum lengkap.")
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Akurasi Cek Data Aktif", format_percent(eval_result["accuracy"]))
        metric_cols[1].metric("Data Sesuai", f"{int((eval_result['pred_df']['status_prediksi'] == 'Sesuai').sum()):,}")
        metric_cols[2].metric("Data Berbeda", f"{int((eval_result['pred_df']['status_prediksi'] == 'Berbeda').sum()):,}")
        metric_cols[3].metric("Total Dicek", f"{len(eval_result['pred_df']):,}")

        col_a, col_b = st.columns([0.9, 1.1])
        with col_a:
            cm = eval_result["cm"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"],
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix Cek Data Aktif",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_b:
            report_df = eval_result["report"].copy()
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            report_df[numeric_cols] = report_df[numeric_cols].round(4)
            st.write("Classification Report")
            st.dataframe(report_df, use_container_width=True, height=330)

        st.markdown("#### Contoh Data dengan Label Asli dan Prediksi Model")
        status_filter = st.radio("Tampilkan status", ["Semua", "Sesuai", "Berbeda"], horizontal=True)
        pred_view = eval_result["pred_df"].copy()
        if status_filter != "Semua":
            pred_view = pred_view[pred_view["status_prediksi"] == status_filter]
        st.dataframe(pred_view.head(200), use_container_width=True, height=420)

with tab_wordcloud:
    st.subheader("WordCloud Ulasan")
    st.caption("WordCloud mengikuti pilihan data, filter label, tanggal, dan pencarian kata pada sidebar.")

    if active_df.empty:
        st.warning("Tidak ada data yang sesuai dengan filter.")
    else:
        wc_fig = make_wordcloud(active_df)
        if wc_fig is None:
            st.warning("WordCloud tidak dapat dibuat karena final_text kosong.")
        else:
            st.pyplot(wc_fig, use_container_width=True)

        col_neg, col_pos = st.columns(2)
        with col_neg:
            st.markdown("#### Negative")
            neg_fig = make_wordcloud(active_df[active_df["label"] == "Negative"]) if "label" in active_df.columns else None
            if neg_fig is None:
                st.info("Tidak ada data Negative pada filter aktif.")
            else:
                st.pyplot(neg_fig, use_container_width=True)

        with col_pos:
            st.markdown("#### Positive")
            pos_fig = make_wordcloud(active_df[active_df["label"] == "Positive"]) if "label" in active_df.columns else None
            if pos_fig is None:
                st.info("Tidak ada data Positive pada filter aktif.")
            else:
                st.pyplot(pos_fig, use_container_width=True)

with tab_prediksi:
    st.subheader("Prediksi Sentimen Ulasan Baru")
    st.caption("Masukkan review berbahasa Inggris. Dashboard akan melakukan preprocessing, TF-IDF transform, lalu prediksi dengan model Naive Bayes.")

    default_review = ""
    user_review = st.text_area("Masukkan review", value=default_review, height=150)

    if st.button("Analisis Sentimen", type="primary"):
        if not user_review.strip():
            st.warning("Masukkan teks review terlebih dahulu.")
        else:
            label, confidence, prob_df, processed = predict_review(user_review, bundle)

            if label == "Positive":
                st.success(f"Hasil Prediksi: {label}" + (f" | Confidence: {confidence * 100:.2f}%" if confidence is not None else ""))
            else:
                st.error(f"Hasil Prediksi: {label}" + (f" | Confidence: {confidence * 100:.2f}%" if confidence is not None else ""))

            if not prob_df.empty:
                fig_prob = px.bar(
                    prob_df,
                    x="Sentimen",
                    y="Probabilitas",
                    text="Probabilitas",
                    title="Probabilitas Prediksi Model",
                    color="Sentimen",
                    color_discrete_map={"Positive": "#22C55E", "Negative": "#EF4444"},
                )
                fig_prob.update_traces(texttemplate="%{text:.2%}", textposition="outside")
                fig_prob.update_yaxes(range=[0, 1])
                st.plotly_chart(fig_prob, use_container_width=True)

            with st.expander("Lihat hasil preprocessing"):
                st.write("Case Folding")
                st.code(processed["case_folding"])

                st.write("Cleansing")
                st.code(processed["cleansing"])

                st.write("Normalization")
                st.code(processed["normalization"])

                st.write("Tokenizing")
                st.write(processed["tokenizing"])

                st.write("Filtering")
                st.write(processed["filtering"])

                st.write("Stemming")
                st.write(processed["stemming"])

                st.write("Final Text")
                st.code(processed["final_text"])

with tab_data:
    st.subheader("Tabel Dataset")
    st.caption(f"Menampilkan {len(active_df):,} baris berdasarkan filter sidebar.")

    show_cols = [
        col
        for col in [
            "review_id",
            "review",
            "case_folding",
            "cleansing",
            "normalization",
            "tokenizing",
            "filtering",
            "stemming",
            "final_text",
            "label",
            "label_encoded",
            "voted_up",
            "timestamp_created",
            "date",
        ]
        if col in active_df.columns
    ]

    st.dataframe(active_df[show_cols], use_container_width=True, height=560)

    csv_data = active_df[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download data hasil filter sebagai CSV",
        data=csv_data,
        file_name="hasil_filter_dashboard_apex.csv",
        mime="text/csv",
    )
