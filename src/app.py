"""
app.py
Streamlit Web Application for ReviewShield.
Interactive Deceptive Review Detection System with real-time inference,
VADER + RoBERTa dual-sentiment comparison, Explainable AI (XAI) Word Attribution,
Batch CSV dataset scanner, and model analytics.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import inspect, text
from src.config import (
    DB_BACKEND,
    DB_HOST,
    DB_NAME,
    DB_PATH,
    DB_PORT,
    MODELS_DIR,
    get_db_engine,
)

# Page Configuration
st.set_page_config(
    page_title="ReviewShield - Deceptive Review Detection (XAI + RoBERTa)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dashboard styling
st.markdown("""
    <style>
    .stApp {
        background: #F3F5F1;
        color: #1C302B;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stSidebar"] {
        background: #173D35;
        border-right: 1px solid #28564C;
    }
    [data-testid="stSidebar"] * {
        color: #EDF4EF;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        color: #FFFFFF;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #173D35;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #587067;
        margin-bottom: 2rem;
    }
    .badge-deceptive {
        background-color: #B84132;
        color: white;
        padding: 7px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1rem;
    }
    .badge-genuine {
        background-color: #24745A;
        color: white;
        padding: 7px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1rem;
    }
    .xai-box {
        background-color: #FFFFFF;
        border: 1px solid #D7E0D9;
        border-radius: 6px;
        padding: 18px;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #263B34;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE4DD;
        border-radius: 6px;
        padding: 12px 14px;
    }
    .stButton button[kind="primary"] {
        background: #B84132;
        border: 1px solid #B84132;
        color: #FFFFFF;
    }
    .stButton button[kind="primary"]:hover {
        background: #963629;
        border-color: #963629;
        color: #FFFFFF;
    }
    @media (max-width: 768px) {
        .block-container {
            padding: 1.25rem 1rem 2rem;
        }
        .main-header {
            font-size: 1.85rem;
        }
        .sub-header {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer():
    from src.inference import ReviewAnalyzer

    return ReviewAnalyzer()


@st.cache_resource
def get_database_engine():
    return get_db_engine()


@st.fragment(run_every=10)
def render_database_analytics():
    st.subheader("ReviewShield Database Overview")
    st.caption(f"Live snapshot · refreshed at {datetime.now().strftime('%H:%M:%S')} · updates every 10 seconds")

    try:
        engine = get_database_engine()
        required_tables = {"raw_reviews", "engineered_features", "system_logs"}
        if not required_tables.issubset(inspect(engine).get_table_names()):
            from src.db_setup import init_db

            init_db()

        with engine.connect() as conn:
            raw_count = conn.execute(text("SELECT COUNT(*) FROM raw_reviews")).scalar()
            feature_count = conn.execute(text("SELECT COUNT(*) FROM engineered_features")).scalar()
            log_count = conn.execute(text("SELECT COUNT(*) FROM system_logs")).scalar()

        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Ingested Raw Reviews", f"{raw_count:,}")
        dc2.metric("Engineered Feature Vectors", f"{feature_count:,}")
        dc3.metric("System Log Events", f"{log_count:,}")

        st.divider()
        st.subheader("System Execution Log History")
        logs_df = pd.read_sql(
            "SELECT log_id, stage, message, logged_at FROM system_logs ORDER BY logged_at DESC LIMIT 10",
            engine,
        )
        st.dataframe(logs_df, width="stretch")

    except Exception as e:
        database_target = (
            str(DB_PATH)
            if DB_BACKEND == "sqlite"
            else f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        st.error(
            f"Database analytics could not open {DB_BACKEND} database "
            f"{database_target}."
        )
        if DB_BACKEND == "postgresql":
            st.markdown("#### Restore the database connection")
            st.markdown(
                "1. Start your PostgreSQL service or Docker container.\n"
                "2. Set `DB_BACKEND=postgresql` and your credentials in the project `.env`.\n"
                "3. Create the database if needed: `createdb -U postgres reviewshield`.\n"
                "4. Run `python -m src.db_setup`, then refresh this page."
            )
        else:
            st.markdown(
                f"The local SQLite database is created automatically. Check that the project "
                f"data folder is writable and that `{DB_PATH}` is available."
            )
        with st.expander("Connection details"):
            st.code(str(e))


def main():
    st.markdown('<div class="main-header">🛡️ ReviewShield AI Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deceptive Review Detection with Explainable AI (XAI) & VADER + RoBERTa Dual-Sentiment Analysis</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.title("ReviewShield")
    page = st.sidebar.radio(
        "Workspace",
        [
            "🔍 Single Review Scanner",
            "📂 Batch CSV Scanner",
            "📊 Model Insights & Benchmarks",
            "🗄️ Database Analytics",
        ],
        key="reviewshield_navigation",
    )

    # PAGE 1: Single Review Scanner
    if page == "🔍 Single Review Scanner":
        st.subheader("Interactive Single Review Deception Scanner")
        st.write("Enter any product or hotel review below along with its star rating to analyze authenticity.")

        sample_reviews = {
            "Custom Input": "",
            "Deceptive Review Sample (Suspiciously Positive)": "This is hands down the most incredible product ever created! Absolutely flawless perfection in every single detail, changed my life forever within seconds. Best purchase of my entire lifetime, everyone MUST buy this right now!",
            "Genuine Review Sample (Balanced Feedback)": "The build quality is good and the screen is clear, but battery life is average. Takes about 2 hours to fully charge. Overall decent value for the price.",
        }

        sample_choice = st.selectbox("Select Sample Input (Optional):", list(sample_reviews.keys()))
        default_text = sample_reviews[sample_choice]

        with st.form("single_review_form"):
            review_text = st.text_area(
                "Review Text Content",
                value=default_text,
                height=140,
                placeholder="Paste review text here...",
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                rating = st.slider("Star Rating Given", min_value=1.0, max_value=5.0, value=5.0, step=0.5)

            submit_btn = st.form_submit_button("Analyze Review Authenticity 🚀", type="primary")

        analyzer = get_analyzer()
        has_result = False
        result = None

        if submit_btn and review_text.strip():
            with st.spinner("Analyzing text with RoBERTa & VADER feature pipelines..."):
                result = analyzer.analyze(review_text, rating)
                has_result = True
        elif default_text and not submit_btn:
            result = analyzer.analyze(default_text, rating)
            has_result = True

        if has_result:
            st.divider()
            res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

            with res_col1:
                if result["is_deceptive"]:
                    st.markdown('<span class="badge-deceptive">⚠️ FLAG: DECEPTIVE REVIEW DETECTED</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-genuine">✅ VERIFIED: GENUINE REVIEW</span>', unsafe_allow_html=True)

                st.write("")
                st.markdown(f"**Classification Summary:** `{result['classification']}`")

            with res_col2:
                st.metric("Deceptive Probability", f"{result['deceptive_percentage']:.1f}%")

            with res_col3:
                st.metric("Authenticity Confidence", f"{100.0 - result['deceptive_percentage']:.1f}%")

            if result["risk_factors"]:
                st.write("")
                st.warning("⚠️ **Detected Risk Signals:**\n" + "\n".join([f"- {rf}" for rf in result["risk_factors"]]))

        st.divider()

        # EXPLAINABLE AI (XAI) WORD ATTRIBUTION SECTION
        if has_result:
            st.subheader("🧠 Explainable AI (XAI) — Word-Level Attribution Scanner")
            st.write("Visual breakdown showing exact word contributions to the deception score based on model weights.")

            # Legend Bar
            st.markdown(
                """
                <div style='margin-bottom: 12px; font-size: 0.95rem;'>
                    <b>Legend:</b> 
                    <span style='background-color:#7F1D1D; color:#FCA5A5; padding:2px 8px; border-radius:4px; margin-right:8px;'>🔴 Suspicious Deceptive Keyword</span>
                    <span style='background-color:#064E3B; color:#6EE7B7; padding:2px 8px; border-radius:4px; margin-right:8px;'>🟢 Authentic Signal Keyword</span>
                    <span style='color:#9CA3AF;'>⚪ Neutral Word</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Highlight Box
            explained_html_str = result.get("explained_html", analyzer.explain_review_words(review_text) if hasattr(analyzer, "explain_review_words") else review_text)
            st.markdown(
                f'<div class="xai-box">{explained_html_str}</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Dual Sentiment & Stylometric Breakdown
        if has_result:
            st.subheader("🤖 Dual-Engine Sentiment & Stylometric Breakdown")
            f = result["features"]

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("VADER Sentiment Score", f"{f['sentiment_score']:.2f}")
            sc2.metric("RoBERTa Contextual Score", f"{f['roberta_sentiment']:.2f}")
            sc3.metric("VADER vs RoBERTa Dissonance", f"{f['vader_roberta_dissonance']:.2f}", delta_color="inverse")

            st.write("")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("RoBERTa Rating Gap", f"{f['roberta_rating_sentiment_gap']:.2f}")
            fc2.metric("Punctuation / 100 chars", f"{f['punctuation_freq']:.2f}")
            fc3.metric("Vocab Diversity (TTR)", f"{f['vocab_diversity']:.2f}")
            fc4.metric("Readability Score", f"{f['readability_score']:.1f}")

    # PAGE 2: Batch CSV Scanner
    elif page == "📂 Batch CSV Scanner":
        st.subheader("Bulk Dataset Deception Scanner")
        st.write("Upload a CSV file containing review text and star ratings to scan hundreds of reviews in bulk.")

        uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                if raw_df.empty:
                    st.warning("The uploaded CSV has no review rows to scan.")
                    st.stop()

                st.success(f"Successfully loaded CSV with **{len(raw_df):,}** rows and columns: `{list(raw_df.columns)}`")

                col_map1, col_map2, col_map3 = st.columns(3)
                with col_map1:
                    text_default_idx = 0
                    for idx, c in enumerate(raw_df.columns):
                        if c.lower() in ["review_text", "text", "text_", "review", "content"]:
                            text_default_idx = idx
                            break
                    text_col = st.selectbox("Select Review Text Column", list(raw_df.columns), index=text_default_idx)

                with col_map2:
                    rating_default_idx = 0
                    for idx, c in enumerate(raw_df.columns):
                        if c.lower() in ["rating", "stars", "score", "star_rating"]:
                            rating_default_idx = idx
                            break
                    rating_col = st.selectbox("Select Star Rating Column", list(raw_df.columns), index=rating_default_idx)

                with col_map3:
                    max_rows = st.number_input("Max Rows to Process", min_value=1, max_value=min(len(raw_df), 10000), value=min(len(raw_df), 200), step=50)

                run_batch_btn = st.button("Run Batch Deception Scanner 🚀", type="primary", width="stretch")

                if run_batch_btn:
                    process_df = raw_df.head(max_rows).copy()
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"Processing batch reviews... {int(pct * 100)}% complete")

                    results_df = get_analyzer().analyze_dataframe(
                        process_df,
                        text_col=text_col,
                        rating_col=rating_col,
                        progress_callback=update_progress,
                    )

                    status_text.success("Batch analysis complete! 🎉")
                    st.divider()

                    total_scanned = len(results_df)
                    deceptive_count = (results_df["Classification"] == "DECEPTIVE").sum()
                    genuine_count = total_scanned - deceptive_count
                    deceptive_pct = (deceptive_count / total_scanned) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Scanned Reviews", f"{total_scanned:,}")
                    m2.metric("Deceptive Reviews Flagged", f"{deceptive_count:,}", delta=f"{deceptive_pct:.1f}%", delta_color="inverse")
                    m3.metric("Genuine Reviews", f"{genuine_count:,}")
                    m4.metric("Deception Ratio", f"{deceptive_pct:.1f}%")

                    st.divider()

                    st.subheader("Detailed Scan Results Table")
                    filter_option = st.radio("Filter Table View", ["All Reviews", "Deceptive Only ⚠️", "Genuine Only ✅"], horizontal=True)

                    if filter_option == "Deceptive Only ⚠️":
                        display_df = results_df[results_df["Classification"] == "DECEPTIVE"]
                    elif filter_option == "Genuine Only ✅":
                        display_df = results_df[results_df["Classification"] == "GENUINE"]
                    else:
                        display_df = results_df

                    st.dataframe(display_df, width="stretch")

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Processed CSV Results",
                        data=csv_data,
                        file_name="reviewshield_batch_analysis.csv",
                        mime="text/csv",
                        type="primary",
                        width="stretch",
                    )

            except Exception as e:
                st.error(f"Unable to process or analyze the uploaded CSV: {e}")

    # PAGE 3: Model Insights & Benchmarks
    elif page == "📊 Model Insights & Benchmarks":
        st.subheader("Model Evaluation & Benchmarking Matrix")

        metrics_file = MODELS_DIR / "metrics_report.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)

            models_df = pd.DataFrame(metrics_data).T
            models_df = models_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
            models_df.columns = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

            st.dataframe(models_df.style.highlight_max(axis=0, color="#10B981"), width="stretch")

            st.divider()
            st.subheader("Hybrid Feature Importances & Word Coefficients")
            fig_path = MODELS_DIR / "feature_importance.png"
            if fig_path.exists():
                st.image(str(fig_path), width="stretch")
        else:
            st.warning("Model metrics report not found. Run `src/train_model.py` to generate benchmarking charts.")

    # PAGE 4: Database Analytics
    elif page == "🗄️ Database Analytics":
        render_database_analytics()


if __name__ == "__main__":
    main()
