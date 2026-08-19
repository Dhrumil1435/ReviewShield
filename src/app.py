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
import pandas as pd
import streamlit as st
from sqlalchemy import text
from src.config import MODELS_DIR, get_db_engine
from src.inference import ReviewAnalyzer

# Page Configuration
st.set_page_config(
    page_title="ReviewShield - Deceptive Review Detection (XAI + RoBERTa)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Mode / Glassmorphism Aesthetic)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }
    .badge-deceptive {
        background-color: #EF4444;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
    }
    .badge-genuine {
        background-color: #10B981;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
    }
    .xai-box {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer():
    return ReviewAnalyzer()


def main():
    st.markdown('<div class="main-header">🛡️ ReviewShield AI Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deceptive Review Detection with Explainable AI (XAI) & VADER + RoBERTa Dual-Sentiment Analysis</div>',
        unsafe_allow_html=True,
    )

    try:
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.info("Please make sure `src/train_model.py` has been executed to generate the model artifacts.")
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🔍 Single Review Scanner",
            "📂 Batch CSV Scanner",
            "📊 Model Insights & Benchmarks",
            "🗄️ Database Analytics",
        ],
    )

    # PAGE 1: Single Review Scanner
    if page == "🔍 Single Review Scanner":
        st.subheader("Analyze Review Authenticity & Word-Level Attributions")
        st.write("Input a product review and star rating to calculate deceptive probability using hybrid TF-IDF + VADER + RoBERTa scores.")

        col1, col2 = st.columns([3, 2])

        with col1:
            default_review = (
                "ABSOLUTELY PERFECT ITEM!!!!!! Best quality I have ever seen on Amazon! "
                "Super fast 1-day delivery, 100% recommended to everyone!!!"
            )
            review_text = st.text_area(
                "Review Text",
                value=default_review,
                height=160,
                placeholder="Enter review text here...",
            )
            rating = st.slider("Star Rating Given", min_value=1.0, max_value=5.0, value=5.0, step=0.5)
            analyze_btn = st.button("Run Deception Scanner 🚀", type="primary", use_container_width=True)

        with col2:
            if analyze_btn or review_text:
                result = analyzer.analyze(review_text, rating)

                st.markdown("### Detection Result")
                prob = result["deceptive_percentage"]

                if result["is_deceptive"]:
                    st.markdown(
                        f'<span class="badge-deceptive">⚠️ DECEPTIVE / COMPUTER-GENERATED ({prob}%)</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<span class="badge-genuine">✅ ORIGINAL / AUTHENTIC ({(100 - prob):.1f}% Genuine)</span>',
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.progress(result["deceptive_probability"])
                st.caption(f"Deceptive Probability Score: **{prob:.2f}%**")

                if result["risk_factors"]:
                    st.markdown("#### 🚨 Risk Indicator Flags")
                    for flag in result["risk_factors"]:
                        st.warning(flag)
                else:
                    st.success("No abnormal stylometric or sentiment dissonance risk flags detected.")

        st.divider()

        # EXPLAINABLE AI (XAI) WORD ATTRIBUTION SECTION
        if analyze_btn or review_text:
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
        if analyze_btn or review_text:
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
                    max_rows = st.number_input("Max Rows to Process", min_value=5, max_value=min(len(raw_df), 10000), value=min(len(raw_df), 200), step=50)

                run_batch_btn = st.button("Run Batch Deception Scanner 🚀", type="primary", use_container_width=True)

                if run_batch_btn:
                    process_df = raw_df.head(max_rows).copy()
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(pct):
                        progress_bar.progress(pct)
                        status_text.text(f"Processing batch reviews... {int(pct * 100)}% complete")

                    results_df = analyzer.analyze_dataframe(
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

                    st.dataframe(display_df, use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Processed CSV Results",
                        data=csv_data,
                        file_name="reviewshield_batch_analysis.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

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

            st.dataframe(models_df.style.highlight_max(axis=0, color="#10B981"), use_container_width=True)

            st.divider()
            st.subheader("Hybrid Feature Importances & Word Coefficients")
            fig_path = MODELS_DIR / "feature_importance.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
        else:
            st.warning("Model metrics report not found. Run `src/train_model.py` to generate benchmarking charts.")

    # PAGE 4: Database Analytics
    elif page == "🗄️ Database Analytics":
        st.subheader("PostgreSQL Data Warehouse Overview")

        try:
            engine = get_db_engine()
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
            st.dataframe(logs_df, use_container_width=True)

        except Exception as e:
            st.error(f"Could not connect to PostgreSQL database: (psycopg2.OperationalError) {e}")


if __name__ == "__main__":
    main()
