"""
app.py
Streamlit Web Application for ReviewShield.
Interactive Deceptive Review Detection System with real-time inference,
stylometric breakdown, risk indicator alerts, and model performance analytics.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pandas as pd
import streamlit as st
from PIL import Image
from sqlalchemy import text
from src.config import MODELS_DIR, get_db_engine
from src.inference import ReviewAnalyzer

# Page Configuration
st.set_page_config(
    page_title="ReviewShield - Deceptive Review Detection",
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
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_analyzer():
    return ReviewAnalyzer()


def main():
    st.markdown('<div class="main-header">🛡️ ReviewShield Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Deceptive & Computer-Generated Review Detection Engine</div>',
        unsafe_allow_html=True,
    )

    # Initialize Engine
    try:
        analyzer = get_analyzer()
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        st.info("Please make sure `src/train_model.py` has been executed to generate the model artifacts.")
        return

    # Navigation Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🔍 Single Review Scanner", "📊 Model Insights & Benchmarks", "🗄️ Database Analytics"],
    )

    # PAGE 1: Single Review Scanner
    if page == "🔍 Single Review Scanner":
        st.subheader("Analyze Review Authenticity")
        st.write("Input a product review and assigned star rating to calculate deceptive probability.")

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

                # Risk Factor Alerts
                if result["risk_factors"]:
                    st.markdown("#### 🚨 Risk Indicator Flags")
                    for flag in result["risk_factors"]:
                        st.warning(flag)
                else:
                    st.success("No abnormal stylometric risk flags detected.")

        st.divider()

        # Detailed Stylometric Feature Breakdown
        if analyze_btn or review_text:
            st.subheader("Subconscious Stylometric DNA Breakdown")
            f = result["features"]

            fc1, fc2, fc3, fc4, fc5 = st.columns(5)
            fc1.metric("Rating-Sentiment Gap", f"{f['rating_sentiment_gap']:.2f}")
            fc2.metric("Sentiment Score", f"{f['sentiment_score']:.2f}")
            fc3.metric("Punctuation / 100 chars", f"{f['punctuation_freq']:.2f}")
            fc4.metric("Vocab Diversity (TTR)", f"{f['vocab_diversity']:.2f}")
            fc5.metric("Readability Score", f"{f['readability_score']:.1f}")

    # PAGE 2: Model Insights & Benchmarks
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
            st.subheader("Random Forest Feature Importances")
            fig_path = MODELS_DIR / "feature_importance.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
        else:
            st.warning("Model metrics report not found. Run `src/train_model.py` to generate benchmarking charts.")

    # PAGE 3: Database Analytics
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
            st.error(f"Could not connect to PostgreSQL database: {e}")


if __name__ == "__main__":
    main()
