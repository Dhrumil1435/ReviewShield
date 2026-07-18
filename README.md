# ReviewShield

## 📝 Project Description

**ReviewShield** is an advanced, industry-grade data science framework engineered to detect fraudulent, deceptive, or AI-generated reviews on e-commerce platforms. 

Traditional online review moderation systems rely heavily on basic text sentiment analysis. However, standard sentiment classifiers frequently fail to catch sophisticated modern fakes because fraudulent reviews can easily be written to mimic a perfect positive or negative tone. **ReviewShield solves this critical vulnerability by shifting the core analytical focus from *what* is written to *how* it is written.**

### 🔍 Core Methodology

The system operates on a hybrid detection engine combining two core pillars of data science:

1. **Stylometric Profiling (Text Psychology):** 
   By leveraging Natural Language Processing (NLP), the pipeline breaks down the textual DNA of a review. It extracts subconscious writing patterns that naturally vary between authentic customer experiences and fabricated content. The model calculates and evaluates:
   * **Structural Metrics:** Average sentence length, word count distributions, and vocabulary diversity (Lexical Richness).
   * **Syntactic Patters:** Punctuation frequency (e.g., over-use of exclamation marks) and Part-of-Speech (POS) density distributions (ratio of verbs and adjectives to nouns).
   * **Readability Indices:** Automated scoring metrics (such as the Flesch-Kincaid Grade Level) to evaluate text complexity and structural naturalness.

2. **Semantic Sentiment Mismatch:**
   The framework computes the exact computational sentiment score of the review text using specialized lexicons and cross-references it against the user's numerical star rating. A stark mathematical deviation between the expressed text emotion and the assigned rating numerical score flags an immediate anomaly.

---

### 🚀 Key Features & Architectural Impact

* **Data-Centric Pipeline:** Implements a robust relational database layer using **PostgreSQL/SQLite** to securely store raw ingestion data, track multi-dimensional engineered features, and maintain historical inference logs.
* **Traditional Machine Learning Excellence:** Utilizes high-performance classifiers (such as Random Forest and Support Vector Machines) optimized to handle text feature vectors efficiently without requiring massive, expensive GPU deep learning architectures.
* **Explainable Data Dashboard:** Features a live, production-ready interactive interface built with **Streamlit**. Users can paste individual reviews to receive an instant "Deceptive Probability Score" paired with a clear, visual breakdown of the underlying stylometric anomalies.

> **Project Impact:** By delivering a lightweight, computationally efficient, and highly explainable detection mechanism, ReviewShield provides digital marketplaces with an immediate defense line to preserve platform integrity, combat review manipulation, and protect consumer trust.
