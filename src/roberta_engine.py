"""
roberta_engine.py
Deep Learning Contextual Sentiment Engine using RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest).
Computes contextual sentiment probabilities and compound scores (-1.0 to +1.0).
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class RobertaSentimentEngine:
    _instance = None

    def __new__(cls, model_name=MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(RobertaSentimentEngine, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.tokenizer = None
            cls._instance.model = None
            cls._instance.initialized = False
        return cls._instance

    def _load_model(self):
        if not self.initialized:
            print(f"Loading RoBERTa model '{self.model_name}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.initialized = True
            print("RoBERTa model loaded successfully.")

    def analyze_text(self, text: str) -> dict:
        self._load_model()
        
        text_str = str(text).strip()
        if not text_str:
            return {
                "roberta_compound": 0.0,
                "roberta_neg": 0.0,
                "roberta_neu": 1.0,
                "roberta_pos": 0.0,
            }

        encoded_input = self.tokenizer(
            text_str,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            output = self.model(**encoded_input)
            scores = output[0][0].numpy()
            probabilities = softmax(scores)

        prob_neg = float(probabilities[0])
        prob_neu = float(probabilities[1])
        prob_pos = float(probabilities[2])

        compound_score = round(prob_pos - prob_neg, 4)

        return {
            "roberta_compound": compound_score,
            "roberta_neg": round(prob_neg, 4),
            "roberta_neu": round(prob_neu, 4),
            "roberta_pos": round(prob_pos, 4),
        }


def get_roberta_sentiment(text: str) -> float:
    engine = RobertaSentimentEngine()
    res = engine.analyze_text(text)
    return res["roberta_compound"]


if __name__ == "__main__":
    engine = RobertaSentimentEngine()
    sample = "This product completely blew my mind! Outstanding quality and lightning fast shipping."
    print("Sample Output:", engine.analyze_text(sample))
