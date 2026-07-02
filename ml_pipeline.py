import re
import string
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

VECTORIZER_PATH = "models/vectorizer.joblib"
MODEL_PATH = "models/classifier.joblib"

SUSPICIOUS_KEYWORDS = [
    "verify your account", "urgent", "suspended", "click here",
    "confirm your password", "limited time", "act now", "winner",
    "bank account", "social security", "update your billing",
    "unusual activity", "security alert", "login attempt",
]

URGENT_WORDS = ["urgent", "immediately", "act now", "expire", "suspended", "warning"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text):
    lowered = text.lower()
    url_count = len(re.findall(r"http\S+|www\.\S+", lowered))
    suspicious_word_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lowered)
    has_urgent_language = any(word in lowered for word in URGENT_WORDS)
    exclamation_count = text.count("!")
    uppercase_ratio = (
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    )

    return {
        "url_count": url_count,
        "suspicious_word_count": suspicious_word_count,
        "has_urgent_language": has_urgent_language,
        "exclamation_count": exclamation_count,
        "uppercase_ratio": round(uppercase_ratio, 3),
    }


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["email_text", "label"])
    df["clean_text"] = df["email_text"].apply(clean_text)
    return df


def train_model(csv_path, model_type="random_forest"):
    df = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)

    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="phishing"),
        "recall": recall_score(y_test, y_pred, pos_label="phishing"),
        "f1_score": f1_score(y_test, y_pred, pos_label="phishing"),
    }

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)

    return metrics


def load_model():
    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)
    return vectorizer, clf


def predict_email(text):
    vectorizer, clf = load_model()

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = clf.predict(vec)[0]
    probabilities = clf.predict_proba(vec)[0]
    classes = list(clf.classes_)
    phishing_prob = probabilities[classes.index("phishing")]

    features = extract_features(text)

    return prediction, round(float(phishing_prob), 4), features
