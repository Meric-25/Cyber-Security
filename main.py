import argparse
import os

import database
import ml_pipeline


def cmd_train(args):
    print(f"Training model: {args.data} (model={args.model})")
    metrics = ml_pipeline.train_model(args.data, model_type=args.model)

    print("\n--- Training Results ---")
    for key, value in metrics.items():
        print(f"{key:12s}: {value:.4f}")
    print("Model saved: models/classifier.joblib, models/vectorizer.joblib")


def cmd_analyze(args):
    if args.text:
        email_text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            email_text = f.read()
    else:
        print("Error: either --text or --file must be provided.")
        return

    if not os.path.exists(ml_pipeline.MODEL_PATH):
        print("Error: train the model first with 'python main.py train --data <csv>'.")
        return

    prediction, probability, features = ml_pipeline.predict_email(email_text)

    database.save_analysis_result(email_text, prediction, probability, features)

    print("\n--- Analysis Result ---")
    print(f"Prediction        : {prediction.upper()}")
    print(f"Phishing probability: {probability * 100:.2f}%")
    print(f"URL count          : {features['url_count']}")
    print(f"Suspicious words   : {features['suspicious_word_count']}")
    print(f"Urgent language    : {'Yes' if features['has_urgent_language'] else 'No'}")
    print(f"Exclamation count  : {features['exclamation_count']}")
    print(f"Uppercase ratio    : {features['uppercase_ratio']}")


def cmd_history(args):
    rows = database.get_history(limit=args.limit)
    if not rows:
        print("No analysis records yet.")
        return

    print(f"\n--- Last {len(rows)} Analyses ---")
    for row_id, snippet, prediction, probability, analyzed_at in rows:
        short_snippet = snippet[:50].replace("\n", " ")
        print(f"[{row_id}] {analyzed_at} | {prediction.upper():10s} "
              f"({probability*100:.1f}%) | {short_snippet}...")


def cmd_stats(args):
    stats = database.get_statistics()
    print("\n--- Statistics ---")
    print(f"Total analyzed     : {stats['total_analyzed']}")
    print(f"Phishing detected  : {stats['phishing_detected']}")
    print(f"Legitimate         : {stats['legitimate']}")


def main():
    database.init_db()

    parser = argparse.ArgumentParser(description="Phishing Email Detection System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Training CSV file")
    train_parser.add_argument(
        "--model", choices=["random_forest", "logistic_regression"],
        default="random_forest", help="Classifier to use"
    )
    train_parser.set_defaults(func=cmd_train)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an email")
    analyze_parser.add_argument("--text", help="Email text to analyze")
    analyze_parser.add_argument("--file", help="Email file to analyze")
    analyze_parser.set_defaults(func=cmd_analyze)

    history_parser = subparsers.add_parser("history", help="List past analyses")
    history_parser.add_argument("--limit", type=int, default=20)
    history_parser.set_defaults(func=cmd_history)

    stats_parser = subparsers.add_parser("stats", help="Show overall statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
