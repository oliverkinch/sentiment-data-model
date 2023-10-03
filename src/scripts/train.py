"""Script to fine-tune a transformer model for sentiment analysis.

Usage:
    python src/scripts/train.py
"""


from src.sentiment_model.train_transformer import train_transformer_model


if __name__ == "__main__":
    train_transformer_model()
