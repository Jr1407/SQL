# main.py

from src.dataset import load_text2sql_dataset
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    train_data, test_data = load_text2sql_dataset()
    tokenizer, model = load_model_and_tokenizer()
    train_model(model, tokenizer, train_data)
    evaluate_model(model, tokenizer, test_data)


if __name__ == "__main__":
    main()
