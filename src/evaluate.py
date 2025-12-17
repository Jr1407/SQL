# src/evaluate.py

import torch


def evaluate_model(model, tokenizer, test_dataset):
    """
    Evaluates model using exact-match accuracy for SQL queries
    """

    model.eval()
    correct = 0

    for sample in test_dataset:
        prompt = sample["sql_prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128
            )

        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        ground_truth = sample["sql"]

        if ground_truth.strip() in prediction.strip():
            correct += 1

    accuracy = correct / len(test_dataset)
    print(f"Exact Match Accuracy: {accuracy:.4f}")
