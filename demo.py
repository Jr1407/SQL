# demo.py
from src.model import load_model_and_tokenizer

def run_demo():
    tokenizer, model = load_model_and_tokenizer()

    prompts = [
        "Sort the employees based on their salary in ascending order",
        "Find the highest salary of_toggle the employees in IT department",
        "Find the sum of salary of the employees in HR department",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

        sql = tokenizer.decode(output[0], skip_special_tokens=True)

        print("\nPrompt:")
        print(prompt)
        print("\nGenerated SQL:")
        print(sql)
        print("-" * 60)


if __name__ == "__main__":
    run_demo()
