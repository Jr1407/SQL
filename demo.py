# demo.py 
from src.model import load_model_and_tokenizer

def run_demo():
    tokenizer, model = load_model_and_tokenizer()

    questions = [
        "Find the sum of salary of the employees in HR department",
        "Find the highest salary of employees in IT department",
    ]

    for question in questions:
        prompt = f"""
Convert the following question into a valid SQL query.
Return ONLY the SQL query, no explanation.

Question:
{question}

SQL:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0
        )

        sql = tokenizer.decode(output[0], skip_special_tokens=True)

        print("\nQuestion:")
        print(question)
        print("\nGenerated SQL:")
        print(sql)
        print("-" * 60)


if __name__ == "__main__":
    run_demo()
