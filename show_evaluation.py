
import json
from pathlib import Path

results_path = Path(__file__).resolve().parent / "evaluation_results.json"

with open(results_path, "r", encoding="utf-8") as file:
    results = json.load(file)

for i, item in enumerate(results, start=1):
    print(f"{i}. {item['question']}")
    print(f"   Faithfulness: {item['faithfulness']:.4f}")
    print(f"   Answer Relevancy: {item['answer_relevancy']:.4f}")
    print()

print(f"Total questions: {len(results)}")
