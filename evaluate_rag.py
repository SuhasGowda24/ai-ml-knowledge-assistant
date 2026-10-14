
import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from rag_pipeline import create_rag_components, ask_question

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

DATASET_PATH = PROJECT_DIR / "evaluation_dataset.csv"
OUTPUT_PATH = PROJECT_DIR / "evaluation_results.json"


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError("evaluation_dataset.csv was not found.")

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {"question", "ground_truth"}
    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            "CSV must contain question and ground_truth columns."
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY was not found in your .env file.")

    vectorstore, llm = create_rag_components()

    # Use RAGAS's modern LLM interface with the existing Groq model.
    groq_openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        )

    ragas_llm = llm_factory(
        "openai/gpt-oss-120b",
        provider="openai",
        client=groq_openai_client,
        adapter="instructor",
        temperature=0.0,
        max_tokens=2048,
    )

    # Use the same local embedding model for evaluation.
    ragas_embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        normalize_embeddings=True,
    )

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    relevancy_metric = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Load previously saved evaluation results.
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = []

    # Identify questions that already have both valid scores.
    completed_questions = {
        item["question"]
        for item in results
        if item.get("faithfulness") is not None
        and item.get("answer_relevancy") is not None
    }

    for index, row in dataset.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip()

        if question in completed_questions:
            print(f"Skipping already evaluated question {index + 1}")
            continue

        print(f"\nEvaluating question {index + 1}/{len(dataset)}")
        print(f"Question: {question}")

        answer, sources = ask_question(question, vectorstore, llm)

        documents = vectorstore.similarity_search(question, k=8)
        contexts = [
            doc.page_content.strip()
            for doc in documents
            if doc.page_content.strip()
        ]

        faithfulness_score = None
        relevancy_score = None

        try:
            faithfulness_result = faithfulness_metric.score(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            faithfulness_score = float(faithfulness_result.value)

            relevancy_result = relevancy_metric.score(
                user_input=question,
                response=answer,
            )
            relevancy_score = float(relevancy_result.value)

            print(f"Faithfulness: {faithfulness_score:.4f}")
            print(f"Answer Relevancy: {relevancy_score:.4f}")

        except Exception as error:
            import traceback
            print(f"Metric calculation failed: {error}")
            traceback.print_exc()
            
        
        new_result = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "sources": [
                {
                    "source": source.get("source"),
                    "page": source.get("page"),
                }
                for source in sources
            ],
        }

        # Update an existing record instead of creating duplicates.
        existing_index = next(
            (
                i for i, item in enumerate(results)
                if item.get("question") == question
            ),
            None,
        )

        if existing_index is not None:
            results[existing_index] = new_result
        else:
            results.append(new_result)

        # Save progress immediately after each question.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2, ensure_ascii=False)

        if (
            faithfulness_score is not None
            and relevancy_score is not None
        ):
            completed_questions.add(question)

        print(f"Progress saved: {len(results)} result records")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    
    valid_faithfulness = [
        item["faithfulness"] for item in results
        if item.get("faithfulness") is not None
    ]
    valid_relevancy = [
        item["answer_relevancy"] for item in results
        if item.get("answer_relevancy") is not None
    ]

    total_questions = len(dataset)
    dataset_questions = set(
    dataset["question"].astype(str).str.strip()
)

    fully_evaluated = sum(
        1 for item in results
        if item.get("question") in dataset_questions
        and item.get("faithfulness") is not None
        and item.get("answer_relevancy") is not None
    )

    print("\n========== RAGAS EVALUATION RESULTS ==========")
    print(f"Dataset questions: {total_questions}")
    print(f"Questions with both scores: {fully_evaluated}/{total_questions}")

    if fully_evaluated == total_questions:
        print(
            "Average Faithfulness: "
            f"{sum(valid_faithfulness) / total_questions:.4f}"
        )
        print(
            "Average Answer Relevancy: "
            f"{sum(valid_relevancy) / total_questions:.4f}"
        )
        print("Evaluation status: COMPLETE")
    else:
        print("Evaluation status: INCOMPLETE")
        print(
            f"Questions still needing evaluation: "
            f"{total_questions - fully_evaluated}"
        )

        if valid_faithfulness:
            print(
                "Current Faithfulness average (partial): "
                f"{sum(valid_faithfulness) / len(valid_faithfulness):.4f}"
            )

        if valid_relevancy:
            print(
                "Current Answer Relevancy average (partial): "
                f"{sum(valid_relevancy) / len(valid_relevancy):.4f}"
            )

    print(f"Detailed results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
