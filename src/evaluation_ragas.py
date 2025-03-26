import json
import pandas as pd
import numpy as np
import google.generativeai as genai
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas import evaluate

# Set up Gemini API Key from .env file
import os
import sys
from dotenv import load_dotenv

# Change path to load .env from one directory back
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

google_api_key = os.getenv("GEMINI_API_KEY")
if not google_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

genai.configure(api_key=google_api_key)

class Evaluator:
    def __init__(self, golden_set_path, model_output_path, results_excel, overall_results_txt):
        self.golden_set_path = golden_set_path
        self.model_output_path = model_output_path
        self.results_excel = results_excel
        self.overall_results_txt = overall_results_txt
        self.golden_set = self.load_json(self.golden_set_path)
        self.model_output = self.load_json(self.model_output_path)

    def load_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self):
        queries = [entry["query"] for entry in self.golden_set]
        golden_answers = [entry["golden_set"] for entry in self.golden_set]
        contexts = [entry["context"] for entry in self.model_output]
        model_outputs = [entry["output"] for entry in self.model_output]

        # Prepare dataset as a dictionary
        dataset = {
            "queries": queries,
            "contexts": contexts,
            "responses": model_outputs,
            "references": golden_answers
        }

        # Compute scores
        scores = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness, answer_relevancy])

        results = []
        for i in range(len(queries)):
            results.append({
                "Query": queries[i],
                "Golden Set": golden_answers[i],
                "Model Output": model_outputs[i],
                "Context": contexts[i],
                "Context Precision": scores["context_precision"][i],
                "Context Recall": scores["context_recall"][i],
                "Faithfulness": scores["faithfulness"][i],
                "Answer Relevancy": scores["answer_relevancy"][i]
            })

        self.save_results(results)
        self.save_overall_results(scores)

    def save_results(self, results):
        df = pd.DataFrame(results)
        df.to_excel(self.results_excel, index=False)

    def save_overall_results(self, scores):
        overall_results = {
            "Overall Context Precision": np.mean(scores["context_precision"]),
            "Overall Context Recall": np.mean(scores["context_recall"]),
            "Overall Faithfulness": np.mean(scores["faithfulness"]),
            "Overall Answer Relevancy": np.mean(scores["answer_relevancy"])
        }
        with open(self.overall_results_txt, "w") as f:
            for key, value in overall_results.items():
                f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    evaluator = Evaluator(
        "assets/golden_set_ragas.json",
        "assets/model_output_ragas.json",
        "assets/evaluation_results_ragas.xlsx",
        "assets/overall_evaluation_results_ragas.txt"
    )
    evaluator.evaluate()
    print("Evaluation completed. Results saved in assets/evaluation_results_ragas.xlsx and assets/overall_evaluation_results_ragas.txt.")