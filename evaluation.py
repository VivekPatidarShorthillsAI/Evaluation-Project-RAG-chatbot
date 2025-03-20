import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    """
    Class to compute various evaluation metrics between model output and ground truth.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_cosine_similarity(self, text1, text2):
        """
        Compute cosine similarity between two texts using their embeddings.
        """
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        cos_sim = cosine_similarity(emb1, emb2)[0][0]
        return cos_sim

    def tokenize(self, text):
        """
        Tokenize the text by converting it to lowercase and splitting on whitespace.
        """
        return text.lower().split()

    def compute_token_metrics(self, model_text, ground_text):
        """
        Compute token-level precision, recall, and F1 score based on token overlap.
        """
        model_tokens = set(self.tokenize(model_text))
        ground_tokens = set(self.tokenize(ground_text))
        
        # Avoid division by zero
        if not model_tokens or not ground_tokens:
            return 0.0, 0.0, 0.0

        common_tokens = model_tokens.intersection(ground_tokens)
        precision = len(common_tokens) / len(model_tokens)
        recall = len(common_tokens) / len(ground_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        return precision, recall, f1

    def evaluate_instance(self, query, ground_truth, model_answer):
        """
        Evaluate a single instance and return a dictionary with the metrics.
        """
        # Compute cosine similarity between ground truth and model output
        cos_sim = self.compute_cosine_similarity(ground_truth, model_answer)
        
        # Compute token-level precision, recall, and F1 score
        precision, recall, f1 = self.compute_token_metrics(model_answer, ground_truth)
        
        # LLM-based Faithfulness metric (placeholder) - using cosine similarity here as a proxy.
        llm_faithfulness = cos_sim
        
        return {
            "Query": query,
            "Model_Output": model_answer,
            "Ground_Truth": ground_truth,
            "Cosine_Similarity": cos_sim,
            "Token_Precision": precision,
            "Token_Recall": recall,
            "Token_F1": f1,
            "LLM_Based_Faithfulness": llm_faithfulness
        }

def load_json_file(file_path):
    """
    Load and return JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_evaluation_to_excel(evaluation_results, output_file):
    """
    Save the evaluation results to an Excel file.
    """
    df = pd.DataFrame(evaluation_results)
    df.to_excel(output_file, index=False)
    print(f"Evaluation metrics saved to {output_file}")

def evaluate_answers(golden_set_file, model_output_file):
    """
    Load the golden set and model output files, compute evaluation metrics,
    and return the results as a list of dictionaries.
    """
    golden_set = load_json_file(golden_set_file)
    model_output_set = load_json_file(model_output_file)
    
    evaluator = Evaluator()
    evaluations = []
    
    # Assuming both JSON files align by index
    for gold_item, model_item in zip(golden_set, model_output_set):
        query = gold_item.get("query", "")
        ground_truth = gold_item.get("ground_truth", "")
        model_answer = model_item.get("ground_truth", "")
        eval_metrics = evaluator.evaluate_instance(query, ground_truth, model_answer)
        evaluations.append(eval_metrics)
    
    return evaluations

def main():
    # File paths
    golden_set_file = 'Golden_set.json'
    model_output_file = 'Model_Output_Set.json'
    output_excel_file = 'Evaluation_Metrics.xlsx'
    
    print("Starting evaluation...")
    evaluations = evaluate_answers(golden_set_file, model_output_file)
    save_evaluation_to_excel(evaluations, output_excel_file)
    print("Evaluation complete.")

if __name__ == '__main__':
    main()
