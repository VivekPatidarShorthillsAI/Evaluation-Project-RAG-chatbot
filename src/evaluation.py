import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
import bert_score

class Evaluator:
    def __init__(self, golden_set_path, model_output_path, results_excel, overall_results_txt):
        self.golden_set_path = golden_set_path
        self.model_output_path = model_output_path
        self.results_excel = results_excel
        self.overall_results_txt = overall_results_txt
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
        self.golden_set = self.load_json(self.golden_set_path)
        self.model_output = self.load_json(self.model_output_path)

    def load_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def calculate_cosine_similarity(self, text1, text2):
        vec1 = self.model.encode([text1])
        vec2 = self.model.encode([text2])
        return cosine_similarity(vec1, vec2)[0][0]
    
    def calculate_precision_recall_f1(self, pred, gold):
        pred_tokens = set(pred.split())
        gold_tokens = set(gold.split())
        
        true_positives = len(pred_tokens & gold_tokens)
        precision = true_positives / len(pred_tokens) if pred_tokens else 0
        recall = true_positives / len(gold_tokens) if gold_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_llm_faithfulness(self, pred, gold):
        pred_tokens = Counter(pred.split())
        gold_tokens = Counter(gold.split())
        hallucinated_tokens = sum((pred_tokens - gold_tokens).values())
        faithfulness_score = 1 - (hallucinated_tokens / len(pred.split()) if pred.split() else 0)
        return max(faithfulness_score, 0)
    
    def calculate_bert_score(self, pred, gold):
        P, R, F1 = bert_score.score([pred], [gold], lang="en", verbose=False)
        return float(F1.mean())
    
    def evaluate(self):
        results = []
        precision_scores, recall_scores, f1_scores, faithfulness_scores, similarity_scores, bert_scores = [], [], [], [], [], []
        
        for gold, pred in zip(self.golden_set, self.model_output):
            query = gold["query"]
            gold_text = gold["golden_set"]
            pred_text = pred["output"]
            
            precision, recall, f1 = self.calculate_precision_recall_f1(pred_text, gold_text)
            faithfulness = self.calculate_llm_faithfulness(pred_text, gold_text)
            cosine_sim = self.calculate_cosine_similarity(pred_text, gold_text)
            bert_f1 = self.calculate_bert_score(pred_text, gold_text)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            faithfulness_scores.append(faithfulness)
            similarity_scores.append(cosine_sim)
            bert_scores.append(bert_f1)
            
            results.append({
                "Query": query,
                "Golden Set": gold_text,
                "Model Output": pred_text,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "LLM Faithfulness": faithfulness,
                "Cosine Similarity": cosine_sim,
                "BERT Score": bert_f1
            })
        
        self.save_results(results)
        self.save_overall_results(precision_scores, recall_scores, f1_scores, faithfulness_scores, similarity_scores, bert_scores)
    
    def save_results(self, results):
        df = pd.DataFrame(results)
        df.to_excel(self.results_excel, index=False)
    
    def save_overall_results(self, precision_scores, recall_scores, f1_scores, faithfulness_scores, similarity_scores, bert_scores):
        overall_results = {
            "Overall Precision": np.mean(precision_scores),
            "Overall Recall": np.mean(recall_scores),
            "Overall F1 Score": np.mean(f1_scores),
            "Overall LLM Faithfulness": np.mean(faithfulness_scores),
            "Overall Cosine Similarity": np.mean(similarity_scores),
            "Overall BERT Score": np.mean(bert_scores)
        }
        with open(self.overall_results_txt, "w") as f:
            for key, value in overall_results.items():
                f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    evaluator = Evaluator(
        "assets/golden_set.json", 
        "assets/model_output.json", 
        "assets/evaluation_results.xlsx", 
        "assets/overall_evaluation_results.txt"
    )
    evaluator.evaluate()
    print("Evaluation completed. Results saved in assets/evaluation_results.xlsx and assets/overall_evaluation_results.txt.")
