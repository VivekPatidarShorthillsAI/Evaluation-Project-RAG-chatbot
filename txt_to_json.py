import json
import re

def convert_txt_to_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = file.read()
    
    # Split the content into questions and answers
    qa_pairs = re.split(r"\n\n", data.strip())
    
    results = []
    for pair in qa_pairs:
        match = re.match(r"Question: (.*?)\nAnswer: (.*)", pair, re.DOTALL)
        if match:
            question, answer = match.groups()
            results.append({"query": question.strip(), "ground_truth": answer.strip()})
    
    # Write to JSON file
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

# File paths
input_file = "text_files/question and answer.txt"
output_file = "Golden_set.json"

# Convert the text file to JSON
convert_txt_to_json(input_file, output_file)

print(f"JSON file '{output_file}' has been created successfully.")
