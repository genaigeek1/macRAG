
from datasets import load_dataset
import evaluate

# Load Dolly 15K
dolly = load_dataset("databricks/databricks-dolly-15k")
sample = dolly["train"].select(range(5))

# Dummy LLM Prediction (Replace with your LLM)
predictions = [example["response"] for example in sample]
references = [example["response"] for example in sample]  # Ground truth

# Evaluate using ROUGE
rouge = evaluate.load("rouge")
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE scores:", results)
