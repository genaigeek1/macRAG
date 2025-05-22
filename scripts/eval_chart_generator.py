
import matplotlib.pyplot as plt

# Dummy before and after scores
before = {"rouge1": 0.21, "rougeL": 0.18}
after = {"rouge1": 0.35, "rougeL": 0.30}

labels = list(before.keys())
before_scores = list(before.values())
after_scores = list(after.values())

x = range(len(labels))
plt.bar(x, before_scores, width=0.4, label='Before', align='center')
plt.bar([i + 0.4 for i in x], after_scores, width=0.4, label='After', align='center')

plt.xticks([i + 0.2 for i in x], labels)
plt.ylabel('Score')
plt.title('Evaluation Before vs After Fine-Tuning/RAG')
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/llm_eval_comparison_chart.png")
plt.show()
