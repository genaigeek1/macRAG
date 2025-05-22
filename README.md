
# 🧠 Local RAG Application on MacBook M2 (Open Source Only)

This guide walks you through building a fully local Retrieval-Augmented Generation (RAG) pipeline using only open-source components. It is optimized for MacBooks with M1/M2 chips and 16GB RAM.

---

## ⚙️ Environment Setup

```bash
brew install python@3.11
brew install ollama
brew install cmake
pip install virtualenv

python3 -m venv rag-env
source rag-env/bin/activate
pip install --upgrade pip
```

---

## 📦 Install Dependencies

```bash
pip install langchain llama-index sentence-transformers \
            chromadb faiss-cpu \
            transformers accelerate \
            bitsandbytes evaluate \
            scikit-learn peft \
            tqdm pandas
```

---

## 🤖 LLM: Run Locally with Ollama

```bash
brew install ollama
ollama pull mistral
```

In Python:

```python
from langchain.llms import Ollama
llm = Ollama(model="mistral")
```

---

## 🗃️ Vector Store (FAISS)

```python
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

texts = ["doc1 content", "doc2 content"]
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

db = FAISS.from_texts(texts, embedding_model)
```

---

## 🔍 RAG with LangChain

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

query = "What is this document about?"
print(qa_chain.run(query))
```

---

## 🧠 Reranking (Optional)

```bash
pip install cross-encoder
```

```python
from cross_encoder import CrossEncoder
re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is the topic?"
retrieved_docs = ["doc1", "doc2"]
scores = re_ranker.predict([(query, doc) for doc in retrieved_docs])
```

---

## 🧪 Evaluation

```python
import evaluate
rouge = evaluate.load("rouge")

results = rouge.compute(
    predictions=["predicted answer"],
    references=["expected answer"]
)
print(results)
```

---

## 🔁 Fine-tuning (Optional)

- Use HuggingFace PEFT + LoRA with QLoRA script
- Sample datasets: Alpaca, Dolly, OpenAssistant

Explore: https://github.com/huggingface/trl/blob/main/examples/sft_trainer.py

---
