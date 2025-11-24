# Medical Question-Answering Assistant using Fine-Tuned Mistral-7B + RAG

This project builds a **medical question-answering assistant** using a **fine-tuned Mistral-7B LLM** combined with a **Retrieval-Augmented Generation (RAG)** pipeline.  
The assistant provides structured, safety-oriented answers for respiratory health topics such as **COPD, Asthma, Pneumonia, Tuberculosis, and Influenza** using evidence retrieved from **WHO fact sheets**.

The system is designed for **educational and research purposes**, demonstrating how modern LLMs can be aligned with domain knowledge and safe medical reasoning.

---

## 🚀 Project Highlights

- **Model:** Mistral-7B Instruct (LoRA fine-tuned)
- **Frameworks:** PyTorch, Hugging Face Transformers, FAISS, Sentence-Transformers
- **Knowledge Base:** WHO respiratory-disease documents
- **Technique:** Retrieval-Augmented Generation (RAG)
- **Task:** Medical question answering with structured outputs
- **Evaluation:** LLM-as-a-Judge + RAGAS-style metrics

---

## 🧠 Dataset Overview

### **1. Fine-Tuning Dataset**
- **Base:** Subset of **MedMCQA** (2000 train, 200 eval)
- **Custom:** 15 handcrafted medical instructions and Q&A examples  
- **Format:** Instruction → Input → Output (SFT format)

### **2. RAG Corpus**
WHO public fact sheets for 5 diseases:
- COPD  
- Asthma  
- Pneumonia  
- Influenza  
- Tuberculosis  

Each document was:
- cleaned
- split into multi-section chunks
- embedded using MiniLM
- indexed using FAISS (cosine similarity)

---

## 🔧 RAG Pipeline & Preprocessing

| Step | Description |
|------|-------------|
| **Text Cleaning** | Removed headers, footers, artifacts, spacing |
| **Chunking** | Multi-section chunks (~500–1000 tokens) |
| **Embedding Model** | MiniLM-L6-V2 |
| **Indexing** | FAISS (cosine similarity) |
| **Thresholding** | Optional similarity cutoff for irrelevant chunks |

---

## 🏗 Model Architecture

### **1. Fine-Tuned LLM**
- **Base Model:** Mistral-7B-Instruct-v0.2  
- **Fine-Tuning:** LoRA (rank 16)  
- **Quantization:** 4-bit QLoRA (bitsandbytes)  
- **Goal:** Improve medical reasoning, safety, and structured answer format  
- **Hardware:** Tesla P100 (Kaggle)

### **2. RAG Integration**
The assistant uses a structured three-step pipeline:

1. **Retrieve** relevant WHO disease chunks using FAISS  
2. **Construct Prompt** with structured medical template  
3. **Generate Answer** following this format:
   - Summary  
   - Possible Causes  
   - Red Flags  
   - Recommended Actions  
   - Safety Disclaimer  

---

## ⚙️ Training Configuration

| Component | Setting |
|----------|---------|
| Fine-Tuning | LoRA + 4-bit quantization |
| Optimizer | AdamW |
| Learning Rate | 2e-4 |
| Batch Size | 2 |
| Gradient Accumulation | 8 |
| Max Sequence Length | 512 |
| Epochs | 2 |
| Attention | FlashAttention/SDPA |
| Padding | Dynamic (Data Collator) |

The LoRA weights were merged with the base model for lightweight inference.

---

## 📊 RAGAS-Style Evaluation

Additional automated evaluation includes:

- **Answer Relevance**
- **Faithfulness** (answer grounded in retrieved context)
- **Answer Correctness**
- **Context Precision** (retrieval quality)
- **Context Recall** (coverage)

All evaluations use offline **FLAN-T5** due to Kaggle’s no-internet environment.

### **Interpretation**
- Answers are generally relevant with occasional broad responses  
- Factuality depends on chunk quality and RAG coverage  
- Safety is strong due to structured outputs + fine-tuning style data  

---

## 📦 Deployment

This model is not deployed due to high GPU requirements.
Inference is performed via Kaggle notebooks or local GPU setups.

---

## Author

Mohamed Dilrose P M
M.Sc. Statistics | Data Scientist
📧 mohameddilrose2018@gmail.com
🔗 https://www.linkedin.com/in/mohamed-dilrose-365554230/
