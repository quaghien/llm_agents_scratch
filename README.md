# LLM-RAG-AGENTS

This project is a modular workspace for independently exploring key components in modern LLM systems, ranging from core transformer implementation to RAG pipelines and autonomous agents.

It includes:

- **Interview Questions**: A curated collection of 100+ real-world interview questions on LLMs, used by top companies like Google, NVIDIA, Meta, Microsoft, and Fortune 500s.
- **Building LLMs from Scratch**: Implementing core transformer models to understand architecture and evolution from GPT to LLaMA 3.2.
- **Training Optimization Techniques**: Exploring memory-efficient training, gradient strategies, and architectural improvements.
- **Reinforcement Learning for LLMs**: Implementing post-training fine-tuning with alignment methods like PPO and DPO.
- **RAG Systems**: Building standalone Retrieval-Augmented Generation modules with custom retrievers and vector databases.
- **Autonomous Agents**: Designing agent frameworks for reasoning, planning, and tool use across multiple steps and memory contexts.


<br><br>

## 1. Interview Question Bank

A curated collection of over 100 real-world interview questions related to Large Language Models (LLMs).  
These questions are compiled based on actual interview processes from companies such as Google, NVIDIA, Meta, Microsoft, and other Fortune 500 firms.

The questions are organized into 15 categories to support systematic learning:

- Prompt Engineering & Basics of LLM  
- Retrieval Augmented Generation (RAG)  
- Document Digitization & Chunking  
- Embedding Models  
- Vector Database Internals  
- Advanced Search Algorithms  
- LLM Internal Architecture  
- Supervised Fine-Tuning  
- Preference Alignment (RLHF/DPO)  
- Evaluation of LLMs  
- Hallucination Mitigation  
- Deployment Strategies  
- Agent-based Systems  
- Prompt Hacking  
- Miscellaneous + Case Studies

<img src="Interview/images/interviewprep.jpg" width="60%" alt="model">

<br><br>

## 2. Building LLMs from Scratch

### 2.1 My implementation of Transformer models based on *"Attention is All You Need" (Google Brain, 2017)*
<img src="LLMs-from-scratch/transformer_from_scratch/images/model2.jpg" width="60%" alt="model">

### 2.2 `gpt-from-scratch.ipynb` – A standalone implementation of the GPT model
### 2.3 `llama3.2-from-scratch.ipynb` – A standalone implementation of the LLaMA 3.2 architecture
### 2.4 `convert_gpt_to_llama2.ipynb` – A step-by-step conversion from GPT to LLaMA 2
### 2.5 `convert_llama2_to_llama3_3.2.ipynb` – A progressive transformation from LLaMA 2 to LLaMA 3, then 3.1, and finally 3.2

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">

<br><br>

## 3. Training Optimization Techniques

### 3.1 Efficient weight initialization, gradient checkpointing, mixed-precision training

### 3.2 Implementation of memory-efficient attention mechanisms (e.g., FlashAttention)

<br><br>

## 4. Reinforcement Learning for LLMs

### 4.1 PPO-based fine-tuning to align model outputs with human preferences

### 4.2 Direct Preference Optimization (DPO) for simpler reward-free alignment

<br><br>

## 5. RAG (Retrieval-Augmented Generation)

### 5.1 Custom retriever design with local and remote vector stores

### 5.2 Full RAG pipelines integrating retriever + generator with context-aware grounding

<br><br>

## 6. Autonomous Agents

### 6.1 Designing LLM-powered agents with planning, memory, and tool-use capabilities

### 6.2 Multi-step reasoning agents using tools like search, calculators, or APIs

<br><br>