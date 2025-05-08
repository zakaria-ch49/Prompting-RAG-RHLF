
<div>
<img src="https://i0.wp.com/thelettertwo.com/wp-content/uploads/2024/02/ai-economy-rag-explained-1600x914-1.jpg?w=1600&ssl=1" width=1100>
<div>


<div align="center" style="font-family: 'Times New Roman', Times, serif; color: #3498db;">
  <h1 style="font-size: 10em; margin-bottom: 180px;">
    Projet Machine Learning :<br>
    `Prompting(RAG), RHLF`
  </h1>
</div>

<div align="right" style="font-family: 'Times New Roman', Times, serif; 
                          color: #8e44ad;
                          font-style: italic;
                          font-weight: bold;
                          margin-right: 15%;
                          margin-top: 10px;
                          font-size: 1.3em;">
   Realised : SABI Houssame & CHADADI Zakaria
</div>

&nbsp;

---

<!-- Le reste de votre contenu ici -->





## Introduction

This project explores three key techniques in modern AI:

- **Prompting**: the art of crafting effective instructions (prompts) to guide language models.
- **RAG (Retrieval-Augmented Generation)**: combines document retrieval with text generation to improve the relevance and accuracy of responses.
- **RHLF (Reinforcement Learning from Human Feedback)**: a training approach that uses human feedback to align models with desired behavior, improving safety and usefulness.

These methods are essential for building advanced and trustworthy AI systems.

---

## Summary

1. [🎯 Project Goals](#project-goals)  
2. [⚙️ Package Installation](#package-installation)  
3. [📦 Library Imports](#library-imports)  
4. [💡 Example Code Explanation](#example-code-explanation)

---

## 🎯 Project Goals

- Understand the role of prompting in interacting with LLMs (Large Language Models).
- Implement a simple Retrieval-Augmented Generation (RAG) pipeline.
- Learn the basics of Reinforcement Learning from Human Feedback (RHLF).

---

## ⚙️ Package Installation

- **LlamaIndex**: A data framework for building LLM (Large Language Model) applications by indexing and querying data with ease.
  ```bash
  %pip install llama-index-llms-openrouter
  !pip install llama-index
  %pip install llama-index-embeddings-huggingface
  %pip install llama-index-embeddings-instructor ```
- **OpenAI**: An AI research and deployment company offering advanced language models like GPT for natural language understanding and generation.
- **Streamlit**: A Python library used to quickly build and share interactive web apps for data science and machine learning projects.
  ```
  pip install requests pymupdf streamlit
  ```

---

 ## 📦 Library Imports
 ### ➡️ Download and Save Educational PDFs from OpenStax
 <div>
 <img src="https://github.com/user-attachments/assets/fcd36e56-928a-4b69-9d5b-d8f266e5ef77" width=1000>
 </div>
 <br>
This Python script imports the os and requests modules, creates a folder named educational_docs (if it doesn’t already exist) to store PDFs, and defines a dictionary called pdf_sources that maps file names to URLs of OpenStax documents (currently commented out, so inactive). For each entry in the dictionary, the script attempts to download the PDF via an HTTP request, checks if the response is valid and contains a proper PDF header (%PDF), then saves the file locally if successful. <br>
If the download fails or the content is not a valid PDF, it prints an error message. Finally, it confirms the outcome of each operation through console messages.
<br>
<br>
⚠️ Key Note: URLs are commented out → uncomment them to enable downloads.

<br>
<br>
<br>

### ➡️ Set Up a DeepSeek Chat Model via OpenRouter API with LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/589c0846-2689-4666-9647-bba25b68052b" width=1000>
 </div>
 <br>

In this part code configures a language model (LLM) via OpenRouter (a service providing access to multiple AI models) using the llama_index library. It imports the required classes (OpenRouter for integration, ChatMessage—unused here), initializes an LLM client with a hardcoded API key (⚠️ security risk), and specifies the free deepseek/deepseek-chat:free model, limiting responses to 256 tokens and setting a 4096-token context window. Commented lines suggest using environment variables for secure API key storage (best practice ignored here).<br>
This snippet acts as a foundational setup for interacting with LLMs via OpenRouter but requires adjustments (security fixes, query logic) for real-world use.
<br>
<br>
<br>

### ➡️ Initialize Qwen2-7B Embedding Model via HuggingFace in LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/a8595f88-7ce2-4813-a959-3d5e5ac600ee" width=1000>
 </div>
 <br>
It imports the HuggingFaceEmbedding class to generate vector representations of text. Two options are shown:

1. Commented-out line (`embed_model = HuggingFaceEmbedding()`) :
   - would load the default `BAAI/bge-small-en model`, a lightweight and efficient embedding model optimized for English.

2. Active line :
   - explicitly loads the Alibaba-NLP/gte-Qwen2-7B-instruct model, a far larger and more powerful model (7 billion parameters) designed for advanced multilingual or complex tasks.

#### Key implications:
- Using Alibaba-NLP/gte-Qwen2-7B-instruct implies a need for high-quality embeddings (`deep contextual understanding, multilingual support`) but requires significant computational resources (GPU/RAM).

- The commented option (`bge-small-en`) offers a lightweight alternative for simpler tasks or resource-constrained environments.

- No additional parameters (`device, pooling...`) are specified, so default settings of HuggingFaceEmbedding are used.

<br>
<br>
<br>

### ➡️ Set Up a Multilingual Embedding Model (Qwen2-7B) in LlamaIndex
 <div>
 <img src="https://github.com/user-attachments/assets/e27e33bd-8ae0-4d9f-b9d4-c83f0796b6f5" width=1000>
 </div>
 <br>

- Global Integration: All LlamaIndex components (`indexers, query engines...`) will automatically use this model to generate embeddings.

- Advanced Model: `The gte-Qwen2-7B-instruct` model is designed for complex tasks (e.g., contextual understanding, multilingual Q&A) but requires significant GPU/RAM resources.

- Simple Customization: A single line of code defines the default behavior for the entire library.

Typical Use Cases:

- Building a multilingual semantic search system.

- Generating high-quality embeddings for classification or clustering.

- Experimenting with large-scale models for specialized tasks.

⚠️ Notes:

No additional parameters (`device to force GPU usage`) are specified : relies on HuggingFaceEmbedding default settings.

The chosen model is resource-intensive : verify hardware availability before execution.

<br>
<br>
<br>

### ➡️ Create a VectorStoreIndex from Files in ‘educational_docs’ Folder
 <div>
 <img src="https://github.com/user-attachments/assets/e27e33bd-8ae0-4d9f-b9d4-c83f0796b6f5" width=1000>
 </div>
 <br>
----
Loads documents from the educational_docs directory using SimpleDirectoryReader.

Constructs a vector index (VectorStoreIndex) from the documents, automatically converting text into embeddings with LlamaIndex's default model.

The minimalist setup leverages predefined configurations, making it ideal for prototyping, but lacks customization (e.g., embedding models, storage settings) for advanced use cases.
 
