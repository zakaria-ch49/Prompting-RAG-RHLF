
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

  ### -Modules:
  Uses os to create an educational_docs folder and requests to fetch files.

  ### -Sources:
  A dictionary (pdf_sources) maps filenames to URLs (but commented out, so empty).

  ### -Download:
  For each URL, checks if the HTTP response is valid and if the content is a PDF (via the %PDF header).

  ### -Save:
  Writes valid PDFs to the folder.

  ### -Feedback:
  Clear success/error messages in the console.

⚠️ Key Note: URLs are commented out → uncomment them to enable downloads.

