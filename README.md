# 🧠 Local LLM-Powered Data Assistant for Public Health (Wayne County Project)

## Overview
A privacy-preserving, offline AI assistant built using [LangChain](https://www.langchain.com/), [Ollama](https://ollama.com/), and [LLaMA 3](https://llama.meta.com/llama3) to enable natural language queries over sensitive human services datasets, including Fee and Residential data. This project was developed during my internship at the **Wayne County Health, Human & Veterans Services (HHVS)** department.

## 🔧 Features
- ✅ Query structured CSV data using natural language
- 🔒 Fully offline, privacy-respecting architecture (no cloud dependencies)
- 📄 Dynamic routing between multiple datasets
- 🧠 Model flexibility – supports LLaMA 3, GPT, and Copilot (can be swapped)
- 📊 Built-in analytics – count missing values, calculate averages, filter by dates, etc.
- 💡 Extendable – can be enhanced with memory, chatbot interface, and RAG

## 🗂 Datasets Used
- `Fee.csv` – Client-level service data
- `Residential.csv` – Records of residential placements
- `Dictionary.csv` – Column descriptions for interpretability
- `Client.csv` – Demographic and identifying info

## 🚀 Tech Stack
- Python + Pandas
- LangChain (Agents + Tools)
- Ollama (for local LLM running LLaMA 3)
- Terminal-based Q&A interface
- Optional: Streamlit frontend (future step)

## 📸 Screenshots / Demo (Optional)
> _You can upload screenshots or GIFs from your CLI interface or demo screenshots here._

## 🔄 Future Work
- Integrate **retrieval-augmented generation (RAG)** with data dictionary
- Add **memory and chat history**
- Deploy a **web-based chatbot interface** (e.g., using Streamlit or Gradio)
- Support **multiple model backends** (Copilot, GPT-4, Claude)

## 🤝 Acknowledgements
Special thanks to **Wayne County HHVS**, **Ali Asadi**, and the IT team for providing guidance, data access, and support throughout the project.
