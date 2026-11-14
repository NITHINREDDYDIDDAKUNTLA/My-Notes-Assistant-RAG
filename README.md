# My Notes Assistant - RAG Chatbot

A RAG chatbot that answers questions ONLY from my class notes PDF. No hallucination - if answer is not in notes, it says not found.

### How I built it
- PDF Loader to read notes
- Recursive Text Splitter (chunk 500, overlap 100)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- FAISS Vector DB for similarity search
- Groq LLM (llama-3.1-8b-instant) for answer
- Streamlit UI with chat_input fixed at bottom like ChatGPT

### How to run
1. Replace sample_notes.pdf with your own notes PDF name in app.py line 17
2. Create .env file with GROQ_API_KEY=your_key
3. pip install -r requirements.txt
4. streamlit run app.py

### Features
- Chat box at bottom
- Shows answer + source from PDF
- Private notes stays private via .gitignore

Tech: Python, LangChain, FAISS, Groq, Streamlit
