# Agentic-RAG

## Setup

1. Create and activate a Python 3.10 virtual environment.
2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

4. Put your `.pdf`, `.txt`, or `.md` source files inside `data/`.
5. Build the FAISS index:

   ```bash
   python src/ingest.py
   ```

6. Start the Streamlit app:

   ```bash
   streamlit run src/app.py
   ```
