# ResumeLens — AI Resume Parser

A Streamlit app by **Shlok Sharma** to extract structured information from PDF/DOCX resumes and compare candidates against a job description. Includes ranked results and CSV/JSON export.

## Run locally

1. Install Python 3.10+.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. Paste a Groq API key into the sidebar. Alternatively, set `GROQ_API_KEY` in your environment or add it to `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your-key"
   ```

The app uses `openai/gpt-oss-120b` by default. If that model is not enabled for your Groq account, choose a currently available Groq model in the sidebar or set `GROQ_MODEL`.

## Deploy free on Streamlit Community Cloud

1. Create a **public or private** GitHub repository and upload these project files (do not upload `.env`, API keys, or resumes).
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository, branch, and `app.py`, then deploy.
4. In the deployed app's **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. Reboot the app. The public URL will be shown by Streamlit after deployment.

## Inputs and limitations

- Supported: text-based PDF and DOCX; up to 20 files, 10 MB each.
- Scanned/image-only PDFs are not OCR'd. Add an OCR service or local OCR (such as Tesseract) if needed.
- No database or persistent resume storage is implemented. Resume text is sent to Groq for processing; review the provider's current data handling/retention terms before using real candidate documents.
- Scores are advisory only. A human must review original resumes and make hiring decisions. Do not use protected attributes or automate rejection.

## APIs / services needed

**Required:** a Groq account and Groq API key (used for structured resume extraction and job matching). No separate OCR, database, or hosting API is required for the starter build.

**Optional:** OCR API/service for scanned PDFs; database/storage if you need saved candidates; transactional email if you want notifications. Streamlit Community Cloud can host this app without a separate hosting API key.
