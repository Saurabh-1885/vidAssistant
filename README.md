# 🎬 AI Meeting Assistant

An AI-powered meeting/video assistant that turns a YouTube link or an uploaded audio/video file into a searchable, chattable meeting record — transcript, summary, action items, key decisions, and a RAG-powered Q&A chat — all from a single Streamlit UI.

## ✨ Features

- **Flexible input** — paste a YouTube URL or upload a local audio/video file.
- **Local transcription** — runs OpenAI's Whisper locally, no third-party transcription API needed.
- **Hindi → English translation** — automatically detects and translates Hindi speech before downstream processing.
- **LLM-powered summarization** — built with LangChain's LCEL and Mistral's free-tier API to generate:
  - A concise meeting summary
  - Action items
  - Key decisions
  - Open questions
- **Chat with your meeting** — a RAG pipeline (ChromaDB + HuggingFace embeddings) lets you ask follow-up questions grounded in the actual transcript.
- **Export** — download results as PDF or TXT.
- **Clean, dark, animated UI** — built entirely in Streamlit with custom CSS (smooth transitions, hover states, live pipeline status).

## 🏗️ How it works

```
YouTube URL / File Upload
        │
        ▼
  Audio extraction (yt-dlp / ffmpeg)
        │
        ▼
  Local Whisper transcription
        │
        ▼
  Hindi → English translation (if needed)
        │
        ▼
  LangChain (LCEL) + Mistral API
        │
        ├──► Summary
        ├──► Action items
        ├──► Key decisions
        └──► Open questions
        │
        ▼
  Chunk + embed transcript (HuggingFace embeddings)
        │
        ▼
  Store in ChromaDB  ──►  RAG chat (ask questions about the meeting)
        │
        ▼
  Export as PDF / TXT
```

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Audio ingestion | yt-dlp, ffmpeg |
| Transcription | OpenAI Whisper (local) |
| Translation | Hindi → English (googletrans / deep-translator) |
| Orchestration | LangChain (LCEL) |
| LLM | Mistral API (free tier) |
| Embeddings | HuggingFace (sentence-transformers) |
| Vector store | ChromaDB |
| Export | fpdf2 / reportlab |

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/ai-video-assistant.git
cd ai-video-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** You'll also need `ffmpeg` installed at the OS level (required by both `yt-dlp` and Whisper).
> - macOS: `brew install ffmpeg`
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Windows: [download from ffmpeg.org](https://ffmpeg.org/download.html) and add it to your PATH

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

## 🚀 Usage

```bash
streamlit run app.py
```

1. Paste a YouTube URL or provide a local file path in the sidebar.
2. Choose the language mode.
3. Click **Analyse** — the app extracts audio, transcribes it, translates if needed, and runs summarization + extraction.
4. Review the summary, action items, key decisions, and open questions.
5. Use the chat panel to ask follow-up questions about the meeting.
6. Export the results as a PDF or TXT file.

## 📁 Project Structure

```
.
├── app.py                     # Streamlit UI entry point
├── core/
│   ├── transcriber.py         # Whisper transcription logic
│   ├── summarize.py           # LangChain/Mistral summarization + title generation
│   ├── extractor.py           # Action items, decisions, open questions extraction
│   └── rag_engine.py          # ChromaDB + HuggingFace RAG chat pipeline
├── utils/
│   └── audio_processor.py     # YouTube/file audio extraction and chunking
├── requirements.txt
├── .env                        # API keys (not committed)
└── README.md
```

## ⚠️ Known Limitations

- Whisper transcription runs locally and can be slow on CPU-only machines; a GPU is recommended for longer meetings.
- The Mistral free-tier API has rate limits, so very long transcripts may need chunking or retries.
- Translation quality for Hindi → English depends on the translation library used and may need tuning for domain-specific terms.

## 🗺️ Roadmap / Ideas

- [ ] Support additional languages beyond Hindi
- [ ] Speaker diarization (who said what)
- [ ] Multi-meeting search across a history of transcripts
- [ ] Docker support for one-command setup

## 📄 License

[MIT](LICENSE) — feel free to use and adapt.

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Mistral AI](https://mistral.ai/)
- [ChromaDB](https://www.trychroma.com/)
- [Hugging Face](https://huggingface.co/)
