# 🤖 AI Agent

A production-grade autonomous AI agent built with LangGraph, FastAPI, and multi-provider LLM fallback. Capable of real-time web search, live weather, Python code execution with auto-fix, multi-language code generation, math calculations, and currency conversion.

🌐 **Live Demo:** https://ai-agent-p13l.onrender.com

---

## ✨ Features

- 🔍 **Real-time web search** — breaking news, current events, live data
- 🌤️ **Live weather** — any city worldwide
- 🐍 **Python code execution** — runs code, shows output, auto-fixes errors
- 💻 **Multi-language code generation** — Python, Java, JavaScript, C++, Go, Rust, SQL, and 15+ more
- 🧮 **Math & calculations** — percentages, formulas, unit conversions
- 💱 **Live currency conversion** — real-time exchange rates
- 🧠 **Conversation memory** — remembers full session context
- ⚡ **Multi-provider LLM fallback** — Groq → Gemini → OpenRouter, never goes down
- 🌐 **Multilingual** — responds in Hindi, English, Hinglish, or any language

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12 |
| AI Framework | LangGraph, LangChain |
| LLM Providers | Groq (Llama 3.3 70B), Google Gemini 2.0, OpenRouter |
| Search | Tavily API |
| Weather | wttr.in |
| Currency | ExchangeRate API |
| Frontend | HTML, CSS, Vanilla JS |
| Deployment | Render |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/kailashv2/ai-agent.git
cd ai-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

### 5. Run locally
```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000

---

## 🔑 Getting API Keys (All Free)

| Service | Link | Free Tier |
|---------|------|-----------|
| Groq | https://console.groq.com | 100K tokens/day |
| Google Gemini | https://aistudio.google.com/apikey | 1500 requests/day |
| OpenRouter | https://openrouter.ai | Multiple free models |
| Tavily | https://app.tavily.com | 1000 searches/month |

---

## 📁 Project Structure

```
ai-agent/
├── backend/
│   ├── agent.py         # LLM provider management, routing, code handling
│   ├── tools.py         # Search, weather, calculator, code executor, currency
│   ├── main.py          # FastAPI server, routes
│   ├── requirements.txt
│   └── static/
│       └── index.html   # Served frontend
├── frontend/
│   └── index.html       # Chat UI source
├── Dockerfile
├── .env.example
└── README.md
```

## 🧠 Architecture

```
User Message
     ↓
  Router
  ├── Code Request? → LLM generates code → Execute → Auto-fix if error → Return output
  ├── Needs Tools?  → LangGraph ReAct Agent → Tools (Search/Weather/Calculator/Currency)
  └── General?      → Direct LLM response (fastest)
     ↓
Multi-Provider LLM Fallback
  Groq (primary) → Gemini (secondary) → OpenRouter (tertiary)
```

## 💬 Example Queries

```
"What is the latest news about AI?"
"Write fibonacci series in Python"
"Weather in Mumbai"
"100 USD to INR"
"Write a REST API in Java"
"What is machine learning?"
"Calculate 18% of 25000"
"Write bubble sort in C++"
"Latest NEET 2026 news"
"What is quantum computing?"
```
## 🚀 Deployment

### Render (Recommended)
1. Fork this repo
2. Create new Web Service on [Render](https://render.com)
3. Set Root Directory to `backend`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add environment variables
7. Deploy

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

<p align="center">Built with ❤️ by <a href="https://github.com/kailashv2">Kailash</a></p>
