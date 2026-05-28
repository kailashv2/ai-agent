from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
try:
    from langchain.agents import create_react_agent as _cra
    from langgraph.prebuilt import create_react_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent
from tools import get_all_tools, run_python_code
from dotenv import load_dotenv
import re
import os

load_dotenv()

SYSTEM_PROMPT = """You are a world-class AI agent - expert in coding, research, analysis, math, science, history, finance, law basics, medicine basics, current events, creative writing, and more.

TOOLS AVAILABLE:
- search_tool: Breaking news, current events, today's updates
- general_search_tool: Facts, people, places, companies, how-to, explanations
- get_weather: Live weather for any city worldwide
- calculate: Math, percentages, unit conversions, formulas
- run_python_code: Execute Python - algorithms, data processing, simulations
- convert_currency: Live exchange rates for all major currencies

BEHAVIOR RULES:
1. Current/recent info -> search first, NEVER answer from memory.
2. Search with specific queries. Bad: "AI news". Good: "latest AI developments May 2026".
3. Irrelevant search result -> search again with better keywords immediately.
4. ONLY report what sources say. Never hallucinate or mix unrelated results.
5. Always cite source name and date for news.
6. Weather -> get_weather tool always.
7. Math/calculation -> calculate tool always.
8. Follow-up ("tell me more", "what about", "aur batao", "and then") -> continue previous topic, never restart.
9. No results after 2 searches -> admit it, suggest official source.
10. Respond in the same language the user writes in (Hindi, English, Hinglish, etc).
11. Be direct, structured, accurate. Use bullet points for lists, headers for long answers."""

CODE_KEYWORDS = {
    # General coding triggers
    "write code", "write a code", "write program", "write script",
    "code karo", "code banao", "bana do code", "likho code",
    "write a function", "write a class", "write a script",
    "write a", "create a", "build a", "make a", "implement",
    "code of", "code for", "example of", "show me code",
    "show me", "give me code", "give code", "deta code",

    # Algorithms and data structures
    "fibonacci", "factorial", "palindrome", "linked list",
    "bubble sort", "merge sort", "quick sort", "binary search",
    "dynamic programming", "dfs", "bfs", "leetcode", "hackerrank",
    "binary tree", "avl tree", "heap", "trie", "graph traversal",
    "dijkstra", "knapsack", "matrix", "recursion", "backtracking",
    "sliding window", "two pointer", "memoization",

    # Web and backend
    "rest api", "restful api", "crud api", "graphql api",
    "flask api", "fastapi", "django", "express server",
    "spring boot", "node server", "websocket", "middleware",
    "authentication", "jwt", "oauth", "login system",
    "microservice", "docker", "kubernetes",

    # Frontend
    "react component", "vue component", "angular component",
    "html page", "css style", "landing page", "navbar",
    "login form", "signup form", "dashboard ui",
    "responsive design", "tailwind", "bootstrap",

    # Database
    "sql query", "mongodb query", "database schema",
    "orm model", "migration", "crud operation",
    "join query", "aggregation", "index",

    # DevOps and tools
    "dockerfile", "docker compose", "github actions",
    "ci cd pipeline", "nginx config", "bash script",
    "cron job", "makefile", "terraform",

    # AI and ML
    "machine learning code", "neural network code", "deep learning code",
    "train model code", "linear regression code", "classification code",
    "clustering code", "nlp code", "chatbot code", "recommendation system code",

    # Games
    "snake game", "tic tac toe", "chess", "tetris",
    "todo app", "calculator app", "weather app", "chat app",

    # Language specific
    "python code", "java code", "javascript code", "cpp code",
    "c++ code", "golang code", "rust code", "kotlin code",
    "typescript code", "swift code", "php code", "ruby code",
    "scala code", "dart code", "r code", "matlab code",

    # New tech
    "blockchain", "smart contract", "solidity", "web3",
    "llm", "langchain", "openai api", "huggingface",
    "pytorch", "tensorflow", "keras", "numpy", "pandas",
    "data visualization", "matplotlib", "seaborn",
    "api integration", "webhook", "rate limiting", "caching",
    "redis", "celery", "kafka", "rabbitmq",
}

NON_PYTHON_LANGUAGES = {
    "java", "javascript", "js", "typescript", "ts", "c++", "cpp", "c#",
    "csharp", "rust", "go", "golang", "kotlin", "swift", "php", "ruby",
    "scala", "dart", "flutter", "react", "vue", "angular", "html", "css",
    "sql", "bash", "shell", "r language", "matlab", "perl", "haskell"
}

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
]

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

conversation_store = {}


def _build_groq(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


def _build_gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )


def _build_openrouter(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=0.1,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
    )


def is_code_request(message: str) -> bool:
    return any(kw in message.lower() for kw in CODE_KEYWORDS)


def is_python_executable(message: str) -> bool:
    return not any(lang in message.lower() for lang in NON_PYTHON_LANGUAGES)


def detect_language(message: str) -> str:
    msg = message.lower()
    for lang in NON_PYTHON_LANGUAGES:
        if lang in msg:
            return lang
    return "python"


def extract_code_block(text: str) -> str:
    match = re.search(r"```(?:\w+)?\n?([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    lines = text.strip().splitlines()
    code_lines = [
        l for l in lines
        if not l.strip().startswith((
            "Here", "This", "The ", "Sure", "I ", "Let",
            "Below", "Above", "Note", "You", "We", "To "
        ))
    ]
    return "\n".join(code_lines).strip()


def format_code_response(raw: str, lang: str, execution_result: str = None) -> str:
    match = re.search(r"```(?:\w+)?\n?([\s\S]*?)```", raw)
    code = match.group(1).strip() if match else extract_code_block(raw)
    code_block = f"```{lang}\n{code}\n```"
    if execution_result:
        return f"{code_block}\n\n**Output:**\n```\n{execution_result}\n```"
    return code_block


def handle_python_request(user_message: str, llm) -> str:
    prompt = (
        f"Write clean, complete, working Python code for: {user_message}\n\n"
        "- Return ONLY code inside a ```python block\n"
        "- Include print() statements to show output\n"
        "- Handle edge cases\n"
        "- No text outside the code block"
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content
    code = extract_code_block(raw)

    if not code:
        return raw

    result = run_python_code.invoke({"code": code})

    if "Error:" in result:
        fix_prompt = (
            f"Fix this Python code:\n```python\n{code}\n```\n"
            f"Error: {result}\n"
            "Return ONLY the corrected ```python block."
        )
        fixed_raw = llm.invoke([HumanMessage(content=fix_prompt)]).content
        fixed_code = extract_code_block(fixed_raw)
        if fixed_code:
            result = run_python_code.invoke({"code": fixed_code})
            return format_code_response(fixed_raw, "python", result)

    return format_code_response(raw, "python", result)


def handle_other_language_request(user_message: str, lang: str, llm) -> str:
    prompt = (
        f"Write clean, production-quality {lang} code for: {user_message}\n"
        f"- Return code inside a ```{lang} block\n"
        "- Follow best practices for this language\n"
        "- Brief inline comments for complex logic\n"
        "- After code: 2 lines on what it does and how to run it"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


def handle_code_request(user_message: str, llm) -> str:
    if is_python_executable(user_message):
        return handle_python_request(user_message, llm)
    return handle_other_language_request(user_message, detect_language(user_message), llm)


def build_messages(session_id: str) -> list:
    history = conversation_store[session_id][-4:]
    trimmed = []
    for msg in history:
        content = msg.content[:400] + "..." if len(msg.content) > 400 else msg.content
        trimmed.append(msg.__class__(content=content))
    return [SystemMessage(content=SYSTEM_PROMPT)] + trimmed


REALTIME_KEYWORDS = {
    "news", "latest", "current", "today", "yesterday", "recent",
    "price", "stock", "weather", "score", "who is", "what happened",
    "when did", "where is", "result", "election", "match", "exam",
    "update", "announce", "launched", "released", "died", "arrested",
    "won", "lost", "rupee", "dollar", "usd", "inr", "convert",
    "calculate", "percent", "temperature", "humidity", "rate",
}

def needs_tools(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in REALTIME_KEYWORDS)

def run_agent(session_id: str, user_message: str) -> str:
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append(HumanMessage(content=user_message))

    all_providers = []
    if os.getenv("GROQ_API_KEY"):
        for m in GROQ_MODELS:
            all_providers.append(("groq", m))
    if os.getenv("GEMINI_API_KEY"):
        all_providers.append(("gemini", None))
    if os.getenv("OPENROUTER_API_KEY"):
        for m in OPENROUTER_MODELS:
            all_providers.append(("openrouter", m))

    last_error = None

    for provider, model in all_providers:
        try:
            if provider == "groq":
                llm = _build_groq(model)
            elif provider == "gemini":
                llm = _build_gemini()
            else:
                llm = _build_openrouter(model)

            if is_code_request(user_message):
                response = handle_code_request(user_message, llm)
            elif needs_tools(user_message):
                agent = create_react_agent(llm, get_all_tools())
                result = agent.invoke({"messages": build_messages(session_id)})
                response = result["messages"][-1].content
            else:
                response = llm.invoke(build_messages(session_id)).content

            conversation_store[session_id].append(AIMessage(content=response))
            return response

        except Exception as e:
            last_error = str(e)
            continue

    response = "All providers unavailable. Please try again in a few minutes."
    conversation_store[session_id].append(AIMessage(content=response))
    return response