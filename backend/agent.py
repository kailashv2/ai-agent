from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import get_all_tools, run_python_code
from dotenv import load_dotenv
import re
import os

load_dotenv()

SYSTEM_PROMPT = """You are a world-class AI agent — expert in coding, research, analysis, math, science, history, finance, law basics, medicine basics, current events, creative writing, and more.

TOOLS AVAILABLE:
- search_tool: Breaking news, current events, today's updates
- general_search_tool: Facts, people, places, companies, how-to, explanations
- get_weather: Live weather for any city worldwide
- calculate: Math, percentages, unit conversions, formulas
- run_python_code: Execute Python — algorithms, data processing, simulations
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
    "write code", "code", "program", "script", "fibonacci", "algorithm",
    "function", "implement", "banao", "likho", "code karo", "bana do",
    "factorial", "prime", "palindrome", "reverse", "linked list", "binary",
    "recursion", "loop", "array", "matrix", "pattern", "series", "sorting",
    "searching", "class", "object", "inheritance", "api", "fetch", "crud",
    "rest", "regex", "parse", "encrypt", "decrypt", "hash", "stack", "queue",
    "tree", "graph", "dynamic programming", "greedy", "backtracking", "thread",
    "async", "decorator", "generator", "iterator", "lambda", "closure",
    "interview question", "data structure", "leetcode", "hackerrank",
    "bubble sort", "merge sort", "quick sort", "binary search", "dfs", "bfs",
    "number system", "calculator", "todo", "snake game", "tic tac toe"
}

NON_PYTHON_LANGUAGES = {
    "java", "javascript", "js", "typescript", "ts", "c++", "cpp", "c#",
    "csharp", "rust", "go", "golang", "kotlin", "swift", "php", "ruby",
    "scala", "dart", "flutter", "react", "vue", "angular", "html", "css",
    "sql", "bash", "shell", "r language", "matlab", "perl", "haskell"
}

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

OPENROUTER_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

conversation_store = {}


def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if groq_key:
        for model in GROQ_MODELS:
            try:
                return ChatGroq(model=model, temperature=0.1, groq_api_key=groq_key)
            except Exception:
                continue

    if gemini_key:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.1,
                google_api_key=gemini_key
            )
        except Exception:
            pass

    if openrouter_key:
        for model in OPENROUTER_MODELS:
            try:
                return ChatOpenAI(
                    model=model,
                    temperature=0.1,
                    openai_api_key=openrouter_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                )
            except Exception:
                continue

    raise RuntimeError("All LLM providers exhausted. Check your API keys in .env")


def is_code_request(message: str) -> bool:
    msg = message.lower()
    return any(kw in msg for kw in CODE_KEYWORDS)


def is_python_executable(message: str) -> bool:
    msg = message.lower()
    return not any(lang in msg for lang in NON_PYTHON_LANGUAGES)


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
            "Here", "This", "The ", "Sure", "I ", "Let", "Below",
            "Above", "Note", "You", "We", "To "
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
        "Requirements:\n"
        "- Return ONLY code inside a ```python block\n"
        "- Include print() statements to demonstrate output\n"
        "- Handle edge cases properly\n"
        "- No text outside the code block"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content
    code = extract_code_block(raw)

    if not code:
        return raw

    execution_result = run_python_code.invoke({"code": code})

    if "Error:" in execution_result:
        fix_prompt = (
            f"This Python code has an error:\n```python\n{code}\n```\n"
            f"Error: {execution_result}\n\n"
            "Fix it. Return ONLY the corrected ```python block. No explanations."
        )
        fixed = llm.invoke([HumanMessage(content=fix_prompt)])
        fixed_code = extract_code_block(fixed.content)
        if fixed_code:
            execution_result = run_python_code.invoke({"code": fixed_code})
            return format_code_response(fixed.content, "python", execution_result)

    return format_code_response(raw, "python", execution_result)


def handle_other_language_request(user_message: str, lang: str, llm) -> str:
    prompt = (
        f"Write clean, production-quality {lang} code for: {user_message}\n\n"
        "Requirements:\n"
        f"- Return code inside a ```{lang} block\n"
        "- Follow best practices and conventions for this language\n"
        "- Add brief inline comments for complex logic\n"
        "- After the code block, write 2-3 lines: what it does and how to compile/run it"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def handle_code_request(user_message: str, llm) -> str:
    if is_python_executable(user_message):
        return handle_python_request(user_message, llm)
    else:
        lang = detect_language(user_message)
        return handle_other_language_request(user_message, lang, llm)


def run_agent(session_id: str, user_message: str) -> str:
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append(HumanMessage(content=user_message))

    try:
        llm = get_llm()

        if is_code_request(user_message):
            response = handle_code_request(user_message, llm)
        else:
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + conversation_store[session_id][-20:]
            agent = create_react_agent(llm, get_all_tools())
            result = agent.invoke({"messages": messages})
            response = result["messages"][-1].content

    except RuntimeError:
        response = "All AI providers are currently unavailable. Please check your API keys."
    except Exception as e:
        response = f"Something went wrong: {str(e)}"

    conversation_store[session_id].append(AIMessage(content=response))
    return response