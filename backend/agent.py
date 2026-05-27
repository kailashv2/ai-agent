from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from tools import get_all_tools
from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """You are an expert AI agent with access to powerful tools. Today's date: use get_current_time tool if needed.

YOUR TOOLS:
- search_tool (news): Real-time news search — use for current events, recent news, today's updates
- general_search_tool: General web search — use for facts, people, places, how-to questions  
- get_weather: Live weather for any city
- calculate: Math, percentages, formulas
- run_python_code: Execute Python code for complex tasks
- convert_currency: Live currency conversion

STRICT RULES:
1. For ANY current/recent information → ALWAYS use search tools. Never answer from memory.
2. Use SPECIFIC search queries. Bad: "paper leak". Good: "AKTU CN paper leak May 2026 rescheduled date".
3. If first search gives irrelevant results → search again with different keywords.
4. Only report what search results actually say. Never mix unrelated results.
5. Always mention source/date when reporting news.
6. For weather → always use get_weather tool, never guess.
7. For math → always use calculate tool.
8. For ANY coding request like "write code", "create program", "make script", "code karo", "program banao" → ALWAYS use run_python_code tool to actually execute it and show real output. Never just explain or show code without running it first.
9. If genuinely no results found after 2 searches → say so honestly and suggest official sources.
10. Be concise, structured, and accurate. Use bullet points for lists.
11. MEMORY: You have full conversation history. Always refer to previous messages when user says "what about", "and that", "tell me more", "what do you think" — these refer to the previous topic, not a new one.

IMPORTANT: When user says "write code", "code karo", "program banao", "create a script" → run_python_code tool is MANDATORY. No exceptions.
IMPORTANT: When user asks a follow-up like "what do you think", "what about you", "tell me more" → it refers to the PREVIOUS topic in conversation. Never treat it as a new unrelated question.

You are reliable, fast, and always use the right tool for the job."""

conversation_store = {}

def run_agent(session_id: str, user_message: str) -> str:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    tools = get_all_tools()
    agent = create_react_agent(llm, tools)

    if session_id not in conversation_store:
        conversation_store[session_id] = []

    conversation_store[session_id].append(
        HumanMessage(content=user_message)
    )

    # Keep last 20 messages for better memory
    history = conversation_store[session_id][-20:]
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    try:
        result = agent.invoke({"messages": messages})
        response = result["messages"][-1].content
    except Exception as e:
        response = f"I encountered an error while processing your request. Please try again. (Error: {str(e)})"

    conversation_store[session_id].append(
        AIMessage(content=response)
    )

    return response