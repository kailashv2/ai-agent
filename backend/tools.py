from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from datetime import datetime
import httpx
import pytz
import os

# --- Search Tools ---
def get_search_tool():
    return TavilySearch(
        max_results=8,
        topic="news",
        days=7,
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )

def get_general_search_tool():
    return TavilySearch(
        max_results=8,
        topic="general",
        tavily_api_key=os.getenv("TAVILY_API_KEY")
    )

# --- Time Tool ---
@tool
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Returns current date and time. Optionally pass a timezone like 'America/New_York' or 'Asia/Kolkata'."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return f"Current date and time in {timezone}: {now.strftime('%A, %d %B %Y %I:%M %p')}"
    except Exception:
        now = datetime.now()
        return f"Current date and time: {now.strftime('%A, %d %B %Y %I:%M %p')}"

# --- Calculator Tool ---
@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Examples: '2 + 2', '15% of 50000', 'sqrt(144)', '2 ** 10'
    For percentages like '15% of 50000', convert to '0.15 * 50000'.
    """
    try:
        import math
        expression = expression.replace('%', '/100*') if 'of' not in expression.lower() else expression
        if '% of' in expression.lower():
            parts = expression.lower().split('% of')
            expression = f"{parts[0].strip()} / 100 * {parts[1].strip()}"
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith('_')}
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result:,}" if isinstance(result, (int, float)) else f"Result: {result}"
    except Exception as e:
        return f"Could not calculate: {str(e)}"

# --- Weather Tool ---
@tool
def get_weather(city: str) -> str:
    """Gets current weather for any city. Example: 'Mumbai', 'Delhi', 'London'."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = httpx.get(url, timeout=10)
        data = response.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        desc = current["weatherDesc"][0]["value"]
        humidity = current["humidity"]
        wind = current["windspeedKmph"]
        return (
            f"Weather in {city_name}, {country}:\n"
            f"🌡️ Temperature: {temp_c}°C (feels like {feels_like}°C)\n"
            f"☁️ Condition: {desc}\n"
            f"💧 Humidity: {humidity}%\n"
            f"💨 Wind: {wind} km/h"
        )
    except Exception as e:
        return f"Could not fetch weather for {city}. Try a different city name."

# --- Code Execution Tool ---
@tool
def run_python_code(code: str) -> str:
    """
    Executes Python code and returns the output.
    Use for calculations, data processing, algorithms, or any Python task.
    Only use safe, non-destructive code.
    """
    import sys
    from io import StringIO
    import math, json, re, datetime as dt

    stdout_capture = StringIO()
    sys.stdout = stdout_capture

    try:
        safe_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len,
                "int": int, "float": float, "str": str, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "bool": bool,
                "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "enumerate": enumerate, "zip": zip,
                "sorted": sorted, "reversed": reversed, "map": map,
                "filter": filter, "isinstance": isinstance, "type": type,
            },
            "math": math,
            "json": json,
            "re": re,
            "datetime": dt,
        }
        exec(code, safe_globals)
        output = stdout_capture.getvalue()
        return f"Output:\n{output}" if output else "Code executed successfully (no output)."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__

# --- Currency Tool ---
@tool
def convert_currency(amount_and_currencies: str) -> str:
    """
    Converts currency. Input format: '100 USD to INR' or '50 EUR to GBP'.
    Supports: USD, INR, EUR, GBP, JPY, AUD, CAD, SGD, AED, etc.
    """
    try:
        parts = amount_and_currencies.lower().split()
        amount = float(parts[0])
        from_cur = parts[1].upper()
        to_cur = parts[3].upper()
        url = f"https://api.exchangerate-api.com/v4/latest/{from_cur}"
        response = httpx.get(url, timeout=10)
        data = response.json()
        rate = data["rates"][to_cur]
        result = amount * rate
        return f"{amount:,.2f} {from_cur} = {result:,.2f} {to_cur}\n(Rate: 1 {from_cur} = {rate:.4f} {to_cur})"
    except Exception as e:
        return f"Could not convert currency. Try format: '100 USD to INR'"

def get_all_tools():
    return [
        get_search_tool(),
        get_general_search_tool(),
        get_current_time,
        calculate,
        get_weather,
        run_python_code,
        convert_currency,
    ]