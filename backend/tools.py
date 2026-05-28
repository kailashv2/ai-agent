from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from datetime import datetime
from io import StringIO
import httpx
import pytz
import sys
import math
import json
import re
import datetime as dt
import os


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


@tool
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Returns current date and time for a given timezone like Asia/Kolkata or America/New_York."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return f"Current date and time in {timezone}: {now.strftime('%A, %d %B %Y %I:%M %p')}"
    except Exception:
        return f"Current date and time: {datetime.now().strftime('%A, %d %B %Y %I:%M %p')}"


@tool
def calculate(expression: str) -> str:
    """Evaluates math expressions. Examples: 2+2, sqrt(144), 15% of 50000, 2**10"""
    try:
        if "% of" in expression.lower():
            parts = expression.lower().split("% of")
            expression = f"{parts[0].strip()} / 100 * {parts[1].strip()}"
        safe_dict = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"Result: {result:,}" if isinstance(result, (int, float)) else f"Result: {result}"
    except Exception as e:
        return f"Could not calculate: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Gets current weather for any city like Mumbai, Delhi, London, New York."""
    try:
        response = httpx.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        data = response.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        city_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        return (
            f"Weather in {city_name}, {country}:\n"
            f"Temperature: {current['temp_C']}C (feels like {current['FeelsLikeC']}C)\n"
            f"Condition: {current['weatherDesc'][0]['value']}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind: {current['windspeedKmph']} km/h"
        )
    except Exception:
        return f"Could not fetch weather for {city}. Try a different city name."


@tool
def run_python_code(code: str) -> str:
    """Executes Python code and returns real output. Use for ANY coding request, algorithms, data processing."""
    capture = StringIO()
    sys.stdout = capture
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
                "input": lambda _: "",
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "StopIteration": StopIteration,
                "Exception": Exception,
                "BaseException": BaseException,
                "NotImplementedError": NotImplementedError,
                "ZeroDivisionError": ZeroDivisionError,
                "NameError": NameError,
                "RuntimeError": RuntimeError,
            },
            "math": math,
            "json": json,
            "re": re,
            "datetime": dt,
        }
        exec(code, safe_globals)
        output = capture.getvalue()
        return f"Output:\n{output}" if output.strip() else "Code ran successfully with no output."
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = sys.__stdout__


@tool
def convert_currency(amount_and_currencies: str) -> str:
    """Converts currency. Format: 100 USD to INR or 50 EUR to GBP. Supports all major currencies."""
    try:
        parts = amount_and_currencies.lower().split()
        amount = float(parts[0])
        from_cur = parts[1].upper()
        to_cur = parts[3].upper()
        data = httpx.get(f"https://api.exchangerate-api.com/v4/latest/{from_cur}", timeout=10).json()
        rate = data["rates"][to_cur]
        return f"{amount:,.2f} {from_cur} = {amount * rate:,.2f} {to_cur}\n(Rate: 1 {from_cur} = {rate:.4f} {to_cur})"
    except Exception:
        return "Could not convert currency. Use format: 100 USD to INR"


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