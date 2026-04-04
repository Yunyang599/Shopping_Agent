"""
Shopping Agent
==============
Describe what you want to buy, and the agent finds real products,
shows them in a table, and uses AI to compare and recommend the best one.

Requirements:
  - OPENAI_API_KEY  (required)
  - SERPAPI_KEY     (optional — enables live Google Shopping results)

Usage:
  python shopping_agent.py
"""

import os
import json
import sys
import requests
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from dotenv import load_dotenv

load_dotenv()

console = Console()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    console.print("[bold red]Error:[/bold red] OPENAI_API_KEY is not set.")
    console.print("Create a [cyan].env[/cyan] file from [cyan].env.example[/cyan] and add your key.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Step 1 — Parse the user query with LLM
# ---------------------------------------------------------------------------

def parse_query(user_input: str) -> dict:
    """Extract structured search parameters from a natural-language shopping query."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a shopping assistant. "
                    "Extract structured information from the user's query and return JSON with these fields:\n"
                    "  search_query  – an optimized search string for Google Shopping\n"
                    "  product_type  – short label for the kind of product\n"
                    "  budget        – price limit as a string, or null\n"
                    "  key_features  – list of important features the user mentioned\n"
                    "  use_case      – what the user will use this product for"
                ),
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"search_query": user_input, "product_type": user_input,
                "budget": None, "key_features": [], "use_case": "general use"}


# ---------------------------------------------------------------------------
# Step 2a — Fetch live products via SerpAPI (Google Shopping)
# ---------------------------------------------------------------------------

def fetch_live_products(search_query: str, max_results: int = 8) -> list[dict]:
    """Return up to max_results products from Google Shopping (requires SERPAPI_KEY)."""
    params = {
        "engine":  "google_shopping",
        "q":       search_query,
        "api_key": SERPAPI_KEY,
        "num":     max_results,
    }
    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        console.print(f"[yellow]SerpAPI request failed:[/yellow] {exc}")
        return []

    products = []
    for item in data.get("shopping_results", [])[:max_results]:
        products.append({
            "title":   item.get("title", "Unknown"),
            "price":   item.get("price", "N/A"),
            "source":  item.get("source", "Unknown"),
            "rating":  item.get("rating"),
            "reviews": item.get("reviews"),
            "link":    item.get("link", ""),
        })
    return products


# ---------------------------------------------------------------------------
# Step 2b — Fallback: ask the LLM to suggest products from its knowledge
# ---------------------------------------------------------------------------

def fetch_ai_products(query_info: dict) -> list[dict]:
    """Ask the LLM to suggest realistic products when live search is unavailable."""
    prompt = (
        f"The user wants: {query_info.get('product_type', 'a product')}.\n"
        f"Use case: {query_info.get('use_case', 'general use')}.\n"
        f"Budget: {query_info.get('budget') or 'not specified'}.\n"
        f"Important features: {', '.join(query_info.get('key_features', [])) or 'none specified'}.\n\n"
        "Suggest 5–6 specific, real products that are currently sold. "
        "Return a JSON object with a 'products' array. "
        "Each product must have: title, price (string with currency symbol), "
        "source (retailer name), rating (float 1–5 or null), reviews (integer or null)."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable shopping expert. Return only valid JSON."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return data.get("products", [])
    except (json.JSONDecodeError, AttributeError):
        return []


# ---------------------------------------------------------------------------
# Step 3 — Display products in a rich table
# ---------------------------------------------------------------------------

def display_products(products: list[dict]) -> None:
    table = Table(
        title="[bold]Products Found[/bold]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="blue",
        show_lines=True,
    )
    table.add_column("#",       style="dim",        width=3,  justify="right")
    table.add_column("Product", style="bold white",  width=42)
    table.add_column("Price",   style="bold green",  width=10)
    table.add_column("Rating",  style="yellow",      width=10)
    table.add_column("Reviews", style="cyan",        width=9,  justify="right")
    table.add_column("Store",   style="magenta",     width=16)

    for i, p in enumerate(products, 1):
        title = p.get("title", "")
        title_display = (title[:40] + "…") if len(title) > 41 else title

        rating  = p.get("rating")
        reviews = p.get("reviews")

        rating_str  = f"⭐ {rating:.1f}" if isinstance(rating, (int, float)) else (str(rating) if rating else "—")
        reviews_str = f"{reviews:,}"     if isinstance(reviews, int)         else (str(reviews) if reviews else "—")

        source = p.get("source", "")
        source_display = (source[:14] + "…") if len(source) > 15 else source

        table.add_row(
            str(i),
            title_display,
            p.get("price", "N/A"),
            rating_str,
            reviews_str,
            source_display,
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Step 4 — LLM comparison and recommendation
# ---------------------------------------------------------------------------

def compare_products(products: list[dict], query_info: dict) -> str:
    """Ask the LLM to compare the products and recommend the best one."""
    products_summary = "\n".join(
        f"{i}. {p.get('title')} — {p.get('price')} "
        f"(Rating: {p.get('rating') or 'N/A'}, Reviews: {p.get('reviews') or 'N/A'}, Store: {p.get('source')})"
        for i, p in enumerate(products, 1)
    )

    prompt = (
        f"The user is looking for: {query_info.get('product_type', 'a product')}\n"
        f"Use case: {query_info.get('use_case', 'general use')}\n"
        f"Budget: {query_info.get('budget') or 'not specified'}\n"
        f"Desired features: {', '.join(query_info.get('key_features', [])) or 'none specified'}\n\n"
        f"Products:\n{products_summary}\n\n"
        "Please:\n"
        "1. Give a short pro/con summary for each product (1–2 sentences each).\n"
        "2. Name your top pick and explain why in 2–3 sentences.\n"
        "3. Add any buying tips or things to watch out for."
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert, unbiased shopping advisor. "
                    "Be concise, honest, and helpful. Use plain text with clear section labels."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run():
    console.print(
        Panel.fit(
            "[bold cyan]Shopping Agent[/bold cyan]\n"
            "[dim]Describe what you want — the agent finds, compares, and recommends.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )

    if SERPAPI_KEY:
        console.print("[dim]Live Google Shopping enabled (SerpAPI)[/dim]")
    else:
        console.print(
            "[dim yellow]No SERPAPI_KEY found — product suggestions will come from AI knowledge.[/dim yellow]"
        )

    while True:
        console.print("\n[bold]What would you like to buy?[/bold] [dim](type 'quit' to exit)[/dim]")
        try:
            user_input = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        # ── 1. Parse query ──────────────────────────────────────────────────
        with console.status("[cyan]Analyzing your request…[/cyan]", spinner="dots"):
            query_info = parse_query(user_input)

        search_label = query_info.get("search_query") or user_input
        console.print(f"\n[bold]Search:[/bold] [cyan]{search_label}[/cyan]")
        if query_info.get("budget"):
            console.print(f"[bold]Budget:[/bold] [green]{query_info['budget']}[/green]")
        if query_info.get("key_features"):
            console.print(f"[bold]Features:[/bold] {', '.join(query_info['key_features'])}")

        # ── 2. Fetch products ───────────────────────────────────────────────
        products: list[dict] = []

        if SERPAPI_KEY:
            with console.status("[cyan]Fetching live products from Google Shopping…[/cyan]", spinner="dots"):
                products = fetch_live_products(search_label)

        if not products:
            if SERPAPI_KEY:
                console.print("[yellow]Live search returned no results — falling back to AI suggestions.[/yellow]")
            with console.status("[cyan]Generating product suggestions…[/cyan]", spinner="dots"):
                products = fetch_ai_products(query_info)

        if not products:
            console.print("[red]Could not find any products. Please try a different description.[/red]")
            continue

        # ── 3. Display table ────────────────────────────────────────────────
        display_products(products)

        # ── 4. AI comparison ────────────────────────────────────────────────
        with console.status("[cyan]Comparing products with AI…[/cyan]", spinner="dots"):
            recommendation = compare_products(products, query_info)

        console.print(
            Panel(
                recommendation,
                title="[bold green]AI Analysis & Recommendation[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    run()
