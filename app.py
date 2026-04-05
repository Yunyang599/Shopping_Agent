"""
Shopping Agent — Web App
========================
Run with:  streamlit run app.py
"""

import os
import re
import json
import requests
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY    = os.getenv("SERPAPI_KEY")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(
    page_title="Shopping Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.stAppDeployButton { display: none; }
[data-testid="manage-app-button"] { display: none; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
iframe[title="streamlit_cloud_sharelink"] { display: none !important; }

.stApp { background: #f0f2f6; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 48px 20px 8px;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}
.hero p { color: #666; font-size: 1.05rem; margin: 0; }

/* ── Search bar ── */
.stTextInput input {
    border-radius: 50px !important;
    padding: 14px 24px !important;
    font-size: 1rem !important;
    border: 2px solid #ddd !important;
    background: white !important;
    color: #1a1a2e !important;
    caret-color: #667eea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
.stTextInput input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15) !important;
}

/* ── Search button ── */
div[data-testid="stButton"] > button {
    border-radius: 50px !important;
    padding: 12px 0 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4) !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

/* ── Product card ── */
.product-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.2s, box-shadow 0.2s;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
}
.product-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.13);
}
.product-img-wrap {
    width: 100%;
    height: 190px;
    background: #f8f8f8;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.product-img-wrap img {
    max-width: 100%;
    max-height: 180px;
    object-fit: contain;
    padding: 8px;
}
.product-img-placeholder {
    width: 100%;
    height: 190px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4rem;
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}
.product-body { padding: 14px 16px 16px; flex: 1; display: flex; flex-direction: column; }
.product-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #222;
    margin-bottom: 10px;
    line-height: 1.4;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    flex: 1;
}
.product-price {
    font-size: 1.45rem;
    font-weight: 800;
    color: #e53935;
    margin-bottom: 6px;
}
.product-rating { font-size: 0.82rem; color: #777; margin-bottom: 12px; }
.stars { color: #f59e0b; letter-spacing: 1px; }
.product-store { font-size: 0.78rem; color: #999; margin-bottom: 10px; }

.btn-view {
    display: block;
    text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    padding: 9px 0;
    border-radius: 10px;
    text-decoration: none !important;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: auto;
    transition: opacity 0.2s;
}
.btn-view:hover { opacity: 0.85; }
.btn-nolink {
    display: block;
    text-align: center;
    background: #e8e8e8;
    color: #aaa !important;
    padding: 9px 0;
    border-radius: 10px;
    font-size: 0.85rem;
    margin-top: auto;
}

/* ── AI Panel ── */
.ai-panel {
    background: white;
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-left: 5px solid #667eea;
    margin-top: 8px;
    white-space: pre-wrap;
    line-height: 1.7;
    color: #333;
    font-size: 0.95rem;
}

/* ── Spinner text ── */
[data-testid="stSpinner"] p, .stSpinner p { color: #333 !important; }

/* ── Section label ── */
.section-label {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1a1a2e;
    margin: 28px 0 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Auth check ───────────────────────────────────────────────────────────────
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛒 Shopping Agent</h1>
    <p>Describe what you want — AI finds, compares, and recommends the best options</p>
</div>
""", unsafe_allow_html=True)

# ── Search bar ────────────────────────────────────────────────────────────────
_, mid, _ = st.columns([1, 5, 1])
with mid:
    query  = st.text_input("q", placeholder="e.g.  wireless headphones under $80  ·  gaming mouse  ·  standing desk",
                           label_visibility="collapsed")
    search = st.button("🔍  Search", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Core functions ────────────────────────────────────────────────────────────

def parse_query(user_input: str) -> dict:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a shopping assistant. Extract structured search parameters "
                    "from the user's query and return JSON with: "
                    "search_query, product_type, budget (string or null), "
                    "key_features (list of strings), use_case (string)."
                ),
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"search_query": user_input, "product_type": user_input,
                "budget": None, "key_features": [], "use_case": "general use"}


def fetch_live_products(search_query: str, max_results: int = 8) -> list[dict]:
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
    except Exception:
        return []

    products = []
    for item in data.get("shopping_results", [])[:max_results]:
        raw_source = item.get("source", "")
        clean_source = re.sub(r"<[^>]+>", "", raw_source).strip()
        products.append({
            "title":     item.get("title", "Unknown"),
            "price":     item.get("price", "N/A"),
            "source":    clean_source,
            "rating":    item.get("rating"),
            "reviews":   item.get("reviews"),
            "link":      item.get("link") or item.get("product_link") or "",
            "thumbnail": item.get("thumbnail", ""),
        })
    return products


def parse_price(price_str) -> float | None:
    """Extract a numeric value from a price string like '$1,295.00'."""
    if not price_str or price_str == "N/A":
        return None
    nums = re.findall(r"[\d]+\.?\d*", price_str.replace(",", ""))
    return float(nums[0]) if nums else None


def filter_by_budget(products: list[dict], budget_str: str | None) -> list[dict]:
    """Remove products that exceed the stated budget."""
    if not budget_str:
        return products
    nums = re.findall(r"[\d]+\.?\d*", budget_str.replace(",", ""))
    if not nums:
        return products
    max_price = float(nums[-1])  # use last number (e.g. "under 1500" → 1500)
    return [p for p in products if parse_price(p.get("price")) is None
            or parse_price(p.get("price")) <= max_price]


def fetch_ai_products(query_info: dict, original_query: str = "") -> list[dict]:
    search_q = query_info.get('search_query') or original_query
    product_type = query_info.get('product_type') or original_query
    prompt = (
        f"The user is shopping for: '{search_q}'.\n"
        f"Product type: {product_type}\n"
        f"Use case: {query_info.get('use_case')}.\n"
        f"Budget: {query_info.get('budget') or 'not specified'}.\n"
        f"Desired features: {', '.join(query_info.get('key_features', []))}.\n\n"
        f"List 6 specific real products that are EXACTLY '{search_q}' or directly related to '{search_q}'. "
        "Do NOT suggest unrelated products. "
        "Return a JSON object with a 'products' array. "
        "Each item must have: title, price (string with currency symbol), "
        "source (retailer name), rating (float 1-5 or null), reviews (integer or null), "
        "link (a real product page URL if you know it, otherwise empty string), "
        "thumbnail (empty string)."
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledgeable shopping expert. Return only valid JSON."},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("products", [])
    except Exception:
        return []


def compare_products(products: list[dict], query_info: dict) -> str:
    summary = "\n".join(
        f"{i}. {p.get('title')} — {p.get('price')} "
        f"(Rating: {p.get('rating') or 'N/A'}, Reviews: {p.get('reviews') or 'N/A'}, Store: {p.get('source')})"
        for i, p in enumerate(products, 1)
    )
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert, unbiased shopping advisor. Be concise, helpful, and honest.",
            },
            {
                "role": "user",
                "content": (
                    f"User wants: {query_info.get('product_type')} for {query_info.get('use_case')}.\n"
                    f"Budget: {query_info.get('budget') or 'not specified'}.\n"
                    f"Key features: {', '.join(query_info.get('key_features', []))}.\n\n"
                    f"Products:\n{summary}\n\n"
                    "1. Short pro/con for each product (1–2 sentences).\n"
                    "2. Your top pick with a clear reason.\n"
                    "3. Any useful buying tips or things to watch out for."
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

PLACEHOLDERS = ["🎧", "💻", "📱", "🖥️", "🎮", "⌨️", "🖱️", "📦"]

def render_stars(rating) -> str:
    if not isinstance(rating, (int, float)):
        return ""
    full  = int(rating)
    half  = "½" if (rating - full) >= 0.5 else ""
    empty = 5 - full - len(half)
    return "★" * full + half + "☆" * empty


def product_card(p: dict, idx: int) -> str:
    title   = p.get("title", "Unknown Product")
    price   = p.get("price", "N/A")
    source  = p.get("source", "")
    rating  = p.get("rating")
    reviews = p.get("reviews")
    link    = p.get("link", "").strip()
    thumb   = p.get("thumbnail", "").strip()

    # Image
    if thumb:
        image_html = (
            f'<div class="product-img-wrap">'
            f'<img src="{thumb}" alt="{title}" onerror="this.parentElement.innerHTML=\'📦\'">'
            f'</div>'
        )
    else:
        emoji = PLACEHOLDERS[idx % len(PLACEHOLDERS)]
        image_html = f'<div class="product-img-placeholder">{emoji}</div>'

    # Rating row
    rating_html = ""
    if rating:
        stars_str = render_stars(rating)
        rev_str   = f" ({reviews:,})" if isinstance(reviews, int) else (f" ({reviews})" if reviews else "")
        rating_html = f'<div class="product-rating"><span class="stars">{stars_str}</span> {rating:.1f}{rev_str}</div>'

    # Store
    store_html = f'<div class="product-store">🏪 {source}</div>' if source else ""

    # Button
    btn = (
        f'<a href="{link}" target="_blank" class="btn-view">View Product →</a>'
        if link
        else '<span class="btn-nolink">No link available</span>'
    )

    return f"""
    <div class="product-card">
        {image_html}
        <div class="product-body">
            <div class="product-title">{title}</div>
            <div class="product-price">{price}</div>
            {rating_html}
            {store_html}
            {btn}
        </div>
    </div>
    """


# ── Main flow ─────────────────────────────────────────────────────────────────

if search and query:
    # 1 — Parse
    with st.spinner("Analyzing your request…"):
        query_info = parse_query(query)

    label = query_info.get("search_query") or query
    st.markdown(f'<div class="section-label">Results for: <em>{label}</em></div>', unsafe_allow_html=True)

    chips = []
    if query_info.get("budget"):
        chips.append(f"💰 Budget: {query_info['budget']}")
    for f in query_info.get("key_features", [])[:4]:
        chips.append(f"✓ {f}")
    if chips:
        st.caption("  ·  ".join(chips))

    # 2 — Fetch products
    products: list[dict] = []
    if SERPAPI_KEY:
        with st.spinner("Fetching live products from Google Shopping…"):
            products = fetch_live_products(label)
        products = filter_by_budget(products, query_info.get("budget"))
        if not products:
            st.info("Live search returned no results — switching to AI suggestions.")

    if not products:
        with st.spinner("Generating product suggestions with AI…"):
            products = fetch_ai_products(query_info, query)
        products = filter_by_budget(products, query_info.get("budget"))

    if not products:
        st.error("No products found. Try a different description.")
        st.stop()

    # 3 — Product grid (4 columns)
    COLS = 4
    cols = st.columns(COLS)
    for i, prod in enumerate(products):
        with cols[i % COLS]:
            st.markdown(product_card(prod, i), unsafe_allow_html=True)

    # 4 — AI comparison
    st.markdown('<div class="section-label">🤖 AI Analysis & Recommendation</div>', unsafe_allow_html=True)
    with st.spinner("Comparing products…"):
        analysis = compare_products(products, query_info)

    st.markdown(f'<div class="ai-panel">{analysis}</div>', unsafe_allow_html=True)

elif search and not query:
    st.warning("Please enter what you're looking for.")
