"""
backend.py - AgroMind Smart Backend (advanced)

Features:
- /recommend : advanced, weather-aware, language-aware crop recommendations (final outputs in user's language)
- /ask       : AI-like Q&A (local FAQ fuzzy match + optional OpenAI)
- /tips      : submit/list community tips (local JSON)
- /crops     : list crop DB
- /health    : status

Design goals:
- Farmer-friendly input: user can send location + simple controls (water availability, goal)
  (No need to know pH/moisture — the system infers reasonable defaults)
- Language autodetect & translation: if translation API configured, responses returned in user's language
- Weather-aware: if OpenWeatherMap API key provided, fetch forecast and factor into decisions (3-month projection heuristic)
- Optional OpenAI integration for richer /ask responses

Set environment variables (optional):
- OPENWEATHER_API_KEY=<your key>
- LIBRETRANSLATE_URL (e.g. https://libretranslate.de) and LIBRETRANSLATE_KEY (if needed)
- OPENAI_API_KEY=<your key>
- USD_TO_INR (overrides default conversion)

Run:
  pip install flask flask-cors requests langdetect python-dotenv
  # optional:
  pip install openai
  python backend.py
"""

import os
import json
import math
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# language detection
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agromind-smart-backend")

app = Flask(__name__)
CORS(app)

# ---------- Config ----------
USD_TO_INR = float(os.getenv("USD_TO_INR", "82.0"))
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.de")
LIBRETRANSLATE_KEY = os.getenv("LIBRETRANSLATE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY

# local tips store
TIPS_FILE = Path("community_tips.json")
if not TIPS_FILE.exists():
    TIPS_FILE.write_text("[]")

# ---------- Minimal crop DB (extendable) ----------
CROPS = [
    {"crop": "Wheat", "pH_range": (6.0,7.5), "moisture": (0.35,0.8), "temp": (5,30),
     "yield_kg_ha": 4000, "price_usd_per_ton": 300, "sustainability":90},
    {"crop": "Rice", "pH_range": (5.0,7.5), "moisture": (0.6,1.0), "temp": (20,35),
     "yield_kg_ha": 5000, "price_usd_per_ton": 400, "sustainability":70},
    {"crop": "Maize", "pH_range": (5.5,7.5), "moisture": (0.35,0.75), "temp": (18,32),
     "yield_kg_ha": 6000, "price_usd_per_ton": 230, "sustainability":85},
    {"crop": "Millet", "pH_range": (5.0,8.5), "moisture": (0.15,0.45), "temp": (20,35),
     "yield_kg_ha": 1500, "price_usd_per_ton": 500, "sustainability":98},
    {"crop": "Pulses", "pH_range": (5.0,7.5), "moisture": (0.2,0.6), "temp": (15,32),
     "yield_kg_ha": 1200, "price_usd_per_ton": 800, "sustainability":95},
    {"crop": "Potato", "pH_range": (5.0,6.5), "moisture": (0.45,0.85), "temp": (10,25),
     "yield_kg_ha": 20000, "price_usd_per_ton": 150, "sustainability":75},
    {"crop": "Tomato", "pH_range": (5.5,7.0), "moisture": (0.5,0.85), "temp": (18,30),
     "yield_kg_ha": 60000, "price_usd_per_ton": 500, "sustainability":72},
    # add more as required
]

COMPANIONS = {
    "Maize": ["Beans (N-fixing)", "Squash (ground cover)"],
    "Tomato": ["Basil", "Marigold"],
    "Potato": ["Beans", "Cabbage"],
    "Wheat": ["Clover (green manure)"],
    "Rice": ["Azolla (green manure)"],
}

# Simple FAQ
FAQ = {
    "how to raise ph": "Apply agricultural lime. Amount depends on soil texture; do a soil test.",
    "how to reduce ph": "Elemental sulfur lowers pH over time; follow small-scale trials and retest.",
    "water saving": "Use drip irrigation, mulch and schedule irrigation by growth stage.",
    "pest control": "IPM: monitoring, biological control, crop rotation and minimal safe pesticides as needed."
}

# ---------- Helpers ----------
def usd_to_inr(usd):
    return int(round(usd * USD_TO_INR))

def acres_to_hectares(acres):
    return acres * 0.404686

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def detect_language(text):
    if not text:
        return "en"
    if LANGDETECT_AVAILABLE:
        try:
            return detect(text)
        except Exception:
            return "en"
    # fallback to en
    return "en"

def translate_text(text, source="auto", target="en"):
    """
    Translate using LibreTranslate if available (configured), else return original.
    Expects LIBRETRANSLATE_URL env; no key required for publicly available instances.
    """
    if not LIBRETRANSLATE_URL:
        return text
    try:
        payload = {"q": text, "source": source, "target": target, "format": "text"}
        headers = {}
        if LIBRETRANSLATE_KEY:
            headers["Authorization"] = f"Bearer {LIBRETRANSLATE_KEY}"
        resp = requests.post(f"{LIBRETRANSLATE_URL}/translate", json=payload, headers=headers, timeout=8)
        if resp.ok:
            return resp.json().get("translatedText") or text
    except Exception as e:
        logger.warning("Translate failed: %s", e)
    return text

def fetch_weather_forecast(lat, lon):
    """
    If OPENWEATHER_API_KEY present, fetch 7-day forecast (One Call).
    We'll return a simple aggregated object: avg_temp_next_30_days, expected_rain_days (count), trend
    If no key, return None to indicate fallback.
    """
    if not OPENWEATHER_API_KEY:
        return None
    try:
        # One Call 3.0 or 2.5 older API may vary; try the free One Call endpoint
        url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly&units=metric&appid={OPENWEATHER_API_KEY}"
        resp = requests.get(url, timeout=8)
        if not resp.ok:
            logger.warning("OpenWeather call failed: %s", resp.text[:200])
            return None
        j = resp.json()
        daily = j.get("daily", [])[:30]  # may be shorter
        if not daily:
            return None
        temps = [((d.get("temp") or {}).get("day") or 0) for d in daily]
        avg_temp = sum(temps) / len(temps) if temps else None
        rain_days = sum(1 for d in daily if d.get("rain", 0) > 2)
        # trend: compare avg of first half vs second half
        half = len(daily)//2 or 1
        first_avg = sum(temps[:half]) / half
        second_avg = sum(temps[half:]) / (len(temps)-half or 1)
        trend = "warming" if second_avg > first_avg + 0.5 else ("cooling" if second_avg + 0.5 < first_avg else "stable")
        return {"avg_temp_30": avg_temp, "rain_days_30": rain_days, "trend": trend, "source": "openweathermap"}
    except Exception as e:
        logger.exception("Weather fetch failed")
        return None

def infer_soil_pH_from_region(country=None, state=None):
    """
    Simple heuristic defaults for farmers who don't know pH.
    This is intentionally rough: real app should call SoilGrids or require a quick test.
    """
    # Very simplified: many Indian soils slightly acidic: default 6.2; arid soils more alkaline 7.2
    if country and country.lower() in ("india", "in", "bharat"):
        # if we can detect some southern/western states, adjust slightly
        if state:
            s = state.lower()
            if any(x in s for x in ("rajasthan", "gujarat", "punjab", "haryana")):
                return 7.0
        return 6.2
    # default
    return 6.5

def faq_match(q):
    """Very simple substring/fuzzy match using lowercase keys."""
    ql = (q or "").lower()
    if not ql:
        return None
    # direct substring
    for k, v in FAQ.items():
        if k in ql or ql in k:
            return v
    # naive token overlap
    best = None
    best_score = 0
    for k, v in FAQ.items():
        ks = set(k.split())
        qs = set(ql.split())
        score = len(ks & qs)
        if score > best_score:
            best_score = score
            best = v
    if best_score > 0:
        return best
    return None

def openai_answer(question):
    if not (OPENAI_API_KEY and OPENAI_AVAILABLE):
        return None
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini" if "gpt-4o-mini" in openai.Model.list().data else "gpt-3.5-turbo",
            messages=[{"role":"system","content":"You are an expert agronomist providing concise practical advice to smallholder farmers."},
                      {"role":"user","content":question}],
            max_tokens=300,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("OpenAI error")
        return None

# ---------- Core scoring and recommendation ----------
def score_crop_for_conditions(crop, soil_pH, water_access, forecast):
    """
    Score crop for farmer given:
      - crop metadata
      - soil_pH (inferred)
      - water_access: "low"/"medium"/"high"
      - forecast: dict from fetch_weather_forecast (or None)
    Returns (score [0..1], explanations[])
    """
    explanations = []
    score = 0.0

    # pH: best at mid-range
    phmin, phmax = crop["pH_range"]
    phmid = (phmin + phmax)/2.0
    phdiff = abs(soil_pH - phmid)
    phwidth = max(0.1, phmax - phmin)
    phscore = max(0.0, 0.5 - (phdiff / phwidth) * 0.5)
    score += phscore
    explanations.append(f"pH_match={phscore:.2f}")

    # water accessibility influence (maps to moisture match)
    # water_access: "low" -> treat moisture as 0.25, "medium"->0.55, "high"->0.8
    wa_map = {"low":0.25, "medium":0.55, "high":0.8}
    wa = wa_map.get((water_access or "medium").lower(), 0.55)
    mmin, mmax = crop["moisture"]
    # if wa in preferred moisture range -> good
    if mmin <= wa <= mmax:
        mscore = 0.35
    else:
        gap = min(abs(mmin - wa), abs(mmax - wa))
        mscore = max(0.0, 0.35 - gap*0.8)
    score += mscore
    explanations.append(f"water_match={mscore:.2f}")

    # weather forecast adjustment
    rain_adj = 0.0
    temp_adj = 0.0
    if forecast:
        # if forecast says lots of rain and crop prefers high moisture => small bonus
        if forecast.get("rain_days_30", 0) > 6 and crop["moisture"][1] > 0.7:
            rain_adj += 0.05
        # if forecast warming and crop prefers cooler temps -> penalty
        if forecast.get("trend") == "warming":
            tmin,tmax = crop["temp"]
            # if crop ideal max < average threshold -> small penalty
            if tmax < 30:
                temp_adj -= 0.03
    score += rain_adj + temp_adj
    explanations.append(f"forecast_adj={rain_adj+temp_adj:+.2f}")

    # clamp
    score = max(0.0, min(1.0, score))
    return score, explanations

def estimate_yield_and_profit_for_crop(crop, farm_acres):
    ha = acres_to_hectares(farm_acres)
    yield_kg = int(round(crop["yield_kg_ha"] * ha))
    inr_per_kg = usd_to_inr(crop["price_usd_per_ton"]) / 1000.0
    profit_inr = int(round(yield_kg * inr_per_kg))
    return yield_kg, profit_inr

def simulate_projection(crop, farm_acres, years=3):
    base_yield, base_profit = estimate_yield_and_profit_for_crop(crop, farm_acres)
    sims = []
    for y in range(1, years+1):
        variation = random.uniform(-0.12, 0.15)  # realistic variance
        yld = int(round(base_yield * (1 + variation)))
        prof = int(round(yld * (usd_to_inr(crop["price_usd_per_ton"]) / 1000.0)))
        sims.append({"year": y, "yield_kg": yld, "profit_inr": prof})
    return sims

# ---------- End helpers ----------

# ---------- Endpoints ----------

@app.route("/health")
def health():
    return jsonify({"status":"ok", "time": datetime.utcnow().isoformat()+"Z"})

@app.route("/crops")
def crops_list():
    out = []
    for c in CROPS:
        out.append({"crop": c["crop"], "pH_range": c["pH_range"], "moisture": c["moisture"], "temp": c["temp"], "yield_kg_ha": c["yield_kg_ha"], "price_usd_per_ton": c["price_usd_per_ton"], "sustainability": c["sustainability"]})
    return jsonify({"crops": out})

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Accepts JSON:
    {
      "location": {"lat": .., "lon": ..} OR {"city":"Name","country":"India"},
      "water_access": "low"/"medium"/"high",
      "goal": "profit"/"sustainability"/"balanced",
      "farm_acres": 2,
      "language": "auto"  # optional, if browser sends navigator.language
    }
    Returns: recommendations (language translated if possible)
    """

    data = request.get_json(silent=True) or {}
    logger.info("recommend payload: %s", data)

    # language handling: frontend may send user's locale (like "en-IN" or "hi")
    user_locale = (data.get("language") or "auto")
    # We'll detect language of returned text and translate to user's primary language when possible.

    # location handling: prefer lat/lon if provided
    lat = None; lon = None; city = None; country = None
    loc = data.get("location") or {}
    if isinstance(loc, dict):
        lat = loc.get("lat"); lon = loc.get("lon")
        city = loc.get("city"); country = loc.get("country")
    # water access & goal
    water_access = (data.get("water_access") or "medium").lower()
    goal = (data.get("goal") or "balanced").lower()
    farm_acres = safe_float(data.get("farm_acres"), 2.0)

    # 1) Weather
    forecast = None
    if lat and lon:
        forecast = fetch_weather_forecast(lat, lon)
    else:
        # try city -> geocode via Nominatim (no key) as a fallback
        if city:
            try:
                geo = requests.get("https://nominatim.openstreetmap.org/search", params={"q": f"{city} {country or ''}", "format":"json", "limit":1}, headers={"User-Agent":"agromind-demo"}, timeout=6)
                if geo.ok:
                    j = geo.json()
                    if j:
                        lat = float(j[0]["lat"]); lon = float(j[0]["lon"])
                        forecast = fetch_weather_forecast(lat, lon)
            except Exception:
                forecast = None

    # 2) Soil pH inference (we avoid asking farmer pH)
    inferred_pH = infer_soil_pH_from_region(country=country, state=None)  # can be improved with reverse geocoding
    # If we have lat/lon, we could integrate SoilGrids (not included here) — mention as next step.

    # 3) Score crops
    recs = []
    for c in CROPS:
        score, expl = score_crop_for_conditions(c, inferred_pH, water_access, forecast)
        yld, profit = estimate_yield_and_profit_for_crop(c, farm_acres)
        proj = simulate_projection(c, farm_acres, years=3)
        warnings = []
        # create simple warnings
        if c["temp"][1] < 28 and forecast and forecast.get("trend") == "warming":
            warnings.append("Forecast warming may stress this crop.")
        if score < 0.4:
            warnings.append("Low match to local conditions.")
        # adjust final score by goal preference
        if goal == "profit":
            final = 0.55*score + 0.35*(profit/1000000.0) + 0.10*(c["sustainability"]/100.0)
        elif goal == "sustainability":
            final = 0.6*score + 0.3*(c["sustainability"]/100.0) + 0.1*(profit/1000000.0)
        else:
            final = 0.6*score + 0.2*(profit/1000000.0) + 0.2*(c["sustainability"]/100.0)

        recs.append({
            "crop": c["crop"],
            "score_base": round(score,3),
            "final_score": round(max(0.0, min(1.0, final)),3),
            "estimated_yield_kg": yld,
            "estimated_profit_inr": profit,
            "sustainability": c["sustainability"],
            "sim_projection": proj,
            "warnings": warnings,
            "companion": COMPANIONS.get(c["crop"], []),
            "explanations": expl,
            "market_trend_hint": forecast.get("trend") if forecast else "no_forecast"
        })

    # sort by final_score then profit
    recs = sorted(recs, key=lambda r: (-r["final_score"], -r["estimated_profit_inr"]))
    primary = recs[0]["crop"] if recs else None
    backup = recs[1]["crop"] if len(recs)>1 else None

    response = {
        "query": {"water_access": water_access, "goal": goal, "farm_acres": farm_acres, "location": {"lat": lat, "lon": lon, "city": city, "country": country}, "inferred_pH": inferred_pH},
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "primary": primary,
        "backup": backup,
        "recommendations": recs,
        "forecast": forecast
    }

    # 4) Translation: if user's locale is provided and is not english, translate text fields
    # user_locale may be like "hi-IN" or "en-US"; extract language code
    user_lang = "en"
    if isinstance(user_locale, str) and user_locale.lower() not in ("auto", ""):
        user_lang = user_locale.split("-")[0]
    # detect need to translate
    if user_lang != "en":
        # translate summary strings and per-crop advice fields
        def translate_obj(obj):
            if isinstance(obj, str):
                return translate_text(obj, source="auto", target=user_lang)
            if isinstance(obj, dict):
                return {k: translate_obj(v) for k,v in obj.items()}
            if isinstance(obj, list):
                return [translate_obj(x) for x in obj]
            return obj
        # translate only human fields
        response["primary_translated"] = translate_text(response["primary"], source="auto", target=user_lang) if response["primary"] else None
        for r in response["recommendations"]:
            # translate warnings and companion and explanations
            r["warnings_translated"] = [translate_text(w, source="auto", target=user_lang) for w in r.get("warnings",[])]
            r["companion_translated"] = [translate_text(w, source="auto", target=user_lang) for w in r.get("companion",[])]
            r["explanations_translated"] = [translate_text(w, source="auto", target=user_lang) for w in r.get("explanations",[])]
    return jsonify(response)

@app.route("/ask", methods=["POST"])
def ask():
    """
    Body:
    { "question": "How to increase yield", "language": "auto" }
    Returns: answer (tries local FAQ, then OpenAI optional). Answer is translated to user's language if requested.
    """
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or "").strip()
    user_locale = (data.get("language") or "auto")
    if not q:
        return jsonify({"error":"Provide 'question' in JSON body"}), 400

    # detect language of question and translate to English for processing if needed
    q_lang = detect_language(q) if LANGDETECT_AVAILABLE else "en"
    q_en = q
    if q_lang != "en":
        # translate to English before processing
        q_en = translate_text(q, source=q_lang, target="en")
    # 1) local FAQ
    local = faq_match(q_en)
    if local:
        answer = local
        source = "local_faq"
    else:
        # try OpenAI (optional)
        ai_ans = openai_answer(q_en)
        if ai_ans:
            answer = ai_ans
            source = "openai"
        else:
            # fallback heuristic
            ql = q_en.lower()
            if "ph" in ql:
                answer = "Soil pH affects nutrient availability. Apply lime to raise pH, sulfur to lower it. Do a soil test."
            elif "water" in ql or "irrig" in ql:
                answer = "Use drip irrigation and mulching. Monitor soil moisture and irrigate by crop stage."
            else:
                answer = "Give more details (crop, location, what's wrong) for precise advice."
            source = "heuristic"

    # translate answer back to user's language if requested
    user_lang = "en"
    if isinstance(user_locale, str) and user_locale.lower() not in ("auto",""):
        user_lang = user_locale.split("-")[0]
    if user_lang != "en":
        translated = translate_text(answer, source="en", target=user_lang)
    else:
        translated = answer

    return jsonify({"question": q, "answer": translated, "source": source, "timestamp": datetime.utcnow().isoformat()+"Z"})

@app.route("/tips", methods=["GET","POST"])
def tips():
    if request.method == "GET":
        try:
            data = json.loads(TIPS_FILE.read_text())
        except Exception:
            data = []
        return jsonify({"tips": data})
    else:
        body = request.get_json(silent=True) or {}
        tip = (body.get("tip") or "").strip()
        author = (body.get("author") or "anonymous").strip()
        if not tip:
            return jsonify({"error":"Provide 'tip' field"}), 400
        item = {"tip": tip, "author": author, "time": datetime.utcnow().isoformat()+"Z"}
        try:
            arr = json.loads(TIPS_FILE.read_text())
        except Exception:
            arr = []
        arr.insert(0, item)
        TIPS_FILE.write_text(json.dumps(arr, indent=2, ensure_ascii=False))
        return jsonify({"saved": item}), 201

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service":"AgroMind Smart Backend", "endpoints":["/recommend","/ask","/tips","/crops","/health"]})

# ---------- run ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info("Starting AgroMind smart backend on port %s", port)
    app.run(host="0.0.0.0", port=port, debug=True)
