// frontend/src/App.js
import React, { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:5000"; // change if backend hosted elsewhere

function SmallBar({ value, max = 1, color = "green" }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  const bg = color === "green" ? "bg-green-500" : "bg-yellow-500";
  return (
    <div className="w-full bg-gray-100 h-2 rounded">
      <div className={`${bg} h-2`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export default function App() {
  const [coords, setCoords] = useState(null);
  const [city, setCity] = useState("");
  const [form, setForm] = useState({
    water_access: "medium",
    goal: "balanced",
    farm_acres: 2,
  });
  const [loading, setLoading] = useState(false);
  const [recs, setRecs] = useState(null);
  const [error, setError] = useState(null);

  const [question, setQuestion] = useState("");
  const [qa, setQa] = useState(null);

  const [tips, setTips] = useState([]);
  const [tipText, setTipText] = useState("");
  const [tipAuthor, setTipAuthor] = useState("");

  const userLang = navigator.language || "en";

  useEffect(() => {
    // attempt geolocation
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setCoords({ lat: pos.coords.latitude, lon: pos.coords.longitude });
        },
        (err) => {
          console.log("Geo blocked or failed:", err.message);
        }
      );
    }
    fetchTips();
  }, []);

  function update(e) {
    const { name, value } = e.target;
    setForm(f => ({ ...f, [name]: value }));
  }

  async function fetchTips() {
    try {
      const r = await fetch(`${API_BASE}/tips`);
      const j = await r.json();
      setTips(j.tips || []);
    } catch (e) {
      console.error(e);
    }
  }

  async function submitRecommend(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setRecs(null);
    const payload = {
      location: coords ? { lat: coords.lat, lon: coords.lon } : (city ? { city } : {}),
      water_access: form.water_access,
      goal: form.goal,
      farm_acres: parseFloat(form.farm_acres) || 2,
      language: userLang
    };
    try {
      const resp = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!resp.ok) throw new Error("Server error");
      const j = await resp.json();
      setRecs(j);
    } catch (err) {
      setError(err.message || "Failed");
    }
    setLoading(false);
  }

  async function askQuestion(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setQa(null);
    try {
      const resp = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ question, language: userLang })
      });
      const j = await resp.json();
      setQa(j);
    } catch (e) {
      setQa({ error: "Failed to fetch answer" });
    }
  }

  async function submitTip(e) {
    e.preventDefault();
    if (!tipText.trim()) return;
    try {
      await fetch(`${API_BASE}/tips`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ tip: tipText, author: tipAuthor || "anonymous" })
      });
      setTipText(""); setTipAuthor("");
      fetchTips();
    } catch (e) { console.error(e); }
  }

  return (
    <div className="min-h-screen bg-green-50 p-6 font-sans">
      <div className="max-w-6xl mx-auto">
        <header className="bg-green-700 text-white p-4 rounded-lg shadow mb-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold">AgroMind Smart</h1>
              <p className="text-sm opacity-90">Auto language + weather-aware crop advisor</p>
            </div>
            <div className="text-xs">Locale: {userLang}</div>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <section className="lg:col-span-2 bg-white p-4 rounded shadow">
            <h2 className="font-semibold text-lg text-green-700">Quick Setup</h2>
            <p className="text-sm text-gray-600">Allow location for best results (optional). Or type your city name below.</p>

            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm">Detected location</label>
                <input type="text" readOnly value={coords ? `${coords.lat.toFixed(3)}, ${coords.lon.toFixed(3)}` : "Not available"} className="mt-1 p-2 border rounded w-full bg-gray-50" />
              </div>
              <div>
                <label className="block text-sm">Or enter city</label>
                <input value={city} onChange={e=>setCity(e.target.value)} placeholder="e.g., Vellore" className="mt-1 p-2 border rounded w-full" />
              </div>
            </div>

            <form onSubmit={submitRecommend} className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="text-sm">Water availability</label>
                <select name="water_access" value={form.water_access} onChange={update} className="mt-1 p-2 border rounded w-full">
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div>
                <label className="text-sm">Goal</label>
                <select name="goal" value={form.goal} onChange={update} className="mt-1 p-2 border rounded w-full">
                  <option value="balanced">Balanced</option>
                  <option value="profit">Max profit</option>
                  <option value="sustainability">Soil / sustainability</option>
                </select>
              </div>
              <div>
                <label className="text-sm">Farm size (acres)</label>
                <input name="farm_acres" value={form.farm_acres} onChange={update} type="number" step="0.1" className="mt-1 p-2 border rounded w-full" />
              </div>

              <div className="md:col-span-3 flex gap-2 mt-2">
                <button className="bg-green-600 text-white px-4 py-2 rounded shadow" disabled={loading}>{loading ? "Analyzing..." : "Get Recommendations"}</button>
                <button type="button" onClick={()=>{ setRecs(null); setError(null); }} className="px-4 py-2 border rounded">Reset</button>
              </div>
            </form>

            {error && <div className="mt-3 text-red-600">{error}</div>}

            {recs && (
              <div className="mt-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Recommendations</h3>
                  <div className="text-sm">Primary: <strong>{recs.primary}</strong> | Backup: <strong>{recs.backup}</strong></div>
                </div>

                <div className="mt-3 space-y-4">
                  {recs.recommendations.map((r, i)=>(
                    <article key={i} className="border rounded p-3 bg-green-50">
                      <div className="flex justify-between">
                        <div>
                          <h4 className="text-xl font-bold text-green-800">{r.crop}</h4>
                          <div className="text-sm italic text-gray-700">{(r.explanations||[]).join(" • ")}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm">Score</div>
                          <div className="text-2xl font-bold">{r.final_score}</div>
                        </div>
                      </div>

                      <div className="mt-3 grid grid-cols-1 md:grid-cols-4 gap-3">
                        <div>
                          <div className="text-xs text-gray-600">Profit (INR)</div>
                          <div className="font-semibold">₹{r.estimated_profit_inr.toLocaleString()}</div>
                          <SmallBar value={r.estimated_profit_inr} max={500000} />
                        </div>
                        <div>
                          <div className="text-xs text-gray-600">Yield (kg)</div>
                          <div className="font-semibold">{r.estimated_yield_kg.toLocaleString()}</div>
                          <SmallBar value={r.estimated_yield_kg} max={80000} color="yellow" />
                        </div>
                        <div>
                          <div className="text-xs text-gray-600">Sustainability</div>
                          <div className="font-semibold">{r.sustainability}%</div>
                          <SmallBar value={r.sustainability/100} max={1} />
                        </div>
                        <div>
                          <div className="text-xs text-gray-600">Risks</div>
                          <div className="text-sm">{r.warnings.length ? r.warnings.join("; ") : "Low"}</div>
                        </div>
                      </div>

                      <details className="mt-2">
                        <summary className="cursor-pointer text-sm text-green-700">Show year-by-year projection</summary>
                        <div className="mt-2 text-sm">
                          {r.sim_projection.map(s => (
                            <div key={s.year} className="flex justify-between">
                              <div>Year {s.year}</div>
                              <div>Yield: {s.yield_kg.toLocaleString()} kg • ₹{s.profit_inr.toLocaleString()}</div>
                            </div>
                          ))}
                        </div>
                      </details>
                    </article>
                  ))}
                </div>
              </div>
            )}
          </section>

          <aside className="space-y-4">
            <div className="bg-white p-3 rounded shadow">
              <h4 className="font-semibold text-green-700">Ask AgroMind</h4>
              <form onSubmit={askQuestion} className="mt-2">
                <textarea value={question} onChange={e=>setQuestion(e.target.value)} placeholder="e.g., How to reduce soil acidity?" className="w-full p-2 border rounded h-28" />
                <div className="flex gap-2 mt-2">
                  <button className="bg-blue-600 text-white px-3 py-1 rounded">Ask</button>
                  <button type="button" onClick={()=>{ setQuestion(""); setQa(null); }} className="px-3 py-1 border rounded">Clear</button>
                </div>
              </form>
              <div className="mt-2">
                {qa && <div className="bg-gray-50 p-2 rounded text-sm"><div className="text-xs text-gray-500">Source: {qa.source}</div><div className="mt-1">{qa.answer}</div></div>}
              </div>
            </div>

            <div className="bg-white p-3 rounded shadow">
              <h4 className="font-semibold text-green-700">Community Tips</h4>
              <form onSubmit={submitTip} className="mt-2 space-y-2">
                <input value={tipAuthor} onChange={e=>setTipAuthor(e.target.value)} placeholder="Your name (optional)" className="w-full p-2 border rounded" />
                <textarea value={tipText} onChange={e=>setTipText(e.target.value)} placeholder="Share a tip..." className="w-full p-2 border rounded h-20" />
                <div className="flex gap-2">
                  <button className="bg-green-600 text-white px-3 py-1 rounded">Share</button>
                  <button type="button" onClick={()=>{ setTipAuthor(""); setTipText(""); }} className="px-3 py-1 border rounded">Clear</button>
                </div>
              </form>

              <div className="mt-3 max-h-48 overflow-auto space-y-2">
                {tips.length === 0 && <div className="text-sm text-gray-500">No tips yet.</div>}
                {tips.map((t,i)=>(
                  <div key={i} className="p-2 bg-gray-50 rounded">
                    <div className="text-sm">{t.tip}</div>
                    <div className="text-xs text-gray-500 mt-1">— {t.author} • {new Date(t.time).toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </div>

          </aside>

        </main>

        <footer className="mt-6 text-center text-xs text-gray-500">
          AgroMind Smart — recommendations incorporate weather forecasts when available and auto-translate results.
        </footer>
      </div>
    </div>
  );
}
