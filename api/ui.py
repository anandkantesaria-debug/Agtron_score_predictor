"""Mobile-responsive HTML UI served at GET / by the FastAPI application."""

from __future__ import annotations


def render_mobile_ui(blends: list[str]) -> str:
    """Return a mobile-friendly single-page HTML interface for /predict and /recommend."""
    blend_options = "\n".join(f'<option value="{b}"></option>' for b in blends)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
  <title>Coffee Color Predictor</title>
  <style>
    :root {{
      --brand: #2E8B57;
      --brand-dark: #246d44;
      --bg: #f7f9fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
    }}
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background: var(--bg);
      color: var(--text);
      padding: 16px;
      max-width: 640px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 8px;
    }}
    h1 {{
      font-size: 1.45rem;
      font-weight: 700;
    }}
    .header-links {{
      display: flex;
      gap: 10px;
      align-items: center;
      font-size: 0.82rem;
    }}
    .badge {{
      background: #dcfce7;
      color: #166534;
      padding: 4px 8px;
      border-radius: 999px;
      font-weight: 600;
    }}
    .docs-link {{
      color: var(--brand);
      text-decoration: none;
      font-weight: 600;
    }}
    .tabs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      background: #e5e7eb;
      padding: 4px;
      border-radius: 10px;
      margin-bottom: 16px;
    }}
    .tab-btn {{
      border: none;
      background: transparent;
      padding: 10px;
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--muted);
      border-radius: 8px;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: var(--card);
      color: var(--text);
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      margin-bottom: 16px;
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }}
    @media (max-width: 480px) {{
      .form-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    .field {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    label {{
      font-size: 0.85rem;
      font-weight: 600;
      color: #374151;
    }}
    input, select {{
      width: 100%;
      padding: 11px 12px;
      font-size: 1rem;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      background: #fff;
      color: var(--text);
    }}
    input:focus, select:focus {{
      outline: 2px solid var(--brand);
      border-color: transparent;
    }}
    .action-btn {{
      width: 100%;
      height: 56px;
      font-size: 1.15rem;
      font-weight: 700;
      background: var(--brand);
      color: #fff;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .action-btn:hover {{
      background: var(--brand-dark);
    }}
    .action-btn:disabled {{
      opacity: 0.65;
      cursor: wait;
    }}
    .result-box {{
      margin-top: 16px;
      padding: 16px;
      border-radius: 10px;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      text-align: center;
      display: none;
    }}
    .metric-label {{
      font-size: 0.85rem;
      color: #166534;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-value {{
      font-size: 2.2rem;
      font-weight: 800;
      color: #14532d;
      margin-top: 4px;
    }}
    .error-box {{
      margin-top: 14px;
      padding: 12px;
      border-radius: 8px;
      background: #fef2f2;
      border: 1px solid #fecaca;
      color: #991b1b;
      font-size: 0.9rem;
      display: none;
    }}
    .rec-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
      text-align: left;
    }}
    .rec-stat {{
      background: #fff;
      border: 1px solid #dcfce7;
      border-radius: 8px;
      padding: 10px;
    }}
    .rec-stat span {{
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .rec-stat strong {{
      font-size: 1.1rem;
      color: var(--text);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 0.88rem;
      background: #fff;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      text-align: right;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
      color: #374151;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    .hidden {{
      display: none;
    }}
  </style>
</head>
<body>
  <header>
    <h1>☕ Coffee Color Predictor</h1>
    <div class="header-links">
      <span id="health-badge" class="badge">Model Ready</span>
      <a class="docs-link" href="/docs">API Docs</a>
    </div>
  </header>

  <datalist id="blend-list">
    {blend_options}
  </datalist>

  <div class="tabs">
    <button id="tab-pred-btn" class="tab-btn active" onclick="switchTab('pred')">Prediction</button>
    <button id="tab-rec-btn" class="tab-btn" onclick="switchTab('rec')">Recommendation</button>
  </div>

  <!-- TAB 1: PREDICTION -->
  <section id="tab-pred" class="card">
    <form id="pred-form" onsubmit="submitPredict(event)">
      <div class="form-grid">
        <div class="field">
          <label for="p-month">Month</label>
          <select id="p-month" required>
            <option value="1" selected>1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9">9</option>
            <option value="10">10</option>
            <option value="11">11</option>
            <option value="12">12</option>
          </select>
        </div>
        <div class="field">
          <label for="p-blend">Blend Code</label>
          <input id="p-blend" list="blend-list" inputmode="numeric" value="284" required />
        </div>
        <div class="field">
          <label for="p-consumed">Consumed Weight (lb)</label>
          <input id="p-consumed" type="number" step="0.1" min="1" max="1000" value="500" required />
        </div>
        <div class="field">
          <label for="p-temp">Roast Temp (°F)</label>
          <input id="p-temp" type="number" step="0.1" min="300" max="600" value="440" required />
        </div>
        <div class="field">
          <label for="p-quench">Quench Applied (lb)</label>
          <input id="p-quench" type="number" step="0.1" min="0" max="150" value="73" required />
        </div>
      </div>
      <button id="pred-submit" type="submit" class="action-btn">PREDICT</button>
    </form>

    <div id="pred-error" class="error-box"></div>
    <div id="pred-result" class="result-box">
      <div class="metric-label">Predicted Agtron</div>
      <div id="pred-value" class="metric-value">--</div>
    </div>
  </section>

  <!-- TAB 2: RECOMMENDATION -->
  <section id="tab-rec" class="card hidden">
    <form id="rec-form" onsubmit="submitRecommend(event)">
      <div class="form-grid">
        <div class="field">
          <label for="r-month">Month</label>
          <select id="r-month" required>
            <option value="1" selected>1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
            <option value="5">5</option>
            <option value="6">6</option>
            <option value="7">7</option>
            <option value="8">8</option>
            <option value="9">9</option>
            <option value="10">10</option>
            <option value="11">11</option>
            <option value="12">12</option>
          </select>
        </div>
        <div class="field">
          <label for="r-blend">Blend Code</label>
          <input id="r-blend" list="blend-list" inputmode="numeric" value="284" required />
        </div>
        <div class="field">
          <label for="r-consumed">Consumed Weight (lb)</label>
          <input id="r-consumed" type="number" step="0.1" min="1" max="1000" value="500" required />
        </div>
        <div class="field">
          <label for="r-temp">Current Temp (°F)</label>
          <input id="r-temp" type="number" step="0.1" min="300" max="600" value="440" required />
        </div>
        <div class="field">
          <label for="r-quench">Current Quench (lb)</label>
          <input id="r-quench" type="number" step="0.1" min="0" max="150" value="73" required />
        </div>
        <div class="field">
          <label for="r-target">Target Agtron (15–65)</label>
          <input id="r-target" type="number" step="0.1" min="15" max="65" value="45.0" required />
        </div>
      </div>
      <button id="rec-submit" type="submit" class="action-btn">RECOMMEND SETTINGS</button>
    </form>

    <div id="rec-error" class="error-box"></div>
    <div id="rec-result" class="result-box">
      <div class="metric-label">Expected Agtron</div>
      <div id="rec-expected" class="metric-value">--</div>
      <div class="rec-grid">
        <div class="rec-stat"><span>Recommended Temp</span><strong id="rec-temp">--</strong></div>
        <div class="rec-stat"><span>Recommended Quench</span><strong id="rec-quench">--</strong></div>
        <div class="rec-stat"><span>Current Predicted</span><strong id="rec-current">--</strong></div>
        <div class="rec-stat"><span>Gap to Target</span><strong id="rec-gap">--</strong></div>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Temp (°F)</th>
            <th>Quench (lb)</th>
            <th>Expected</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody id="rec-table-body"></tbody>
      </table>
    </div>
  </section>

  <script>
    function switchTab(tab) {{
      const isPred = tab === 'pred';
      document.getElementById('tab-pred').classList.toggle('hidden', !isPred);
      document.getElementById('tab-rec').classList.toggle('hidden', isPred);
      document.getElementById('tab-pred-btn').classList.toggle('active', isPred);
      document.getElementById('tab-rec-btn').classList.toggle('active', !isPred);
    }}

    function formatError(data) {{
      if (data && Array.isArray(data.detail)) {{
        return data.detail.map(e => e.msg).join('; ');
      }}
      return (data && data.detail) ? String(data.detail) : 'Request failed.';
    }}

    async function submitPredict(e) {{
      e.preventDefault();
      const btn = document.getElementById('pred-submit');
      const errBox = document.getElementById('pred-error');
      const resBox = document.getElementById('pred-result');
      errBox.style.display = 'none';
      btn.disabled = true;
      btn.textContent = 'PREDICTING...';
      try {{
        const resp = await fetch('/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            Month: document.getElementById('p-month').value,
            Blend: document.getElementById('p-blend').value,
            Consumed: parseFloat(document.getElementById('p-consumed').value),
            Temp: parseFloat(document.getElementById('p-temp').value),
            Quench: parseFloat(document.getElementById('p-quench').value)
          }})
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          errBox.textContent = formatError(data);
          errBox.style.display = 'block';
          resBox.style.display = 'none';
        }} else {{
          document.getElementById('pred-value').textContent = data.predicted_agtron.toFixed(1);
          resBox.style.display = 'block';
        }}
      }} catch (err) {{
        errBox.textContent = 'Network error: ' + err.message;
        errBox.style.display = 'block';
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'PREDICT';
      }}
    }}

    async function submitRecommend(e) {{
      e.preventDefault();
      const btn = document.getElementById('rec-submit');
      const errBox = document.getElementById('rec-error');
      const resBox = document.getElementById('rec-result');
      errBox.style.display = 'none';
      btn.disabled = true;
      btn.textContent = 'CALCULATING...';
      try {{
        const resp = await fetch('/recommend', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            Month: document.getElementById('r-month').value,
            Blend: document.getElementById('r-blend').value,
            Consumed: parseFloat(document.getElementById('r-consumed').value),
            Temp: parseFloat(document.getElementById('r-temp').value),
            Quench: parseFloat(document.getElementById('r-quench').value),
            target_agtron: parseFloat(document.getElementById('r-target').value)
          }})
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          errBox.textContent = formatError(data);
          errBox.style.display = 'block';
          resBox.style.display = 'none';
        }} else {{
          document.getElementById('rec-expected').textContent = data.expected_agtron.toFixed(1);
          document.getElementById('rec-temp').textContent = data.recommended_temp + ' °F';
          document.getElementById('rec-quench').textContent = data.recommended_quench + ' lb';
          document.getElementById('rec-current').textContent = data.current_predicted_agtron.toFixed(1);
          document.getElementById('rec-gap').textContent = data.gap_to_target.toFixed(1);
          const tbody = document.getElementById('rec-table-body');
          tbody.innerHTML = data.top_recommendations.map((row, idx) => `
            <tr>
              <td>${{idx + 1}}</td>
              <td>${{row.temp}}</td>
              <td>${{row.quench}}</td>
              <td>${{row.expected_agtron.toFixed(1)}}</td>
              <td>${{row.error.toFixed(2)}}</td>
            </tr>
          `).join('');
          resBox.style.display = 'block';
        }}
      }} catch (err) {{
        errBox.textContent = 'Network error: ' + err.message;
        errBox.style.display = 'block';
      }} finally {{
        btn.disabled = false;
        btn.textContent = 'RECOMMEND SETTINGS';
      }}
    }}
  </script>
</body>
</html>
"""

