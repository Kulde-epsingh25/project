from __future__ import annotations

from datetime import datetime


def build(charts, stats):
    tab_buttons = []
    sections = []
    for i, (title, b64) in enumerate(charts.items()):
        active = "active" if i == 0 else ""
        tab_buttons.append(f"<button class='tab {active}' data-target='p{i}'>{title}</button>")
        sections.append(
            f"""
            <section class='page {active}' id='p{i}'>
              <h2>{title}</h2>
              <p class='page-note'>Real historical data and model output. Use tabs to navigate all ten analysis layers.</p>
              <img src='data:image/png;base64,{b64}' alt='{title}' />
            </section>
            """
        )

    chips = [
        ("Latest BTC", stats.get("latest_btc", "N/A")),
        ("All Time High", stats.get("ath", "N/A")),
        ("30D Avg Vol", stats.get("avg_vol_30", "N/A")),
        ("10Y Return", stats.get("ret_10y", "N/A")),
        ("Best RMSE", stats.get("best_rmse", "N/A")),
        ("Best DirAcc", stats.get("best_dir", "N/A")),
    ]
    chip_html = "".join([f"<div class='chip'><span>{k}</span><strong>{v}</strong></div>" for k, v in chips])

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Crypto TSA Dashboard</title>
  <style>
    :root {{
      --bg:#050a1a;
      --panel:#101d3f;
      --panel2:#0a1128;
      --cyan:#00d4ff;
      --purple:#7c3aed;
      --amber:#f59e0b;
      --green:#10b981;
      --red:#ef4444;
      --text:#e2e8f0;
      --dim:#94a3b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Courier New", monospace;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 8%, rgba(0,212,255,0.14) 0%, transparent 28%),
        radial-gradient(circle at 84% 16%, rgba(124,58,237,0.18) 0%, transparent 34%),
        radial-gradient(circle at 35% 78%, rgba(16,185,129,0.10) 0%, transparent 32%),
        linear-gradient(180deg, #050a1a 0%, #030611 100%);
    }}
    header {{
      padding: 28px 22px 16px;
      border-bottom: 1px solid rgba(0, 212, 255, 0.25);
      background: linear-gradient(180deg, rgba(16, 29, 63, 0.9), rgba(5, 10, 26, 0.92));
    }}
    .kicker {{ color: var(--amber); font-size: 12px; letter-spacing: 1px; margin-bottom: 4px; }}
    header h1 {{ margin: 0; font-size: 26px; color: var(--cyan); }}
    header p {{ margin: 6px 0 0; color: var(--dim); }}
    .badge-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .badge {{
      font-size: 11px;
      color: var(--text);
      border: 1px solid rgba(148,163,184,0.35);
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(10,17,40,0.65);
    }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 5;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 14px;
      background: rgba(10, 17, 40, 0.95);
      backdrop-filter: blur(4px);
      border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }}
    .tab {{
      border: 1px solid rgba(0,212,255,0.35);
      background: rgba(7,14,34,0.55);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
      font-family: inherit;
      font-size: 12px;
      transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
    }}
    .tab:hover {{
      transform: translateY(-1px);
      border-color: rgba(0,212,255,0.7);
    }}
    .tab.active {{
      background: linear-gradient(90deg, rgba(0,212,255,0.30), rgba(124,58,237,0.25));
      border-color: var(--cyan);
    }}
    main {{ max-width: 1250px; margin: 0 auto; padding: 16px; }}
    .stats-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .chip {{
      background: linear-gradient(135deg, rgba(16,29,63,0.95), rgba(10,17,40,0.9));
      border: 1px solid rgba(148,163,184,0.25);
      border-radius: 12px;
      padding: 12px;
    }}
    .chip span {{ display:block; font-size:11px; color:var(--dim); margin-bottom:3px; }}
    .chip strong {{ font-size:17px; color:var(--text); }}
    .page {{ display: none; animation: fade .28s ease-in; }}
    .page.active {{ display: block; }}
    .page h2 {{ color: var(--cyan); font-size: 18px; margin: 8px 0; }}
    .page-note {{ color: var(--dim); font-size: 12px; margin: 4px 0 10px; }}
    .page img {{ width: 100%; border: 1px solid rgba(148,163,184,0.24); border-radius: 10px; background: var(--panel); }}
    footer {{
      margin-top: 24px;
      padding: 14px 20px;
      border-top: 1px solid rgba(148,163,184,0.25);
      color: var(--dim);
      text-align: center;
      font-size: 12px;
    }}
    @keyframes fade {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  </style>
</head>
<body>
  <header>
    <div class='kicker'>CRYPTOCURRENCY TIME SERIES ANALYTICS PLATFORM</div>
    <h1>Crypto Time Series Analytics Dashboard</h1>
    <p>Real data | 4 models | 10 pages | Source: {stats.get('source', 'N/A')} | Generated: {now}</p>
    <div class='badge-row'>
      <span class='badge'>6 Assets</span>
      <span class='badge'>ARIMA • SARIMA • Prophet-Like • LSTM</span>
      <span class='badge'>Self-contained HTML</span>
      <span class='badge'>No external CDN</span>
    </div>
  </header>
  <nav>{''.join(tab_buttons)}</nav>
  <main>
    <div class='stats-row'>{chip_html}</div>
    {''.join(sections)}
  </main>
  <footer>
    Assets: {stats.get('assets','N/A')} | Rows: {stats.get('rows','N/A')} | Models: {stats.get('models','N/A')} | Source: {stats.get('source','N/A')}
  </footer>
  <script>
    const tabs = [...document.querySelectorAll('.tab')];
    const pages = [...document.querySelectorAll('.page')];
    tabs.forEach(btn => {{
      btn.addEventListener('click', () => {{
        tabs.forEach(t => t.classList.remove('active'));
        pages.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const page = document.getElementById(btn.dataset.target);
        if (page) page.classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""
