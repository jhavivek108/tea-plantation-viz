"""Step 4 - Prepare the Medium article and preview it on localhost.

Usage:
    python src/medium.py build                 # -> medium/images/ + medium/preview.html
    python src/medium.py serve [--port 8000]   # build, then preview at http://127.0.0.1:8000

The article itself is medium/article.md. Medium has no tables and cannot fetch images
from your computer, so this script:
  * copies the charts into medium/images/, numbered in the order they appear;
  * renders the article's two tables as images;
  * builds preview.html, a Medium-style page with a "Copy article" button that puts
    the text on the clipboard as rich text, with a marker where each image goes.
"""
from __future__ import annotations

import argparse
import html
import http.server
import re
import shutil
import socketserver
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mistune
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEDIUM = ROOT / "medium"
IMAGES = MEDIUM / "images"
FIG = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

FIGURE_MAP = {  # Medium image name -> chart produced by src/analysis.py
    "01_hero_heatmap.png": "02_seasonality_heatmap.png",
    "03_annual_production.png": "01_annual_production.png",
    "04_seasonal_profile.png": "03_seasonal_profile.png",
    "05_small_grower_share.png": "04_small_grower_share.png",
    "06_covid_impact.png": "07_covid_impact.png",
    "08_model_comparison.png": "09_model_comparison.png",
    "09_feature_importance.png": "11_feature_importance.png",
    "10_actual_vs_forecast.png": "10_actual_vs_forecast_FY2025-26.png",
    "11_forecast.png": "12_forecast_FY2026-27.png",
}
TAGS = ["Data Science", "Machine Learning", "Time Series Forecasting", "Python", "India"]

SURFACE, INK, INK2, LINE, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"


def table_image(rows: list[list[str]], header: list[str], path: Path, title: str,
                col_widths: list[float], highlight_row: int | None = None) -> None:
    """Render a small table in the same visual style as the charts."""
    n = len(rows)
    fig_h = 1.15 + 0.46 * (n + 1)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 1)
    ax.invert_yaxis()
    xs = [0]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w)
    font = {"family": ["Segoe UI", "DejaVu Sans"]}
    for x, h in zip(xs, header):
        ax.text(x + 0.01, 0.5, h, fontsize=10, color=INK2, fontweight="semibold", va="center", **font)
    ax.axhline(1, color="#c3c2b7", linewidth=1)
    for r, row in enumerate(rows, start=1):
        if highlight_row == r - 1:
            ax.axhspan(r, r + 1, color="#e8f0fb", zorder=0)
        for c, (x, v) in enumerate(zip(xs, row)):
            ax.text(x + 0.01, r + 0.5, v, fontsize=11, color=INK, va="center",
                    fontweight="semibold" if (highlight_row == r - 1 and c == 0) else "normal", **font)
        ax.axhline(r + 1, color=LINE, linewidth=0.8)
    fig.suptitle(title, x=0.125, y=0.98, ha="left", fontsize=13, fontweight="semibold", color=INK, **font)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_images() -> None:
    if IMAGES.exists():
        shutil.rmtree(IMAGES)
    IMAGES.mkdir(parents=True)
    for new, old in FIGURE_MAP.items():
        shutil.copy2(FIG / old, IMAGES / new)

    table_image(
        header=["Check", "Result"],
        rows=[["Months printed in two releases agree", "3,276 duplicate cells, all identical"],
              ["Big growers + small growers = Total", "Every row, every month"],
              ["Areas → states → regions → All India", "Every month, within 0.01 M kg"],
              ["CTC + Orthodox + Green = Total", "Every month"],
              ["Printed annual totals = sum of their months", "14 of 14"],
              ["Final annual totals = sum of final months", "30 of 30"],
              ["Provisional 2024 months vs final 2024 total", "18.75 M kg short (flagged)"]],
        col_widths=[0.52, 0.48], highlight_row=6,
        path=IMAGES / "02_table_checks.png", title="29 integrity checks on the curated dataset")

    s = pd.read_csv(REPORTS / "model_backtest_summary.csv")
    s = s.sort_values("india_WAPE_pct")
    rows = [[m, f"{w:.1f}%", f"{mae:.1f}", f"{a:.1f}%"]
            for m, w, mae, a in zip(s.model, s.india_WAPE_pct, s.india_MAE, s.abs_annual_error_pct)]
    table_image(
        header=["Model", "Monthly WAPE", "Monthly MAE (M kg)", "Annual-total error"],
        rows=rows, col_widths=[0.34, 0.2, 0.24, 0.22], highlight_row=0,
        path=IMAGES / "07_table_results.png",
        title="Backtest results: mean of four held-out years (lower is better)")


CSS = r"""
:root{--bg:#ffffff;--ink:#242424;--ink2:#6b6b6b;--rule:#e6e6e6;--code:#f2f2f2;--bar:#fafafa;--accent:#1a8917;
 --note:#fff4c2;--note-ink:#5c4a00;
 --serif:"Charter","Bitstream Charter","Sitka Text",Cambria,Georgia,serif;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;
 --mono:"Cascadia Mono",Menlo,Consolas,monospace;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121212;--ink:#e6e6e6;--ink2:#a8a8a8;--rule:#2e2e2e;
 --code:#1e1e1e;--bar:#1a1a1a;--accent:#4fbf4b;--note:#3d3410;--note-ink:#f4e3a0;color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink)}
.bar{position:sticky;top:0;z-index:5;background:var(--bar);border-bottom:1px solid var(--rule);font:14px/1.4 var(--sans)}
.bar .in{max-width:980px;margin-inline:auto;padding:10px 20px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;justify-content:space-between}
.bar b{font-weight:600}
.bar .muted{color:var(--ink2)}
.btn{font:600 14px var(--sans);background:var(--accent);color:#fff;border:0;border-radius:999px;padding:8px 16px;cursor:pointer}
.btn:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.status{font-size:13px;color:var(--accent);min-width:10ch}
article{max-width:680px;margin-inline:auto;padding:40px 20px 80px}
h1{font:700 42px/1.2 var(--sans);letter-spacing:-.016em;margin:0 0 10px;text-wrap:balance}
h3.subtitle{font:400 22px/1.4 var(--sans);color:var(--ink2);margin:0 0 32px}
h2{font:700 24px/1.3 var(--sans);letter-spacing:-.012em;margin:48px 0 6px}
h3{font:700 20px/1.35 var(--sans);margin:32px 0 4px}
p,li{font:20px/1.6 var(--serif);letter-spacing:-.003em}
p{margin:14px 0}
ul,ol{padding-left:28px;margin:14px 0} li{margin:8px 0;padding-left:4px}
a{color:inherit;text-decoration:underline}
strong{font-weight:700}
code{font:15px var(--mono);background:var(--code);padding:2px 4px;border-radius:3px}
pre{background:var(--code);padding:20px;overflow-x:auto;border-radius:4px;margin:24px 0}
pre code{background:none;padding:0;font-size:14px;line-height:1.55}
figure{margin:36px 0}
figure img{width:100%;height:auto;display:block}
figcaption{font:14px/1.45 var(--sans);color:var(--ink2);text-align:center;margin-top:10px}
.upload{display:block;font:12px var(--mono);color:var(--ink2);text-align:center;margin-top:4px}
hr{border:0;text-align:center;margin:44px 0;height:auto}
hr::before{content:"· · ·";letter-spacing:.6em;color:var(--ink2);font-size:22px}
mark{background:var(--note);color:var(--note-ink);padding:0 4px;border-radius:3px}
.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:40px}
.tag{font:14px var(--sans);background:var(--code);border-radius:999px;padding:6px 14px;color:var(--ink)}
@media (max-width:600px){h1{font-size:32px}h3.subtitle{font-size:19px}p,li{font-size:18px}}
"""

JS = r"""<script>
// Copy the article as rich text for the Medium editor. Images cannot be pasted from
// localhost, so each one becomes a marker line naming the file to upload there.
const btn=document.getElementById('copy'),status=document.getElementById('status');
btn.addEventListener('click',async()=>{
  const clone=document.getElementById('story').cloneNode(true);
  clone.querySelectorAll('figure').forEach(f=>{const p=document.createElement('p');
    p.textContent='[UPLOAD IMAGE '+f.dataset.file+' | caption: '+(f.querySelector('figcaption')?.textContent||'')+']';f.replaceWith(p);});
  clone.querySelectorAll('.tags').forEach(t=>t.remove());
  const htmlStr=clone.innerHTML,text=clone.innerText;
  try{await navigator.clipboard.write([new ClipboardItem({'text/html':new Blob([htmlStr],{type:'text/html'}),'text/plain':new Blob([text],{type:'text/plain'})})]);
      status.textContent='Copied. Paste into a new Medium story.';}
  catch(e){const r=document.createRange();r.selectNodeContents(document.getElementById('story'));const s=getSelection();s.removeAllRanges();s.addRange(r);
      status.textContent='Copy blocked: the article is selected, press Ctrl+C.';}
});
</script>"""


def build_preview() -> None:
    md_text = (MEDIUM / "article.md").read_text(encoding="utf-8")
    body = mistune.create_markdown(plugins=["strikethrough"])(md_text)
    # first H3 directly after the title is Medium's subtitle
    body = re.sub(r"(</h1>\s*)<h3>", r'\1<h3 class="subtitle">', body, count=1)
    body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
        lambda g: (f'<figure data-file="{g.group(1)}"><img src="images/{g.group(1)}" alt="{g.group(2)}">'
                   f'<figcaption>{g.group(2)}</figcaption><span class="upload">{g.group(1)}</span></figure>'),
        body)
    body = body.replace("[REPO_URL]", '<mark>[REPO_URL: add your GitHub link before publishing]</mark>')
    n_img = body.count("<figure")
    words = len(re.findall(r"\w+", re.sub(r"<pre>.*?</pre>", "", re.sub(r"<[^>]+>", " ", body), flags=re.S)))
    tags = "".join(f'<span class="tag">{html.escape(t)}</span>' for t in TAGS)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Medium Draft Preview</title><style>{CSS}</style></head>
<body><div class="bar"><div class="in"><span><b>Medium draft preview</b> <span class="muted">· {words:,} words · ~{round(words / 230)} min read · {n_img} images in medium/images</span></span>
<span><span class="status" id="status" role="status"></span> <button class="btn" id="copy">Copy article</button></span></div></div>
<article id="story">{body}<div class="tags">{tags}</div></article>{JS}</body></html>"""
    (MEDIUM / "preview.html").write_text(page, encoding="utf-8")
    print(f"Built medium/preview.html: {words:,} words, {n_img} images")


def build() -> None:
    build_images()
    build_preview()


def serve(port: int) -> None:
    build()
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(MEDIUM))
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Preview at http://127.0.0.1:{port}/preview.html  (Ctrl+C to stop)", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "serve"])
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    build() if a.command == "build" else serve(a.port)
