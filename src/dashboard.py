"""Step 3 - Build an offline, self-contained HTML dashboard.

Usage:  python src/dashboard.py        ->  reports/dashboard.html (open it in any browser)

Run src/extract.py and src/analysis.py first. The dashboard embeds the figures from
reports/figures and the curated monthly data, so it needs no internet connection,
server, or extra packages. The "Explore" tab draws its chart in the browser from the
embedded data: pick any area / state / zone, grower segment and year range.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
OUT = REPORTS / "dashboard.html"

TABS = [
    ("overview", "Overview", [
        ("01_annual_production.png", "Calendar-year production, North vs South India"),
        ("02_seasonality_heatmap.png", "All-India production by month and year"),
    ]),
    ("seasonality", "Seasonality & shocks", [
        ("03_seasonal_profile.png", "Typical month, North vs South"),
        ("08_stl_decomposition.png", "Trend, seasonal and residual components"),
        ("07_covid_impact.png", "COVID-19 impact, 2020 vs 2019"),
    ]),
    ("structure", "Growers, states & products", [
        ("04_small_grower_share.png", "Small-grower share of production"),
        ("05_state_share.png", "State share of production"),
        ("06_orthodox_production.png", "Orthodox tea production"),
    ]),
    ("model", "Model & forecast", [
        ("09_model_comparison.png", "Backtest error by model and year"),
        ("10_actual_vs_forecast_FY2025-26.png", "Held-out year FY 2025-26"),
        ("11_feature_importance.png", "What the Random Forest relies on"),
        ("12_forecast_FY2026-27.png", "FY 2026-27 forecast"),
    ]),
]


def img_tag(name: str, alt: str) -> str:
    path = FIG / name
    if not path.exists():
        return f'<p class="missing">Missing {html.escape(name)}: run src/analysis.py first.</p>'
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" loading="lazy">'


def fmt(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


def main() -> None:
    master = pd.read_csv(DATA / "tea_production_master.csv")
    stats = json.loads((REPORTS / "statistics.json").read_text())
    qa = pd.read_csv(DATA / "qa_checks.csv")
    backtest = pd.read_csv(REPORTS / "model_backtest_summary.csv")

    mon = master[(master.period_type == "month") & (master.breakdown == "grower")]
    allind = mon[(mon.geography == "All India") & (mon.segment == "Total")].set_index("period")["production_mkg"]
    ann = master[(master.period_type != "month") & (master.geography == "All India")].set_index("period")["production_mkg"]
    cy25, cy24 = allind.loc["2025-01":"2025-12"].sum(), ann["2024"]
    fy26, fy25 = allind.loc["2025-04":"2026-03"].sum(), ann["2024-25"]
    sg = stats["small_growers"]
    cat = stats["category"]
    fc = stats["forecast_FY2026_27"]

    kpis = [
        ("Production, calendar 2025", f"{fmt(cy25, 1)}", "M kg", f"{100 * (cy25 / cy24 - 1):+.1f}% vs 2024", cy25 >= cy24),
        ("Production, FY 2025-26", f"{fmt(fy26, 1)}", "M kg", f"{100 * (fy26 / fy25 - 1):+.1f}% vs FY 2024-25", fy26 >= fy25),
        ("Small-grower share, 2025", f"{sg['sg_share_2025_pct']}", "%", f"{sg['sg_share_2025_pct'] - sg['sg_share_2019_pct']:+.1f} pts vs 2019", None),
        ("Orthodox tea, 2025", f"{fmt(cat['orthodox_mkg']['2025'], 1)}", "M kg", f"{cat['orthodox_share_pct']['2025']}% of output", None),
        ("Forecast, FY 2026-27", f"{fmt(fc['total_mkg'], 0)}", "M kg", f"{fc['vs_FY2025_26_pct']:+.1f}% vs FY 2025-26 · {fc['model']}", None),
    ]
    kpi_html = "\n".join(
        f'<div class="kpi"><div class="kpi-label">{html.escape(l)}</div>'
        f'<div class="kpi-value">{v}<span class="unit">{u}</span></div>'
        f'<div class="kpi-delta {"up" if good is True else "down" if good is False else ""}">{html.escape(d)}</div></div>'
        for l, v, u, d, good in kpis)

    tab_buttons = "".join(
        f'<button role="tab" id="tab-{k}" aria-controls="panel-{k}" data-tab="{k}">{html.escape(t)}</button>'
        for k, t, _ in TABS) + '<button role="tab" id="tab-explore" aria-controls="panel-explore" data-tab="explore">Explore data</button>' \
        + '<button role="tab" id="tab-quality" aria-controls="panel-quality" data-tab="quality">Data quality</button>'

    panels = []
    for k, _, figs in TABS:
        cards = "".join(f'<figure class="card"><figcaption>{html.escape(c)}</figcaption>{img_tag(f, c)}</figure>'
                        for f, c in figs)
        extra = ""
        if k == "model":
            rows = "".join(
                f"<tr><td>{html.escape(r.model)}</td><td>{r.india_WAPE_pct:.1f}%</td><td>{r.india_MAE:.1f}</td>"
                f"<td>{r.abs_annual_error_pct:.1f}%</td><td>{r.series_R2:.3f}</td></tr>"
                for r in backtest.itertuples())
            extra = ('<div class="card"><div class="card-title">Backtest summary: mean of four held-out years</div>'
                     '<div class="table-wrap"><table><thead><tr><th>Model</th><th>Monthly WAPE</th>'
                     '<th>Monthly MAE (M kg)</th><th>Annual error</th><th>Series R²</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table></div>'
                     '<p class="note">WAPE = total absolute error as a share of actual production; lower is better. '
                     'The Random Forest is the strongest ML model, but its gap to the seasonal-median baseline is not '
                     f'statistically significant (Wilcoxon p = {stats["model"]["significance_tests"]["Random forest vs Seasonal median"]["wilcoxon_p"]:.2f}).</p></div>')
        panels.append(f'<section role="tabpanel" id="panel-{k}" aria-labelledby="tab-{k}" class="grid">{extra}{cards}</section>')

    qa_rows = "".join(
        f'<tr><td><span class="pill {r.result.lower()}">{r.result}</span></td><td>{html.escape(r.check)}</td>'
        f'<td>{html.escape(str(r.detail))}</td></tr>' for r in qa.itertuples())
    panels.append(
        '<section role="tabpanel" id="panel-quality" aria-labelledby="tab-quality">'
        '<div class="card"><div class="card-title">Integrity checks run by src/extract.py</div>'
        f'<div class="table-wrap"><table><thead><tr><th>Result</th><th>Check</th><th>Detail</th></tr></thead><tbody>{qa_rows}</tbody></table></div>'
        '<p class="note">Apr–Dec 2024 monthly figures were published only as provisional. They sum to 18.75 M kg below the '
        'Tea Board\'s final 2024 total; the gap is almost entirely North India.</p></div></section>')

    # Explorer data: every monthly grower-split row, compact
    ex = mon[["period", "geography", "geo_level", "segment", "production_mkg", "status"]]
    order = ["All India", "North India", "South India", "Assam", "West Bengal", "Assam Valley", "Cachar",
             "Dooars", "Terai", "Darjeeling", "Others (NE & HP etc.)", "Tamil Nadu", "Kerala", "Karnataka"]
    explore_data = {
        "geos": order,
        "rows": [[r.period, order.index(r.geography), r.segment, r.production_mkg, int(r.status == "Provisional")]
                 for r in ex.itertuples()],
    }
    panels.append(
        '<section role="tabpanel" id="panel-explore" aria-labelledby="tab-explore">'
        '<div class="card"><div class="controls">'
        '<label>Geography <select id="ex-geo"></select></label>'
        '<label>Growers <select id="ex-seg"><option value="Total">All growers</option>'
        '<option value="BG">Big growers (estates)</option><option value="SG">Small growers</option></select></label>'
        '<label>From <select id="ex-from"></select></label><label>To <select id="ex-to"></select></label>'
        '<label>View <select id="ex-view"><option value="month">Monthly</option><option value="year">Calendar-year totals</option></select></label>'
        '</div><div class="ex-summary" id="ex-summary"></div>'
        '<div class="chart" id="ex-chart"><div class="tip" id="ex-tip" hidden></div></div>'
        '<p class="note">Hollow points are provisional figures (Apr-2024 to Jan-2025). A partial year in the yearly view is marked "partial".</p>'
        '<div class="table-wrap tall"><table id="ex-table"><thead><tr><th>Period</th><th>Production (M kg)</th><th>Status</th></tr></thead><tbody></tbody></table></div>'
        '</div></section>')

    page = TEMPLATE.replace("{{KPIS}}", kpi_html).replace("{{TABS}}", tab_buttons) \
        .replace("{{PANELS}}", "\n".join(panels)) \
        .replace("{{DATA}}", json.dumps(explore_data, separators=(",", ":"))) \
        .replace("{{TAKEAWAY}}", html.escape(
            f"Output is flat at about 1,300–1,400 M kg a year and 2025 set the high mark ({fmt(cy25, 1)} M kg). "
            f"The big shift is structural: small growers now make {sg['sg_share_2025_pct']}% of India's tea, up from "
            f"{sg['sg_share_2019_pct']}% in 2019."))
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) - open it in a browser")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>India Tea Production Dashboard</title>
<style>
:root{--bg:#f7f7f5;--panel:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#898781;--line:#e1e0d9;--axis:#c3c2b7;
--accent:#2a78d6;--accent2:#eb6834;--ok:#006300;--bad:#b42f2f;--warn:#8a5a00;--chip:#eef3fb;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--panel:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#898781;
--line:#2c2c2a;--axis:#383835;--accent:#3987e5;--accent2:#d95926;--ok:#0ca30c;--bad:#e66767;--warn:#eda100;--chip:#1c2a3d;color-scheme:dark}
 figure.card img{background:#fcfcfb;border-radius:6px}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1280px;margin-inline:auto;padding:24px 20px 48px}
header h1{font-size:26px;font-weight:600;margin:0 0 4px;text-wrap:balance}
header p{margin:0;color:var(--fg2);max-width:80ch}
header .src{font-size:12px;color:var(--muted);margin-top:6px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,200px),1fr));gap:12px;margin:20px 0}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi-label{font-size:12px;color:var(--fg2)}
.kpi-value{font-size:28px;font-weight:600;margin-top:2px}
.kpi-value .unit{font-size:14px;font-weight:400;color:var(--fg2);margin-left:4px}
.kpi-delta{font-size:12px;color:var(--fg2)} .kpi-delta.up{color:var(--ok)} .kpi-delta.down{color:var(--bad)}
[role=tablist]{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin-bottom:16px}
[role=tab]{font:inherit;background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;color:var(--fg2);cursor:pointer}
[role=tab][aria-selected=true]{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}
[role=tab]:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0;min-width:0}
figcaption,.card-title{font-weight:600;margin-bottom:10px}
figure.card img{width:100%;height:auto;display:block}
.table-wrap{overflow:auto} .table-wrap.tall{max-height:360px;margin-top:12px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{position:sticky;top:0;background:var(--panel);color:var(--fg2);font-weight:600;font-size:12px}
.note{font-size:12px;color:var(--fg2);margin:10px 0 0;max-width:100ch}
.pill{display:inline-block;font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px}
.pill.pass{color:var(--ok);border:1px solid currentColor} .pill.fail{color:var(--bad);border:1px solid currentColor}
.pill.info{color:var(--warn);border:1px solid currentColor}
.controls{display:flex;flex-wrap:wrap;gap:12px 16px;align-items:end}
.controls label{display:flex;flex-direction:column;font-size:12px;color:var(--fg2);gap:4px}
select{font:inherit;padding:5px 8px;border:1px solid var(--axis);border-radius:6px;background:var(--panel);color:var(--fg)}
.ex-summary{margin:14px 0 4px;color:var(--fg2)} .ex-summary b{color:var(--fg)}
.chart{position:relative;min-height:340px}
.chart svg{display:block;width:100%;height:auto}
.chart text{fill:var(--muted);font-size:11px}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--axis);border-radius:6px;
 padding:4px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);white-space:nowrap}
.missing{color:var(--bad)}
section[hidden]{display:none}
</style></head><body><div class="wrap">
<header><h1>India Tea Production, 2017 – March 2026</h1>
<p>{{TAKEAWAY}}</p>
<p class="src">Source: Tea Board of India production statistics (7 PDF releases), curated by src/extract.py. Units: million kg of made tea.</p></header>
<div class="kpis">{{KPIS}}</div>
<nav role="tablist" aria-label="Dashboard sections">{{TABS}}</nav>
{{PANELS}}
</div>
<script>
const DATA = {{DATA}};
// ---- tabs (remembers the last one via the URL hash)
const tabs = [...document.querySelectorAll('[role=tab]')];
function show(key){
  tabs.forEach(t=>{const on=t.dataset.tab===key;t.setAttribute('aria-selected',on);
    document.getElementById('panel-'+t.dataset.tab).hidden=!on;});
  history.replaceState(null,'','#'+key);
  if(key==='explore') draw();
}
tabs.forEach(t=>t.addEventListener('click',()=>show(t.dataset.tab)));
// ---- explorer
const geoSel=document.getElementById('ex-geo'),segSel=document.getElementById('ex-seg'),
 fromSel=document.getElementById('ex-from'),toSel=document.getElementById('ex-to'),viewSel=document.getElementById('ex-view');
DATA.geos.forEach((g,i)=>geoSel.add(new Option(g,i)));
const years=[...new Set(DATA.rows.map(r=>r[0].slice(0,4)))].sort();
years.forEach(y=>{fromSel.add(new Option(y,y));toSel.add(new Option(y,y));});
fromSel.value=years[0];toSel.value=years[years.length-1];
[geoSel,segSel,fromSel,toSel,viewSel].forEach(s=>s.addEventListener('change',draw));
const nf=(x,d=1)=>x.toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d});
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const label=p=>p.length>4?MON[+p.slice(5,7)-1]+' '+p.slice(0,4):p;
function series(){
  const g=+geoSel.value,s=segSel.value,a=fromSel.value,b=toSel.value>=fromSel.value?toSel.value:fromSel.value;
  let pts=DATA.rows.filter(r=>r[1]===g&&r[2]===s&&r[0].slice(0,4)>=a&&r[0].slice(0,4)<=b)
     .map(r=>({p:r[0],v:r[3],prov:r[4]})).sort((x,y)=>x.p<y.p?-1:1);
  if(viewSel.value==='year'){
    const m=new Map();pts.forEach(d=>{const y=d.p.slice(0,4);const o=m.get(y)||{p:y,v:0,prov:0,n:0};o.v+=d.v;o.prov|=d.prov;o.n++;m.set(y,o);});
    pts=[...m.values()].map(o=>({...o,partial:o.n<12}));
  }
  return pts;
}
function draw(){
  const box=document.getElementById('ex-chart'),tip=document.getElementById('ex-tip');
  const pts=series(),W=Math.max(320,box.clientWidth),H=320,m={l:52,r:16,t:12,b:34};
  box.querySelector('svg')?.remove();
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.setAttribute('role','img');
  svg.setAttribute('aria-label','Production chart for '+DATA.geos[+geoSel.value]);
  const el=(t,a)=>{const e=document.createElementNS(ns,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
  const css=getComputedStyle(document.documentElement),c=k=>css.getPropertyValue(k).trim();
  const max=Math.max(1,...pts.map(d=>d.v)),step=Math.pow(10,Math.floor(Math.log10(max)))/(max/Math.pow(10,Math.floor(Math.log10(max)))>5?1:2);
  const top=Math.ceil(max/step)*step,x=i=>m.l+(pts.length<2?(W-m.l-m.r)/2:i*(W-m.l-m.r)/(pts.length-1)),y=v=>m.t+(H-m.t-m.b)*(1-v/top);
  for(let v=0;v<=top+1e-9;v+=step){el('line',{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v),stroke:c(v===0?'--axis':'--line')});
    el('text',{x:m.l-8,y:y(v)+4,'text-anchor':'end'}).textContent=nf(v,step<1?1:0);}
  const every=Math.ceil(pts.length/Math.max(2,Math.floor((W-m.l-m.r)/70)));
  pts.forEach((d,i)=>{if(i%every===0)el('text',{x:x(i),y:H-12,'text-anchor':'middle'}).textContent=viewSel.value==='year'?d.p:label(d.p);});
  if(viewSel.value==='year'){
    const bw=Math.min(36,(W-m.l-m.r)/pts.length*0.6);
    pts.forEach((d,i)=>el('rect',{x:x(i)-bw/2,y:y(d.v),width:bw,height:y(0)-y(d.v),rx:3,fill:c('--accent'),opacity:d.partial?0.45:1}));
  }else{
    el('polyline',{points:pts.map((d,i)=>x(i)+','+y(d.v)).join(' '),fill:'none',stroke:c('--accent'),'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'});
    pts.forEach((d,i)=>{if(d.prov)el('circle',{cx:x(i),cy:y(d.v),r:3.5,fill:c('--panel'),stroke:c('--accent'),'stroke-width':1.5});});
  }
  const hit=el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent'});
  const dot=el('circle',{r:5,fill:c('--accent'),stroke:c('--panel'),'stroke-width':2,visibility:'hidden'});
  hit.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect(),sx=(ev.clientX-r.left)*W/r.width;
    const i=Math.max(0,Math.min(pts.length-1,Math.round((sx-m.l)/((W-m.l-m.r)/Math.max(1,pts.length-1)))));const d=pts[i];if(!d)return;
    dot.setAttribute('cx',x(i));dot.setAttribute('cy',y(d.v));dot.setAttribute('visibility','visible');
    tip.hidden=false;tip.textContent=(viewSel.value==='year'?d.p:label(d.p))+': '+nf(d.v)+' M kg'+(d.prov?' (provisional)':'')+(d.partial?' (partial year)':'');
    tip.style.left=Math.min(r.width-160,x(i)*r.width/W+10)+'px';tip.style.top=(y(d.v)*r.height/H-34)+'px';});
  hit.addEventListener('mouseleave',()=>{tip.hidden=true;dot.setAttribute('visibility','hidden');});
  box.insertBefore(svg,tip);
  const tot=pts.reduce((s,d)=>s+d.v,0),pk=pts.reduce((a,d)=>d.v>a.v?d:a,{v:-1});
  document.getElementById('ex-summary').innerHTML=pts.length?
    `<b>${DATA.geos[+geoSel.value]}</b> · ${segSel.options[segSel.selectedIndex].text} · total <b>${nf(tot)}</b> M kg over ${pts.length} ${viewSel.value==='year'?'years':'months'} · peak ${viewSel.value==='year'?pk.p:label(pk.p)} (${nf(pk.v)} M kg)`:'No data for this selection.';
  const tb=document.querySelector('#ex-table tbody');tb.replaceChildren(...pts.slice().reverse().map(d=>{const tr=document.createElement('tr');
    [viewSel.value==='year'?d.p:label(d.p),nf(d.v,2),d.prov?'Provisional':(d.partial?'Final (partial year)':'Final')].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.appendChild(td);});return tr;}));
}
window.addEventListener('resize',()=>{if(!document.getElementById('panel-explore').hidden)draw();});
show(tabs.some(t=>t.dataset.tab===location.hash.slice(1))?location.hash.slice(1):'overview');
</script></body></html>
"""

if __name__ == "__main__":
    main()
