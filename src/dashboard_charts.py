"""Dashboard chart set - varied chart types for each dashboard tab.

Usage:  python src/dashboard_charts.py   ->  reports/figures/dashboard/*.png

Visualisation only: reads the curated data and the saved model outputs from
src/extract.py and src/analysis.py, and changes no data, model or result.
The map uses India state boundaries from geoBoundaries (gbOpen IND ADM1, CC BY 2.5 IN,
DataMeet / Election Commission of India), saved in data/external/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUT = REPORTS / "figures" / "dashboard"
GEOJSON = ROOT / "data" / "external" / "india_states_simplified.geojson"

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, RED = CAT[0], CAT[1], CAT[7]
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ZONE = {"North India": BLUE, "South India": ORANGE}
STATES = ["Assam", "West Bengal", "Others (North)", "Tamil Nadu", "Kerala", "Karnataka"]
STATE_COLOR = dict(zip(STATES, CAT))
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
})


def titled(ax, title, sub=""):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


# ------------------------------------------------------------------- data --
def load():
    m = pd.read_csv(DATA / "tea_production_master.csv")
    mon = m[m.period_type == "month"].copy()
    mon["date"] = pd.to_datetime(mon["date"])
    g = mon[mon.breakdown == "grower"]
    c = mon[mon.breakdown == "category"]
    ann = m[(m.period_type == "calendar_year") & (m.geography == "All India")].set_index("period")["production_mkg"]
    ann.index = ann.index.astype(int)
    india = g[(g.geography == "All India") & (g.segment == "Total")].set_index("date")["production_mkg"].sort_index()
    ann.loc[2025] = india.loc["2025"].sum()
    return m, g, c, ann.sort_index(), india


# -------------------------------------------------------------- overview --
AREA_LONLAT = {  # approximate centre of each tea-growing area
    "Assam Valley": (93.9, 26.9), "Cachar": (92.8, 24.8), "Darjeeling": (88.26, 27.04),
    "Terai": (88.35, 26.55), "Dooars": (89.4, 26.75), "Tamil Nadu": (76.75, 11.35),
    "Kerala": (77.06, 10.1), "Karnataka": (75.75, 13.3),
}
TEA_STATES = {"IN-AS": "Assam", "IN-WB": "West Bengal", "IN-TN": "Tamil Nadu", "IN-KL": "Kerala", "IN-KA": "Karnataka"}


def draw_states(ax, gj, share, norm):
    for f in gj["features"]:
        iso = f["properties"]["shapeISO"]
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        name = TEA_STATES.get(iso)
        face = SEQ(norm(share[name])) if name else "#efeeea"
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=face, edgecolor=SURFACE, linewidth=0.6))


def bubbles(ax, rows, scale, label=True, offsets=None):
    for area, prod, sg in rows:
        if area not in AREA_LONLAT:
            continue
        x, y = AREA_LONLAT[area]
        ax.scatter(x, y, s=prod * scale, color=ORANGE, alpha=0.75, edgecolor=INK, linewidth=0.8, zorder=5)
        if label:
            dx, dy = (offsets or {}).get(area, (0.35, 0.0))
            ax.annotate(f"{area}\n{prod:,.0f} M kg · {sg:.0f}% small growers", (x, y), xytext=(x + dx, y + dy),
                        fontsize=8.5, color=INK, zorder=6, va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.7} if abs(dx) + abs(dy) > 0.6 else None)


def chart_map(g):
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))
    y25 = g[(g.year == 2025)]
    st = y25[(y25.segment == "Total") & y25.geography.isin(["Assam", "West Bengal", "Tamil Nadu", "Kerala", "Karnataka"])]
    total = y25[(y25.geography == "All India") & (y25.segment == "Total")]["production_mkg"].sum()
    share = (100 * st.groupby("geography")["production_mkg"].sum() / total).to_dict()
    ar = y25[y25.geo_level == "area"].pivot_table(index="geography", columns="segment", values="production_mkg", aggfunc="sum")
    rows = [(a, ar.loc[a, "Total"], 100 * ar.loc[a, "SG"] / ar.loc[a, "Total"]) for a in ar.index]
    norm = Normalize(0, 55)

    fig = plt.figure(figsize=(12, 7.6))
    ax = fig.add_axes([0.0, 0.0, 0.5, 0.86])
    ax2 = fig.add_axes([0.52, 0.30, 0.47, 0.56])
    for a in (ax, ax2):
        draw_states(a, gj, share, norm)
        a.set_aspect("equal")
        a.axis("off")
    ax.set_xlim(68, 98); ax.set_ylim(6.5, 37.2)
    ax2.set_xlim(87.4, 96.4); ax2.set_ylim(23.6, 28.4)
    ax.add_patch(Rectangle((87.4, 23.6), 9, 4.8, fill=False, edgecolor=INK2, linewidth=1, linestyle="-"))
    ax.text(87.4, 28.7, "zoom →", fontsize=8.5, color=INK2)
    bubbles(ax, [r for r in rows if r[0] in ("Tamil Nadu", "Kerala", "Karnataka")], 2.2,
            offsets={"Karnataka": (-6.5, 1.8), "Tamil Nadu": (1.6, 1.4), "Kerala": (1.4, -1.4)})
    bubbles(ax, [r for r in rows if r[0] not in ("Tamil Nadu", "Kerala", "Karnataka")], 2.2, label=False)
    bubbles(ax2, [r for r in rows if r[0] not in ("Tamil Nadu", "Kerala", "Karnataka")], 6,
            offsets={"Assam Valley": (-1.6, 1.15), "Cachar": (0.9, -0.35), "Darjeeling": (-0.4, 0.9),
                     "Terai": (-0.1, -1.55), "Dooars": (1.4, -0.95)})
    ax2.set_title("North-East India (zoomed)", loc="left", fontsize=10, color=INK2)
    for a in (ax, ax2):
        for v in (0, 1):
            a.spines[["left", "bottom"][v]].set_visible(False)
    # colour scale + bubble key
    cax = fig.add_axes([0.56, 0.20, 0.36, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=cax, orientation="horizontal")
    cb.set_label("State share of All-India production, 2025 (%)", color=INK2)
    cb.outline.set_visible(False)
    kax = fig.add_axes([0.56, 0.03, 0.36, 0.12])
    kax.axis("off")
    kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    for i, v in enumerate((50, 200, 600)):
        kax.scatter(0.12 + i * 0.28, 0.5, s=v * 2.2, color=ORANGE, alpha=0.75, edgecolor=INK, linewidth=0.8)
        kax.text(0.12 + i * 0.28 + 0.07, 0.5, f"{v} M kg", va="center", fontsize=9, color=INK2)
    other = 100 - sum(share.values())
    fig.text(0.0, 0.97, "Where India's tea grows, 2025", fontsize=14, fontweight="bold")
    fig.text(0.0, 0.935, f"States shaded by share of national output; bubbles sized by each area's production (main map "
             f"scaled smaller than the zoom). Other northern states ({other:.1f}%) are combined in the source and not mapped.",
             fontsize=9.5, color=INK2)
    fig.text(0.0, 0.0, "Boundaries: geoBoundaries gbOpen IND ADM1 (DataMeet / ECI, CC BY 2.5 IN). Bubble positions approximate.",
             fontsize=7.5, color=MUTED)
    save(fig, "d01_map.png")


def chart_waterfall(ann):
    yrs = list(ann.index)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(0, ann.iloc[0], color=MUTED, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.text(0, ann.iloc[0] + 8, f"{ann.iloc[0]:,.0f}", ha="center", fontsize=9, color=INK2)
    level = ann.iloc[0]
    for i, y in enumerate(yrs[1:], start=1):
        d = ann[y] - ann[yrs[i - 1]]
        ax.bar(i, d, bottom=level, color=BLUE if d >= 0 else RED, width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.plot([i - 1 + 0.3, i - 0.3], [level, level], color=BASE, linewidth=1)
        ax.text(i, level + d + (8 if d >= 0 else -8), f"{d:+.0f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9, color=INK2)
        level += d
    n = len(yrs)
    ax.bar(n, level, color=INK2, width=0.6, edgecolor=SURFACE, linewidth=2)
    ax.text(n, level + 8, f"{level:,.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(n + 1), [str(yrs[0])] + [str(y) for y in yrs[1:]] + ["2025\ntotal"])
    ax.set_ylim(1150, 1450)
    ax.set_ylabel("All-India production (million kg)")
    titled(ax, "Year-on-year swings: two big drops, both recovered",
           "Waterfall of calendar-year production 2017 → 2025 · blue = rise, red = fall · axis starts at 1,150 M kg")
    save(fig, "d02_waterfall.png")


def squarify(values, x, y, w, h):
    """Squarified treemap layout (Bruls et al.) - returns rectangles in input order."""
    vals = list(values)
    total = sum(vals)
    vals = [v * w * h / total for v in vals]
    rects, i = [], 0
    while i < len(vals):
        short = min(w, h)
        row, best = [vals[i]], None
        j = i + 1

        def worst(r):
            s = sum(r)
            return max(max(s * s / (short * short * v), short * short * v / (s * s)) for v in r)

        best = worst(row)
        while j < len(vals) and worst(row + [vals[j]]) <= best:
            row.append(vals[j]); best = worst(row); j += 1
        s = sum(row)
        if w >= h:
            cw = s / h
            cy = y
            for v in row:
                rects.append((x, cy, cw, v / cw)); cy += v / cw
            x += cw; w -= cw
        else:
            ch = s / w
            cx = x
            for v in row:
                rects.append((cx, y, v / ch, ch)); cx += v / ch
            y += ch; h -= ch
        i = j
    return rects


def chart_treemap(g):
    y25 = g[(g.year == 2025) & (g.geo_level == "area") & (g.segment.isin(["BG", "SG"]))]
    t = y25.groupby(["state", "geography", "segment"])["production_mkg"].sum().reset_index()
    t["state"] = t["state"].replace({"Others (North)": "Others (North)"})
    t = t.sort_values("production_mkg", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axis("off")
    rects = squarify(t.production_mkg.values, 0, 0, 100, 60)
    total = t.production_mkg.sum()
    for (x, y, w, h), r in zip(rects, t.itertuples()):
        col = STATE_COLOR[r.state]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=col, edgecolor=SURFACE, linewidth=2,
                               alpha=1.0 if r.segment == "SG" else 0.62))
        if w > 9 and h > 7:
            name = r.geography.replace("Others (NE & HP etc.)", "Others")
            ax.text(x + 0.8, y + 0.9, f"{name}\n{'Small growers' if r.segment == 'SG' else 'Estates'}\n"
                    f"{r.production_mkg:,.0f} M kg ({100 * r.production_mkg / total:.1f}%)",
                    fontsize=8.5 if w > 8 else 7.2, color="white", va="top", fontweight="semibold")
    ax.set_xlim(0, 100); ax.set_ylim(60, 0)
    handles = [Rectangle((0, 0), 1, 1, color=STATE_COLOR[s]) for s in STATES]
    ax.legend(handles, STATES, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8.5)
    titled(ax, "Who makes India's tea: every area × grower block, 2025",
           f"Treemap, area ∝ production ({total:,.0f} M kg) · colour = state · solid = small growers, lighter = estates")
    save(fig, "d03_treemap.png")


# ----------------------------------------------------------- seasonality --
def chart_tea_clock(g):
    fig, ax = plt.subplots(figsize=(7.6, 7.6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    width = 2 * np.pi / 12 * 0.86
    north = g[(g.geography == "North India") & (g.segment == "Total") & (g.year <= 2025)].groupby("month_num")["production_mkg"].median()
    south = g[(g.geography == "South India") & (g.segment == "Total") & (g.year <= 2025)].groupby("month_num")["production_mkg"].median()
    ax.bar(theta, south.values, width=width, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label="South India")
    ax.bar(theta, north.values, width=width, bottom=south.values, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="North India")
    ax.set_xticks(theta, MONTHS)
    ax.set_yticks([50, 100, 150]); ax.set_yticklabels(["50", "100", "150 M kg"], fontsize=8, color=MUTED)
    ax.set_rlabel_position(100)
    ax.grid(color=GRID, linewidth=0.7)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.text(0.02, 1.0, "The tea clock: a year of production", fontsize=13, fontweight="bold", va="bottom")
    fig.text(0.02, 0.975, "Median monthly output 2019–2025, stacked North + South · winter gap is North India's dormancy",
             fontsize=9.5, color=INK2, va="bottom")
    save(fig, "d04_tea_clock.png")


def chart_cycle(india):
    d = india.loc[:"2025-12"].to_frame("v")
    d["m"], d["y"] = d.index.month, d.index.year
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for mth in range(1, 13):
        s = d[d.m == mth].sort_values("y")
        xs = (mth - 1) + np.linspace(0.12, 0.88, len(s))
        ax.plot(xs, s.v, color=BLUE, linewidth=1.6, marker="o", markersize=3.5)
        ax.hlines(s.v.mean(), xs[0], xs[-1], color=ORANGE, linewidth=2)
        for yr, xv, vv in zip(s.y, xs, s.v):
            if (yr, mth) in ((2020, 4), (2024, 12)):
                ax.annotate(f"{MONTHS[mth - 1]} {yr}", (xv, vv), xytext=(4, -12), textcoords="offset points",
                            fontsize=8, color=RED)
    for b in range(1, 12):
        ax.axvline(b, color=GRID, linewidth=0.8)
    ax.set_xticks(np.arange(12) + 0.5, MONTHS)
    ax.set_xlim(0, 12)
    ax.set_ylabel("All-India production (million kg)")
    ax.plot([], [], color=BLUE, marker="o", label="Each year 2019 → 2025 (left to right)")
    ax.plot([], [], color=ORANGE, linewidth=2, label="7-year mean for that month")
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, 230)
    titled(ax, "Cycle plot: the seasonal shape is stable, years only nudge it",
           "Each panel is one calendar month; its line shows that month across 2019–2025")
    save(fig, "d05_cycle_plot.png")


def chart_shock_lollipop(india):
    d = india.loc[:"2025-12"].to_frame("v")
    d["m"] = d.index.month
    d["norm"] = d.groupby("m")["v"].transform("median")
    d["dev"] = 100 * (d.v / d.norm - 1)
    d = d[d.index >= "2019-03-01"]  # Jan/Feb North India output is ~0, so % deviations are not meaningful
    d = d[~d.m.isin([1, 2])]
    fig, ax = plt.subplots(figsize=(11.5, 5))
    cols = [RED if v < -15 else (BLUE if v > 15 else MUTED) for v in d.dev]
    ax.vlines(d.index, 0, d.dev, color=cols, linewidth=1.6)
    ax.scatter(d.index, d.dev, color=cols, s=26, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.axhline(0, color=BASE, linewidth=1)
    for lim in (-15, 15):
        ax.axhline(lim, color=GRID, linewidth=1)
    for dt in d.dev.nsmallest(3).index:
        ax.annotate(f"{dt:%b %Y}: {d.loc[dt, 'dev']:.0f}%", (dt, d.loc[dt, "dev"]), xytext=(6, -2),
                    textcoords="offset points", fontsize=8.5, color=INK2, va="top")
    ax.set_ylabel("Departure from that month's 2019–2025 median (%)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    titled(ax, "Shocks stand out against the seasonal norm",
           "Lollipop chart, Mar–Dec months only (Jan–Feb North India output ≈ 0) · red/blue beyond ±15%")
    save(fig, "d06_shock_lollipop.png")


# ------------------------------------------------------------- structure --
def chart_dumbbell(g):
    a = g[(g.geo_level == "area") & g.year.isin([2019, 2025]) & g.segment.isin(["SG", "Total"])]
    p = a.pivot_table(index=["geography", "year"], columns="segment", values="production_mkg", aggfunc="sum")
    sh = (100 * p["SG"] / p["Total"]).unstack("year").sort_values(2025)
    sh.index = [i.replace("Others (NE & HP etc.)", "Other northern states") for i in sh.index]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(sh))
    ax.hlines(y, sh[2019], sh[2025], color=BASE, linewidth=3)
    ax.scatter(sh[2019], y, s=70, color=MUTED, zorder=3, label="2019", edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(sh[2025], y, s=90, color=BLUE, zorder=3, label="2025", edgecolor=SURFACE, linewidth=1.5)
    for yi, (a19, a25) in zip(y, sh[[2019, 2025]].values):
        ax.text(max(a19, a25) + 1.5, yi, f"{a25 - a19:+.1f} pts", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, sh.index)
    ax.axvline(50, color=BASE, linewidth=1)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Small-grower share of the area's production (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    titled(ax, "Small growers gained ground in every area, most in the Dooars",
           "Dumbbell plot: small-grower share 2019 (grey) → 2025 (blue) · 2024 values not used")
    save(fig, "d07_dumbbell.png")


def chart_marimekko(g):
    y = g[(g.year == 2025) & g.segment.isin(["BG", "SG"])]
    y = y[y.geography.isin(["Assam", "West Bengal", "Others (NE & HP etc.)", "Tamil Nadu", "Kerala", "Karnataka"])]
    p = y.pivot_table(index="geography", columns="segment", values="production_mkg", aggfunc="sum")
    p = p.loc[["Assam", "West Bengal", "Tamil Nadu", "Kerala", "Others (NE & HP etc.)", "Karnataka"]]
    tot = p.sum(axis=1)
    widths = 100 * tot / tot.sum()
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = 0
    for (state, r), w in zip(p.iterrows(), widths):
        sgp = 100 * r["SG"] / (r["SG"] + r["BG"])
        ax.add_patch(Rectangle((x, 0), w, sgp, facecolor=ORANGE, edgecolor=SURFACE, linewidth=2))
        ax.add_patch(Rectangle((x, sgp), w, 100 - sgp, facecolor=BLUE, edgecolor=SURFACE, linewidth=2))
        name = state.replace("Others (NE & HP etc.)", "Others")
        if w > 3.5:
            ax.text(x + w / 2, sgp / 2, f"{sgp:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="semibold")
            ax.text(x + w / 2, sgp + (100 - sgp) / 2, f"{100 - sgp:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="semibold")
        if w >= 9:
            ax.text(x + w / 2, -4, f"{name}\n{w:.1f}% of output", ha="center", va="top", fontsize=8.5, color=INK2)
        elif w >= 1:  # narrow columns: stagger labels so they don't collide
            ax.text(x + w / 2, -4 if name == "Kerala" else -13, f"{name}\n{w:.1f}%", ha="center", va="top", fontsize=8, color=INK2)
        else:
            ax.text(101, 3, f"{name}\n({w:.1f}%)", fontsize=7.5, color=MUTED, va="bottom")
        x += w
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_ylabel("Share within the state (%)")
    ax.grid(False)
    ax.legend([Rectangle((0, 0), 1, 1, color=ORANGE), Rectangle((0, 0), 1, 1, color=BLUE)],
              ["Small growers", "Big growers (estates)"], ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    titled(ax, "Marimekko: state size and who grows the tea, 2025",
           "Column width = state's share of national output · column split = small growers vs estates")
    fig.subplots_adjust(bottom=0.22)
    save(fig, "d08_marimekko.png")


def chart_slope(c):
    d = c[(c.year.isin([2019, 2025])) & c.geography.isin(["North India", "South India"])]
    p = d.pivot_table(index=["geography", "year"], columns="segment", values="production_mkg", aggfunc="sum")
    fig, axes = plt.subplots(1, 3, figsize=(11, 5), sharey=False)
    for ax, cat in zip(axes, ["CTC", "Orthodox", "Green"]):
        for zone in ("North India", "South India"):
            v = [100 * p.loc[(zone, yr), cat] / p.loc[(zone, yr), "Total"] for yr in (2019, 2025)]
            ax.plot([0, 1], v, color=ZONE[zone], marker="o", markersize=7, linewidth=2)
            ax.text(-0.06, v[0], f"{v[0]:.1f}%", ha="right", va="center", fontsize=9, color=INK2)
            ax.text(1.06, v[1], f"{v[1]:.1f}%", ha="left", va="center", fontsize=9, color=INK2)
        ax.set_xticks([0, 1], ["2019", "2025"])
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(f"{cat} share", loc="left", fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
    axes[0].plot([], [], color=BLUE, marker="o", label="North India")
    axes[0].plot([], [], color=ORANGE, marker="o", label="South India")
    fig.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.01, 0.9))
    fig.suptitle("Slope charts: how the tea mix shifted, 2019 → 2025", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.01, 0.95, "Share of each zone's output by tea type · each panel has its own scale", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "d09_slope_tea_types.png")


# ----------------------------------------------------------------- model --
MODEL_ORDER = ["Seasonal median", "Random forest", "Ridge regression", "Gradient boosting", "Seasonal naive"]


def chart_radar():
    s = pd.read_csv(REPORTS / "model_backtest_summary.csv").set_index("model").loc[MODEL_ORDER]
    metrics = {"MAE": "india_MAE", "RMSE": "india_RMSE", "WAPE": "india_WAPE_pct",
               "Annual error": "abs_annual_error_pct", "Series R²": "series_R2"}
    score = pd.DataFrame({k: (s[v].min() / s[v] if k != "Series R²" else s[v] / s[v].max()) for k, v in metrics.items()})
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.8, 7.4), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    style = {"Seasonal median": (ORANGE, 2.4, 0.12), "Random forest": (BLUE, 2.4, 0.12)}
    for mname in MODEL_ORDER:
        vals = score.loc[mname].tolist() + score.loc[mname].tolist()[:1]
        col, lw, fill = style.get(mname, (BASE, 1.3, 0))
        ax.plot(ang, vals, color=col, linewidth=lw, label=mname)
        if fill:
            ax.fill(ang, vals, color=col, alpha=fill)
    ax.set_xticks(ang[:-1], list(metrics))
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks([0.6, 0.8, 1.0]); ax.set_yticklabels(["0.6", "0.8", "1.0 = best"], fontsize=8, color=MUTED)
    ax.grid(color=GRID)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    fig.text(0.02, 1.0, "Radar: model scorecard across five metrics", fontsize=13, fontweight="bold", va="bottom")
    fig.text(0.02, 0.975, "Mean of 4 backtest years · each metric scaled so the best model = 1 · grey = other models",
             fontsize=9.5, color=INK2, va="bottom")
    save(fig, "d10_radar_models.png")


def chart_error_heatmap():
    p = pd.read_csv(REPORTS / "backtest_predictions.csv", parse_dates=["date"])
    a = p.groupby(["model", "fold", "date"])[["y", "pred"]].sum().reset_index()
    a["ape"] = 100 * (a.y - a.pred).abs() / a.y
    a["m"] = a.date.dt.month
    h = a.groupby(["model", "m"])["ape"].mean().unstack().loc[MODEL_ORDER]
    order = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
    h = h[order]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(h.values, cmap=LinearSegmentedColormap.from_list("e", ["#f0efec", "#f3b49f", "#e34948", "#8e1b1b"]),
                   aspect="auto", vmin=0, vmax=max(40, np.nanmax(h.values)))
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v > 25 else INK)
    ax.set_xticks(range(12), [MONTHS[k - 1] for k in order])
    ax.set_yticks(range(len(h)), h.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Mean absolute % error", color=INK2); cb.outline.set_visible(False)
    titled(ax, "Where forecasts go wrong: error by model and month",
           "Mean absolute % error of the All-India monthly forecast over 4 backtest years · financial-year order Apr → Mar")
    save(fig, "d11_error_heatmap.png")


def chart_violin():
    p = pd.read_csv(REPORTS / "backtest_predictions.csv", parse_dates=["date"])
    a = p.groupby(["model", "fold", "date"])[["y", "pred"]].sum().reset_index()
    a["e"] = a.y - a.pred
    data = [a[a.model == m]["e"].values for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, m in zip(parts["bodies"], MODEL_ORDER):
        body.set_facecolor(BLUE if m == "Random forest" else (ORANGE if m == "Seasonal median" else BASE))
        body.set_edgecolor(SURFACE); body.set_alpha(0.75)
    for i, d in enumerate(data, start=1):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.vlines(i, q1, q3, color=INK, linewidth=3)
        ax.scatter(i, med, color=SURFACE, edgecolor=INK, s=30, zorder=3)
        ax.scatter(np.full(len(d), i) + np.random.default_rng(1).uniform(-0.06, 0.06, len(d)), d, s=7, color=INK2, alpha=0.35, zorder=2)
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks(range(1, 6), MODEL_ORDER)
    ax.set_ylabel("Forecast error, actual − predicted (M kg)")
    titled(ax, "Violin plot: error distribution of each model",
           "48 All-India monthly errors per model (4 backtest years) · bar = middle 50%, dot = median")
    save(fig, "d12_violin_errors.png")


def chart_fan(india):
    fc = pd.read_csv(REPORTS / "forecast_FY2026-27.csv", parse_dates=["month"]).set_index("month")["ml_forecast_mkg"]
    p = pd.read_csv(REPORTS / "backtest_predictions.csv")
    a = p[p.model == "Random forest"].groupby(["fold", "date"])[["y", "pred"]].sum()
    rel = (a.y / a.pred - 1).values
    hist = india.loc["2024-04":]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for lo, hi, alpha in ((2.5, 97.5, 0.12), (10, 90, 0.2), (25, 75, 0.32)):
        ql, qh = np.percentile(rel, [lo, hi])
        ax.fill_between(fc.index, fc * (1 + ql), fc * (1 + qh), color=BLUE, alpha=alpha, linewidth=0,
                        label=f"{hi - lo:.0f}% band")
    ax.plot(hist.index, hist.values, color=INK, linewidth=1.8, label="Actual")
    ax.plot(fc.index, fc.values, color=BLUE, linewidth=2, marker="o", markersize=4, label="Random Forest forecast")
    ax.axvline(pd.Timestamp("2026-03-16"), color=BASE, linewidth=1)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b-%y"))
    ax.set_ylabel("All-India production (million kg)")
    ax.set_ylim(0, 260)
    ax.legend(loc="upper left", ncol=5, fontsize=8.5)
    titled(ax, f"Fan chart: FY 2026-27 outlook ≈ {fc.sum():,.0f} M kg",
           "Bands from the spread of the model's own backtest errors (empirical 50 / 80 / 95% ranges)")
    save(fig, "d13_fan_forecast.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, g, c, ann, india = load()
    print("Dashboard charts:")
    chart_map(g); chart_waterfall(ann); chart_treemap(g)
    chart_tea_clock(g); chart_cycle(india); chart_shock_lollipop(india)
    chart_dumbbell(g); chart_marimekko(g); chart_slope(c)
    chart_radar(); chart_error_heatmap(); chart_violin(); chart_fan(india)


if __name__ == "__main__":
    main()
