"""Step 5 - Fill the DAV Micro Project report template.

Usage:  python src/report.py   ->  report/DAV_Micro_Project_Report.docx

Reads the template report/template.docx (copied from the faculty's
"DAV Micro Project_Report_Format.docx"), leaves page 1 (student details) untouched,
and fills Parts 1-5 with this project's dataset, model, statistics, blog and results.
Also produces the extra charts the format asks for (histogram, box plot, scatter,
correlation heatmap) and renders the source PDFs as evidence images.
Run src/extract.py and src/analysis.py first.
"""
from __future__ import annotations

import re
import shutil
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
OUTDIR = ROOT / "report"
ASSETS = OUTDIR / "assets"
TEMPLATE = OUTDIR / "template.docx"
OUT = OUTDIR / "DAV_Micro_Project_Report_final.docx"

BASE_URL = "https://www.teaboard.gov.in/pdf/"
RELEASES = [  # file, what it covers
    ("Website_Production_Data_for_2019_2019_20_FINAL_pdf2416.pdf", "Jan 2019 – Mar 2020 (2019 and FY 2019-20), final"),
    ("Production_Data_for_2020_2020_21_pdf4425.pdf", "Jan 2020 – Mar 2021 (2020 and FY 2020-21), final"),
    ("Production_data_for_2021_and_2021_22_pdf223.pdf", "Jan 2021 – Mar 2022 (2021 and FY 2021-22), final"),
    ("Production_2022_and_2022_23_pdf5959.pdf", "Jan 2022 – Mar 2023 (2022 and FY 2022-23), final"),
    ("State_Region_wise_tea_production_2024_and_2024_25_Apr_Dec_pdf4332.pdf", "Jan 2023 – Jan 2025 (2023, 2024, FY 2023-24, FY 2024-25 to Jan); Apr-2024 onwards provisional"),
    ("Annual_Production_2025_and_2025_26_pdf5398.pdf", "Jan 2025 – Mar 2026 (2025 and FY 2025-26), final"),
    ("Production_Region_wise_FINAL_pdf9516.pdf", "Annual totals 2017–2024 and FY 2017-18 to FY 2024-25, final"),
]

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BASE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
})


def titled(ax, title, sub=""):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=24 if sub else 10)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")


# ------------------------------------------------------------- statistics --
def compute() -> dict:
    p = pd.read_csv(REPORTS / "backtest_predictions.csv", parse_dates=["date"])
    agg = p.groupby(["model", "fold", "date"])[["y", "pred"]].sum().reset_index()
    agg["e"] = agg.y - agg.pred
    agg["ae"] = agg.e.abs()

    def metrics(g):
        mse = (g.e ** 2).mean()
        return {"MAE": g.ae.mean(), "MSE": mse, "RMSE": mse ** 0.5,
                "R2": 1 - (g.e ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum(),
                "WAPE": 100 * g.ae.sum() / g.y.sum()}

    test = {m: metrics(g) for m, g in agg[agg.fold == "2025-26"].groupby("model")}
    folds = {m: pd.DataFrame([metrics(gg) for _, gg in g.groupby("fold")]).mean().to_dict()
             for m, g in agg.groupby("model")}
    w = agg.pivot_table(index=["fold", "date"], columns="model", values="ae")
    tt = {}
    for a, b in [("Random forest", "Seasonal naive"), ("Random forest", "Seasonal median"),
                 ("Seasonal median", "Seasonal naive")]:
        r = stats.ttest_rel(w[a], w[b])
        tt[(a, b)] = {"t": r.statistic, "df": len(w) - 1, "p": r.pvalue, "diff": (w[a] - w[b]).mean(),
                      "wilcoxon_p": stats.wilcoxon(w[a], w[b]).pvalue}
    shapiro = stats.shapiro(w["Random forest"] - w["Seasonal naive"])
    rf_all = agg[agg.model == "Random forest"]
    bias = stats.ttest_1samp(rf_all.e, 0)
    fy26 = agg[(agg.fold == "2025-26") & (agg.model == "Random forest")].sort_values("date")
    series26 = p[(p.fold == "2025-26") & (p.model == "Random forest")]
    train_rows = len(p[(p.fold == "2025-26")]) // p.model.nunique()
    return {"agg": agg, "test": test, "folds": folds, "tt": tt, "shapiro": shapiro, "bias": bias,
            "bias_mean": rf_all.e.mean(), "fy26": fy26, "series26": series26, "test_rows": train_rows}


# ---------------------------------------------------------------- figures --
def figures(S: dict) -> None:
    FIG.mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(DATA / "tea_production_master.csv")
    india = m[(m.period_type == "month") & (m.breakdown == "grower") & (m.geography == "All India")
              & (m.segment == "Total")].copy()

    # 13 histogram of forecast errors
    e = S["agg"][S["agg"].model == "Random forest"]["e"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(e, bins=12, color=BLUE, edgecolor=SURFACE, linewidth=2)
    ax.axvline(0, color=BASE, linewidth=1.2)
    ax.axvline(e.mean(), color=ORANGE, linewidth=2)
    ax.text(e.mean() + 0.8, ax.get_ylim()[1] * 0.92, f"mean {e.mean():+.1f}", color=INK2, fontsize=9)
    ax.set_xlabel("Forecast error, actual − predicted (million kg, All-India month)")
    ax.set_ylabel("Number of months")
    titled(ax, "Random Forest errors are centred on zero",
           f"48 held-out months, FY 2022-23 to FY 2025-26 · one-sample t-test of bias p = {S['bias'].pvalue:.2f}")
    fig.tight_layout(); fig.savefig(FIG / "13_error_histogram.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 14 box plot by calendar month
    d = india[india.year <= 2025]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = [d[d.month_num == k]["production_mkg"].values for k in range(1, 13)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                    flierprops={"marker": "o", "markerfacecolor": RED, "markeredgecolor": SURFACE, "markersize": 7})
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=BLUE)
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("All-India production (million kg)")
    titled(ax, "Each month has its own level; outliers are the 2020 lockdown and Dec-2024",
           "Distribution of All-India production by calendar month, 2019–2025 (7 years per box)")
    fig.tight_layout(); fig.savefig(FIG / "14_monthly_boxplot.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 15 scatter actual vs predicted (series level, FY 2025-26)
    s = S["series26"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for zone, col in (("North India", BLUE), ("South India", ORANGE)):
        z = s[s.zone == zone]
        ax.scatter(z.y, z.pred, s=34, color=col, edgecolor=SURFACE, linewidth=1, label=zone, alpha=0.9)
    lim = max(s.y.max(), s.pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1.2, label="Perfect forecast (y = x)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(axis="both")
    ax.set_xlabel("Actual production (million kg)")
    ax.set_ylabel("Predicted production (million kg)")
    r2 = 1 - ((s.y - s.pred) ** 2).sum() / ((s.y - s.y.mean()) ** 2).sum()
    ax.legend(loc="upper left", frameon=False)
    titled(ax, "Predicted vs actual, every series in FY 2025-26",
           f"Random Forest · 18 area × grower series × 12 months = {len(s)} points · R² = {r2:.3f}")
    fig.tight_layout(); fig.savefig(FIG / "15_actual_vs_predicted_scatter.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 16 correlation heatmap of numeric features and target
    leaf = pd.read_csv(DATA / "monthly_area_grower.csv", parse_dates=["date"])
    leaf["series"] = leaf.area + "|" + leaf.grower
    P = leaf.pivot(index="date", columns="series", values="production_mkg")
    rows = []
    for c in P.columns:
        y = P[c]
        h = pd.DataFrame({"v": y.shift(12), "m": y.index.month}, index=y.index)
        rows.append(pd.DataFrame({
            "Production (target)": y, "Same month −1 yr": y.shift(12), "Same month −2 yr": y.shift(24),
            "Median of month": h.groupby("m")["v"].transform(lambda v: v.expanding().median()),
            "Mean of month": h.groupby("m")["v"].transform(lambda v: v.expanding().mean()),
            "Month (sine)": np.sin(2 * np.pi * y.index.month / 12),
            "Month (cosine)": np.cos(2 * np.pi * y.index.month / 12)}))
    corr = pd.concat(rows).dropna().corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#f0efec", "#1c5cab"])
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=2, linecolor=SURFACE,
                cbar_kws={"label": "Pearson correlation", "shrink": 0.8}, ax=ax, annot_kws={"fontsize": 9})
    ax.grid(False)
    ax.tick_params(axis="x", rotation=35)
    titled(ax, "History features correlate 0.96–0.98 with the target",
           "Pearson correlation of numeric model inputs and the target, all series 2021–2026")
    fig.tight_layout(); fig.savefig(FIG / "16_feature_correlation.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # Source evidence: render the first page of two original PDFs
    for name, out in (("Annual_Production_2025_and_2025_26_pdf5398.pdf", "source_pdf_2025.png"),
                      ("Production_Region_wise_FINAL_pdf9516.pdf", "source_pdf_regionwise.png")):
        pdf = pdfium.PdfDocument(str(RAW / name))
        pdf[0].render(scale=1.6).to_pil().save(ASSETS / out)
        pdf.close()


# ------------------------------------------------------------- docx XML --
W_NS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
PIC_NS = 'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'


def runs(text: str, size: int | None = None, italic: bool = False, color: str | None = None) -> str:
    """Text with **bold** segments -> w:r runs."""
    out = []
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if not part:
            continue
        rpr = ("<w:b/>" if i % 2 else "") + ("<w:i/>" if italic else "") + \
              (f'<w:color w:val="{color}"/>' if color else "") + (f'<w:sz w:val="{size}"/>' if size else "")
        out.append(f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


def P(text: str = "", style: str | None = None, align: str | None = None, size: int | None = None,
      italic: bool = False, color: str | None = None, keep_next: bool = False, after: int | None = None) -> str:
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + ("<w:keepNext/>" if keep_next else "") + \
          (f'<w:spacing w:after="{after}"/>' if after is not None else "") + (f'<w:jc w:val="{align}"/>' if align else "")
    return f'<w:p>{"<w:pPr>" + ppr + "</w:pPr>" if ppr else ""}{runs(text, size, italic, color)}</w:p>'


def bullet(text: str) -> str:
    return P(text, style="ListBullet")


def H2(text: str) -> str:
    return P(text, style="Heading2")


LINKS: list[str] = []  # external hyperlink targets, added to document.xml.rels


def hyperlink(url: str, size: int = 19) -> str:
    """Clickable link; zero-width spaces in the visible text only let it wrap in narrow cells."""
    LINKS.append(url)
    shown = escape(url).replace("/", "/​").replace("_", "_​")
    return (f'<w:hyperlink r:id="rIdLink{len(LINKS)}" w:history="1"><w:r><w:rPr><w:color w:val="0563C1"/>'
            f'<w:u w:val="single"/><w:sz w:val="{size}"/></w:rPr><w:t xml:space="preserve">{shown}</w:t></w:r></w:hyperlink>')


def table(header: list[str], rows: list[list[str]], widths: list[int], bold_first_col: bool = False) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

    def cell(txt, w, head=False, bold=False):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="EAF0FF"/>' if head else ""
        t = f"**{txt}**" if (head or bold) and txt else txt
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}<w:vAlign w:val="center"/></w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:before="20" w:after="20"/></w:pPr>'
                f'{hyperlink(txt) if txt.startswith("http") else runs(t, 19)}</w:p></w:tc>')

    head = '<w:tr><w:trPr><w:tblHeader/><w:jc w:val="center"/></w:trPr>' + \
           "".join(cell(h, w, head=True) for h, w in zip(header, widths)) + "</w:tr>"
    body = "".join('<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>' +
                   "".join(cell(str(v), w, bold=(bold_first_col and j == 0)) for j, (v, w) in enumerate(zip(r, widths)))
                   + "</w:tr>" for r in rows)
    return (f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
            f'<w:jc w:val="center"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
            f'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{head}{body}</w:tbl>')


class Images:
    def __init__(self):
        self.items: list[tuple[str, Path]] = []

    def add(self, path: Path, caption: str, width_in: float = 6.6) -> str:
        n = len(self.items) + 1
        rid = f"rIdRpt{n}"
        self.items.append((rid, path))
        with open(path, "rb") as f:
            f.seek(16)
            w, h = struct.unpack(">II", f.read(8))
        cx = int(width_in * 914400)
        cy = int(cx * h / w)
        pid = 5000 + n
        drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                   f'<wp:docPr id="{pid}" name="Figure {n}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks {W_NS} noChangeAspect="1"/></wp:cNvGraphicFramePr>'
                   f'<a:graphic {W_NS}><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                   f'<pic:pic {PIC_NS}><pic:nvPicPr><pic:cNvPr id="{pid}" name="{path.name}"/><pic:cNvPicPr/></pic:nvPicPr>'
                   f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                   f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
                   f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>')
        return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="40"/><w:jc w:val="center"/></w:pPr><w:r>{drawing}</w:r></w:p>'
                + P(f"Figure {n}: {caption}", align="center", size=18, italic=True, color="52514E", after=200))


def body_elements(doc: str) -> tuple[str, list[str], str]:
    """Split document.xml into (prefix up to <w:body>, top-level body elements, suffix)."""
    start = doc.index("<w:body>") + len("<w:body>")
    end = doc.rindex("</w:body>")
    inner = doc[start:end]
    elems, depth, i0 = [], 0, None
    for mt in re.finditer(r"<(/?)(w:[A-Za-z]+)\b[^>]*?(/?)>", inner):
        closing, selfclose = mt.group(1) == "/", mt.group(3) == "/"
        if not closing and depth == 0:
            i0 = mt.start()
        if selfclose:
            if depth == 0:
                elems.append(inner[i0:mt.end()])
            continue
        depth += -1 if closing else 1
        if depth == 0 and closing:
            elems.append(inner[i0:mt.end()])
    return doc[:start], elems, doc[end:]


# --------------------------------------------------------------- content --
def build(S: dict) -> None:
    img = Images()
    stats_json = __import__("json").loads((REPORTS / "statistics.json").read_text())
    master = pd.read_csv(DATA / "tea_production_master.csv")
    leaf = pd.read_csv(DATA / "monthly_area_grower.csv")
    t, f = S["test"], S["folds"]
    rf, sm, sn = t["Random forest"], t["Seasonal median"], t["Seasonal naive"]
    main_t = S["tt"][("Random forest", "Seasonal naive")]
    rf_sm = S["tt"][("Random forest", "Seasonal median")]
    sm_sn = S["tt"][("Seasonal median", "Seasonal naive")]
    fc = stats_json["forecast_FY2026_27"]
    W2 = [5112, 5112]
    R = {}  # template element index -> replacement XML (list joined)

    # ---- PART 1
    R[14] = [
        P("**Project problem statement:** India is the world's second-largest tea producer, making about 1.3–1.4 "
          "billion kg of tea a year. Planning across the industry (plucking labour at estates, factory capacity, "
          "auction supply and export contracts) depends on knowing how much tea will be made in each month of the "
          "coming year, yet the Tea Board of India publishes production only as PDF tables and no public forecast exists."),
        P("This project curates the Tea Board's monthly releases into one dataset (January 2019 – March 2026), analyses "
          "trend, seasonality and structural change, and asks: **can historical production data be used to forecast "
          "monthly tea production 12 months ahead?** The variable studied is **monthly made-tea production in million kg "
          "(M kg)** for each producing area and grower group (big estates and small growers)."),
    ]
    R[16] = [table(["Parameter", "Details"], [
        ["Dataset Name", "Indian Tea Production Statistics, State/Region-wise and Month-wise (curated as tea_production_master.csv)"],
        ["Source / Website", "Tea Board of India, Ministry of Commerce & Industry, Government of India, Kolkata (www.teaboard.gov.in); seven releases, all URLs listed in 1.3"],
        ["Working URL", BASE_URL + "Annual_Production_2025_and_2025_26_pdf5398.pdf"],
        ["Date Accessed", "13 October 2026"],
        ["Number of Records", f"{len(master):,} rows in the curated master table; {len(leaf):,} rows in the modelling table (18 series × 87 months)"],
        ["Number of Variables", f"{master.shape[1]} columns in the master table; 10 input features + 1 target in the modelling table"],
        ["Target Variable", "production_mkg: monthly made-tea production (million kg) of one producing area and grower group"],
        ["Input Variables", "Same month 1 and 2 years earlier, median and mean of that calendar month in prior years, month (sine, cosine, one-hot), producing area, grower type (BG/SG), disrupted-month flag"],
        ["Time Period", "Monthly, January 2019 – March 2026 (87 months); official annual totals 2017–2025"],
        ["Geographic Coverage", "All India: 9 producing areas (Assam Valley, Cachar, Dooars, Terai, Darjeeling, other North-Indian states, Tamil Nadu, Kerala, Karnataka) across North and South India"],
    ], W2, bold_first_col=False)]
    R[19] = [
        "KEEP",
        P("The data comes from seven official Tea Board releases. Each was downloaded from the Board's website on 13 October 2026; "
          "the URLs below were re-checked and return the same files (identical byte size)."),
        table(["Release coverage", "Working URL"],
              [[c, BASE_URL + n] for n, c in RELEASES],
              [3400, 6824]),
        P(""),
        P("**Why the source is reliable.** The Tea Board is the statutory body that regulates the Indian tea industry under the "
          "Tea Act, 1953. Its production statistics are compiled from the monthly returns that registered tea manufacturers "
          "file with the Board, and they are the official figures used by government and industry. The data is also "
          "internally consistent: the 29 checks described in 1.5 reproduce the Board's own subtotals and annual totals to "
          "within 0.01 M kg."),
        img.add(ASSETS / "source_pdf_2025.png", "Original source: first page of the Tea Board release 'Final tea production in "
                "2025-26 (Apr–Mar)', showing State/Region rows and BG / SG / Total columns per month.", 6.4),
        img.add(ASSETS / "source_pdf_regionwise.png", "Original source: Tea Board 'Production (Region wise) – Final', "
                "annual totals used to reconcile the monthly data.", 4.6),
    ]
    R[21] = [table(["Variable", "Description", "Data Type", "Unit", "Role"], [
        ["X1  lag12", "Production in the same month one year earlier", "Float", "M kg", "Feature"],
        ["X2  lag24", "Production in the same month two years earlier (filled with lag12 if unavailable)", "Float", "M kg", "Feature"],
        ["X3  hist_median", "Median of the same calendar month across all earlier years", "Float", "M kg", "Feature"],
        ["X4  hist_mean", "Mean of the same calendar month across all earlier years", "Float", "M kg", "Feature"],
        ["X5  month_sin, month_cos", "Cyclical encoding of the month: sin and cos of 2π·month/12", "Float", "–", "Feature"],
        ["X6  month_num", "Calendar month 1–12 (one-hot encoded)", "Categorical", "–", "Feature"],
        ["X7  area", "Producing area (9 levels, e.g. Assam Valley, Dooars, Tamil Nadu)", "Categorical", "–", "Feature"],
        ["X8  grower", "BG = big growers (estates), SG = small growers", "Categorical", "–", "Feature"],
        ["X9  lag12_disrupted", "1 if the year-earlier month was disrupted (Mar–May 2020 lockdown, Dec 2024 provisional)", "Binary", "0/1", "Feature"],
        ["Y  production_mkg", "Monthly made-tea production of one area × grower group", "Float", "M kg", "Target"],
    ], [2100, 3924, 1400, 1100, 1700])]
    R[24] = [bullet("**Missing-value treatment:** the source tables have no missing cells. Lag features are undefined for the first "
                    "12 or 24 months of each series: rows without a 12-month lag (2019) are excluded from modelling, and a missing "
                    "24-month lag is filled with the 12-month lag.")]
    R[25] = [bullet("**Duplicate removal:** every month is printed in both a calendar-year and a financial-year table (3,276 duplicate "
                    "cells). All duplicates were verified identical and removed. Where a month exists as both Provisional and Final "
                    "(January 2025), the Final figure is kept.")]
    R[26] = [bullet("**Outlier treatment:** the box plot and STL residuals show two shocks, the COVID-19 lockdown (March–May 2020) and "
                    "the provisional December 2024 (z = −4.7 against other Decembers). They are published values, so they are kept "
                    "in the dataset but excluded as training targets, flagged with the lag12_disrupted feature, and kept in "
                    "evaluation so models are scored on shocks they could not foresee.")]
    R[27] = [bullet("**Encoding categorical variables:** area, grower type and month are one-hot encoded; the month is also encoded "
                    "cyclically (sine and cosine) so December and January are close.")]
    R[28] = [bullet("**Feature scaling / normalization:** StandardScaler (zero mean, unit variance) is applied to numeric features for "
                    "Ridge regression only; the tree-based models use raw values.")]
    R[29] = [bullet(f"**Train-test split:** chronological, never shuffled. Main split: train on targets from January 2020 to March 2025, "
                    f"test on FY 2025-26 (April 2025 – March 2026, {S['test_rows']} rows = 18 series × 12 months). For robustness, a "
                    f"rolling-origin backtest repeats this for four held-out financial years (2022-23 to 2025-26)."),
             P("**Data integrity checks (29, all passed):** duplicates agree; BG + SG = Total in every row; areas sum to states, "
               "states to regions and regions to All India; CTC + Orthodox + Green = Total; all 14 printed period totals and all "
               "30 official final annual totals reconcile with the monthly data. One informational gap: the provisional "
               "Apr–Dec 2024 months sum to 18.75 M kg less than the Board's final 2024 total; they are flagged in a status column.")]

    # ---- PART 2
    R[34] = ["KEEP", P("**Selected problem type:** time-series forecasting, framed as supervised **regression** (direct "
                       "12-month-ahead forecasts of a numerical target). Algorithms: **Ridge (linear) regression, Random Forest "
                       "regression and Histogram Gradient Boosting**, compared against two time-series reference models: the "
                       "seasonal-naive forecast (same month last year) and the seasonal-median forecast (median of that calendar "
                       "month in all earlier years).")]
    R[36] = [bullet("**Define input features and target:** target Y = monthly production of each of 18 series (9 areas × big/small "
                    "growers). Features X1–X9 (see 1.4) use only information available at the forecast date, the end of March.")]
    R[37] = [bullet("**Split into training and testing sets:** chronological split; test year FY 2025-26, plus a four-year "
                    "rolling-origin backtest (train on all data before each April, test on the following 12 months).")]
    R[38] = [bullet("**Train the selected model:** scikit-learn pipelines. Ridge (α = 1, scaled inputs); Random Forest (500 trees, "
                    "min_samples_leaf = 2); Histogram Gradient Boosting (400 iterations, learning rate 0.05, 15 leaves).")]
    R[39] = [bullet("**Generate predictions:** each model predicts all 216 series-months of the test year; series forecasts are summed "
                    "to give the All-India monthly forecast. The best model is then refitted on all data to forecast FY 2026-27.")]
    R[40] = [bullet("**Explain the model and assumptions:** the Random Forest averages many decision trees, each splitting on the "
                    "seasonal-history features; it captures non-linear differences between areas and grower groups without "
                    "assuming a linear form. Assumptions: the seasonal pattern is stable over time; no future information is "
                    "used; shocks (lockdown, provisional Dec-2024) are not representative of normal months.")]
    rows_test = [[mname, f"{v['MAE']:.2f}", f"{v['MSE']:.1f}", f"{v['RMSE']:.2f}", f"{v['R2']:.3f}", f"{v['WAPE']:.1f}%"]
                 for mname, v in sorted(t.items(), key=lambda kv: kv[1]["MAE"])]
    rows_bt = [[mname, f"{v['MAE']:.2f}", f"{v['MSE']:.1f}", f"{v['RMSE']:.2f}", f"{v['R2']:.3f}", f"{v['WAPE']:.1f}%"]
               for mname, v in sorted(f.items(), key=lambda kv: kv[1]["WAPE"])]
    R[42] = ["KEEP",
             P("**Test year FY 2025-26** (All-India monthly totals, n = 12; MAE and RMSE in million kg):", keep_next=True),
             table(["Model", "MAE", "MSE", "RMSE", "R²", "WAPE"], rows_test, [2924, 1460, 1460, 1460, 1460, 1460]),
             P(""),
             P("**Rolling-origin backtest**, mean of four held-out years (FY 2022-23 to FY 2025-26):", keep_next=True),
             table(["Model", "MAE", "MSE", "RMSE", "R²", "WAPE"], rows_bt, [2924, 1460, 1460, 1460, 1460, 1460]),
             P(f"**Interpretation:** the Random Forest is the best machine-learning model (test-year MAE {rf['MAE']:.2f} M kg, "
               f"RMSE {rf['RMSE']:.2f}, R² {rf['R2']:.3f}). It halves the error of the seasonal-naive baseline (MAE "
               f"{sn['MAE']:.2f}, R² {sn['R2']:.3f}) but performs the same as the seasonal-median baseline (MAE {sm['MAE']:.2f}, "
               f"R² {sm['R2']:.3f}). WAPE (total absolute error ÷ total actual) is also reported because MAPE is unstable in "
               f"January–February, when North India produces almost nothing.")]
    fy = S["fy26"]
    R[44] = [table(["Sample (month)", "Actual Y (M kg)", "Predicted Y (M kg)", "Error = Actual − Predicted"],
                   [[f"{i + 1} ({d:%b %Y})", f"{a:.2f}", f"{pr:.2f}", f"{a - pr:+.2f}"]
                    for i, (d, a, pr) in enumerate(zip(fy.date, fy.y, fy.pred))]
                   + [["FY 2025-26 total", f"{fy.y.sum():.2f}", f"{fy.pred.sum():.2f}",
                       f"{fy.y.sum() - fy.pred.sum():+.2f} ({100 * (fy.pred.sum() / fy.y.sum() - 1):+.1f}%)"]],
                   [2556, 2556, 2556, 2556])]
    R[46] = [P("**Actual vs Predicted graphs:**", keep_next=True),
             img.add(FIG / "10_actual_vs_forecast_FY2025-26.png", "Actual vs predicted All-India monthly production, held-out year "
                     "FY 2025-26 (Random Forest and seasonal-median baseline)."),
             img.add(FIG / "15_actual_vs_predicted_scatter.png", "Predicted vs actual for all 216 series-months of FY 2025-26; "
                     "points close to the y = x line are accurate forecasts.", 5.0)]

    # ---- PART 3
    R[52] = [P("**Selected test:** Paired t-test (two-tailed). "
               "**Reason:** both models forecast the same 48 held-out months (4 years × 12), so their absolute errors are paired "
               "observations; the paired t-test checks whether the mean difference in error is zero. Its normality assumption "
               f"holds: Shapiro–Wilk on the 48 differences gives W = {S['shapiro'].statistic:.3f}, p = {S['shapiro'].pvalue:.2f}.")]
    R[54] = [P("**H₀ (Null Hypothesis):** the mean absolute forecast error of the Random Forest equals that of the seasonal-naive "
               "baseline (mean paired difference μd = 0)."),
             P("**H₁ (Alternative Hypothesis):** the mean absolute forecast errors differ (μd ≠ 0).")]
    R[56] = [table(["Parameter", "Result"], [
        ["Significance level (α)", "0.05"],
        ["Test statistic", f"t = {main_t['t']:.3f}"],
        ["Degrees of freedom", f"{main_t['df']}"],
        ["p-value", f"{main_t['p']:.4f}"],
        ["Decision", "Reject H₀" if main_t["p"] < 0.05 else "Fail to reject H₀"],
        ["Interpretation", f"The Random Forest's errors are on average {abs(main_t['diff']):.1f} M kg per month lower than the "
                           f"seasonal-naive baseline, a statistically significant improvement (p = {main_t['p']:.3f} < 0.05)."],
    ], W2)]
    R[57] = [P(""), P("**Supporting tests** (same 48 paired months; Wilcoxon signed-rank shown as a non-parametric check):",
                      keep_next=True),
             table(["Comparison / test", "Statistic", "p-value", "Decision at α = 0.05"], [
                 ["Paired t: Random Forest vs seasonal median", f"t = {rf_sm['t']:.3f}, df = 47", f"{rf_sm['p']:.3f} (Wilcoxon {rf_sm['wilcoxon_p']:.3f})",
                  "Fail to reject H₀: no significant difference"],
                 ["Paired t: seasonal median vs seasonal naive", f"t = {sm_sn['t']:.3f}, df = 47", f"{sm_sn['p']:.4f} (Wilcoxon {sm_sn['wilcoxon_p']:.4f})",
                  "Reject H₀: the seasonal median is significantly better"],
                 ["One-sample t: Random Forest bias (mean error = 0)", f"t = {S['bias'].statistic:.3f}, df = 47, mean {S['bias_mean']:+.2f}",
                  f"{S['bias'].pvalue:.3f}", "Fail to reject H₀: forecasts are unbiased"],
                 ["Kendall τ: trend in annual production 2017–2025", f"τ = {stats_json['trend']['kendall_tau']:.2f}",
                  f"{stats_json['trend']['kendall_p']:.2f}", "No significant long-run trend"],
                 ["Linear regression: small-grower share 2019–2025", f"slope +{stats_json['small_growers']['trend_pp_per_year']:.2f} pp/yr, R² {stats_json['small_growers']['trend_r2']:.2f}",
                  "< 0.001", "Significant structural shift to small growers"],
             ], [3300, 2300, 2100, 2524]),
             P("**Conclusion of validation:** the machine-learning improvement over a naive forecast is statistically supported, "
               "but the Random Forest is not significantly better than the seasonal-median baseline: the stable seasonal pattern "
               "already captures most of the predictable signal.")]

    # ---- PART 4
    R[62] = [table(["Section", "What to Include", "In this project's blog"], [
        ["Title / Hook", "A compelling title connected to the real-world problem",
         "“Forecasting India's Tea Production: Why the Calendar Beat Machine Learning”; subtitle: seven Tea Board PDFs, three ML models, four years of testing"],
        ["Introduction", "Why the problem matters and what question is being investigated",
         "India makes ~1.3–1.4 billion kg of tea a year; the question: from the end of March, can we forecast each month of the next year?"],
        ["The Data", "Dataset source, size, variables and credibility",
         "Seven official Tea Board releases, 87 months, 4,746 rows, BG/SG split and tea types, 29 integrity checks"],
        ["Visual Story", "Charts, graphs, dashboards, infographics",
         "11 visuals: production heatmap, annual bars, seasonal profile, small-grower share, COVID impact, model comparison, results table, feature importance, actual vs forecast, FY 2026-27 forecast; plus an offline interactive dashboard"],
        ["Conclusion", "Key takeaway for a general reader",
         "With production history alone the calendar is nearly the best forecast; weather data is needed to do better"],
        ["References", "Dataset, articles, tools and other sources",
         "Tea Board of India releases (URLs in 1.3); Python: pandas, scikit-learn, SciPy, statsmodels, matplotlib, seaborn, pdfplumber"],
    ], [1900, 3400, 4924])]
    R[65] = [bullet("**Histogram / distribution plot:** distribution of the Random Forest's forecast errors (Figure below).")]
    R[66] = [bullet("**Box plot:** All-India production by calendar month, showing seasonality and the two outlier months.")]
    R[67] = [bullet("**Bar chart:** annual production by region; backtest error by model and year; COVID-19 impact by month.")]
    R[68] = [bullet("**Scatter plot:** predicted vs actual for every series-month of the test year (Part 2.4).")]
    R[69] = [bullet("**Correlation heatmap:** correlations between the numeric model inputs and the target.")]
    R[70] = [bullet("**Time-series plot:** month × year production heatmap, small-grower share over time, STL decomposition and the "
                    "FY 2026-27 forecast.")]
    R[71] = [bullet("**Actual vs Predicted plot:** held-out FY 2025-26 (Part 2.4).")]
    R[72] = [bullet("**Confusion matrix / ROC curve:** not applicable; this is a regression / forecasting problem."),
             P("**Selected visualizations from the blog and analysis:**", keep_next=True),
             img.add(FIG / "02_seasonality_heatmap.png", "Time-series heatmap: All-India production by month and year (blog hero image)."),
             img.add(FIG / "01_annual_production.png", "Bar chart: calendar-year production, North vs South India, 2017–2025."),
             img.add(FIG / "14_monthly_boxplot.png", "Box plot: All-India production by calendar month, 2019–2025."),
             img.add(FIG / "04_small_grower_share.png", "Time-series plot: small-grower share of production, with linear trend."),
             img.add(FIG / "16_feature_correlation.png", "Correlation heatmap of the numeric model inputs and the target.", 5.4),
             img.add(FIG / "09_model_comparison.png", "Bar chart: backtest error (WAPE) of every model in every held-out year."),
             img.add(FIG / "13_error_histogram.png", "Histogram of Random Forest forecast errors over 48 held-out months."),
             img.add(FIG / "12_forecast_FY2026-27.png", "Forecast for FY 2026-27 with an approximate 80% interval."),
             img.add(ASSETS / "dashboard_explore.png", "Offline interactive dashboard (reports/dashboard.html): headline figures, "
                     "tabs and a data explorer for any area, grower group and period."),
             ]
    R[73] = [P("**Blog / publication link:** to be added after publishing on Medium (the publication-ready draft is "
               "medium/article.md, with a local preview at http://127.0.0.1:8000/preview.html).")]
    R[74] = [P("**Blog title:** Forecasting India's Tea Production: Why the Calendar Beat Machine Learning")]

    # ---- PART 5
    R[77] = [table(["Model / Method", "Main Metric", "Value", "Key Finding"], [
        ["Model 1: Random Forest regression", "MAE / RMSE / R² (test FY 2025-26)", f"{rf['MAE']:.2f} / {rf['RMSE']:.2f} / {rf['R2']:.3f}",
         f"Best ML model; backtest WAPE {f['Random forest']['WAPE']:.1f}%"],
        ["Model 2: Ridge regression", "MAE / RMSE / R² (test FY 2025-26)",
         f"{t['Ridge regression']['MAE']:.2f} / {t['Ridge regression']['RMSE']:.2f} / {t['Ridge regression']['R2']:.3f}",
         f"Linear model; backtest WAPE {f['Ridge regression']['WAPE']:.1f}%"],
        ["Model 3: Gradient boosting", "MAE / RMSE / R² (test FY 2025-26)",
         f"{t['Gradient boosting']['MAE']:.2f} / {t['Gradient boosting']['RMSE']:.2f} / {t['Gradient boosting']['R2']:.3f}",
         f"Backtest WAPE {f['Gradient boosting']['WAPE']:.1f}%"],
        ["Baseline: seasonal median", "MAE / RMSE / R² (test FY 2025-26)", f"{sm['MAE']:.2f} / {sm['RMSE']:.2f} / {sm['R2']:.3f}",
         f"Matches the Random Forest; lowest backtest WAPE ({f['Seasonal median']['WAPE']:.1f}%)"],
        ["Baseline: seasonal naive", "MAE / RMSE / R² (test FY 2025-26)", f"{sn['MAE']:.2f} / {sn['RMSE']:.2f} / {sn['R2']:.3f}",
         "Weakest; every ML model beats it"],
        ["Statistical test: paired t-test (RF vs naive)", "t, df, p-value", f"t = {main_t['t']:.2f}, df = 47, p = {main_t['p']:.3f}",
         "Improvement over naive is significant; RF vs seasonal median is not (p = %.2f)" % rf_sm["p"]],
        ["Forecasting: FY 2026-27 (Random Forest)", "Total production", f"{fc['total_mkg']:,.0f} M kg",
         f"{fc['vs_FY2025_26_pct']:+.1f}% vs FY 2025-26; range ≈ {fc['annual_range_from_backtest_mkg'][0]:,.0f}–{fc['annual_range_from_backtest_mkg'][1]:,.0f} M kg"],
    ], [2700, 2300, 2100, 3124])]
    R[80] = [bullet("**Larger or more diverse dataset:** extend back before 2019 with older Tea Board releases and add district-level "
                    "data, so models see more than seven seasons.")]
    R[81] = [bullet("**Additional features:** India Meteorological Department rainfall and temperature, soil moisture, and auction "
                    "prices. The residuals point to weather as the missing signal.")]
    R[82] = [bullet("**Alternative ML algorithms:** SARIMA / ETS and Prophet as statistical benchmarks; LightGBM; hierarchical "
                    "forecasting that keeps area forecasts consistent with state and national totals.")]
    R[83] = [bullet("**Ensemble / advanced modelling:** combine the Random Forest with the seasonal median (an early test averaged "
                    "9.6% WAPE), or model the correction to the seasonal median instead of the level.")]
    R[84] = [bullet("**Real-time data collection:** fetch each new monthly Tea Board PDF automatically and re-run the 29 integrity "
                    "checks and the forecast.")]
    R[85] = [bullet("**Deployment as a dashboard/web/mobile application:** an offline interactive dashboard is already built "
                    "(reports/dashboard.html); next, a hosted web app that refreshes the forecast each month."),
             P("5.3 Conclusion", style="Heading2"),
             P(f"The project delivers a verified dataset of Indian tea production built from seven official Tea Board releases "
               f"(4,746 rows, 29 integrity checks). The analysis shows that national output is flat (no significant trend, "
               f"Kendall p = {stats_json['trend']['kendall_p']:.2f}), that 95% of monthly variation is seasonal, and that small "
               f"growers' share rose from {stats_json['small_growers']['sg_share_2019_pct']}% to "
               f"{stats_json['small_growers']['sg_share_2025_pct']}% between 2019 and 2025 (p < 0.001)."),
             P(f"Historical data can predict monthly production well: the Random Forest explains {100 * rf['R2']:.1f}% of the "
               f"variance in the held-out year (MAE {rf['MAE']:.1f} M kg) and significantly beats a naive forecast (paired t-test "
               f"p = {main_t['p']:.3f}). However, it does not beat a simple seasonal-median baseline (p = {rf_sm['p']:.2f}). With "
               f"production history alone the calendar is close to the best possible forecast; further gains require exogenous "
               f"drivers such as weather. The model forecasts about {fc['total_mkg']:,.0f} M kg for FY 2026-27."),
             P("5.4 References", style="Heading2"),
             *[bullet(f"Tea Board of India, {c}: {BASE_URL}{n}") for n, c in RELEASES],
             bullet("Python libraries: pandas, NumPy, scikit-learn, SciPy, statsmodels (STL), matplotlib, seaborn, pdfplumber, pypdfium2."),
             bullet("Hyndman, R. J. & Athanasopoulos, G., Forecasting: Principles and Practice (3rd ed.), OTexts, for baselines, "
                    "STL and rolling-origin evaluation."),
             ]
    R[88] = ["KEEP", P(""), P("**How each step was implemented in this project:**", keep_next=True),
             table(["Step", "Implementation", "Output"], [
                 ["Real-world problem", "Forecast monthly Indian tea production 12 months ahead", "Problem statement (1.1)"],
                 ["Authentic historical data", "Seven Tea Board PDFs parsed with pdfplumber (src/extract.py)", "tea_production_master.csv, 4,746 rows"],
                 ["Data cleaning & EDA", "De-duplication, 29 integrity checks, STL, trend and seasonality analysis (src/analysis.py)", "16 figures, statistics.json"],
                 ["ML prediction / regression", "Ridge, Random Forest, Gradient Boosting vs seasonal baselines; 4-year backtest", "Metrics, FY 2026-27 forecast"],
                 ["Statistical validation", "Paired t-tests, Wilcoxon, bias test, Kendall trend test", "Part 3 tables"],
                 ["Data visualization blog", "Medium article with 11 visuals (src/medium.py); offline dashboard (src/dashboard.py)", "medium/article.md, reports/dashboard.html"],
                 ["Results & conclusion", "Summary of models, tests and forecast", "Part 5"],
             ], [2500, 4800, 2924])]

    # ---- assemble
    with zipfile.ZipFile(TEMPLATE) as z:
        files = {n: z.read(n) for n in z.namelist()}
    doc = files["word/document.xml"].decode("utf-8")
    prefix, elems, suffix = body_elements(doc)
    out = []
    for i, el in enumerate(elems):
        if i in R:
            out.extend(el if x == "KEEP" else x for x in R[i])
        else:
            out.append(el)
    files["word/document.xml"] = (prefix + "".join(out) + suffix).encode("utf-8")

    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    add = "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                  f'Target="media/report_{k + 1}.png"/>' for k, (rid, _) in enumerate(img.items))
    add += "".join(f'<Relationship Id="rIdLink{k + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
                   f'Target="{escape(u)}" TargetMode="External"/>' for k, u in enumerate(LINKS))
    files["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", add + "</Relationships>").encode("utf-8")
    for k, (_, path) in enumerate(img.items):
        files[f"word/media/report_{k + 1}.png"] = path.read_bytes()

    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(img.items)} figures)")


if __name__ == "__main__":
    if not TEMPLATE.exists():
        raise SystemExit("Copy the faculty template to report/template.docx first")
    S = compute()
    figures(S)
    build(S)
