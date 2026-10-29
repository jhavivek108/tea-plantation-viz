"""Step 1 - Extract and curate Tea Board of India production PDFs into one tidy dataset.

Inputs : data/raw/*.pdf  (7 Tea Board publications, 2017 - Mar 2026)
Outputs: data/processed/tea_production_master.csv   single curated long table
         data/processed/monthly_area_grower.csv     modelling view (area x grower x month)
         data/processed/qa_checks.csv               every integrity check that was run
         data/processed/tea_production_curated.xlsx all of the above + data dictionary

Every PDF page is mapped explicitly (PAGE_MAP) to the month / total each column
group holds, so nothing depends on fragile header parsing. Months that appear in
several files (calendar-year and financial-year tables overlap) are compared and
the Final figure is kept over a Provisional one.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# ---------------------------------------------------------------- page map --
# Each entry: (file stem suffix, page number) -> list of column groups.
# "YYYY-MM" = a month; anything else = a printed total used only for checks.
def months(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]

FILES = {
    "2019": "Website_Production_Data_for_2019_2019_20_FINAL_pdf2416.pdf",
    "2020": "Production_Data_for_2020_2020_21_pdf4425.pdf",
    "2021": "Production_data_for_2021_and_2021_22_pdf223.pdf",
    "2022": "Production_2022_and_2022_23_pdf5959.pdf",
    "2023_24": "State_Region_wise_tea_production_2024_and_2024_25_Apr_Dec_pdf4332.pdf",
    "2025": "Annual_Production_2025_and_2025_26_pdf5398.pdf",
    "annual": "Production_Region_wise_FINAL_pdf9516.pdf",
}

STATE_PAGES: dict[tuple[str, int], list[str]] = {}
CATEGORY_PAGES: list[tuple[str, int]] = []
for y in (2019, 2020, 2021, 2022):
    k = str(y)
    STATE_PAGES[(k, 1)] = months(f"{y}-01", f"{y}-07")
    STATE_PAGES[(k, 2)] = months(f"{y}-08", f"{y}-12") + [f"CY{y}"]
    fy = f"FY{y}-{str(y + 1)[2:]}"
    if y == 2021:  # this file splits the FY table Apr-Oct / Nov-Mar
        STATE_PAGES[(k, 3)] = months(f"{y}-04", f"{y}-10")
        STATE_PAGES[(k, 4)] = months(f"{y}-11", f"{y + 1}-03") + [fy]
    else:
        STATE_PAGES[(k, 3)] = months(f"{y}-04", f"{y}-09")
        STATE_PAGES[(k, 4)] = months(f"{y}-10", f"{y + 1}-03") + [fy]
    CATEGORY_PAGES.append((k, 5))

STATE_PAGES.update({
    ("2023_24", 1): months("2024-01", "2024-07"),
    ("2023_24", 2): months("2024-08", "2024-12") + ["CY2024"],
    ("2023_24", 3): months("2024-04", "2024-09"),
    ("2023_24", 4): months("2024-10", "2025-01") + ["FY2024-25 Apr-Jan"],
    ("2023_24", 5): months("2023-01", "2023-07"),
    ("2023_24", 6): months("2023-08", "2023-12") + ["CY2023"],
    ("2023_24", 7): months("2023-04", "2023-10"),
    ("2023_24", 8): months("2023-11", "2024-03") + ["FY2023-24"],
    ("2025", 1): months("2025-04", "2025-09"),
    ("2025", 2): months("2025-10", "2026-03") + ["FY2025-26"],
    ("2025", 3): months("2025-01", "2025-07"),
    ("2025", 4): months("2025-08", "2025-12") + ["CY2025"],
})
CATEGORY_PAGES += [("2023_24", 9), ("2023_24", 10), ("2025", 5)]

# The 2024 file marks Apr-2024 onwards as provisional ("*Provisional, subject to
# revision from April 24 onwards"). Everything else is published as Final.
def status_for(file_key: str, month: str) -> str:
    if file_key == "2023_24" and month >= "2024-04":
        return "Provisional"
    return "Final"

# --------------------------------------------------------- geography model --
# Canonical row label -> (geo_level, area, state, zone)
GEO = {
    "Assam Valley": ("area", "Assam Valley", "Assam", "North India"),
    "Cachar":       ("area", "Cachar", "Assam", "North India"),
    "Assam":        ("state", None, "Assam", "North India"),
    "Dooars":       ("area", "Dooars", "West Bengal", "North India"),
    "Terai":        ("area", "Terai", "West Bengal", "North India"),
    "Darjeeling":   ("area", "Darjeeling", "West Bengal", "North India"),
    "West Bengal":  ("state", None, "West Bengal", "North India"),
    "Others":       ("area", "Others (NE & HP etc.)", "Others (North)", "North India"),
    "North India":  ("zone", None, None, "North India"),
    "Tamil Nadu":   ("area", "Tamil Nadu", "Tamil Nadu", "South India"),
    "Kerala":       ("area", "Kerala", "Kerala", "South India"),
    "Karnataka":    ("area", "Karnataka", "Karnataka", "South India"),
    "South India":  ("zone", None, None, "South India"),
    "All India":    ("country", None, None, "All India"),
}
LABEL_RE = re.compile(r"^(?:Total\s+)?(" + "|".join(sorted(GEO, key=len, reverse=True)) + r")$")
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
MONTH_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"


def page_lines(pdf_path: Path, page_no: int) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[page_no - 1].extract_text() or ""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def split_label_numbers(line: str) -> tuple[str, list[float]]:
    toks = line.split()
    i = len(toks)
    while i > 0 and NUM_RE.match(toks[i - 1]):
        i -= 1
    return " ".join(toks[:i]), [float(t) for t in toks[i:]]


def parse_state_page(file_key: str, page_no: int, groups: list[str]) -> list[dict]:
    lines = page_lines(RAW / FILES[file_key], page_no)
    # Merge cells whose label wraps around the numbers: "Assam" / <numbers> / "Valley"
    merged, i = [], 0
    while i < len(lines):
        if i + 2 < len(lines):
            (l0, n0), (l1, n1), (l2, n2) = (split_label_numbers(x) for x in lines[i:i + 3])
            if not n0 and not l1 and n1 and not n2 and LABEL_RE.match(f"{l0} {l2}"):
                merged.append(f"{l0} {l2} {lines[i + 1]}")
                i += 3
                continue
        merged.append(lines[i])
        i += 1
    lines = merged
    rows, pending = [], ""
    for ln in lines:
        label, nums = split_label_numbers(ln)
        if not nums:  # label wrapped onto two lines, e.g. "Total West" / "Bengal ..."
            cand = re.sub(r"^Total\s+", "", f"{pending} {label}".strip())
            is_prefix = any(g.startswith(cand) and g != cand for g in GEO)
            pending = f"{pending} {label}".strip() if is_prefix else ""
            continue
        label = f"{pending} {label}".strip() if pending else label
        pending = ""
        m = LABEL_RE.match(re.sub(r"\s+", " ", label))
        if not m:
            continue
        expected = 3 * len(groups)
        if len(nums) != expected:
            raise ValueError(f"{file_key} p{page_no} '{label}': {len(nums)} numbers, expected {expected}")
        for g_i, group in enumerate(groups):
            for c_i, grower in enumerate(("BG", "SG", "Total")):
                rows.append({"file_key": file_key, "page": page_no, "label": m.group(1),
                             "column": group, "grower": grower, "value": nums[3 * g_i + c_i]})
    labels_found = {r["label"] for r in rows}
    missing = set(GEO) - labels_found
    if missing:
        raise ValueError(f"{file_key} p{page_no}: rows not found {missing}")
    return rows


def parse_category_page(file_key: str, page_no: int) -> list[dict]:
    rows = []
    for ln in page_lines(RAW / FILES[file_key], page_no):
        m = re.match(rf"^({MONTH_ABBR})-(\d{{2}})\*?\s+(.*)$", ln)
        if not m:
            continue
        nums = [float(t) for t in m.group(3).split()]
        if len(nums) != 12:
            raise ValueError(f"{file_key} p{page_no} '{ln}': expected 12 numbers")
        month = pd.Period(f"{m.group(1)}-20{m.group(2)}", freq="M").strftime("%Y-%m")
        for z_i, zone in enumerate(("North India", "South India", "All India")):
            for c_i, cat in enumerate(("CTC", "Orthodox", "Green", "Total")):
                rows.append({"file_key": file_key, "page": page_no, "month": month, "zone": zone,
                             "category": cat, "value": nums[4 * z_i + c_i]})
    return rows


def parse_annual() -> pd.DataFrame:
    rows = []
    for ln in page_lines(RAW / FILES["annual"], 1):
        m = re.match(r"^(\d{4}(?:-\d{2})?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", ln)
        if m:
            period = m.group(1)
            ptype = "fiscal_year" if "-" in period else "calendar_year"
            for zone, v in zip(("North India", "South India", "All India"), m.groups()[1:]):
                rows.append({"period_type": ptype, "period": period, "zone": zone, "value": float(v)})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qa: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        qa.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    # ---- state / area x grower tables
    raw = pd.DataFrame([r for (fk, pg), groups in STATE_PAGES.items()
                        for r in parse_state_page(fk, pg, groups)])
    raw["source_file"] = raw["file_key"].map(FILES)
    monthly_raw = raw[raw["column"].str.match(r"^\d{4}-\d{2}$")].copy()
    totals_raw = raw[~raw["column"].str.match(r"^\d{4}-\d{2}$")].copy()
    monthly_raw["status"] = [status_for(f, m) for f, m in zip(monthly_raw.file_key, monthly_raw.column)]

    # Duplicate months across files: compare, then keep Final over Provisional
    dup = monthly_raw.groupby(["column", "label", "grower", "status"])["value"].agg(["min", "max", "count"])
    dup = dup[dup["count"] > 1]
    worst = (dup["max"] - dup["min"]).max()
    check("Duplicate months agree across files (same status)", worst <= 0.011,
          f"{len(dup)} duplicated cells, max disagreement {worst:.2f} M kg")
    prov_vs_final = monthly_raw.pivot_table(index=["column", "label", "grower"], columns="status",
                                            values="value", aggfunc="first").dropna()
    if not prov_vs_final.empty:
        diff = (prov_vs_final["Final"] - prov_vs_final["Provisional"])
        allind = diff.xs(("All India", "Total"), level=["label", "grower"])
        check("Provisional vs Final overlap (informational)", True,
              "; ".join(f"{m}: All India final-provisional = {d:+.2f}" for m, d in allind.items()))
    monthly = (monthly_raw.assign(rank=monthly_raw["status"].map({"Final": 0, "Provisional": 1}))
               .sort_values(["rank", "file_key"]).drop_duplicates(["column", "label", "grower"])
               .drop(columns="rank"))

    # BG + SG = Total
    w = monthly.pivot_table(index=["column", "label"], columns="grower", values="value")
    err = (w["BG"] + w["SG"] - w["Total"]).abs()
    check("BG + SG = Total (every row, every month)", err.max() <= 0.021,
          f"{len(err)} rows, max error {err.max():.2f}; rows > 0.02: {(err > 0.021).sum()}")

    # Hierarchy: areas -> states -> zones -> country
    t = monthly.pivot_table(index=["column", "grower"], columns="label", values="value")
    hier = {
        "Assam": ["Assam Valley", "Cachar"],
        "West Bengal": ["Dooars", "Terai", "Darjeeling"],
        "North India": ["Assam", "West Bengal", "Others"],
        "South India": ["Tamil Nadu", "Kerala", "Karnataka"],
        "All India": ["North India", "South India"],
    }
    for parent, kids in hier.items():
        e = (t[kids].sum(axis=1) - t[parent]).abs()
        check(f"Hierarchy: {' + '.join(kids)} = {parent}", e.max() <= 0.031,
              f"max error {e.max():.2f} M kg over {len(e)} month-grower cells")

    # Printed period totals vs sum of months
    for (fk, col), grp in totals_raw.groupby(["file_key", "column"]):
        if col.startswith("CY"):
            rng = months(f"{col[2:]}-01", f"{col[2:]}-12")
        elif col.startswith("FY"):
            y0 = int(col[2:6])
            end = f"{y0 + 1}-01" if "Apr-Jan" in col else f"{y0 + 1}-03"
            rng = months(f"{y0}-04", end)
        src = monthly_raw[(monthly_raw.file_key == fk) & monthly_raw.column.isin(rng)]
        src = src.drop_duplicates(["column", "label", "grower"])  # CY and FY pages repeat months
        s = src.groupby(["label", "grower"])["value"].sum()
        printed = grp.set_index(["label", "grower"])["value"]
        e = (s - printed).abs()
        check(f"Printed total {col} ({FILES[fk][:28]}...) = sum of its months", e.max() <= 0.12,
              f"max error {e.max():.2f} M kg")

    # ---- category (CTC / Orthodox / Green)
    cat_raw = pd.DataFrame([r for fk, pg in CATEGORY_PAGES for r in parse_category_page(fk, pg)])
    cat_raw["status"] = [status_for(f, m) for f, m in zip(cat_raw.file_key, cat_raw.month)]
    cat_raw["source_file"] = cat_raw["file_key"].map(FILES)
    cat = (cat_raw.assign(rank=cat_raw["status"].map({"Final": 0, "Provisional": 1}))
           .sort_values(["rank", "file_key"]).drop_duplicates(["month", "zone", "category"])
           .drop(columns="rank"))
    cw = cat.pivot_table(index=["month", "zone"], columns="category", values="value")
    e = (cw[["CTC", "Orthodox", "Green"]].sum(axis=1) - cw["Total"]).abs()
    check("Category: CTC + Orthodox + Green = Total", e.max() <= 0.021, f"max error {e.max():.2f}")
    st_tot = t.xs("Total", level="grower")
    for zone in ("North India", "South India", "All India"):
        z = cw.xs(zone, level="zone")["Total"]
        e = (z - st_tot.loc[z.index, zone]).abs()
        check(f"Category total = state-table total ({zone})", e.max() <= 0.021, f"max error {e.max():.2f}")

    # ---- annual summary & reconciliation with monthly data
    annual = parse_annual()
    annual["source_file"] = FILES["annual"]
    mon_tot = st_tot[["North India", "South India", "All India"]]
    recon = []
    for _, r in annual.iterrows():
        p = r["period"]
        rng = months(f"{p}-01", f"{p}-12") if r.period_type == "calendar_year" else \
            months(f"{p[:4]}-04", f"{int(p[:4]) + 1}-03")
        if not set(rng) <= set(mon_tot.index):
            continue
        s = mon_tot.loc[rng, r["zone"]].sum()
        has_prov = monthly[monthly.column.isin(rng)]["status"].eq("Provisional").any()
        recon.append({**r.to_dict(), "sum_of_months": round(s, 2), "gap": round(r["value"] - s, 2),
                      "months_include_provisional": has_prov})
    recon = pd.DataFrame(recon)
    fin = recon[~recon.months_include_provisional]
    bad = fin[fin["gap"].abs() > 0.15]
    check("Annual FINAL totals = sum of FINAL monthly figures", bad.empty,
          f"{len(fin)} period-zone totals checked, max gap {fin['gap'].abs().max():.2f} M kg")
    prov = recon[recon.months_include_provisional]
    qa.append({"check": "Annual FINAL totals vs PROVISIONAL monthly figures (2024)", "result": "INFO",
               "detail": "; ".join(f"{b.period} {b.zone}: final {b.value} vs months {b.sum_of_months} "
                                   f"(gap {b.gap:+.2f})" for b in prov.itertuples())})

    # --------------------------------------------------- build master table --
    def fy(m: str) -> str:
        y, mo = int(m[:4]), int(m[5:])
        s = y if mo >= 4 else y - 1
        return f"{s}-{str(s + 1)[2:]}"

    geo = pd.DataFrame.from_dict(GEO, orient="index", columns=["geo_level", "area", "state", "zone"])
    geo.index.name = "label"
    g = monthly.merge(geo, left_on="label", right_index=True)
    g = g.assign(geography=g["label"].replace({"Others": "Others (NE & HP etc.)"}),
                 breakdown="grower", segment=g["grower"])
    c = cat.assign(geo_level=cat["zone"].map(lambda z: "country" if z == "All India" else "zone"),
                   geography=cat["zone"], area=None, state=None, breakdown="category",
                   segment=cat["category"], column=cat["month"])
    cols = ["column", "geo_level", "geography", "area", "state", "zone", "breakdown", "segment",
            "value", "status", "source_file"]
    master = pd.concat([g[cols], c[cols]], ignore_index=True).rename(columns={"column": "month"})
    master.insert(0, "period_type", "month")
    master.insert(1, "period", master["month"])
    a = annual.assign(geo_level=annual["zone"].map(lambda z: "country" if z == "All India" else "zone"),
                      geography=annual["zone"], area=None, state=None, breakdown="grower",
                      segment="Total", status="Final", month=None)
    master = pd.concat([master, a[["period_type", "period"] + cols[1:] + ["month"]].assign(
        value=a["value"])], ignore_index=True)

    dt = pd.to_datetime(master["month"], format="%Y-%m", errors="coerce")
    master["date"] = dt.dt.strftime("%Y-%m-01")
    master["year"] = dt.dt.year.astype("Int64")
    master["month_num"] = dt.dt.month.astype("Int64")
    master["month_name"] = dt.dt.strftime("%b")
    master["fiscal_year"] = master["month"].map(lambda m: fy(m) if isinstance(m, str) else None)
    master = master.rename(columns={"value": "production_mkg"})
    master = master[["period_type", "period", "date", "year", "month_num", "month_name", "fiscal_year",
                     "geo_level", "geography", "area", "state", "zone", "breakdown", "segment",
                     "production_mkg", "status", "source_file"]]
    master = master.sort_values(["period_type", "period", "breakdown", "geo_level", "geography",
                                 "segment"]).reset_index(drop=True)

    # Modelling view: one row per area x grower x month (the additive leaf level)
    leaf = master[(master.period_type == "month") & (master.geo_level == "area") &
                  (master.breakdown == "grower") & (master.segment != "Total")]
    leaf = leaf[["date", "year", "month_num", "fiscal_year", "zone", "state", "area", "segment",
                 "production_mkg", "status"]].rename(columns={"segment": "grower"})

    n_months = master.loc[master.period_type == "month", "period"].nunique()
    check("Monthly coverage is continuous", n_months == len(months("2019-01", "2026-03")),
          f"{n_months} months, Jan-2019 to Mar-2026")

    qa_df = pd.DataFrame(qa)
    dictionary = pd.DataFrame([
        ("period_type", "month | calendar_year | fiscal_year"),
        ("period", "YYYY-MM for months, YYYY for calendar years, YYYY-YY for fiscal years (Apr-Mar)"),
        ("date", "First day of the month (monthly rows only)"),
        ("fiscal_year", "Indian financial year Apr-Mar the month belongs to"),
        ("geo_level", "area (leaf) < state < zone (North/South India) < country (All India)"),
        ("geography", "Name of the geography on this row"),
        ("area / state / zone", "Hierarchy columns for roll-ups"),
        ("breakdown", "grower = Big/Small grower split; category = CTC/Orthodox/Green split"),
        ("segment", "BG (big growers, estates) | SG (small growers) | CTC | Orthodox | Green | Total"),
        ("production_mkg", "Made-tea production in million kg"),
        ("status", "Final, or Provisional (Apr-2024 to Jan-2025 monthly figures, never re-issued as final)"),
        ("source_file", "Tea Board PDF the figure was taken from"),
    ], columns=["column", "description"])

    master.to_csv(OUT / "tea_production_master.csv", index=False)
    leaf.to_csv(OUT / "monthly_area_grower.csv", index=False)
    qa_df.to_csv(OUT / "qa_checks.csv", index=False)
    recon.to_csv(OUT / "annual_reconciliation.csv", index=False)
    with pd.ExcelWriter(OUT / "tea_production_curated.xlsx") as xw:
        master.to_excel(xw, sheet_name="master", index=False)
        leaf.to_excel(xw, sheet_name="monthly_area_grower", index=False)
        recon.to_excel(xw, sheet_name="annual_reconciliation", index=False)
        qa_df.to_excel(xw, sheet_name="qa_checks", index=False)
        dictionary.to_excel(xw, sheet_name="data_dictionary", index=False)

    print(f"master rows: {len(master):,} | modelling rows: {len(leaf):,} | months: {n_months}")
    print(qa_df.to_string(index=False, max_colwidth=110))


if __name__ == "__main__":
    main()
