# Indian Tea Production Analysis (2017 – Mar 2026)

This project takes seven Tea Board of India PDFs and turns them into one curated dataset. It then runs a statistical analysis and trains and evaluates machine-learning models that forecast production 12 months ahead.

## Run it

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\extract.py          # PDFs -> curated dataset + 29 QA checks
.venv\Scripts\python src\analysis.py         # statistics, 12 figures, models, forecast
.venv\Scripts\python src\build_notebook.py   # same analysis as an executed notebook
.venv\Scripts\python src\dashboard_charts.py # dashboard charts: map, treemap, radar, fan chart...
.venv\Scripts\python src\dashboard.py        # offline dashboard -> reports\dashboard.html
.venv\Scripts\python src\medium.py serve     # Medium article preview at http://127.0.0.1:8000/preview.html
```

The Medium article is `medium/article.md`. `src/medium.py` does three things:
- copies its images into `medium/images/`, numbered in the order they appear;
- renders its two tables as images, because Medium has no tables;
- builds a Medium-style preview page with a **Copy article** button.

The publishing steps are in `medium/PUBLISHING.md`.

| Path | Contents |
|---|---|
| `data/raw/` | The 7 source PDFs (Tea Board, Kolkata) |
| `data/external/` | India state boundaries for the dashboard map: geoBoundaries gbOpen IND ADM1 (DataMeet / Election Commission of India, CC BY 2.5 IN) |
| `data/processed/tea_production_master.csv` | **The single curated dataset** (4,746 rows) |
| `data/processed/tea_production_curated.xlsx` | Master table, modelling view, QA checks, reconciliation and data dictionary in one workbook (ready for Power BI or Tableau) |
| `data/processed/monthly_area_grower.csv` | Modelling view: 9 areas × 2 grower types × 87 months |
| `notebooks/tea_production_analysis.ipynb` | Executed notebook with every output |
| `reports/figures/` | 12 charts |
| `reports/*.csv`, `statistics.json` | Backtest metrics, predictions, forecast, all test statistics |

## The curated dataset

The master table is one tidy table. Each row is one figure for one period, one geography and one segment.

- **Periods:** months from Jan 2019 to Mar 2026 (87 continuous months), plus official annual totals: calendar years 2017–2024 and financial years 2017-18 to 2024-25.
- **Geography:** area (Assam Valley, Cachar, Dooars, Terai, Darjeeling, Others, Tamil Nadu, Kerala, Karnataka), then state, then zone (North or South India), then All India.
- **Breakdowns:** `grower` (BG = big growers/estates, SG = small growers, Total) and `category` (CTC, Orthodox, Green, Total).
- **Status:** `Final` or `Provisional`.

**How it was curated.** The PDFs repeat many months: each year is printed once as a calendar year and once as a financial year (April–March). These duplicates were matched and compared, and they agree exactly in all 3,276 cells. Where a month exists as both a Final and a Provisional figure, the Final one is kept. `extract.py` then checks that:

- BG + SG = Total;
- every level of the hierarchy adds up (area → state → zone → All India);
- CTC + Orthodox + Green = Total;
- every printed annual total matches the sum of its months.

All 29 checks pass, to within 0.01 M kg.

**Known data issue.** The monthly figures for Apr–Dec 2024 were published only as *provisional* and never re-issued as final. They sum to 18.75 M kg less than the Tea Board's final 2024 total, and 18.59 M kg of that gap is in North India. Most of the shortfall appears to sit in December 2024: 29.4 M kg against a normal December of about 73 (z = −4.7). The figures are kept as published and flagged in the `status` column.

## Key findings

1. **Output has no significant trend.** Calendar-year output ranges from 1,258 to 1,396 M kg. Growth from 2017 to 2025 is +0.68% a year (CAGR), and a Kendall trend test is not significant (τ = 0.39, p = 0.18). 2025 was the highest year in the series at 1,395.5 M kg. The two dips were 2020 (−9.5%, COVID) and 2024 (−6.5%).
2. **Seasonality dominates.** It explains 95% of the monthly variation (STL seasonal strength 0.95). North India makes 55% of its tea in July–October and almost nothing in January–February; its monthly coefficient of variation is 62%, against 21% for South India. Kruskal–Wallis tests confirm a month effect in both zones (p < 10⁻⁶).
3. **Small growers now produce most of India's tea.** Their share rose from 49.3% (2019) to 56.6% (2025), a linear trend of +1.12 percentage points a year (R² = 0.96, p < 0.001). Over the same period, big-estate output fell from 705 to 605 M kg. The shift is strongest in West Bengal, where the small-grower share went from 56% to 68%.
4. **Where the tea comes from.** Assam produces about 50% of India's tea, West Bengal about 30% and Tamil Nadu about 12%. These shares barely moved between 2019 and 2025.
5. **Product mix.** CTC is 88% of output. Orthodox tea rebounded 55%, from 96 M kg (2020) to 149 M kg (2025), and is now 10.7% of output. Green tea is 1.35%.
6. **Shocks.** The 2020 lockdown cut April production by 54% and April–May by 38% compared with 2019; output was back at 2019 levels from September.

## Machine-learning model

**Task.** At the end of each March, forecast every month of the next financial year for each of the 18 area × grower series. Summing the series gives the All-India forecast.

**Features.** Production in the same month one and two years earlier, the median and mean of that calendar month in all prior years, month, area, grower type, and a flag when the year-ago month was disrupted.

**Evaluation.** A rolling-origin backtest over four held-out years (FY 2022-23 to FY 2025-26). WAPE is the total absolute error as a percentage of actual production; lower is better.

| Model | All-India monthly WAPE | Annual-total error |
|---|---|---|
| Seasonal median (baseline) | **9.5%** | 3.2% |
| Random forest | 10.1% | 3.6% |
| Ridge regression | 11.0% | 2.8% |
| Gradient boosting | 11.4% | 4.5% |
| Seasonal naive (copy last year) | 13.2% | 3.9% |

**Conclusion.** The ML models beat copying last year's figures, but none beats the seasonal-median baseline.

- Random forest vs the seasonal median: no significant difference (Wilcoxon p = 0.24 on 48 paired monthly errors).
- A version that learns a correction to the baseline, and a 1-month-ahead model using recent momentum, did not beat it either.

From history alone, the best predictor of a month is that month's typical level. Year-to-year departures are driven by weather, so beating the baseline would need outside drivers such as rainfall, temperature or auction prices. That is the natural next step.

**FY 2026-27 forecast (Random forest).** About 1,373 M kg, 0.7% below FY 2025-26. The seasonal-median benchmark gives about 1,356 M kg. Past annual errors imply a range of roughly 1,268–1,418 M kg. See `reports/forecast_FY2026-27.csv`.

## Limitations

- About 7 years of monthly data (18 series) is a small sample for machine learning, so the backtest has only four held-out years.
- The 2024 monthly figures are provisional and inconsistent with the final annual total, which affects the FY 2024-25 test year.
- The models use only past production. They cannot anticipate weather, pest outbreaks or policy changes.
