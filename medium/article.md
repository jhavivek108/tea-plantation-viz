# Forecasting India's Tea Production: Why the Calendar Beat Machine Learning

### I turned seven Tea Board PDFs into one dataset, tested three ML models over four years, and a simple seasonal median still won.

![India's monthly tea production, 2019 to March 2026. The pattern repeats so faithfully that the calendar alone explains 95% of the variation.](01_hero_heatmap.png)

India makes about 1.3 to 1.4 billion kilograms of tea a year, second only to China. The Tea Board of India publishes the numbers every month, split by growing area, by big estates versus small growers, and by type of tea. It is a rich dataset, but it is published only as PDF tables.

I set out to answer a practical question: **standing at the end of March, can you forecast how much tea India will make in each month of the coming year?** I expected machine learning to win comfortably. It did not, and the reason turned out to be the most useful finding of the project.

## TL;DR

- I extracted **87 months** of production (Jan 2019 to Mar 2026) from seven Tea Board PDFs into one tidy table of 4,746 rows, and verified it with **29 integrity checks**.
- National output is **flat**: about 0.7% a year of growth since 2017, not statistically significant. But **small growers' share rose from 49.3% to 56.6%** in seven years, a very steady trend (R² = 0.96).
- Ridge regression, a Random Forest and gradient boosting all **beat a "same as last year" baseline**, but **none beat a seasonal median** (the typical level of each calendar month). The best ML model scored 10.1% monthly error against the baseline's 9.5%; the difference is not significant (Wilcoxon p = 0.24).
- With production history alone, the calendar is close to the best possible forecast. Beating it needs **weather data**, not a bigger model.

---

## The problem: good data, locked in PDFs

There is no API for Indian tea production. Each Tea Board release is a PDF with one wide table per page: nine producing areas down the side, and three columns per month across the top (**BG** for big growers, **SG** for small growers, and their **Total**). A second table splits each month into **CTC**, **Orthodox** and **Green** tea. Everything is in million kilograms (M kg).

The layouts drift between releases. Headers read "Feb, 2026", "Jan- Dec 22" and "2024 (Apr-Dec)", and one header labelled "Apr-Dec" actually runs to January. Labels wrap across lines, and in one file "Assam Valley" is printed above *and* below its row of numbers.

Rather than guess months from those headers, I mapped every page explicitly:

```python
STATE_PAGES[("2025", 1)] = months("2025-04", "2025-09")
STATE_PAGES[("2025", 2)] = months("2025-10", "2026-03") + ["FY2025-26"]  # last block = printed total
```

It is a few dozen lines, readable, and impossible to misinterpret. `pdfplumber` extracts the text, and a small parser stitches wrapped labels back together.

### Letting the data check itself

The releases overlap heavily: every year is printed once as a calendar year and again as a financial year (April to March). That overlap is a gift, because it means almost every number can be checked against another.

![The 29 integrity checks. Everything reconciles to within 0.01 M kg except one gap that sits in data published only as provisional.](02_table_checks.png)

The extraction reproduces the Tea Board's own arithmetic to the second decimal place, with one exception. The monthly figures for April to December 2024 were published only as **provisional** and never re-issued as final. They sum to **18.75 M kg less** than the Board's final 2024 total, and most of the gap appears to sit in December 2024, which came in at 29.4 M kg against a normal December of about 73.

I kept the published figures, flagged them in a `status` column, and made sure no model learned from that month.

---

## What the data says

Before modelling anything, it pays to understand the series. Four findings shaped everything that followed.

### 1. Output is flat

![Calendar-year production. 2025 was the highest year in the data, but the long-run trend is indistinguishable from zero.](03_annual_production.png)

Production grew at **0.68% a year** from 2017 to 2025. A Kendall trend test gives τ = 0.39, **p = 0.18**: not significant. The two dips are 2020 (−9.5%, COVID-19) and 2024 (−6.5%), and each was fully recovered the following year. 2025 set the high mark at 1,395.5 M kg.

### 2. Seasonality is everything

![Median monthly production by region. North India's cycle is extreme; South India barely varies.](04_seasonal_profile.png)

North India, which makes 83% of the country's tea, stops plucking in January and February and produces **55% of its year's tea in July to October**. South India has no winter dormancy and produces 15 to 25 M kg every month. The coefficient of variation is **62% in the North and 21% in the South**. An STL decomposition puts the seasonal strength of the All-India series at **0.95**.

That number is the key to the whole project. When 95% of the variation is a repeating cycle, a model that simply knows the calendar is already most of the way there.

### 3. A quiet change of ownership

![Small growers' share of production. Both regions crossed 50% and kept climbing.](05_small_grower_share.png)

In 2019, estates and smallholders made almost equal amounts. By 2025, small growers made **56.6%** of India's tea. A linear fit gives **+1.12 percentage points a year, R² = 0.96, p < 0.001**. Because total output was flat, estate production fell from 705 to 605 M kg. The shift is fastest in West Bengal, where the small-grower share rose from 56% to 68%.

### 4. Shocks are rare but violent

![2020 production compared with 2019. The lockdown hit exactly when the spring flush began.](06_covid_impact.png)

The 2020 lockdown cut April production by **54%** compared with 2019. Output was back at 2019 levels by September. Shocks like this, and the provisional December 2024, are the only large departures from the seasonal pattern in seven years.

---

## The forecasting setup

**The task.** At the end of each March, forecast every month of the next financial year for each of the **18 additive series**: nine producing areas times big and small growers. Summing them gives the All-India forecast.

**The features** use only information available on the forecast date:

- production in the same month one and two years earlier
- the median and mean of that calendar month across all earlier years
- the month, the producing area and the grower type
- a flag when the year-earlier month was disrupted (the 2020 lockdown or December 2024)

The seasonal history features take a few lines of pandas:

```python
lag12 = y.shift(12)
hist = pd.DataFrame({"v": lag12, "m": y.index.month})
hist_median = hist.groupby("m")["v"].transform(lambda v: v.expanding().median())
```

**The models.** Ridge regression, a Random Forest and histogram gradient boosting from scikit-learn, compared against two baselines:

- **Seasonal naive:** each month equals the same month last year.
- **Seasonal median:** each month equals the median of that calendar month in all earlier years.

**The evaluation** is a rolling-origin backtest. I trained on everything before April 2022 and forecast FY 2022-23, then repeated for the next three years. That gives four out-of-sample years and 48 forecast months. Disrupted months are excluded as training targets but kept in the evaluation, so every model is scored on the shocks it could not have predicted.

```python
for fold in ["2022-23", "2023-24", "2024-25", "2025-26"]:
    start = pd.Timestamp(f"{fold[:4]}-04-01")
    end = start + pd.DateOffset(months=12)
    train = data[(data.date < start) & ~data.target_disrupted]
    test = data[(data.date >= start) & (data.date < end)]
    model.fit(train[FEATURES], train["y"])
    preds = model.predict(test[FEATURES])
```

The headline metric is **WAPE** (weighted absolute percentage error): total absolute error as a percentage of actual production. Unlike MAPE, it does not explode in January, when North India makes almost nothing.

---

## The results

![Mean backtest results over four held-out years. Lower is better.](07_table_results.png)

![Monthly WAPE for every model in every held-out year.](08_model_comparison.png)

All three ML models clearly beat "same as last year". **None of them beat the seasonal median.**

Is the gap real or noise? A Wilcoxon signed-rank test compares two models' absolute errors month by month across all 48 forecasts:

```python
from scipy.stats import wilcoxon

wilcoxon(err["Seasonal median"], err["Seasonal naive"]).pvalue   # 0.006: real improvement
wilcoxon(err["Random forest"], err["Seasonal naive"]).pvalue     # 0.050: borderline
wilcoxon(err["Random forest"], err["Seasonal median"]).pvalue    # 0.243: no detectable difference
```

The Random Forest and the seasonal median are statistically indistinguishable. The extra complexity bought nothing.

### Trying harder

I tried two fixes before accepting the result:

1. **Predict the correction, not the level.** The series differ about 100-fold in size (Assam Valley against Darjeeling), which makes it hard for trees to share what they learn. Predicting the log-ratio to the seasonal median makes every series comparable. Best result: 9.8%, still behind the baseline.
2. **Shorten the horizon to one month.** Surely last month's year-on-year change helps predict next month? It made things *worse*: a simple momentum rule scored 21%. The timing of the peak harvest shifts from year to year, so a strong July often means a weaker August.

---

## Why the calendar wins

The model's own feature importances explain it.

![Permutation importance for the Random Forest on the held-out year FY 2025-26.](09_feature_importance.png)

The Random Forest relies almost entirely on the same signal as the baseline: the typical level of that calendar month. Everything else adds little.

![Held-out FY 2025-26: both forecasts track the season closely and miss the same months.](10_actual_vs_forecast.png)

Look at where the forecasts miss. Both models undershoot April, May and July 2025 and overshoot September, in the same places and by similar amounts. Misses like these are what weather produces: an early or late monsoon, a dry spell in the wrong weeks. Nothing in past production predicts that, so more model capacity cannot recover it.

This is the general lesson: **when a series is dominated by a stable seasonal cycle, a robust seasonal baseline is hard to beat with the series' own history.** To improve on it you need information the series does not contain: rainfall, temperature, soil moisture, or auction prices.

---

## The outlook for FY 2026-27

![Forecast for April 2026 to March 2027. The shaded band is an approximate 80% interval from backtest errors.](11_forecast.png)

Refitted on all data to March 2026, the Random Forest forecasts about **1,373 M kg for FY 2026-27**, 0.7% below FY 2025-26. The seasonal median gives about **1,356 M kg**. Based on past annual errors, a reasonable range is roughly **1,270 to 1,420 M kg**.

---

## What I learned

1. **Build the baseline first, and build a strong one.** "Same as last year" is a weak baseline. The seasonal median is a strong one, and it is the one that tells you whether ML is adding anything.
2. **Test the difference, not just the ranking.** A 0.6-point gap in a leaderboard looks like a result. A paired Wilcoxon test showed it was not.
3. **Curation is half the project.** Explicit page maps and 29 cross-checks turned PDFs into data I could trust, and surfaced a reporting gap in the official 2024 numbers.
4. **A negative result is still a result.** "Production history cannot beat the calendar" tells you exactly where to look next: weather data.
5. **Pick a metric that survives zeros.** WAPE stays sensible when North India's January output is close to nothing; MAPE does not.

## Next steps

The obvious extension is to add district-level rainfall and temperature from the India Meteorological Department as features. That is the information the residuals are asking for.

The full pipeline (PDF extraction, 29 integrity checks, the analysis, the models and an offline dashboard) re-runs end to end with a few Python commands. The code and the curated dataset are on GitHub: [REPO_URL]

*Data source: Tea Board of India, monthly production statistics, January 2019 to March 2026.*
