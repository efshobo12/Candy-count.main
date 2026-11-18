"""World Happiness Report EDA (2015-2019).

Run it either way:

    uv run happiness_eda.py 
    uv run happiness_eda.py --csv 2015=data/2015.csv --csv 2016=data/2016.csv \
                            --csv 2017=data/2017.csv --csv 2018=data/2018.csv \
                            --csv 2019=data/2019.csv

Required packages: pandas, matplotlib, scipy, plotly (+ kaleido for PNG maps).
All figures and tables are saved in ./outputs
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

YEARS = [2015, 2016, 2017, 2018, 2019]
OUT = Path("outputs")

FACTORS = ["gdp", "family", "health", "freedom", "corruption", "generosity"]
REQUIRED = ["country", "score", *FACTORS]

# Country names that changed between yearly files -> one common name
ALIASES = {
    "trinidad & tobago": "Trinidad and Tobago",
    "northern cyprus": "North Cyprus",
    "macedonia": "North Macedonia",
    "taiwan province of china": "Taiwan",
    "hong kong s.a.r., china": "Hong Kong",
    "somaliland region": "Somaliland Region",
}
# Fallback regions for countries that never appear in the 2015/2016 files
EXTRA_REGIONS = {
    "Gambia": "Sub-Saharan Africa",
    "Namibia": "Sub-Saharan Africa",
    "Mauritania": "Sub-Saharan Africa",
}


# ----------------------------------------------------------------- loading
def load_year(path: Path, year: int, overrides: dict = None) -> pd.DataFrame:
    """Read one yearly CSV and rename its columns to a common schema."""
    raw = pd.read_csv(path)
    mapping = {}
    for col in raw.columns:
        n = re.sub(r"[^a-z]", "", col.lower())
        if "country" in n:
            key = "country"
        elif n == "region":
            key = "region"
        elif "rank" in n:
            key = "rank_original"
        elif "score" in n:
            key = "score"
        elif "gdp" in n or "economy" in n:
            key = "gdp"
        elif "family" in n or "socialsupport" in n:
            key = "family"
        elif "health" in n or "lifeexpectancy" in n:
            key = "health"
        elif "freedom" in n:
            key = "freedom"
        elif "corruption" in n or "trust" in n:
            key = "corruption"
        elif "generosity" in n:
            key = "generosity"
        else:
            continue
        if key not in mapping.values():
            mapping[col] = key

    # manual overrides, e.g. --map score=ss --map gdp=22 (applied to every year)
    for key, colname in (overrides or {}).items():
        if colname not in raw.columns:
            sys.exit(f"{path} ({year}): no column named '{colname}'. Found: {list(raw.columns)}")
        mapping = {c: k for c, k in mapping.items() if k != key and c != colname}
        mapping[colname] = key

    # still unmatched? ask the person (only when typing at a terminal)
    unmatched = [k for k in REQUIRED if k not in mapping.values()]
    if unmatched and sys.stdin.isatty():
        print(f"\n{path} ({year}): could not recognise some columns.")
        print(f"Columns in this file: {list(raw.columns)}")
        for key in unmatched:
            while True:
                ans = input(f"  Which column is '{key}'? ").strip()
                if ans in raw.columns:
                    mapping = {c: k for c, k in mapping.items() if c != ans}
                    mapping[ans] = key
                    break
                print("  That column does not exist, try again.")

    df = raw[list(mapping)].rename(columns=mapping)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path} ({year}) is missing columns: {missing}\nFound: {list(raw.columns)}")

    df["country"] = df["country"].astype(str).str.strip()
    df["country"] = df["country"].map(lambda c: ALIASES.get(c.lower(), c))
    for col in ["score", *FACTORS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().sum() < 2:
            sys.exit(f"{path} ({year}): column '{col}' has no usable numbers. "
                     "Check that the right column was matched and that it holds numeric values.")
    if "region" not in df.columns:
        df["region"] = pd.NA
    df["year"] = year
    # same definition of rank for every year (1 = happiest)
    df["rank"] = df["score"].rank(ascending=False, method="min").astype("Int64")
    return df[["year", "country", "region", "rank", "score", *FACTORS]]


def fill_regions(data: pd.DataFrame) -> pd.DataFrame:
    """2017-2019 files have no Region column, so borrow it from 2015/2016."""
    known = data.dropna(subset=["region"]).drop_duplicates("country").set_index("country")["region"].to_dict()
    known.update({k: v for k, v in EXTRA_REGIONS.items() if k not in known})
    data["region"] = data["region"].fillna(data["country"].map(known)).fillna("Unknown")
    return data


def get_paths(args) -> dict:
    paths = {}
    for item in args.csv or []:
        year, _, p = item.partition("=")
        paths[int(year)] = Path(p.strip().strip("\"'"))
    for year in YEARS:
        if year not in paths:
            p = input(f"Path to the {year} CSV file: ").strip().strip("\"'")
            paths[year] = Path(p)
    for year, p in paths.items():
        if not p.is_file():
            sys.exit(f"File not found for {year}: {p}")
    return paths


# ------------------------------------------------------------------ tasks
def task1_histograms(data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, year in zip(axes.flat, YEARS):
        s = data.loc[data.year == year, "score"].dropna()
        ax.hist(s, bins=15, edgecolor="black")
        ax.axvline(s.mean(), color="red", linestyle="--", label=f"mean {s.mean():.2f}")
        ax.set(title=f"Happiness score {year}", xlabel="Score", ylabel="Countries")
        ax.legend()
        print(f"  {year}: mean={s.mean():.3f} median={s.median():.3f} std={s.std():.3f} skew={s.skew():.3f}")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "task1_score_histograms.png", dpi=150)
    plt.close(fig)


def labeled_scatter(data, x, y, xlabel, ylabel, prefix):
    for year in YEARS:
        d = data[data.year == year].dropna(subset=[x, y])
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(d[x], d[y], alpha=0.7)
        for _, r in d.iterrows():
            ax.annotate(r["country"], (r[x], r[y]), fontsize=6, alpha=0.8)
        ax.set(title=f"{ylabel} vs {xlabel} ({year})", xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"{prefix}_{year}.png", dpi=150)
        plt.close(fig)


def task4_3d(data):
    for year in YEARS:
        d = data[data.year == year].dropna(subset=["gdp", "corruption", "generosity"])
        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(d["gdp"], d["corruption"], d["generosity"], c=d["score"], cmap="viridis")
        # label the 10 happiest and 10 least happy countries to keep it readable
        for _, r in pd.concat([d.nlargest(10, "score"), d.nsmallest(10, "score")]).iterrows():
            ax.text(r["gdp"], r["corruption"], r["generosity"], r["country"], fontsize=6)
        ax.set(title=f"GDP, Corruption, Generosity ({year})", xlabel="GDP per Capita",
               ylabel="Corruption (trust)", zlabel="Generosity")
        fig.tight_layout()
        fig.savefig(OUT / f"task4_3d_{year}.png", dpi=150)
        plt.close(fig)


def task5_missing(data):
    lines = []
    for a, b in zip(YEARS, YEARS[1:]):
        ca = set(data.loc[data.year == a, "country"])
        cb = set(data.loc[data.year == b, "country"])
        only_a, only_b = sorted(ca - cb), sorted(cb - ca)
        lines.append(f"{a} -> {b}")
        lines.append(f"  In {a} but not in {b} ({len(only_a)}): {', '.join(only_a) or 'none'}")
        lines.append(f"  In {b} but not in {a} ({len(only_b)}): {', '.join(only_b) or 'none'}")
    text = "\n".join(lines)
    print(text)
    (OUT / "task5_missing_countries.txt").write_text(text, encoding="utf-8")


def task6_correlations(data):
    pairs = [
        ("Score vs Generosity", "score", "generosity"),
        ("Freedom vs Generosity", "freedom", "generosity"),
        ("GDP vs Corruption", "gdp", "corruption"),
        ("GDP vs Freedom", "gdp", "freedom"),
    ]
    rows = []
    for year in YEARS:
        d = data[data.year == year]
        for label, a, b in pairs:
            x = d[[a, b]].dropna()
            rows.append({
                "year": year, "pair": label, "n": len(x),
                "pearson": stats.pearsonr(x[a], x[b])[0],
                "spearman": stats.spearmanr(x[a], x[b])[0],
                "kendall": stats.kendalltau(x[a], x[b])[0],
            })
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    table.to_csv(OUT / "task6_correlations.csv", index=False)


def task7_strongest(data):
    rows = []
    for year in YEARS:
        d = data[data.year == year]
        for f in FACTORS:
            x = d[["score", f]].dropna()
            rows.append({"year": year, "variable": f,
                         "pearson": x["score"].corr(x[f], method="pearson"),
                         "spearman": x["score"].corr(x[f], method="spearman")})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "task7_score_correlations.csv", index=False)
    summary = table.groupby("variable")[["pearson", "spearman"]].mean().sort_values("pearson", ascending=False)
    print("Average correlation with Score across years (strongest first):")
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    summary["pearson"].plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set(title="Mean Pearson correlation with happiness score (2015-2019)", xlabel="Pearson r")
    fig.tight_layout()
    fig.savefig(OUT / "task7_score_correlations.png", dpi=150)
    plt.close(fig)


def task8_maps(data):
    try:
        import plotly.express as px
    except ImportError:
        print("  plotly not installed, skipping maps (uv add plotly kaleido)")
        return
    for year in YEARS:
        d = data[data.year == year]
        fig = px.choropleth(d, locations="country", locationmode="country names", color="score",
                            hover_name="country", color_continuous_scale="RdYlGn",
                            title=f"World Happiness Score {year}")
        fig.write_html(OUT / f"task8_map_{year}.html")
        try:
            fig.write_image(OUT / f"task8_map_{year}.png", width=1400, height=800)
        except Exception:
            print(f"  {year}: saved HTML map only (install kaleido for PNG: uv add kaleido)")


def task9_trends(data):
    pivot = data.pivot_table(index="region", columns="year", values="rank", aggfunc="mean").round(1)
    pivot["change_2015_2019"] = pivot[2019] - pivot[2015]
    print("Mean rank by region (lower = happier):")
    print(pivot.sort_values(2019).to_string())
    pivot.to_csv(OUT / "task9_region_mean_rank.csv")

    fig, ax = plt.subplots(figsize=(12, 7))
    pivot[YEARS].T.plot(ax=ax, marker="o")
    ax.invert_yaxis()
    ax.set(title="Mean happiness rank by region", xlabel="Year", ylabel="Mean rank (1 = happiest)")
    ax.set_xticks(YEARS)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "task9_region_rank_trends.png", dpi=150)
    plt.close(fig)

    wide = data.pivot(index="country", columns="year", values="rank")
    wide["change_2015_2019"] = wide[2015] - wide[2019]  # positive = moved up
    movers = wide.dropna(subset=["change_2015_2019"]).sort_values("change_2015_2019")
    movers.to_csv(OUT / "task9_country_rank_changes.csv")
    print("\nBiggest risers 2015->2019:\n", movers.tail(5)[[2015, 2019, "change_2015_2019"]].to_string())
    print("\nBiggest fallers 2015->2019:\n", movers.head(5)[[2015, 2019, "change_2015_2019"]].to_string())


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="World Happiness Report EDA")
    parser.add_argument("--csv", action="append", metavar="YEAR=PATH",
                        help="CSV for a year, e.g. --csv 2015=data/2015.csv (repeatable)")
    parser.add_argument("--map", action="append", metavar="FIELD=COLUMN",
                        help="tell the script which column is which, e.g. --map score=ss --map gdp=22 "
                             "(fields: country, region, score, gdp, family, health, freedom, corruption, generosity)")
    args = parser.parse_args()

    overrides = {}
    for item in args.map or []:
        field, _, col = item.partition("=")
        if field not in REQUIRED + ["region"]:
            sys.exit(f"Unknown field '{field}'. Use one of: {REQUIRED + ['region']}")
        overrides[field] = col.strip()

    OUT.mkdir(exist_ok=True)
    paths = get_paths(args)
    data = pd.concat([load_year(paths[y], y, overrides) for y in YEARS], ignore_index=True)
    data = fill_regions(data)
    data.to_csv(OUT / "combined_clean_data.csv", index=False)
    print(f"Loaded {len(data)} rows across {len(YEARS)} years.\n")

    print("Task 1: score distribution");          task1_histograms(data)
    print("\nTask 2: GDP vs Health");              labeled_scatter(data, "gdp", "health", "GDP per Capita", "Health", "task2_gdp_health")
    print("Task 3: Freedom vs Generosity");        labeled_scatter(data, "freedom", "generosity", "Freedom", "Generosity", "task3_freedom_generosity")
    print("Task 4: 3D scatter");                   task4_3d(data)
    print("\nTask 5: missing countries");          task5_missing(data)
    print("\nTask 6: correlations");               task6_correlations(data)
    print("\nTask 7: strongest relationships");    task7_strongest(data)
    print("\nTask 8: world maps");                 task8_maps(data)
    print("\nTask 9: rank trends");                task9_trends(data)
    print(f"\nDone. Everything is in {OUT.resolve()}")


if __name__ == "__main__":
    main()