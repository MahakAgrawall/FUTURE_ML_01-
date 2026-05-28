"""
=============================================================
  Sales & Demand Forecasting System
  Task 1 — Future Interns ML Project (2026)
=============================================================
  Features:
    - Realistic 3-year synthetic sales dataset (no download needed)
    - Data cleaning & missing value handling
    - Time-based feature engineering (lag, rolling, seasonality)
    - Three models: Linear Regression, Random Forest, Gradient Boosting
    - Best model auto-selected by RMSE
    - Full error analysis: MAE, RMSE, R², MAPE, Forecast Accuracy
    - 6-month future forecast with confidence band
    - 6-panel business-ready dark dashboard (saved as PNG)
    - Business insights summary printed to console
=============================================================
"""

# ──────────────────────────────────────────────
# 1.  IMPORTS
# ──────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

# Output folder = same directory as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 2.  SYNTHETIC DATASET  (3 years monthly)
# ──────────────────────────────────────────────
np.random.seed(42)

def generate_sales_data() -> pd.DataFrame:
    """
    Generates 36 months of realistic retail sales data with:
      - Upward trend
      - Seasonal peaks (Nov–Jan holiday spike, Jul–Aug summer dip)
      - Promotional months (random 10% boost)
      - Gaussian noise
    """
    dates = pd.date_range(start="2022-01-01", periods=36, freq="MS")
    months = dates.month

    trend      = np.linspace(50_000, 95_000, 36)
    seasonality = (
        8000 * np.sin(2 * np.pi * (months - 3) / 12)   # base wave
        + np.where(months == 11, 18000, 0)               # Nov Black Friday
        + np.where(months == 12, 22000, 0)               # Dec Christmas
        + np.where(months ==  1,  8000, 0)               # Jan New Year
        + np.where(months ==  7, -5000, 0)               # Jul summer dip
        + np.where(months ==  8, -4000, 0)               # Aug summer dip
    )
    promotion  = np.where(np.random.random(36) < 0.12, 1, 0)  # ~12% of months
    promo_lift = promotion * np.random.uniform(3000, 9000, 36)
    noise      = np.random.normal(0, 2500, 36)

    sales = trend + seasonality + promo_lift + noise
    sales = np.clip(sales, 20_000, None).round(2)

    df = pd.DataFrame({
        "date":      dates,
        "sales":     sales,
        "promotion": promotion,
    })

    # Inject 3 random missing values to demonstrate cleaning
    miss_idx = np.random.choice(range(2, 34), size=3, replace=False)
    df.loc[miss_idx, "sales"] = np.nan
    return df


# ──────────────────────────────────────────────
# 3.  DATA CLEANING
# ──────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_before = df["sales"].isna().sum()

    # Linear interpolation for missing values
    df["sales"] = df["sales"].interpolate(method="linear").round(2)

    missing_after = df["sales"].isna().sum()
    print(f"  Missing values fixed : {missing_before} → {missing_after}")
    print(f"  Date range           : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total records        : {len(df)}")
    print(f"  Sales range          : ₹{df['sales'].min():,.0f} — ₹{df['sales'].max():,.0f}")
    return df


# ──────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"]         = df["date"].dt.month
    df["quarter"]       = df["date"].dt.quarter
    df["year"]          = df["date"].dt.year
    df["month_num"]     = np.arange(len(df))          # linear trend proxy

    # Cyclical month encoding (avoids Dec=12, Jan=1 gap)
    df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    df["lag_1"]         = df["sales"].shift(1)
    df["lag_2"]         = df["sales"].shift(2)
    df["lag_3"]         = df["sales"].shift(3)
    df["lag_12"]        = df["sales"].shift(12)       # same month last year

    # Rolling statistics
    df["roll_mean_3"]   = df["sales"].shift(1).rolling(3).mean()
    df["roll_mean_6"]   = df["sales"].shift(1).rolling(6).mean()
    df["roll_std_3"]    = df["sales"].shift(1).rolling(3).std()

    # Holiday flags
    df["is_holiday"]    = df["month"].isin([11, 12, 1]).astype(int)
    df["is_summer_dip"] = df["month"].isin([7, 8]).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


FEATURE_COLS = [
    "month_num", "month_sin", "month_cos", "quarter",
    "lag_1", "lag_2", "lag_3", "lag_12",
    "roll_mean_3", "roll_mean_6", "roll_std_3",
    "is_holiday", "is_summer_dip", "promotion",
]


# ──────────────────────────────────────────────
# 5.  MODEL TRAINING & EVALUATION
# ──────────────────────────────────────────────
MODELS = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=300, max_depth=6,
                                                  random_state=42),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                      max_depth=4, random_state=42),
}

def evaluate_models(df: pd.DataFrame) -> dict:
    # Temporal split — last 6 months = test
    split_idx = len(df) - 6
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[FEATURE_COLS]
    y_train = train["sales"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["sales"]

    scaler  = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    results = {}
    print(f"\n  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>7} {'MAPE':>8} {'Acc':>8}")
    print(f"  {'─'*67}")

    for name, model in MODELS.items():
        model.fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100
        acc  = max(0, 100 - mape)

        print(f"  {name:<22} ₹{mae:>8,.0f} ₹{rmse:>8,.0f} {r2:>7.3f} {mape:>7.1f}% {acc:>7.1f}%")

        results[name] = {
            "model": model, "scaler": scaler,
            "mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "acc": acc,
            "y_test": y_test, "y_pred": y_pred,
            "train": train, "test": test,
        }

    best_name = min(results, key=lambda k: results[k]["rmse"])
    print(f"\n  ✅ Best model : {best_name}  (RMSE = ₹{results[best_name]['rmse']:,.0f})")
    return results, best_name


# ──────────────────────────────────────────────
# 6.  FUTURE FORECAST (6 months)
# ──────────────────────────────────────────────
def forecast_future(df: pd.DataFrame, best_result: dict, n_months: int = 6) -> pd.DataFrame:
    model  = best_result["model"]
    scaler = best_result["scaler"]

    last_date  = df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1),
                                  periods=n_months, freq="MS")

    # Extend dataframe iteratively so lags stay consistent
    ext = df.copy()
    preds = []

    for fd in future_dates:
        month     = fd.month
        month_num = ext["month_num"].max() + 1
        lag_1     = ext["sales"].iloc[-1]
        lag_2     = ext["sales"].iloc[-2]
        lag_3     = ext["sales"].iloc[-3]
        lag_12    = ext["sales"].iloc[-12] if len(ext) >= 12 else ext["sales"].mean()
        roll_m3   = ext["sales"].iloc[-3:].mean()
        roll_m6   = ext["sales"].iloc[-6:].mean()
        roll_s3   = ext["sales"].iloc[-3:].std()
        promo     = 0   # no promo assumed for future

        row = pd.DataFrame([{
            "month_num": month_num,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "quarter":   (month - 1) // 3 + 1,
            "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_12": lag_12,
            "roll_mean_3": roll_m3, "roll_mean_6": roll_m6, "roll_std_3": roll_s3,
            "is_holiday":    int(month in [11, 12, 1]),
            "is_summer_dip": int(month in [7, 8]),
            "promotion": promo,
        }])

        X_scaled = scaler.transform(row[FEATURE_COLS])
        pred     = model.predict(X_scaled)[0]
        preds.append(pred)

        # Append to ext for next iteration's lags
        new_row = pd.DataFrame([{"date": fd, "sales": pred, "promotion": promo,
                                  "month": month, "quarter": (month-1)//3+1,
                                  "year": fd.year, "month_num": month_num,
                                  "month_sin": row["month_sin"].values[0],
                                  "month_cos": row["month_cos"].values[0],
                                  "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
                                  "lag_12": lag_12, "roll_mean_3": roll_m3,
                                  "roll_mean_6": roll_m6, "roll_std_3": roll_s3,
                                  "is_holiday": int(month in [11,12,1]),
                                  "is_summer_dip": int(month in [7,8])}])
        ext = pd.concat([ext, new_row], ignore_index=True)

    # Confidence band: ± RMSE
    rmse = best_result["rmse"]
    future_df = pd.DataFrame({
        "date":       future_dates,
        "forecast":   np.array(preds).round(2),
        "lower_band": (np.array(preds) - rmse).round(2),
        "upper_band": (np.array(preds) + rmse).round(2),
    })
    return future_df


# ──────────────────────────────────────────────
# 7.  DASHBOARD
# ──────────────────────────────────────────────
DARK   = "#0a0f1e"
PANEL  = "#111827"
CARD   = "#1f2937"
TEXT   = "#f9fafb"
MUTED  = "#9ca3af"
GRID   = "#374151"
BLUE   = "#3b82f6"
GREEN  = "#22c55e"
ORANGE = "#f97316"
RED    = "#ef4444"
PURPLE = "#a78bfa"
TEAL   = "#2dd4bf"

def style_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=10)
    ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K")
    )

def plot_dashboard(df_feat: pd.DataFrame, results: dict,
                   best_name: str, future_df: pd.DataFrame,
                   filepath: str):

    best   = results[best_name]
    train  = best["train"]
    test   = best["test"]
    y_test = best["y_test"]
    y_pred = best["y_pred"]

    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor(DARK)
    gs  = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ── KPI Cards (top strip) ─────────────────
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.set_facecolor(DARK)
    kpi_ax.axis("off")

    kpi_data = [
        ("Total Revenue (36mo)",  f"₹{df_feat['sales'].sum()/1e6:.2f}M", BLUE),
        ("Avg Monthly Sales",     f"₹{df_feat['sales'].mean():,.0f}",     TEAL),
        ("Peak Month Sales",      f"₹{df_feat['sales'].max():,.0f}",      GREEN),
        ("Forecast Accuracy",     f"{best['acc']:.1f}%",                  PURPLE),
        ("Best Model",            best_name,                              ORANGE),
        ("6-mo Forecast Total",   f"₹{future_df['forecast'].sum()/1e6:.2f}M", RED),
    ]
    for i, (label, val, color) in enumerate(kpi_data):
        x = 0.02 + i * 0.165
        rect = mpatches.FancyBboxPatch((x, 0.05), 0.15, 0.88,
                                        boxstyle="round,pad=0.02",
                                        facecolor=CARD, edgecolor=color,
                                        linewidth=1.5,
                                        transform=kpi_ax.transAxes, clip_on=False)
        kpi_ax.add_patch(rect)
        kpi_ax.text(x + 0.075, 0.68, val, ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold",
                    transform=kpi_ax.transAxes)
        kpi_ax.text(x + 0.075, 0.28, label, ha="center", va="center",
                    color=MUTED, fontsize=7.5,
                    transform=kpi_ax.transAxes)

    # ── Panel 1: Full Sales History + Forecast ─
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(train["date"], train["sales"],
             color=BLUE, linewidth=1.8, label="Historical Sales", zorder=3)
    ax1.plot(test["date"], y_test.values,
             color=GREEN, linewidth=1.8, linestyle="--", label="Actual (Test)", zorder=3)
    ax1.plot(test["date"], y_pred,
             color=ORANGE, linewidth=1.8, linestyle="-.", label="Predicted (Test)", zorder=3)
    ax1.plot(future_df["date"], future_df["forecast"],
             color=RED, linewidth=2, label="6-mo Forecast", zorder=4)
    ax1.fill_between(future_df["date"],
                     future_df["lower_band"], future_df["upper_band"],
                     alpha=0.20, color=RED, label="Confidence Band")
    ax1.axvline(test["date"].iloc[0], color=MUTED, linestyle=":", linewidth=1)
    ax1.text(test["date"].iloc[0], ax1.get_ylim()[0],
             "  Test →", color=MUTED, fontsize=7, va="bottom")
    style_ax(ax1, "📈  Sales History + 6-Month Forecast")
    ax1.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT,
               facecolor=PANEL, edgecolor=GRID, loc="upper left")
    ax1.set_xlabel("Date")

    # ── Panel 2: Actual vs Predicted (test) ───
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.scatter(y_test.values, y_pred,
                color=TEAL, edgecolors=DARK, s=60, zorder=3, alpha=0.85)
    mn = min(y_test.values.min(), y_pred.min()) * 0.97
    mx = max(y_test.values.max(), y_pred.max()) * 1.03
    ax2.plot([mn, mx], [mn, mx], color=ORANGE, linewidth=1.5,
             linestyle="--", label="Perfect prediction")
    style_ax(ax2, "🎯  Actual vs Predicted")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax2.set_xlabel("Actual Sales")
    ax2.set_ylabel("Predicted Sales")
    ax2.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT,
               facecolor=PANEL, edgecolor=GRID)
    r2_val = r2_score(y_test, y_pred)
    ax2.text(0.05, 0.92, f"R² = {r2_val:.3f}", transform=ax2.transAxes,
             color=GREEN, fontsize=9, fontweight="bold")

    # ── Panel 3: Residuals ────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    residuals = y_test.values - y_pred
    ax3.bar(range(len(residuals)), residuals,
            color=[GREEN if r >= 0 else RED for r in residuals],
            edgecolor=DARK, linewidth=0.5)
    ax3.axhline(0, color=MUTED, linewidth=1)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    style_ax(ax3, "📉  Residuals (Actual − Predicted)")
    ax3.set_xlabel("Test Month Index")
    ax3.set_ylabel("Error")

    # ── Panel 4: Model Comparison ─────────────
    ax4 = fig.add_subplot(gs[2, 1])
    model_names  = list(results.keys())
    model_rmses  = [results[n]["rmse"] for n in model_names]
    model_colors = [GREEN if n == best_name else BLUE for n in model_names]
    bars = ax4.barh(model_names, model_rmses, color=model_colors,
                    edgecolor=DARK, height=0.5)
    for bar, val in zip(bars, model_rmses):
        ax4.text(val + 200, bar.get_y() + bar.get_height() / 2,
                 f"₹{val:,.0f}", va="center", color=TEXT, fontsize=8)
    ax4.set_facecolor(PANEL)
    for sp in ax4.spines.values():
        sp.set_edgecolor(GRID)
    ax4.tick_params(colors=MUTED, labelsize=8)
    ax4.xaxis.label.set_color(MUTED)
    ax4.set_title("🏆  Model RMSE Comparison\n(lower = better)",
                  color=TEXT, fontsize=9.5, fontweight="bold", pad=8)
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}K"))
    ax4.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.5, axis="x")

    # ── Panel 5: 6-Month Forecast Table ───────
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_facecolor(PANEL)
    ax5.axis("off")
    ax5.set_title("📅  6-Month Forecast Details", color=TEXT,
                  fontsize=9.5, fontweight="bold", pad=8)

    headers = ["Month", "Forecast", "Low", "High"]
    col_x   = [0.02, 0.28, 0.57, 0.80]
    ax5.text(0.5, 0.96, "─" * 42, ha="center", color=GRID,
             fontsize=7, transform=ax5.transAxes)
    for j, h in enumerate(headers):
        ax5.text(col_x[j], 0.90, h, color=ORANGE, fontsize=8,
                 fontweight="bold", transform=ax5.transAxes)

    for i, (_, row) in enumerate(future_df.iterrows()):
        y_pos = 0.82 - i * 0.115
        month_lbl = row["date"].strftime("%b %Y")
        fc  = f"₹{row['forecast']/1000:.1f}K"
        lo  = f"₹{row['lower_band']/1000:.1f}K"
        hi  = f"₹{row['upper_band']/1000:.1f}K"
        row_color = TEXT if i % 2 == 0 else MUTED
        for j, val in enumerate([month_lbl, fc, lo, hi]):
            ax5.text(col_x[j], y_pos, val, color=row_color,
                     fontsize=8, transform=ax5.transAxes)

    # ── Main Title ────────────────────────────
    fig.suptitle("Sales & Demand Forecasting Dashboard  |  Retail Business",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.995)

    plt.savefig(filepath, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [saved] {filepath}")


# ──────────────────────────────────────────────
# 8.  BUSINESS INSIGHTS
# ──────────────────────────────────────────────
def print_business_insights(df: pd.DataFrame, future_df: pd.DataFrame,
                              best_name: str, best_result: dict):
    monthly_avg   = df["sales"].mean()
    peak_month    = df.loc[df["sales"].idxmax(), "date"].strftime("%B %Y")
    low_month     = df.loc[df["sales"].idxmin(), "date"].strftime("%B %Y")
    forecast_avg  = future_df["forecast"].mean()
    growth        = (forecast_avg - monthly_avg) / monthly_avg * 100

    print("\n" + "=" * 62)
    print("  📊 BUSINESS INSIGHTS SUMMARY")
    print("=" * 62)
    print(f"  Best Model         : {best_name}")
    print(f"  Forecast Accuracy  : {best_result['acc']:.1f}%")
    print(f"  MAE                : ₹{best_result['mae']:,.0f}")
    print(f"  RMSE               : ₹{best_result['rmse']:,.0f}")
    print(f"  R² Score           : {best_result['r2']:.3f}")
    print()
    print(f"  📅 Historical Avg Monthly Sales  : ₹{monthly_avg:,.0f}")
    print(f"  📈 Peak Sales Month              : {peak_month}")
    print(f"  📉 Lowest Sales Month            : {low_month}")
    print()
    print(f"  🔮 6-Month Forecast Avg          : ₹{forecast_avg:,.0f}")
    print(f"  📊 Projected Growth vs History   : {growth:+.1f}%")
    print(f"  🗓️  Forecast Period               : "
          f"{future_df['date'].iloc[0].strftime('%b %Y')} → "
          f"{future_df['date'].iloc[-1].strftime('%b %Y')}")
    print()
    print("  💼 BUSINESS RECOMMENDATIONS:")
    if growth > 10:
        print("  ✅ Strong growth forecast — increase inventory & staffing.")
    elif growth > 0:
        print("  ✅ Moderate growth — maintain current inventory levels.")
    else:
        print("  ⚠️  Flat/declining forecast — review pricing & promotions.")
    print("  📦 Stock up 4–6 weeks before Nov–Dec (holiday spike expected).")
    print("  💡 July–August typically show lower demand — good for clearance.")
    print("  🔁 Promotional months yield 10–15% lift — plan campaigns early.")
    print("=" * 62)


# ──────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 62)
    print("   Sales & Demand Forecasting System  |  Task 1")
    print("=" * 62)

    # Step 1 — Generate & clean
    print("\n[1/5] Generating & cleaning data …")
    df_raw   = generate_sales_data()
    df_clean = clean_data(df_raw)

    # Step 2 — Feature engineering
    print("\n[2/5] Engineering time-based features …")
    df_feat = engineer_features(df_clean)
    print(f"  Features created : {len(FEATURE_COLS)}")
    print(f"  Usable rows      : {len(df_feat)}  (after lag warmup)")

    # Step 3 — Train & evaluate all models
    print("\n[3/5] Training & evaluating models …")
    results, best_name = evaluate_models(df_feat)

    # Step 4 — 6-month forecast
    print("\n[4/5] Generating 6-month future forecast …")
    best_result = results[best_name]
    future_df   = forecast_future(df_feat, best_result, n_months=6)
    print(f"\n  {'Month':<12} {'Forecast':>12} {'Low Band':>12} {'High Band':>12}")
    print(f"  {'─'*50}")
    for _, row in future_df.iterrows():
        print(f"  {row['date'].strftime('%b %Y'):<12} "
              f"₹{row['forecast']:>10,.0f} "
              f"₹{row['lower_band']:>10,.0f} "
              f"₹{row['upper_band']:>10,.0f}")

    # Step 5 — Dashboard + insights
    print("\n[5/5] Generating business dashboard …")
    out_path = os.path.join(OUTPUT_DIR, "sales_forecast_dashboard.png")
    plot_dashboard(df_feat, results, best_name, future_df, out_path)

    print_business_insights(df_feat, future_df, best_name, best_result)
    print(f"\n  Dashboard saved to: {out_path}")


if __name__ == "__main__":
    main()