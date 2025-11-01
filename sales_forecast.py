# --- Imports & setup ---
import numpy as np                     # numerical operations (arrays, math)
import pandas as pd                    # data wrangling (DataFrame)
import matplotlib.pyplot as plt        # plotting
import seaborn as sns                  # optional styling (not required)
import xgboost as xgb                  # gradient boosting model

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error  # evaluation
# (No shuffle-based split; we'll do a time-aware split manually below.)

# --- Load data ---
file_path = "train.csv"                                                    # CSV path (relative)
data = pd.read_csv(file_path)                                             # read the dataset into a DataFrame

# --- Parse dates & basic cleaning (leak-safe prep) ---
data["Order Date"] = pd.to_datetime(                                      # convert to datetime, robust to day-first
    data["Order Date"], dayfirst=True, errors="coerce"
)
data = data.dropna(subset=["Order Date", "Sales"]).sort_values("Order Date")  # keep valid rows and sort by time

# --- Optional: group to daily level (protects against duplicates in a day) ---
daily = (                                                                  # aggregate to daily total sales
    data.groupby("Order Date", as_index=False)["Sales"].sum()
    .rename(columns={"Order Date": "ds", "Sales": "y"})                    # standard names: ds (date), y (target)
)

# --- Quick EDA: sales trend (distinctive but simple) ---
plt.figure(figsize=(12, 5))                                               # set plot size
plt.plot(daily["ds"], daily["y"], label="Daily Sales")                    # line plot of daily sales
plt.title("Daily Sales Trend")                                            # chart title
plt.xlabel("Date"); plt.ylabel("Sales")                                   # axis labels
plt.grid(True); plt.legend(); plt.tight_layout()                           # cosmetics
plt.show()                                                                 # render

# --- Feature engineering (small, unique, leak-safe) ---
def build_features(df: pd.DataFrame, lags=(1, 7, 14, 28), windows=(7, 14)):
    """
    Create lag features and leak-safe rolling statistics,
    plus simple calendar seasonality encodings (sin/cos).
    """
    out = df.copy()                                                        # work on a copy
    # Calendar features
    out["dow"] = out["ds"].dt.dayofweek                                   # day of week [0..6]
    out["month"] = out["ds"].dt.month                                     # month [1..12]
    # Cyclical encodings to capture seasonality smoothly
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["mon_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["mon_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    # Lag features (classic time-series signal)
    for L in lags:
        out[f"lag_{L}"] = out["y"].shift(L)                                # value L days ago

    # Rolling stats — shift(1) to avoid leaking current day's target
    for w in windows:
        out[f"rmean_{w}"] = out["y"].shift(1).rolling(w).mean()            # rolling mean
        out[f"rstd_{w}"]  = out["y"].shift(1).rolling(w).std()             # rolling std
        out[f"ewm_{w}"]   = out["y"].shift(1).ewm(span=w, adjust=False).mean()  # exponential mean

    # Growth rate week-over-week (safe: based on past values)
    out["grow_7"] = out["y"].pct_change(7)

    return out

feat = build_features(daily).dropna()                                      # drop rows made NaN by lags/rolls

# --- Train/test split (time-aware) ---
split_idx = int(len(feat) * 0.8)                                           # 80% train, 20% test by time
train = feat.iloc[:split_idx]                                              # historical window for training
test  = feat.iloc[split_idx:]                                              # future window for evaluation

# --- Model matrices (features/target) ---
feature_cols = [c for c in feat.columns if c not in {"ds", "y"}]           # all engineered features
X_train = train[feature_cols].astype(np.float32)                            # ensure numeric dtype
y_train = train["y"].astype(float)                                          # target as float
X_test  = test[feature_cols].astype(np.float32)                             # test features
y_test  = test["y"].astype(float)                                           # test target

# --- Baseline for honest comparison (seasonal naive using lag_7) ---
naive_cols = [c for c in feature_cols if c.startswith("lag_")]              # available lags
y_pred_naive7 = test["lag_7"].to_numpy() if "lag_7" in naive_cols else test["lag_1"].to_numpy()

# --- XGBoost regressor (slightly tuned, reproducible) ---
model_xgb = xgb.XGBRegressor(                                              # gradient boosted trees
    objective="reg:squarederror",                                          # regression loss
    n_estimators=400,                                                      # more trees than default
    learning_rate=0.05,                                                    # smaller learning rate
    max_depth=6,                                                           # allow modest interactions
    subsample=0.9, colsample_bytree=0.9,                                   # regularization via sampling
    random_state=42                                                        # reproducibility
)
model_xgb.fit(X_train, y_train)                                            # fit on training window

# --- Forecast & metrics ---
pred = model_xgb.predict(X_test)                                           # model predictions on future window
rmse = np.sqrt(mean_squared_error(y_test, pred))                           # RMSE for the model
rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive7))            # RMSE for the naive baseline
impr = 100.0 * (1 - rmse / rmse_naive) if rmse_naive > 0 else np.nan       # relative improvement vs baseline

print(f"Model RMSE:  {rmse:.2f}")                                          # report model error
print(f"Naive RMSE:  {rmse_naive:.2f}")                                    # report baseline error
print(f"Lift vs Naive-7: {impr:.1f}%")                                     # how much better than seasonal naive

# --- Save an artifact (nice for portfolio/repro) ---
out = test[["ds"]].copy()                                                  # start with dates
out["y_true"] = y_test.to_numpy()                                          # actual sales
out["y_hat"]  = pred                                                       # model forecast
out["y_naive7"] = y_pred_naive7                                            # seasonal naive forecast
out.to_csv("forecast_eval.csv", index=False)                               # write predictions for later review

# --- Plot: actual vs. model vs. naive (clear differentiator) ---
plt.figure(figsize=(12, 5))                                                # set figure size
plt.plot(out["ds"], out["y_true"], label="Actual", linewidth=2)            # actual sales (bold)
plt.plot(out["ds"], out["y_hat"], label="Forecast (xgboost)", alpha=0.9)             # model forecast
plt.plot(out["ds"], out["y_naive7"], label="Naive (lag_7)", alpha=0.6)     # naive seasonal forecast
plt.title("Sales Forecast: Model vs Naive (Holdout)")                      # title
plt.xlabel("Date"); plt.ylabel("Sales")                                    # axis labels
plt.legend(); plt.grid(True); plt.tight_layout()                            # cosmetics
plt.show()                                                                  # render
