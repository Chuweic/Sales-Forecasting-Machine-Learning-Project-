<h1>Sales Forecasting (XGBoost) — Leak-Safe TS + Seasonal Baseline</h1>

<p>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue.svg"></a>
  <a href="#"><img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-GBDT-green"></a>
  <a href="#"><img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-metrics-orange"></a>
</p>

<p>
  Forecast daily sales using <strong>XGBoost</strong> with <strong>leak-safe features</strong> and a clear
  <strong>Naive (lag_7)</strong> baseline for honest lift. The script also saves a CSV artifact for analysis/plots.
</p>

<hr/>

<h2 id="toc">Table of Contents</h2>
<ul>
  <li><a href="#features">Features</a></li>
  <li><a href="#data">Data</a></li>
  <li><a href="#setup">Setup</a></li>
  <li><a href="#run">Run</a></li>
  <li><a href="#outputs">Outputs</a></li>
  <li><a href="#customize">Customize</a></li>
  <li><a href="#naive">What is “Naive (lag_7)”?</a></li>
  <li><a href="#structure">Suggested Repo Structure</a></li>
  <li><a href="#troubleshooting">Troubleshooting</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="features">✨ Features</h2>
<ul>
  <li><strong>Leak-safe feature engineering:</strong> lags (1/7/14/28), rolling mean/std/EMA (shifted), cyclical calendar encodings (DoW/Month as sin/cos), week-over-week growth.</li>
  <li><strong>Baseline to beat:</strong> <strong>Naive (lag_7)</strong> → predict <code>&#x005E;y<sub>t</sub> = y<sub>t-7</sub></code>.</li>
  <li><strong>Evaluation:</strong> RMSE + % lift vs. Naive (lag_7).</li>
  <li><strong>Artifacts:</strong> <code>forecast_eval.csv</code> with timestamps, actuals, <em>Forecast (xgboost)</em>, and naive forecast.</li>
  <li><strong>Plot:</strong> Actual vs <em>Forecast (xgboost)</em> vs <em>Naive (lag_7)</em>.</li>
</ul>

<h2 id="data">📁 Data</h2>
<p>Provide a CSV named <code>train.csv</code> with:</p>
<ul>
  <li><code>Order Date</code> — date of the order (day-first formats supported)</li>
  <li><code>Sales</code> — numeric target</li>
</ul>
<p><em>If names differ, update <code>DATE_COL</code> / <code>TARGET_COL</code> in the script.</em></p>

<h2 id="setup">⚙️ Setup</h2>
<p><strong>macOS / Linux</strong></p>
<pre><code class="language-bash">python3 -m venv .venv &amp;&amp; source .venv/bin/activate
python -m pip install -U pip
python -m pip install xgboost scikit-learn pandas numpy matplotlib seaborn
</code></pre>

<p><strong>Windows (PowerShell)</strong></p>
<pre><code class="language-powershell">py -m venv .venv; .\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install xgboost scikit-learn pandas numpy matplotlib seaborn
</code></pre>

<p><strong>macOS OpenMP tip for XGBoost errors:</strong> <code>brew install libomp</code></p>

<h2 id="run">🚀 Run</h2>
<pre><code class="language-bash">python sales_forecast.py
</code></pre>

<h2 id="outputs">📊 Outputs</h2>
<p><strong>Example console output</strong></p>
<pre><code>Model RMSE:  1124.11
Naive RMSE:  2856.73
Lift vs Naive-7: 60.6%
</code></pre>

<ul>
  <li><code>forecast_eval.csv</code> — predictions + actuals + naive baseline</li>
  <li>A line chart with <strong>Actual</strong>, <strong>Forecast (xgboost)</strong>, <strong>Naive (lag_7)</strong></li>
</ul>

<h2 id="customize">🔧 Customize</h2>
<ul>
  <li><strong>Train/test split:</strong> <code>split_idx = int(len(feat) * 0.8)</code></li>
  <li><strong>Lags/Windows:</strong> in <code>build_features(lags=(1,7,14,28), windows=(7,14))</code></li>
  <li><strong>Model params:</strong> tweak <code>n_estimators</code>, <code>max_depth</code>, <code>learning_rate</code> in <code>XGBRegressor</code></li>
  <li><strong>Different frequency:</strong> swap <code>lag_7</code> for <code>lag_12</code> (monthly) or <code>lag_24</code> (hourly), etc.</li>
</ul>

<h2 id="naive">🧠 What is “Naive (lag_7)”?</h2>
<p><strong>Definition:</strong> Predict today’s value as exactly what it was <strong>7 days ago</strong>.
It captures weekly seasonality and is a strong baseline; your model should beat it to justify complexity.</p>

<h2 id="structure">🗂 Suggested Repo Structure</h2>
<pre><code>.
├── README.md
├── sales_forecast.py
├── train.csv                # your data (omit from public repos if private)
└── forecast_eval.csv        # generated after running
</code></pre>

<h2 id="troubleshooting">🧯 Troubleshooting</h2>
<ul>
  <li><strong>ModuleNotFoundError</strong>: Activate the venv and reinstall (<code>pip install ...</code>).</li>
  <li><strong>macOS XGBoost/OpenMP</strong>: <code>brew install libomp</code>, then reinstall <code>xgboost</code>.</li>
  <li><strong>Metrics errors</strong>: ensure <code>y_test</code>/<code>pred</code> are 1-D numeric arrays (the provided script handles this).</li>
</ul>


