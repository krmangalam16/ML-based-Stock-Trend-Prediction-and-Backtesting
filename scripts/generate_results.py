import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
# ensure project root is on path so "src" package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import utils, indicators, backtest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10,4)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
os.makedirs(OUT_DIR, exist_ok=True)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Parameters
TICKER = 'AAPL'
START = '2016-01-01'
END = '2026-01-01'

# Download and prepare
print('Downloading data...')
df = utils.download_data(TICKER, START, END)
df = utils.clean_data(df)

# Feature engineering (same as notebook)
df_fe = df.copy()
df_fe['Return'] = df_fe['Adj Close'].pct_change()
df_fe['LogReturn'] = np.log(df_fe['Adj Close']).diff()
df_fe['MA10'] = indicators.moving_average(df_fe['Adj Close'], 10)
df_fe['MA20'] = indicators.moving_average(df_fe['Adj Close'], 20)
df_fe['MA50'] = indicators.moving_average(df_fe['Adj Close'], 50)
df_fe['EMA12'] = df_fe['Adj Close'].ewm(span=12, adjust=False).mean()
df_fe['EMA26'] = df_fe['Adj Close'].ewm(span=26, adjust=False).mean()
df_fe['MACD'] = df_fe['EMA12'] - df_fe['EMA26']
df_fe['Volatility20'] = indicators.calculate_volatility(df_fe['Adj Close'], 20)
df_fe['RSI14'] = indicators.calculate_rsi(df_fe['Adj Close'], 14)
df_fe['PriceRange'] = (df_fe['High'] - df_fe['Low']) / df_fe['Low']
df_fe['VolumeChange'] = df_fe['Volume'].pct_change()
df_fe['Lag1'] = df_fe['Return'].shift(1)
df_fe['Lag2'] = df_fe['Return'].shift(2)
df_fe['CloseToMA10'] = df_fe['Adj Close'] / df_fe['MA10'] - 1
df_fe = df_fe.dropna()

df_fe['Target'] = (df_fe['Adj Close'].shift(-1) > df_fe['Adj Close']).astype(int)
df_fe = df_fe.dropna()

feature_cols = ['Return','LogReturn','MA10','MA20','MA50','EMA12','EMA26','MACD','Volatility20','RSI14','PriceRange','VolumeChange','Lag1','Lag2','CloseToMA10']
X, y = utils.prepare_dataset(df_fe, feature_cols, 'Target')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

models = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, kernel='rbf')
}

results = []
fitted = {}
for name, m in models.items():
    m.fit(X_train_s, y_train)
    fitted[name] = m
    preds = m.predict(X_test_s)
    probs = m.predict_proba(X_test_s)[:,1] if hasattr(m, 'predict_proba') else m.decision_function(X_test_s)
    res = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1': f1_score(y_test, preds, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, probs)
    }
    results.append(res)

results_df = pd.DataFrame(results).set_index('Model')
results_df.to_csv(os.path.join(PROJECT_ROOT, 'results.csv'))

# Small grid search for RandomForest
param_grid = {'n_estimators':[50,100],'max_depth':[3,6,None],'min_samples_split':[2,5]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1)
print('Running GridSearchCV...')
grid.fit(X_train_s, y_train)
best_rf = grid.best_estimator_

# Feature importance plot
imp = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(8,6))
imp.plot(kind='barh', ax=ax)
ax.set_title('Feature Importances (Random Forest)')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'feature_importances.png'))
plt.close(fig)

# Model comparison plot
fig, ax = plt.subplots(figsize=(10,5))
results_df[['Accuracy','Precision','Recall','F1','ROC_AUC']].plot(kind='bar', ax=ax)
ax.set_ylim(0,1)
ax.set_title('Model comparison')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'model_comparison.png'))
plt.close(fig)

# Backtest using predictions from tuned RF
preds_test = pd.Series(best_rf.predict(X_test_s), index=X_test_s.index)
positions = backtest.generate_signals(preds_test)
price_test = df_fe['Adj Close'].reindex(X_test_s.index)
bt = backtest.calculate_returns(price_test, positions, position_size=1.0, transaction_cost=0.001)
bt['CumMarket'] = (1 + bt['MarketReturn']).cumprod() - 1
bt['CumStrat'] = (1 + bt['StrategyReturn']).cumprod() - 1

# Cumulative returns plot
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(bt.index, bt['CumMarket'], label='Market')
ax.plot(bt.index, bt['CumStrat'], label='Strategy')
ax.set_title('Cumulative Returns')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'cumulative_returns.png'))
plt.close(fig)

# Performance summary
perf = backtest.performance_summary(bt)

# Write results.txt
with open(os.path.join(PROJECT_ROOT, 'results.txt'), 'w') as f:
    f.write('Model comparison (test set)\n')
    f.write(results_df.to_string())
    f.write('\n\nBest Random Forest params: %s\n' % str(grid.best_params_))
    f.write('\nBacktest performance (market and strategy):\n')
    f.write(str(perf))

print('Results saved: results.csv, results.txt, and plots in images/')
