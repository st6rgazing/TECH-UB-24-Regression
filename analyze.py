from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/model-hw-matplotlib')
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
ROOT = Path(__file__).resolve().parent
SEED = 42

def outdir():
    p = ROOT / 'results'
    p.mkdir(exist_ok=True)
    return p

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + '\n')

def check(df, n):
    assert len(df) == n
    assert not df.isna().any().any(), 'Missing values require handling'

def regression_features(df):
    x = df[['Mileage_K', 'Age_Years', 'Horsepower', 'Num_Owners']].copy()
    x['Accident_yes'] = df.Accident.map({'no': 0, 'yes': 1})
    assert x.notna().all().all()
    return x.astype(float)

def car_decision(car, auction_price, model):
    x = regression_features(pd.DataFrame([car]))
    price = float(model.predict(sm.add_constant(x[model.model.exog_names[1:]], has_constant='add')).iloc[0])
    return 'buy' if price > auction_price + 1500 else 'pass'

def regression():
    out = outdir()
    df = pd.read_csv(ROOT / 'used_car_sales.csv')
    auction = pd.read_csv(ROOT / 'auction_lots.csv')
    check(df, 240)
    check(auction, 40)
    train, test = train_test_split(df, test_size=0.3, random_state=SEED)
    xtr, xte = (regression_features(train), regression_features(test))
    full = sm.OLS(train.Sale_Price, sm.add_constant(xtr)).fit()
    drop = [x for x in xtr.columns if full.pvalues[x] > 0.05]
    reduced = sm.OLS(train.Sale_Price, sm.add_constant(xtr.drop(columns=drop))).fit()
    model = reduced if drop and reduced.rsquared_adj >= full.rsquared_adj else full
    features = model.model.exog_names[1:]
    (out / 'full_model.txt').write_text(full.summary().as_text())
    (out / 'candidate_reduced_model.txt').write_text(reduced.summary().as_text())
    (out / 'v1_model.txt').write_text(model.summary().as_text())
    pd.DataFrame({'coefficient': model.params, 'p_value': model.pvalues, 'ci_lower': model.conf_int()[0], 'ci_upper': model.conf_int()[1]}).to_csv(out / 'coefficients.csv')
    average = float(train.Sale_Price.mean())
    predicted = model.predict(sm.add_constant(xte[features], has_constant='add'))
    metrics = []
    for name, pred in [('baseline', np.full(len(test), average)), ('v1', predicted)]:
        metrics.append({'model': name, 'test_r2': r2_score(test.Sale_Price, pred), 'test_rmse_dollars': root_mean_squared_error(test.Sale_Price, pred)})
    pd.DataFrame(metrics).to_csv(out / 'test_metrics.csv', index=False)
    predictions = test.copy()
    predictions['source_csv_row'] = test.index + 2
    predictions['baseline_prediction'] = average
    predictions['v1_prediction'] = predicted
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    auction['baseline_prediction'] = average
    auction['v1_prediction'] = model.predict(sm.add_constant(regression_features(auction)[features], has_constant='add'))
    auction['profit_if_bought'] = auction.Later_Sale_Price - auction.Auction_Price - 1500
    value = []
    for name in ['baseline', 'v1']:
        bought = auction[f'{name}_prediction'] > auction.Auction_Price + 1500
        auction[f'{name}_decision'] = np.where(bought, 'buy', 'pass')
        value.append({'model': name, 'cars_bought': int(bought.sum()), 'losing_cars_bought': int((bought & (auction.profit_if_bought < 0)).sum()), 'total_profit_dollars': int(auction.loc[bought, 'profit_if_bought'].sum())})
    auction.to_csv(out / 'auction_decisions.csv', index=False)
    pd.DataFrame(value).to_csv(out / 'auction_value.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(test.Sale_Price, predicted, color='black', marker='o', facecolors='none', label='v1')
    ax.axhline(average, color='black', linestyle='--', label='Baseline')
    low, high = (min(test.Sale_Price.min(), predicted.min()), max(test.Sale_Price.max(), predicted.max()))
    ax.plot([low, high], [low, high], ':', color='black', label='Perfect prediction')
    ax.set(xlabel='Actual sale price ($)', ylabel='Predicted sale price ($)', title='Held-out test cars (n=72)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'test_predictions.png', dpi=180)
    plt.close(fig)
    save_json(out / 'summary.json', {'seed': SEED, 'train_rows': len(train), 'test_rows': len(test), 'training_average_price': average, 'features_used': features, 'features_dropped': [x for x in xtr if x not in features], 'full_adjusted_r2': full.rsquared_adj, 'reduced_adjusted_r2': reduced.rsquared_adj, 'v1_adjusted_r2': model.rsquared_adj, 'test_metrics': metrics, 'auction_value': value})
    pd.DataFrame({'source_csv_row': df.index + 2, 'split': np.where(df.index.isin(train.index), 'train', 'test')}).to_csv(out / 'split.csv', index=False)
if __name__ == '__main__':
    regression()
    print('Analysis complete. See results/.')
