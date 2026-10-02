#!/usr/bin/env python3
"""ESS battery project: Batch 1 EDA, group CV/Hold-out, Batch 2 final test.

Notebook model code is embedded here; no notebook or Jupyter runtime is required.
The model/features/search grids match the supplied notebook. Test is evaluated once
per complete invocation, after model selection. --train-only never reads Batch 2.
"""
from __future__ import annotations
import argparse
import sys
import importlib.metadata
from pathlib import Path
import json
import hashlib
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


RANDOM_STATE = 42
FEATURES = ['dq_min', 'log_dq_var']
TARGET_MAPE = 9.1

def dereference(f, node):
    if isinstance(node, h5py.Dataset) and h5py.check_dtype(ref=node.dtype) is not None:
        refs = np.asarray(node).reshape(-1)
        if refs.size != 1:
            raise ValueError(f'단일 reference 예상: {node.name}')
        return f[refs[0]] if refs[0] else None
    return node

def numeric_value(f, node):
    node = dereference(f, node)
    if node is None or not isinstance(node, h5py.Dataset):
        return np.array([], dtype=float)
    if node.attrs.get('MATLAB_empty', 0):
        return np.array([], dtype=float)
    return np.asarray(node, dtype=float).reshape(-1)

def matlab_text(f, node):
    node = dereference(f, node)
    if node is None:
        return ''
    if isinstance(node, h5py.Group):
        # MATLAB string containers can contain a char data field.
        for key in ['data', 'Value', 'value']:
            if key in node:
                return matlab_text(f, node[key])
        return ''
    matlab_class = node.attrs.get('MATLAB_class', b'')
    if matlab_class in [b'string', 'string']:
        # MATLAB opaque string headers are not Unicode code points.
        return ''
    arr = np.asarray(node).reshape(-1)
    if arr.dtype.kind in 'ui':
        return ''.join(chr(int(v)) for v in arr if v).strip()
    if arr.dtype.kind in 'SU':
        return ''.join(v.decode() if isinstance(v, bytes) else str(v) for v in arr).strip()
    return ''

def cell_node(f, batch_group, field, index):
    refs = np.asarray(batch_group[field]).reshape(-1)
    return f[refs[index]] if refs[index] else None

def load_early_records(path, batch_name):
    rows, excluded = [], []
    with h5py.File(path, 'r') as f:
        b = f['batch']
        n_cells = np.asarray(b['cycle_life']).size
        for i in range(n_cells):
            uid = f'{batch_name}:cell_{i}'
            life_arr = numeric_value(f, cell_node(f, b, 'cycle_life', i))
            if life_arr.size != 1 or not np.isfinite(life_arr[0]) or life_arr[0] <= 0:
                excluded.append({'uid': uid, 'reason': '수명 누락/무효'})
                continue
            life = float(life_arr[0])
            if life <= 100:
                excluded.append({'uid': uid, 'reason': '100사이클 예측 대상 이전 EOL'})
                continue
            policy = ''
            for key in ['policy_readable', 'policy']:
                if key in b:
                    policy = matlab_text(f, cell_node(f, b, key, i))
                    if policy:
                        break
            if not policy:
                raise ValueError(f'{uid}: 프로토콜 식별 불가 — 누수 방지 그룹 분할 불가능')
            summary = cell_node(f, b, 'summary', i)
            cycle_numbers = numeric_value(f, summary['cycle'])
            indices = [np.flatnonzero(cycle_numbers == n) for n in [10, 100]]
            if any(len(idx) != 1 for idx in indices):
                excluded.append({'uid': uid, 'reason': '실제 사이클 10 또는 100 없음'})
                continue
            v = np.array([], dtype=float)
            dq = np.array([], dtype=float)
            feature_issue = ''
            try:
                cycle_group = cell_node(f, b, 'cycles', i)
                q_refs = np.asarray(cycle_group['Qdlin']).reshape(-1)
                if len(q_refs) != len(cycle_numbers):
                    raise ValueError('summary와 cycles 길이 불일치')
                q10 = numeric_value(f, f[q_refs[int(indices[0][0])]])
                q100 = numeric_value(f, f[q_refs[int(indices[1][0])]])
                v = numeric_value(f, cell_node(f, b, 'Vdlin', i))
                if not (len(v) == len(q10) == len(q100)) or len(v) < 2:
                    raise ValueError('Qdlin/Vdlin 길이 불일치 또는 결측')
                finite = np.isfinite(v) & np.isfinite(q10) & np.isfinite(q100)
                v, dq = v[finite], (q100 - q10)[finite]
                order = np.argsort(v)
                v, dq = v[order], dq[order]
                if len(v) < 2 or np.any(np.diff(v) <= 0):
                    raise ValueError('전압점 부족 또는 중복')
            except (ValueError, KeyError, TypeError, IndexError) as error:
                v, dq = np.array([]), np.array([])
                feature_issue = str(error)
            barcode = matlab_text(f, cell_node(f, b, 'barcode', i)) if 'barcode' in b else ''
            rows.append({'uid': uid, 'batch_id': batch_name, 'cell_id': i,
                         'charging_policy': policy.strip().lower(), 'barcode': barcode,
                         'cycle_life': life, 'voltage': v, 'delta_q': dq,
                         'feature_issue': feature_issue})
    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=['uid', 'reason'])


class DeltaQFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_points=1000, epsilon=1e-12):
        self.n_points = n_points
        self.epsilon = epsilon

    def fit(self, X, y=None):
        usable = [v for v in X['voltage'] if len(v) >= 2]
        if not usable:
            raise ValueError('train에 유효한 초기 곡선이 없습니다.')
        lo = max(float(v[0]) for v in usable)
        hi = min(float(v[-1]) for v in usable)
        if lo >= hi:
            raise ValueError('train의 공통 전압 구간이 없습니다.')
        self.voltage_grid_ = np.linspace(lo, hi, self.n_points)
        return self

    def transform(self, X):
        rows = []
        lo, hi = self.voltage_grid_[[0, -1]]
        for v, dq in zip(X['voltage'], X['delta_q']):
            if len(v) < 2 or v[0] > lo + 1e-9 or v[-1] < hi - 1e-9:
                rows.append([np.nan, np.nan])
                continue
            values = np.interp(self.voltage_grid_, v, dq)
            rows.append([values.min(), np.log10(values.var(ddof=0) + self.epsilon)])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)

def make_pipeline(model):
    return Pipeline([
        ('features', DeltaQFeatures()),
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scaler', StandardScaler()),
        ('model', model),
    ])


def group_cv(groups, max_splits=3):
    n = min(max_splits, len(np.unique(groups)))
    if n < 2:
        raise ValueError('group CV에 필요한 프로토콜이 부족합니다.')
    return GroupKFold(n_splits=n)

def tune_model(pipe, params, X, y, groups):
    search = GridSearchCV(
        pipe, params, scoring='neg_mean_absolute_percentage_error',
        cv=group_cv(groups), refit=True, n_jobs=1, error_score='raise')
    search.fit(X, y, groups=groups)
    return search

def mape_percent(y_true, y_pred):
    return 100 * mean_absolute_percentage_error(y_true, y_pred)


def load_summary_table(path):
    """노트북의 summary EDA를 재현하며 전체 raw 시계열은 로드하지 않습니다."""
    tables = []
    fields = {'QDischarge': 'QD', 'QCharge': 'QC', 'IR': 'IR',
              'Tmax': 'Tmax', 'Tavg': 'Tavg', 'Tmin': 'Tmin',
              'chargetime': 'chargetime'}
    with h5py.File(path, 'r') as f:
        b = f['batch']
        for i in range(np.asarray(b['cycle_life']).size):
            s = cell_node(f, b, 'summary', i)
            cycles = numeric_value(f, s['cycle'])
            life_values = numeric_value(f, cell_node(f, b, 'cycle_life', i))
            life = float(life_values[0]) if len(life_values) == 1 else np.nan
            policy = matlab_text(f, cell_node(f, b, 'policy_readable', i))
            if not policy:
                policy = matlab_text(f, cell_node(f, b, 'policy', i))
            table = pd.DataFrame({'cell_id': i, 'cycle': cycles,
                                  'cycle_life': life, 'charging_policy': policy})
            for field, name in fields.items():
                values = numeric_value(f, s[field])
                if len(values) != len(cycles):
                    raise ValueError(f'cell {i}: summary.{field} 길이 불일치')
                table[name] = values
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def run_batch1_eda(path, raw, fitted_features, output_dir):
    """EDA는 모델 선택 후 생성하며 모델 전처리/선택에 다시 반영하지 않습니다."""
    eda_dir = output_dir / 'eda'
    eda_dir.mkdir(exist_ok=True)
    summary = load_summary_table(path)
    summary.describe().to_csv(eda_dir / 'summary_describe.csv')
    summary.isna().sum().rename('missing_count').to_csv(eda_dir / 'summary_missing.csv')
    cells = summary.drop_duplicates('cell_id')[['cell_id', 'cycle_life', 'charging_policy']]
    life = cells['cycle_life'].dropna()
    ratio = pd.Series({'Short (<500)': (life < 500).sum(),
                       'Middle (500-1000)': life.between(500, 1000).sum(),
                       'Long (>1000)': (life > 1000).sum()}).to_frame('n_cells')
    ratio['ratio_pct'] = ratio['n_cells'] / len(life) * 100
    ratio.to_csv(eda_dir / 'cycle_life_ratios.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(life, bins=np.linspace(150, 2300, 44), edgecolor='white')
    axes[0].axvline(life.mean(), color='tomato', linestyle='--')
    axes[0].set(xlim=(150, 2300), xlabel='Cycle Life', ylabel='Number of Cells')
    axes[1].boxplot(life.to_numpy())
    axes[1].set(ylabel='Cycle Life', title='Batch 1')
    fig.tight_layout()
    fig.savefig(eda_dir / 'cycle_life_distribution.png', dpi=150)
    plt.close(fig)

    # 곡선 비교도 train에서 fit한 전압 격자만 사용합니다.
    feature_values = fitted_features.transform(raw[['voltage', 'delta_q']])
    features = raw[['uid', 'cell_id', 'cycle_life']].copy()
    features[['dq_min', 'log_dq_var']] = feature_values
    grid = fitted_features.voltage_grid_
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    descriptive = []
    for v, dq in zip(raw['voltage'], raw['delta_q']):
        if len(v) < 2 or v[0] > grid[0] + 1e-9 or v[-1] < grid[-1] - 1e-9:
            descriptive.append([np.nan] * 4)
            continue
        values = np.interp(grid, v, dq)
        descriptive.append([values.mean(), values.var(ddof=0),
                            integrate(np.abs(values), grid), grid[values.argmin()]])
    extra = ['dq_mean', 'dq_var', 'dq_abs_area', 'dq_min_voltage']
    features[extra] = descriptive
    features['life_group'] = np.where(features['cycle_life'] < 500, 'Short',
        np.where(features['cycle_life'] > 1000, 'Long', 'Middle'))
    features.to_csv(eda_dir / 'delta_q_features.csv', index=False)
    features[['cycle_life', 'dq_min', 'log_dq_var'] + extra].corr().to_csv(
        eda_dir / 'delta_q_correlations.csv')
    features.groupby('life_group')[['dq_min', 'log_dq_var'] + extra].median().to_csv(
        eda_dir / 'delta_q_group_medians.csv')
    pd.DataFrame({method: features[['dq_min', 'log_dq_var'] + extra].corrwith(
        features['cycle_life'], method=method) for method in ['pearson', 'spearman']}).to_csv(
        eda_dir / 'delta_q_life_correlations.csv')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for _, row in raw.iterrows():
        v, dq = row['voltage'], row['delta_q']
        if len(v) >= 2:
            axes[0].plot(v, dq, alpha=0.3, linewidth=0.8)
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set(xlabel='Voltage (V)', ylabel='Q100 - Q10 (Ah)', title='Batch 1 Delta Q(V)')
    axes[1].scatter(features['cell_id'], features['dq_min'], label='Minimum Delta Q')
    axes[1].set(xlabel='Cell ID', ylabel='Delta Q (Ah)')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(eda_dir / 'delta_q_features.png', dpi=150)
    plt.close(fig)
    print(f'Batch 1 summary shape: {summary.shape}; EDA 저장: {eda_dir}')


def display(*objects):
    """IPython 없이 터미널에서 표를 표시합니다."""
    for obj in objects:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            print(obj.to_string(max_rows=25))
        else:
            print(obj)
        print()


def main(args):
    RANDOM_STATE = 42
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    MODEL_DATA_DIR = args.data_dir.expanduser().resolve()
    BATCH1_FILE = MODEL_DATA_DIR / '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
    BATCH2_FILE = MODEL_DATA_DIR / '2018-02-20_batchdata_updated_struct_errorcorrect.mat'
    MODEL_OUTPUT_DIR = args.output_dir.expanduser().resolve()
    if not BATCH1_FILE.is_file():
        raise FileNotFoundError(f'Batch 1 파일이 없습니다: {BATCH1_FILE}. --data-dir을 확인하세요.')
    if not args.train_only and not BATCH2_FILE.is_file():
        raise FileNotFoundError(f'Batch 2 파일이 없습니다: {BATCH2_FILE}. --data-dir을 확인하세요.')
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # Batch 2는 여기서 열지 않습니다.
    batch1_raw, batch1_excluded = load_early_records(BATCH1_FILE, 'Batch 1')
    assert not batch1_raw.empty
    display(batch1_excluded)
    display(batch1_raw[['uid', 'charging_policy', 'cycle_life', 'feature_issue']])

    # 특징 처리보다 먼저 셀/프로토콜 그룹 분할
    holdout_splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, valid_idx = next(holdout_splitter.split(
        batch1_raw, groups=batch1_raw['charging_policy']))
    train_raw = batch1_raw.iloc[train_idx].reset_index(drop=True)
    valid_raw = batch1_raw.iloc[valid_idx].reset_index(drop=True)

    assert set(train_raw['uid']).isdisjoint(valid_raw['uid'])
    assert set(train_raw['charging_policy']).isdisjoint(valid_raw['charging_policy'])
    def known_barcodes(frame):
        return {v for v in frame['barcode'] if v and v.lower() not in ['unknown', 'none', 'nan']}
    assert known_barcodes(train_raw).isdisjoint(known_barcodes(valid_raw)), '동일 물리 셀의 중복 분할을 확인하세요.'
    if train_raw['charging_policy'].nunique() < 4:
        raise ValueError('Nested group CV에 필요한 프로토콜 수가 부족합니다.')

    split_manifest = pd.concat([
        train_raw.assign(split='Train'), valid_raw.assign(split='Valid')
    ])[['uid', 'charging_policy', 'barcode', 'split']]
    split_manifest.to_csv(MODEL_OUTPUT_DIR / 'batch1_split_manifest.csv', index=False)
    display(split_manifest.groupby('split').agg(
        n_cells=('uid', 'size'), n_protocols=('charging_policy', 'nunique')))

    # 모델 X에는 초기 곡선만 포함합니다. 타깃/ID/프로토콜은 제외합니다.
    X_train = train_raw[['voltage', 'delta_q']].copy()
    y_train = train_raw['cycle_life'].to_numpy()
    X_valid = valid_raw[['voltage', 'delta_q']].copy()
    y_valid = valid_raw['cycle_life'].to_numpy()
    train_groups = train_raw['charging_policy'].to_numpy()

    candidate_specs = {
        'Dummy': (DummyRegressor(strategy='median'), {}),
        'Ridge': (Ridge(), {'model__alpha': [0.01, 0.1, 1, 10, 100]}),
        'ElasticNet': (ElasticNet(max_iter=30000, random_state=RANDOM_STATE), {
            'model__alpha': [0.1, 1, 10], 'model__l1_ratio': [0.2, 0.8]}),
        'RBF-SVR': (SVR(kernel='rbf'), {
            'model__C': [100, 1000], 'model__gamma': ['scale', 0.1],
            'model__epsilon': [10, 50]}),
        'RandomForest': (RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1), {
            'model__max_depth': [2, 4], 'model__min_samples_leaf': [2, 4]}),
        'GradientBoosting': (GradientBoostingRegressor(random_state=RANDOM_STATE), {
            'model__n_estimators': [50, 100], 'model__learning_rate': [0.03, 0.1],
            'model__max_depth': [1, 2], 'model__min_samples_leaf': [3]}),
    }


    # Nested CV: 바깥 fold로 평가하고 안쪽 fold에서만 하이퍼파라미터를 선택합니다.
    cv_rows, tuning_rows, fitted_candidates = [], [], {}
    outer_splits = list(group_cv(train_groups).split(X_train, y_train, train_groups))

    for name, (model, params) in candidate_specs.items():
        print(f'Nested CV: {name}', flush=True)
        for fold, (fit_idx, eval_idx) in enumerate(outer_splits, start=1):
            assert set(train_groups[fit_idx]).isdisjoint(train_groups[eval_idx])
            search = tune_model(make_pipeline(clone(model)), params,
                                X_train.iloc[fit_idx], y_train[fit_idx], train_groups[fit_idx])
            pred = search.predict(X_train.iloc[eval_idx])
            cv_rows.append({'model': name, 'fold': fold,
                            'MAPE_pct': mape_percent(y_train[eval_idx], pred),
                            'n_fit': len(fit_idx), 'n_eval': len(eval_idx)})
    # 고정된 Train에만 fit. Valid/Test는 아직 사용하지 않음.
        final_search = tune_model(make_pipeline(clone(model)), params,
                                  X_train, y_train, train_groups)
        fitted_candidates[name] = final_search.best_estimator_
        tuning_rows.append({'model': name, 'best_params': final_search.best_params_})

    cv_detail = pd.DataFrame(cv_rows)
    cv_summary = cv_detail.groupby('model')['MAPE_pct'].agg(
        Train_CV_MAPE='mean', CV_std='std').sort_values('Train_CV_MAPE')
    winner = cv_summary.index[0]  # CV 결과만으로 선택하고 이후 변경하지 않습니다.
    selected_model = fitted_candidates[winner]
    display(cv_summary.round(3))
    display(pd.DataFrame(tuning_rows))
    print('CV만으로 확정한 최종 모델:', winner)

    selection_lock = {
        'winner': winner, 'random_state': RANDOM_STATE, 'features': FEATURES,
        'train_uids': train_raw['uid'].tolist(), 'valid_uids': valid_raw['uid'].tolist(),
        'cv_mean_mape': float(cv_summary.loc[winner, 'Train_CV_MAPE']),
        'best_params': next(r['best_params'] for r in tuning_rows if r['model'] == winner),
        'target_reference_mape': TARGET_MAPE,
    }
    (MODEL_OUTPUT_DIR / 'selection_lock.json').write_text(
        json.dumps(selection_lock, ensure_ascii=False, indent=2), encoding='utf-8')
    cv_detail.to_csv(MODEL_OUTPUT_DIR / 'batch1_nested_cv.csv', index=False)

    selection_signature = hashlib.sha256(
        json.dumps(selection_lock, sort_keys=True).encode()).hexdigest()

    # Valid는 모델 선택에 사용하지 않고 고정된 모델의 독립적인 평가에 사용
    valid_scores = {}
    valid_predictions = {}
    for name, model in fitted_candidates.items():
        pred = model.predict(X_valid)
        valid_scores[name] = mape_percent(y_valid, pred)
        valid_predictions[name] = pred

    candidate_performance = cv_summary.copy()
    candidate_performance['Valid_Holdout_MAPE'] = pd.Series(valid_scores)
    candidate_performance['Gap_Train_Valid_pp'] = (
        candidate_performance['Train_CV_MAPE'] - candidate_performance['Valid_Holdout_MAPE'])
    display(candidate_performance.round(3))
    candidate_performance.to_csv(MODEL_OUTPUT_DIR / 'batch1_candidate_performance.csv')

    final_feature_audit = pd.DataFrame(
        selected_model.named_steps['features'].transform(X_train), columns=FEATURES)
    final_feature_audit.insert(0, 'uid', train_raw['uid'].to_numpy())
    display(final_feature_audit)
    final_feature_audit.to_csv(MODEL_OUTPUT_DIR / 'train_feature_audit.csv', index=False)

    batch1_excluded.to_csv(MODEL_OUTPUT_DIR / 'batch1_exclusions.csv', index=False)
    if not args.skip_eda:
        run_batch1_eda(BATCH1_FILE, batch1_raw,
                      selected_model.named_steps['features'], MODEL_OUTPUT_DIR)
    if args.train_only:
        print('Train/Valid 검증 완료. --train-only에 따라 Batch 2를 읽거나 평가하지 않았습니다.')
        return

    test_raw, test_excluded = load_early_records(BATCH2_FILE, 'Batch 2')
    if test_raw.empty:
        raise ValueError('Batch 2에 평가할 셀이 없습니다.')
    overlap = known_barcodes(batch1_raw) & known_barcodes(test_raw)
    if overlap:
        raise ValueError(f'Batch 간 동일 물리 셀 barcode 중복: {overlap}')
    def curve_fingerprint(row):
        if len(row['voltage']) < 2:
            return None
        return hashlib.sha256(
            np.asarray(row['voltage'], dtype='<f8').tobytes()
            + np.asarray(row['delta_q'], dtype='<f8').tobytes()).hexdigest()
    fingerprints1 = {curve_fingerprint(r) for _, r in batch1_raw.iterrows()} - {None}
    fingerprints2 = {curve_fingerprint(r) for _, r in test_raw.iterrows()} - {None}
    if fingerprints1 & fingerprints2:
        raise ValueError('Batch 간 완전히 일치하는 초기 곡선이 있습니다. 중복 셀을 확인하세요.')
    X_test = test_raw[['voltage', 'delta_q']].copy()
    y_test = test_raw['cycle_life'].to_numpy()
    test_pred = selected_model.predict(X_test)  # 최종 Test 예측은 한 번만 수행합니다.
    test_mape = mape_percent(y_test, test_pred)
    train_mape = float(cv_summary.loc[winner, 'Train_CV_MAPE'])
    valid_mape = valid_scores[winner]
    performance = pd.DataFrame({
        'Index': ['Train (Batch 1 CV)', 'Valid (Batch 1 Hold-out)', 'Test (Batch 2)',
                  'Gap (Train-Valid)', 'Gap (Valid-Test)', 'Gap (Target-Test)'],
        'Value': [train_mape, valid_mape, test_mape,
                  train_mape - valid_mape, valid_mape - test_mape, TARGET_MAPE - test_mape],
        'Unit': ['MAPE (%)'] * 3 + ['percentage points'] * 3,
    }).set_index('Index')
    predictions = test_raw[['uid', 'charging_policy', 'cycle_life', 'feature_issue']].copy()
    predictions['predicted_cycle_life'] = test_pred
    predictions['APE_pct'] = 100 * np.abs(test_pred - y_test) / y_test
    test_features = pd.DataFrame(
        selected_model.named_steps['features'].transform(X_test), columns=FEATURES)
    predictions['features_missing_before_imputation'] = test_features.isna().any(axis=1).to_numpy()
    performance.to_csv(MODEL_OUTPUT_DIR / 'selected_model_performance.csv')
    predictions.to_csv(MODEL_OUTPUT_DIR / 'batch2_predictions.csv', index=False)
    test_excluded.to_csv(MODEL_OUTPUT_DIR / 'batch2_exclusions.csv', index=False)
    _ess_final_test_cache = {'winner': winner, 'performance': performance,
        'predictions': predictions, 'excluded': test_excluded,
        'n_train': len(train_raw), 'n_valid': len(valid_raw), 'n_test': len(test_raw),
        'selection_signature': selection_signature}

    print('선택 모델:', _ess_final_test_cache['winner'])
    print('Train/Valid/Test 셀 수:',
          [_ess_final_test_cache[k] for k in ['n_train', 'n_valid', 'n_test']])
    display(_ess_final_test_cache['performance'].round(3))
    display(_ess_final_test_cache['excluded'])
    display(_ess_final_test_cache['predictions'].round(3))
    predictions = _ess_final_test_cache['predictions']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(predictions['cycle_life'], predictions['predicted_cycle_life'])
    lo = min(predictions['cycle_life'].min(), predictions['predicted_cycle_life'].min())
    hi = max(predictions['cycle_life'].max(), predictions['predicted_cycle_life'].max())
    axes[0].plot([lo, hi], [lo, hi], '--', color='gray')
    axes[0].set(xlabel='Actual Cycle Life', ylabel='Predicted Cycle Life', title='Batch 2 Final Test')
    axes[1].scatter(predictions['cycle_life'], predictions['APE_pct'])
    axes[1].set(xlabel='Actual Cycle Life', ylabel='Absolute Percentage Error (%)', title='Error by Lifetime')
    plt.tight_layout()
    fig.savefig(MODEL_OUTPUT_DIR / 'batch2_final_test.png', dpi=150)
    plt.close(fig)

    manifest = {'python': sys.version, 'packages': {
        name: importlib.metadata.version(name)
        for name in ['numpy', 'pandas', 'matplotlib', 'h5py', 'scipy', 'scikit-learn']},
        'batch1_file': str(BATCH1_FILE), 'batch2_file': str(BATCH2_FILE),
        'features': FEATURES, 'random_state': RANDOM_STATE}
    (MODEL_OUTPUT_DIR / 'run_environment.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'전체 실행 완료. 결과: {MODEL_OUTPUT_DIR}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'Data',
                        help='원본 .mat 파일 폴더 (명시적으로 지정하는 것을 권장)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'model_outputs',
                        help='CSV/그래프 출력 폴더')
    parser.add_argument('--skip-eda', action='store_true', help='Batch 1 EDA 출력 생략')
    parser.add_argument('--train-only', action='store_true',
                        help='Batch 1 EDA/Train/Valid만 실행, Batch 2는 읽지 않음')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
