"""
评估与监控模块
包含：模型评估(KS/AUC/Gini/Lift/PR) + 特征监控(PSI/缺失率) + 模型监控(评分分布/指标退化)
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


# ============================================================
# 核心指标计算
# ============================================================

def calculate_ks(y_true, y_pred_proba):
    """计算KS值"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return max(tpr - fpr)


def calculate_auc(y_true, y_pred_proba):
    """计算AUC"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred_proba)


def calculate_gini(y_true, y_pred_proba):
    """计算Gini = 2*AUC - 1"""
    return 2 * calculate_auc(y_true, y_pred_proba) - 1


def calculate_lift(y_true, y_pred_proba, n_bins=10):
    """计算Lift值（按十分位）"""
    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred_proba, dtype=float)

    # 过滤NaN
    valid_mask = ~(np.isnan(y_pred_arr) | np.isnan(y_true_arr))
    y_true_arr = y_true_arr[valid_mask]
    y_pred_arr = y_pred_arr[valid_mask]

    if len(y_pred_arr) < n_bins:
        return pd.DataFrame()

    df = pd.DataFrame({'y_true': y_true_arr, 'y_pred': y_pred_arr})
    df['decile'] = pd.qcut(df['y_pred'], q=n_bins, duplicates='drop', labels=False)
    df['decile'] = n_bins - 1 - df['decile']

    overall_rate = y_true_arr.mean()
    if overall_rate == 0:
        return pd.DataFrame()

    lift_data = []
    for decile in sorted(df['decile'].unique()):
        subset = df[df['decile'] == decile]
        bad_rate = subset['y_true'].mean()
        lift = bad_rate / overall_rate
        lift_data.append({
            'decile': int(decile) + 1,
            'count': len(subset),
            'bad_count': int(subset['y_true'].sum()),
            'bad_rate': round(bad_rate, 4),
            'lift': round(lift, 2)
        })
    return pd.DataFrame(lift_data)


def calculate_pr_auc(y_true, y_pred_proba):
    """计算PR-AUC"""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_pred_proba)


def calculate_psi_score(expected, actual, bins=10):
    """计算PSI (Population Stability Index)"""
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.arange(0, bins + 1) / bins * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_bins = np.histogram(expected, bins=breakpoints)[0]
    actual_bins = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = np.clip(expected_bins / len(expected), 1e-6, None)
    actual_pct = np.clip(actual_bins / len(actual), 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(psi, 4)


# ============================================================
# 模型评估
# ============================================================

def evaluate_model(y_true, y_pred_proba, model_name='model', threshold=0.5):
    """模型全面评估

    Returns:
        dict: 包含所有评估指标
    """
    print(f"\n[Model Evaluation] {model_name}")

    auc = calculate_auc(y_true, y_pred_proba)
    ks = calculate_ks(y_true, y_pred_proba)
    gini = calculate_gini(y_true, y_pred_proba)
    pr_auc = calculate_pr_auc(y_true, y_pred_proba)
    lift_df = calculate_lift(y_true, y_pred_proba)

    top10_lift = lift_df.iloc[0]['lift'] if len(lift_df) > 0 else 0

    # 混淆矩阵
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)
    y_true_arr = np.array(y_true).astype(int)
    tp = ((y_pred == 1) & (y_true_arr == 1)).sum()
    fp = ((y_pred == 1) & (y_true_arr == 0)).sum()
    fn = ((y_pred == 0) & (y_true_arr == 1)).sum()
    tn = ((y_pred == 0) & (y_true_arr == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'model_name': model_name,
        'auc': round(auc, 4),
        'ks': round(ks, 4),
        'gini': round(gini, 4),
        'pr_auc': round(pr_auc, 4),
        'top10_lift': round(top10_lift, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'threshold': threshold,
        'n_samples': len(y_true),
        'bad_rate': round(float(np.mean(y_true_arr)), 4),
    }

    print(f"  AUC:     {auc:.4f} {'✓' if auc >= cfg.AUC_THRESHOLD else '✗'}")
    print(f"  KS:      {ks:.4f} {'✓' if ks >= cfg.KS_THRESHOLD else '✗'}")
    print(f"  Gini:    {gini:.4f} {'✓' if gini >= cfg.GINI_THRESHOLD else '✗'}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  Lift@10: {top10_lift:.2f} {'✓' if top10_lift >= cfg.LIFT_TOP10_THRESHOLD else '✗'}")

    return metrics, lift_df


def evaluate_segment_stability(df, features, y_pred_proba, target='new_target',
                               segment_col='new_org'):
    """分群稳定性评估 — 按机构/月份评估模型表现"""
    results = []

    # 将y_pred_proba对齐到df的index
    if not isinstance(y_pred_proba, pd.Series):
        y_pred_series = pd.Series(np.array(y_pred_proba), index=df.index)
    else:
        y_pred_series = y_pred_proba

    if segment_col in df.columns:
        segments = df[segment_col].unique()
        for seg in segments:
            seg_data = df[df[segment_col] == seg]
            if len(seg_data) < 50:
                continue
            y_true_seg = seg_data[target]
            y_pred_seg = y_pred_series.loc[seg_data.index].values

            try:
                from sklearn.metrics import roc_auc_score
                seg_auc = roc_auc_score(y_true_seg, y_pred_seg)
                seg_ks = calculate_ks(y_true_seg, y_pred_seg)
                results.append({
                    'segment': seg, 'n_samples': len(seg_data),
                    'bad_rate': round(float(y_true_seg.mean()), 4),
                    'auc': round(seg_auc, 4), 'ks': round(seg_ks, 4)
                })
            except Exception:
                pass

    return pd.DataFrame(results)


def generate_report(metrics_list, lift_df_list=None, output_path=None):
    """生成评估报告"""
    if output_path is None:
        output_path = cfg.MODEL_REPORT_PATH

    report = {
        'generated_at': datetime.now().isoformat(),
        'models': metrics_list,
        'thresholds': {
            'auc': cfg.AUC_THRESHOLD,
            'ks': cfg.KS_THRESHOLD,
            'gini': cfg.GINI_THRESHOLD,
            'lift_top10': cfg.LIFT_TOP10_THRESHOLD,
            'psi': cfg.PSI_MODEL_THRESHOLD,
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Report saved: {output_path}")
    return report


# ============================================================
# 监控基线管理
# ============================================================

def save_monitor_baseline(df, features, target='new_target', path=None):
    """保存监控基线（训练集统计信息）"""
    if path is None:
        path = cfg.MONITOR_BASELINE_PATH

    baseline = {
        'created_at': datetime.now().isoformat(),
        'n_samples': len(df),
        'bad_rate': float(df[target].mean()),
        'features': {}
    }

    for feat in features:
        if df[feat].dtype in ['float64', 'int64', 'int32', 'float32']:
            baseline['features'][feat] = {
                'mean': float(df[feat].mean()),
                'std': float(df[feat].std()),
                'min': float(df[feat].min()),
                'max': float(df[feat].max()),
                'median': float(df[feat].median()),
                'missing_rate': float(df[feat].isnull().mean()),
                'percentiles': {
                    'p1': float(df[feat].quantile(0.01)),
                    'p5': float(df[feat].quantile(0.05)),
                    'p25': float(df[feat].quantile(0.25)),
                    'p50': float(df[feat].quantile(0.50)),
                    'p75': float(df[feat].quantile(0.75)),
                    'p95': float(df[feat].quantile(0.95)),
                    'p99': float(df[feat].quantile(0.99)),
                }
            }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)

    print(f"  Monitor baseline saved to {path}")
    return baseline


def load_monitor_baseline(path=None):
    """加载监控基线"""
    if path is None:
        path = cfg.MONITOR_BASELINE_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# 特征监控
# ============================================================

def monitor_features(df_baseline, df_current, features, psi_threshold=None):
    """特征监控：PSI漂移 + 缺失率变化 + 分布变化"""
    if psi_threshold is None:
        psi_threshold = cfg.MONITOR_PSI_THRESHOLD

    print(f"\n[Feature Monitoring] {len(features)} features")

    alerts = []
    for feat in features:
        if feat not in df_baseline.columns or feat not in df_current.columns:
            continue

        # 跳过分类特征的PSI和均值计算
        is_cat = df_baseline[feat].dtype == object or str(df_baseline[feat].dtype) == 'category'

        # PSI（仅数值特征）
        psi_val = 0.0
        if not is_cat:
            try:
                psi_val = calculate_psi_score(
                    df_baseline[feat].dropna().values,
                    df_current[feat].dropna().values
                )
            except Exception:
                psi_val = -1

        # 缺失率变化
        base_missing = df_baseline[feat].isnull().mean()
        curr_missing = df_current[feat].isnull().mean()
        missing_change = curr_missing - base_missing

        # 均值变化（仅数值特征）
        base_mean = 0.0
        curr_mean = 0.0
        if not is_cat:
            base_mean = df_baseline[feat].mean()
            curr_mean = df_current[feat].mean()

        if psi_val > psi_threshold:
            alerts.append({
                'feature': feat, 'type': 'PSI漂移',
                'value': f'PSI={psi_val:.4f}',
                'severity': 'HIGH' if psi_val > 0.25 else 'MEDIUM'
            })

        if abs(missing_change) > cfg.MONITOR_MISSING_CHANGE:
            alerts.append({
                'feature': feat, 'type': '缺失率变化',
                'value': f'{base_missing:.4f}→{curr_missing:.4f}',
                'severity': 'MEDIUM'
            })

    print(f"  Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"    [{alert['severity']}] {alert['feature']}: {alert['type']} - {alert['value']}")

    return alerts


# ============================================================
# 模型监控
# ============================================================

def monitor_model(df_baseline, df_current, model, features,
                  target='new_target'):
    """模型监控：评分分布 + 指标退化"""
    print(f"\n[Model Monitoring]")

    # 预测
    base_proba = model.predict_proba(df_baseline)
    curr_proba = model.predict_proba(df_current)

    # 评分分布PSI
    score_psi = calculate_psi_score(base_proba, curr_proba)

    # 指标对比
    base_auc = calculate_auc(df_baseline[target], base_proba)
    curr_auc = calculate_auc(df_current[target], curr_proba)
    auc_drop = base_auc - curr_auc

    base_ks = calculate_ks(df_baseline[target], base_proba)
    curr_ks = calculate_ks(df_current[target], curr_proba)

    # 坏样本率变化
    base_bad_rate = df_baseline[target].mean()
    curr_bad_rate = df_current[target].mean()

    alerts = []
    if score_psi > cfg.PSI_MODEL_THRESHOLD:
        alerts.append({
            'type': '评分分布漂移', 'value': f'PSI={score_psi:.4f}',
            'severity': 'HIGH' if score_psi > 0.25 else 'MEDIUM'
        })
    if auc_drop > cfg.MONITOR_AUC_DROP:
        alerts.append({
            'type': 'AUC退化', 'value': f'{base_auc:.4f}→{curr_auc:.4f} (drop={auc_drop:.4f})',
            'severity': 'HIGH' if auc_drop > 0.1 else 'MEDIUM'
        })
    if abs(curr_bad_rate - base_bad_rate) > cfg.MONITOR_BAD_RATE_CHANGE:
        alerts.append({
            'type': '坏样本率变化', 'value': f'{base_bad_rate:.4f}→{curr_bad_rate:.4f}',
            'severity': 'MEDIUM'
        })

    monitor_result = {
        'score_psi': round(score_psi, 4),
        'baseline_auc': round(base_auc, 4),
        'current_auc': round(curr_auc, 4),
        'auc_drop': round(auc_drop, 4),
        'baseline_ks': round(base_ks, 4),
        'current_ks': round(curr_ks, 4),
        'baseline_bad_rate': round(base_bad_rate, 4),
        'current_bad_rate': round(curr_bad_rate, 4),
        'alerts': alerts
    }

    print(f"  Score PSI: {score_psi:.4f}")
    print(f"  AUC: {base_auc:.4f} → {curr_auc:.4f} (drop={auc_drop:.4f})")
    print(f"  KS:  {base_ks:.4f} → {curr_ks:.4f}")
    print(f"  Alerts: {len(alerts)}")

    return monitor_result


def generate_monitor_report(monitor_result, feature_alerts, output_path=None):
    """生成监控报告"""
    if output_path is None:
        output_path = cfg.MONITOR_REPORT_PATH

    report = {
        'generated_at': datetime.now().isoformat(),
        'model_monitoring': monitor_result,
        'feature_alerts': feature_alerts,
        'summary': {
            'total_feature_alerts': len(feature_alerts),
            'high_severity': sum(1 for a in feature_alerts if a.get('severity') == 'HIGH'),
            'model_alerts': len(monitor_result.get('alerts', []))
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Monitor report saved: {output_path}")
    return report
