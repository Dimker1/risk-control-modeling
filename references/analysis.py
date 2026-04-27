"""
特征筛选与分析模块
包含：异常月份过滤、缺失率筛选、IV筛选(含机构维度)、PSI筛选(含机构维度)、
      Null Importance去噪、相关性筛选、特征筛选主流程
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg
from references.func import (
    missing_check, calculate_iv, calculate_iv_by_org,
    calculate_psi_by_org, calculate_psi,
    iv_distribution_by_org, psi_distribution_by_org,
    value_ratio_distribution_by_org, export_report_xlsx
)


# ============================================================
# Step 1: 异常月份过滤
# ============================================================

def filter_abnormal_months(data: pd.DataFrame,
                           min_bad: int = None,
                           min_sample: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """过滤异常月份 — 坏样本数过少或总量过少的月份剔除

    Args:
        data: 含 new_date_ym, new_target, new_org 列的数据
        min_bad: 每月最低坏样本数
        min_sample: 每月最低总样本数

    Returns:
        data: 过滤后的数据
        filter_detail: 被过滤的月份明细
    """
    if min_bad is None:
        min_bad = cfg.MIN_YM_BAD_SAMPLE
    if min_sample is None:
        min_sample = cfg.MIN_YM_SAMPLE

    print(f"\n[Step1] Abnormal Month Filter (min_bad={min_bad}, min_sample={min_sample})")

    # 逐月统计
    month_stat = data.groupby(['new_org', 'new_date_ym']).agg(
        坏样本数=('new_target', 'sum'),
        总样本数=('new_target', 'count'),
        坏样率=('new_target', 'mean')
    ).reset_index()

    # 找异常月份
    abnormal = month_stat[
        (month_stat['坏样本数'] < min_bad) | (month_stat['总样本数'] < min_sample)
    ].copy()
    abnormal['剔除原因'] = abnormal.apply(
        lambda x: f"坏样本数{x['坏样本数']}<{min_bad}" if x['坏样本数'] < min_bad
        else f"总样本数{x['总样本数']}<{min_sample}", axis=1
    )

    if len(abnormal) > 0:
        # 剔除异常月份
        abnormal_keys = set(zip(abnormal['new_org'], abnormal['new_date_ym']))
        mask = ~data.apply(lambda x: (x['new_org'], x['new_date_ym']) in abnormal_keys, axis=1)
        data = data[mask].copy()
        print(f"  Filtered {len(abnormal)} abnormal month-org combinations, {mask.sum()} records retained")

    filter_detail = abnormal.rename(columns={
        'new_org': '机构', 'new_date_ym': '年月'
    })

    return data, filter_detail


# ============================================================
# Step 2: 缺失率筛选（含机构维度）
# ============================================================

def filter_missing(data: pd.DataFrame,
                   threshold: float = None,
                   max_org_low: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """缺失率筛选 — 整体缺失率过高或过多机构缺失率过高的特征剔除

    Args:
        data: 数据
        threshold: 整体缺失率阈值
        max_org_low: 最大容忍高缺失机构数

    Returns:
        data: 过滤后的数据
        miss_detail: 缺失率明细（含机构维度）
        miss_filter: 被剔除的特征
        miss_dist: 有值率分布统计
    """
    if threshold is None:
        threshold = cfg.MISSING_THRESHOLD
    if max_org_low is None:
        max_org_low = cfg.MAX_UNSTABLE_ORGS

    print(f"\n[Step2] Missing Rate Filter (threshold={threshold}, max_org_low={max_org_low})")

    # 计算缺失率
    miss_detail, miss_overall = missing_check(data)

    # 整体缺失率过高
    high_missing = miss_detail[miss_detail['整体'] > threshold]['变量'].tolist()

    # 机构维度缺失率：统计每个变量在多少个机构缺失率过高
    exclude_cols = ['变量', '整体']
    org_cols = [c for c in miss_detail.columns if c not in exclude_cols]
    org_missing_count = {}
    for _, row in miss_detail.iterrows():
        var = row['变量']
        low_orgs = sum(1 for org in org_cols if row[org] is not None and row[org] > threshold)
        org_missing_count[var] = low_orgs

    org_high_missing = [var for var, count in org_missing_count.items() if count >= max_org_low]

    # 合并剔除
    to_drop = list(set(high_missing + org_high_missing))
    miss_filter_rows = []
    for var in to_drop:
        reasons = []
        if var in high_missing:
            overall_val = miss_detail[miss_detail['变量'] == var]['整体'].values
            reasons.append(f"整体缺失率{overall_val[0]:.4f}>{threshold}")
        if var in org_high_missing:
            reasons.append(f"{org_missing_count[var]}个机构缺失率>{threshold}")
        miss_filter_rows.append({'变量': var, '处理原因': '; '.join(reasons)})

    miss_filter = pd.DataFrame(miss_filter_rows)

    # 有值率分布统计
    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']
                and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'float16', 'int16']]
    miss_dist = value_ratio_distribution_by_org(data, features)

    # 剔除
    if to_drop:
        data = data.drop(columns=[c for c in to_drop if c in data.columns], errors='ignore')
        print(f"  Dropped {len(to_drop)} features: {to_drop[:10]}{'...' if len(to_drop)>10 else ''}")

    return data, miss_detail, miss_filter, miss_dist


# ============================================================
# Step 3: IV 筛选（含机构维度）
# ============================================================

def filter_iv(data: pd.DataFrame,
              iv_threshold: float = None,
              org_iv_threshold: float = None,
              max_org_low_iv: int = None,
              n_jobs: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IV筛选 — 整体IV过低或过多机构IV过低的特征剔除

    Args:
        data: 数据
        iv_threshold: 整体IV阈值
        org_iv_threshold: 单机构IV阈值
        max_org_low_iv: 最大容忍低IV机构数
        n_jobs: 并行数

    Returns:
        data: 过滤后的数据
        iv_detail: IV明细（含机构维度宽表）
        iv_filter: 被剔除的特征
        iv_dist: IV分布统计
    """
    if iv_threshold is None:
        iv_threshold = cfg.IV_THRESHOLD
    if org_iv_threshold is None:
        org_iv_threshold = cfg.ORG_IV_THRESHOLD
    if max_org_low_iv is None:
        max_org_low_iv = cfg.MAX_ORG_LOW_IV
    if n_jobs is None:
        n_jobs = cfg.N_JOBS

    print(f"\n[Step3] IV Filter (threshold={iv_threshold}, org_threshold={org_iv_threshold}, max_org_low={max_org_low_iv})")

    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']
                and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'float16', 'int16']]
    features = [f for f in features if data[f].nunique() > 1]

    # 计算整体+机构IV
    iv_detail, iv_overall = calculate_iv_by_org(data, features, 'new_target', n_jobs=n_jobs)

    if len(iv_detail) == 0:
        print("  [WARN] No IV calculated")
        return data, iv_detail, pd.DataFrame(columns=['变量', '处理原因']), pd.DataFrame()

    # 整体IV过低
    low_iv_overall = iv_detail[iv_detail['整体'] < iv_threshold]['变量'].tolist()

    # 机构维度：统计每个变量在多少个机构IV过低
    exclude_cols = ['变量', '整体']
    org_cols = [c for c in iv_detail.columns if c not in exclude_cols]
    org_low_iv_count = {}
    for _, row in iv_detail.iterrows():
        var = row['变量']
        low_orgs = sum(1 for org in org_cols if row[org] is not None and row[org] < org_iv_threshold)
        org_low_iv_count[var] = low_orgs

    org_low_iv = [var for var, count in org_low_iv_count.items() if count >= max_org_low_iv]

    # 合并剔除
    to_drop = list(set(low_iv_overall + org_low_iv))
    iv_filter_rows = []
    for var in to_drop:
        reasons = []
        if var in low_iv_overall:
            val = iv_detail[iv_detail['变量'] == var]['整体'].values
            reasons.append(f"整体IV={val[0]:.4f}<{iv_threshold}")
        if var in org_low_iv:
            reasons.append(f"{org_low_iv_count[var]}个机构IV<{org_iv_threshold}")
        iv_filter_rows.append({'变量': var, '处理原因': '; '.join(reasons)})

    iv_filter_df = pd.DataFrame(iv_filter_rows)

    # IV分布统计
    iv_dist = iv_distribution_by_org(iv_detail)

    # 剔除
    if to_drop:
        data = data.drop(columns=[c for c in to_drop if c in data.columns], errors='ignore')
        print(f"  Dropped {len(to_drop)} features, retained {len([c for c in data.columns if c not in ['new_date','new_date_ym','new_target','new_org']])}")

    return data, iv_detail, iv_filter_df, iv_dist


# ============================================================
# Step 4: PSI 筛选（含机构维度）
# ============================================================

def filter_psi(data: pd.DataFrame,
               psi_threshold: float = None,
               max_months_ratio: float = None,
               max_orgs: int = None,
               n_jobs: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """PSI筛选 — 按机构按月计算PSI，不稳定特征剔除

    Returns:
        data: 过滤后的数据
        psi_detail: PSI明细
        psi_filter: 被剔除的特征
        psi_dist: PSI分布统计
    """
    if psi_threshold is None:
        psi_threshold = cfg.PSI_FEATURE_THRESHOLD
    if max_months_ratio is None:
        max_months_ratio = cfg.MAX_MONTHS_RATIO
    if max_orgs is None:
        max_orgs = cfg.MAX_UNSTABLE_ORGS
    if n_jobs is None:
        n_jobs = cfg.N_JOBS

    print(f"\n[Step4] PSI Filter (threshold={psi_threshold}, max_months_ratio={max_months_ratio:.0%}, max_orgs={max_orgs})")

    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']
                and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'float16', 'int16']]
    features = [f for f in features if data[f].nunique() > 1]

    data, psi_detail, psi_filter = calculate_psi_by_org(
        data, features, psi_threshold, max_months_ratio, max_orgs, n_jobs=n_jobs
    )

    # PSI分布统计
    psi_dist = pd.DataFrame()
    if len(psi_detail) > 0:
        psi_dist = psi_distribution_by_org(psi_detail)

    if len(psi_filter) > 0:
        print(f"  Dropped {len(psi_filter)} unstable features: {psi_filter['变量'].tolist()[:10]}")

    return data, psi_detail, psi_filter, psi_dist


# ============================================================
# Step 5: Null Importance 去噪
# ============================================================

def filter_null_importance(data: pd.DataFrame,
                           n_estimators: int = None,
                           max_depth: int = None,
                           gain_threshold: float = None,
                           n_runs: int = 5,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Null Importance去噪 — 标签置换法，剔除碰巧通过IV但实际无预测力的特征

    原理：将标签随机打乱后训练LightGBM，如果特征在打乱标签后仍获得高importance，
    说明该特征的importance是噪声。比较原始importance与置换importance，
    差异不显著的特征视为噪声剔除。

    Args:
        data: 数据
        n_estimators: LightGBM树数量
        max_depth: 最大深度
        gain_threshold: gain差异阈值（原始gain/50分位置换gain < threshold 则剔除）
        n_runs: 置换运行次数
        random_state: 随机种子

    Returns:
        data: 过滤后的数据
        null_imp_detail: Null Importance明细
        null_imp_filter: 被剔除的特征
    """
    if n_estimators is None:
        n_estimators = cfg.NULL_IMPORTANCE_N_ESTIMATORS
    if max_depth is None:
        max_depth = cfg.NULL_IMPORTANCE_MAX_DEPTH
    if gain_threshold is None:
        gain_threshold = cfg.NULL_IMPORTANCE_GAIN_THRESHOLD

    print(f"\n[Step5] Null Importance Filter (n_estimators={n_estimators}, max_depth={max_depth}, gain_threshold={gain_threshold})")

    try:
        import lightgbm as lgb
    except ImportError:
        print("  [WARN] LightGBM not installed, skipping Null Importance")
        return data, pd.DataFrame(), pd.DataFrame()

    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']
                and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'float16', 'int16']]
    features = [f for f in features if data[f].nunique() > 1]

    if len(features) < 2:
        print("  [WARN] Too few features for Null Importance")
        return data, pd.DataFrame(), pd.DataFrame()

    X = data[features].copy()
    y = data['new_target'].copy()

    # 原始importance
    print("  Training original model...")
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        verbose=-1, random_state=random_state, n_jobs=1
    )
    model.fit(X, y)
    original_importance = pd.DataFrame({
        '变量': features,
        '原始gain': model.booster_.feature_importance(importance_type='gain')
    })

    # 置换importance
    print(f"  Running {n_runs} permutation rounds...")
    null_importances = []
    for run in range(n_runs):
        y_shuffled = y.sample(frac=1, random_state=random_state + run + 1).values
        model_null = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
            verbose=-1, random_state=random_state + run + 1, n_jobs=1
        )
        model_null.fit(X, y_shuffled)
        null_imp = model_null.booster_.feature_importance(importance_type='gain')
        null_importances.append(null_imp)

    null_imp_array = np.array(null_importances)  # (n_runs, n_features)
    null_imp_50 = np.percentile(null_imp_array, 50, axis=0)  # 50分位

    # 比较原始gain与置换gain
    null_imp_detail = original_importance.copy()
    null_imp_detail['置换gain_50分位'] = null_imp_50
    null_imp_detail['gain差异'] = null_imp_detail['原始gain'] - null_imp_detail['置换gain_50分位']
    null_imp_detail['gain差异比'] = null_imp_detail.apply(
        lambda x: x['原始gain'] / max(x['置换gain_50分位'], 1e-6), axis=1
    )

    # 差异不显著的特征 = 噪声特征
    noise_features = null_imp_detail[
        null_imp_detail['gain差异'] < gain_threshold
    ]['变量'].tolist()

    null_imp_filter = null_imp_detail[
        null_imp_detail['gain差异'] < gain_threshold
    ][['变量', '原始gain', '置换gain_50分位', 'gain差异']].copy()
    null_imp_filter['处理原因'] = null_imp_filter.apply(
        lambda x: f"gain差异{x['gain差异']:.1f}<{gain_threshold}", axis=1
    )

    # 剔除噪声特征
    if noise_features:
        data = data.drop(columns=[c for c in noise_features if c in data.columns], errors='ignore')
        print(f"  Dropped {len(noise_features)} noise features")

    null_imp_detail = null_imp_detail.sort_values('gain差异', ascending=False).reset_index(drop=True)

    return data, null_imp_detail, null_imp_filter


# ============================================================
# Step 6: 相关性筛选
# ============================================================

def filter_correlation(data: pd.DataFrame,
                       threshold: float = None,
                       top_n: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """相关性筛选 — 高相关特征剔除（保留IV/gain较高的）

    Args:
        data: 数据
        threshold: 相关系数阈值
        top_n: 保留top_n个高gain特征

    Returns:
        data: 过滤后的数据
        corr_detail: 相关性明细
        corr_filter: 被剔除的特征
    """
    if threshold is None:
        threshold = cfg.CORRELATION_THRESHOLD
    if top_n is None:
        top_n = cfg.FEATURE_IMPORTANCE_TOP_N

    print(f"\n[Step6] Correlation Filter (threshold={threshold})")

    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']
                and data[c].dtype in ['float64', 'int64', 'int32', 'float32', 'float16', 'int16']]
    features = [f for f in features if data[f].nunique() > 1]

    if len(features) < 2:
        return data, pd.DataFrame(), pd.DataFrame()

    # 计算相关矩阵
    corr_matrix = data[features].corr().abs()

    # 找高相关对
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = []
    for col in upper.columns:
        for idx in upper.index:
            val = upper.loc[idx, col]
            if pd.notna(val) and val > threshold:
                high_corr_pairs.append({
                    '变量1': idx, '变量2': col, '相关系数': round(val, 4)
                })

    corr_detail = pd.DataFrame(high_corr_pairs)
    if len(corr_detail) == 0:
        print("  No high correlation pairs found")
        return data, corr_detail, pd.DataFrame(columns=['变量', '处理原因'])

    # 按IV值决定保留哪个（用整体IV表作为参考）
    try:
        iv_ref = calculate_iv(data, features, 'new_target', n_jobs=1)
        iv_dict = dict(zip(iv_ref['变量'], iv_ref['IV']))
    except Exception:
        iv_dict = {f: 0 for f in features}

    # 剔除IV较低的那个
    to_drop = set()
    drop_reasons = []
    for _, row in corr_detail.iterrows():
        f1, f2 = row['变量1'], row['变量2']
        if f1 in to_drop or f2 in to_drop:
            continue
        iv1 = iv_dict.get(f1, 0)
        iv2 = iv_dict.get(f2, 0)
        drop_feat = f1 if iv1 <= iv2 else f2
        to_drop.add(drop_feat)
        keep_feat = f2 if drop_feat == f1 else f1
        drop_reasons.append({
            '变量': drop_feat,
            '处理原因': f"与{keep_feat}相关系数={row['相关系数']:.4f}>{threshold}，IV较低({iv_dict.get(drop_feat,0):.4f}<{iv_dict.get(keep_feat,0):.4f})"
        })

    corr_filter = pd.DataFrame(drop_reasons)

    if to_drop:
        data = data.drop(columns=[c for c in to_drop if c in data.columns], errors='ignore')
        print(f"  Dropped {len(to_drop)} correlated features")

    return data, corr_detail, corr_filter


# ============================================================
# 特征筛选主流程
# ============================================================

def run_feature_selection(data: pd.DataFrame,
                          run_null_importance: bool = True,
                          run_psi: bool = True,
                          export_report: bool = True,
                          report_path: str = None) -> Tuple[pd.DataFrame, dict]:
    """特征筛选主流程 — 串联所有筛选步骤

    流程：
    1. 异常月份过滤
    2. 缺失率筛选（含机构维度）
    3. IV筛选（含机构维度）
    4. PSI筛选（含机构维度）
    5. Null Importance去噪
    6. 相关性筛选
    7. 导出Excel报告

    Args:
        data: 格式化后的数据（含 new_date_ym, new_target, new_org 列）
        run_null_importance: 是否运行Null Importance
        run_psi: 是否运行PSI筛选
        export_report: 是否导出Excel报告
        report_path: 报告路径

    Returns:
        data: 筛选后的数据
        report_steps: 报告步骤列表 [(步骤名, DataFrame), ...]
    """
    if report_path is None:
        report_path = cfg.CLEANING_REPORT_PATH

    print("\n" + "=" * 70)
    print("   Feature Selection Pipeline")
    print("=" * 70)

    features_before = len([c for c in data.columns if c not in
                           ['new_date', 'new_date_ym', 'new_target', 'new_org']])
    print(f"  Starting features: {features_before}")

    report_steps = []

    # --- Step 1: 异常月份过滤 ---
    data, month_filter = filter_abnormal_months(data)
    report_steps.append(('Step4-异常月份处理', month_filter))

    # --- Step 2: 缺失率筛选 ---
    data, miss_detail, miss_filter, miss_dist = filter_missing(data)
    report_steps.append(('Step5-有值率分布统计', miss_dist))
    report_steps.append(('Step6-高缺失率处理', miss_filter))
    report_steps.append(('Step6-缺失率明细', miss_detail))

    # --- Step 3: IV筛选 ---
    data, iv_detail, iv_filter_df, iv_dist = filter_iv(data)
    report_steps.append(('Step7-IV明细', iv_detail))
    report_steps.append(('Step7-IV分布统计', iv_dist))
    report_steps.append(('Step7-IV处理', iv_filter_df))

    # --- Step 4: PSI筛选 ---
    if run_psi:
        data, psi_detail, psi_filter, psi_dist = filter_psi(data)
        report_steps.append(('Step8-PSI明细', psi_detail))
        report_steps.append(('Step8-PSI分布统计', psi_dist))
        report_steps.append(('Step8-PSI处理', psi_filter))

    # --- Step 5: Null Importance ---
    if run_null_importance:
        data, null_imp_detail, null_imp_filter = filter_null_importance(data)
        report_steps.append(('Step9-Null Importance处理', null_imp_filter))
        report_steps.append(('Step9-Null Importance明细', null_imp_detail))

    # --- Step 6: 相关性筛选 ---
    data, corr_detail, corr_filter = filter_correlation(data)
    report_steps.append(('Step10-高相关性剔除', corr_filter))
    report_steps.append(('Step10-相关性明细', corr_detail))

    features_after = len([c for c in data.columns if c not in
                          ['new_date', 'new_date_ym', 'new_target', 'new_org']])

    # 汇总
    print(f"\n{'=' * 70}")
    print(f"   Feature Selection Complete")
    print(f"   Before: {features_before} → After: {features_after}")
    print(f"   Dropped: {features_before - features_after}")
    print(f"{'=' * 70}")

    # 导出报告
    if export_report:
        params = {
            'min_ym_bad_sample': cfg.MIN_YM_BAD_SAMPLE,
            'min_ym_sample': cfg.MIN_YM_SAMPLE,
            'missing_ratio': cfg.MISSING_THRESHOLD,
            'overall_iv_threshold': cfg.IV_THRESHOLD,
            'org_iv_threshold': cfg.ORG_IV_THRESHOLD,
            'max_org_threshold': cfg.MAX_ORG_LOW_IV,
            'psi_threshold': cfg.PSI_FEATURE_THRESHOLD,
            'max_months_ratio': cfg.MAX_MONTHS_RATIO,
            'max_orgs': cfg.MAX_UNSTABLE_ORGS,
            'gain_threshold': cfg.NULL_IMPORTANCE_GAIN_THRESHOLD,
            'max_corr': cfg.CORRELATION_THRESHOLD,
            'top_n_keep': cfg.FEATURE_IMPORTANCE_TOP_N,
        }
        export_report_xlsx(report_path, report_steps, params)

    return data, report_steps
