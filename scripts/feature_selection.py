# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "joblib", "tqdm", "openpyxl"]
# ///
"""
风控建模全链路 — Step 4: 特征筛选（6步串联）

用法:
    python scripts/feature_selection.py \
      --input data/loaded_data.parquet \
      --output data/feature_selection/ \
      --missing-threshold 0.6 \
      --iv-threshold 0.02 \
      --psi-threshold 0.10 \
      --correlation-threshold 0.8 \
      --run-null-importance \
      --n-jobs 1

输出:
    <output>/selected_features.json  — 入选特征列表
    <output>/selection_report.xlsx   — 每步筛选明细
    <output>/filtered_data.parquet   — 筛选后的数据
"""
import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 特征筛选6步串联')
    parser.add_argument('--input', required=True, help='输入数据文件路径（parquet）')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--missing-threshold', type=float, default=0.6, help='整体缺失率阈值（默认 0.6）')
    parser.add_argument('--iv-threshold', type=float, default=0.02, help='全局IV阈值（默认 0.02）')
    parser.add_argument('--psi-threshold', type=float, default=0.10, help='PSI阈值（默认 0.10）')
    parser.add_argument('--correlation-threshold', type=float, default=0.8, help='相关系数阈值（默认 0.8）')
    parser.add_argument('--run-null-importance', action='store_true', default=True, help='运行 Null Importance（默认开启）')
    parser.add_argument('--skip-null-importance', action='store_true', help='跳过 Null Importance')
    parser.add_argument('--n-jobs', type=int, default=1, help='并行进程数（默认 1）')
    args = parser.parse_args()

    import pandas as pd
    import config as cfg
    from references.analysis import run_feature_selection

    print("=" * 60)
    print("  Step 4: Feature Selection Pipeline")
    print("=" * 60)

    # 覆盖 config 中的阈值
    cfg.MISSING_THRESHOLD = args.missing_threshold
    cfg.IV_THRESHOLD = args.iv_threshold
    cfg.ORG_IV_THRESHOLD = args.iv_threshold
    cfg.PSI_FEATURE_THRESHOLD = args.psi_threshold
    cfg.CORRELATION_THRESHOLD = args.correlation_threshold

    run_null_importance = args.run_null_importance and not args.skip_null_importance

    data = pd.read_parquet(args.input)
    data, report_steps = run_feature_selection(
        data,
        run_null_importance=run_null_importance,
        run_psi=True,
        export_report=True,
    )

    # 提取特征列表
    features = [c for c in data.columns if c not in
                ['new_date', 'new_date_ym', 'new_target', 'new_org']]

    os.makedirs(args.output, exist_ok=True)

    # 保存特征列表
    features_path = os.path.join(args.output, 'selected_features.json')
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    # 保存筛选后的数据
    data_path = os.path.join(args.output, 'filtered_data.parquet')
    data.to_parquet(data_path, index=False)

    # 复制报告到 output 目录（如果存在）
    report_src = cfg.CLEANING_REPORT_PATH
    report_dst = os.path.join(args.output, 'selection_report.xlsx')
    if os.path.exists(report_src):
        import shutil
        shutil.copy2(report_src, report_dst)

    print(f"\n  Selected features: {len(features)}")
    print(f"  Features list: {features_path}")
    print(f"  Filtered data: {data_path}")
    if os.path.exists(report_dst):
        print(f"  Report: {report_dst}")
    print("  Done!")


if __name__ == '__main__':
    main()
