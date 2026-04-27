# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy"]
# ///
"""
风控建模全链路 — Step 3: 机构样本分析

用法:
    python scripts/org_analysis.py \
      --input data/loaded_data.parquet \
      --output data/org_analysis/
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 机构样本分析')
    parser.add_argument('--input', required=True, help='输入数据文件路径（parquet，需含 new_org 列）')
    parser.add_argument('--oos-orgs', default=None, help='贷外机构列表，逗号分隔（可选）')
    parser.add_argument('--output', required=True, help='输出目录')
    args = parser.parse_args()

    import pandas as pd
    from references.func import org_analysis
    import config as cfg

    print("=" * 60)
    print("  Step 3: Organization Analysis")
    print("=" * 60)

    data = pd.read_parquet(args.input)
    oos_orgs = args.oos_orgs.split(',') if args.oos_orgs else cfg.OOS_ORGS

    org_stat = org_analysis(data, oos_orgs=oos_orgs)

    os.makedirs(args.output, exist_ok=True)
    org_stat.to_csv(os.path.join(args.output, 'org_statistics.csv'), index=False, encoding='utf-8-sig')

    # 打印摘要
    print(f"\n  Org statistics:")
    for org in data['new_org'].unique():
        org_data = data[data['new_org'] == org]
        print(f"    {org}: {len(org_data)} samples, bad_rate={org_data['new_target'].mean():.4f}")

    print(f"\n  Saved to: {args.output}/org_statistics.csv")
    print("  Done!")


if __name__ == '__main__':
    main()
