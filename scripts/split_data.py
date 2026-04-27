# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy"]
# ///
"""
风控建模全链路 — Step 5: 时序数据切分

用法:
    python scripts/split_data.py \
      --input data/feature_selection/filtered_data.parquet \
      --output data/split/ \
      --train-ratio 0.7 \
      --val-ratio 0.15

输出:
    <output>/train.parquet
    <output>/val.parquet
    <output>/test.parquet
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 时序数据切分')
    parser.add_argument('--input', required=True, help='输入数据文件路径（parquet）')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例（默认 0.7）')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例（默认 0.15）')
    args = parser.parse_args()

    import pandas as pd
    from references.modeling import time_based_split
    import config as cfg

    print("=" * 60)
    print("  Step 5: Time-Based Data Split")
    print("=" * 60)

    # 覆盖 config 中的切分比例
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    cfg.TEST_SIZE = test_ratio
    cfg.VAL_SIZE = args.val_ratio / (args.train_ratio + args.val_ratio)  # 相对于非测试部分

    data = pd.read_parquet(args.input)
    # 按日期排序后再切分（贷前场景必须按时序划分）
    data = data.sort_values('new_date').reset_index(drop=True)
    df_train, df_val, df_test = time_based_split(data)

    os.makedirs(args.output, exist_ok=True)
    df_train.to_parquet(os.path.join(args.output, 'train.parquet'), index=False)
    df_val.to_parquet(os.path.join(args.output, 'val.parquet'), index=False)
    df_test.to_parquet(os.path.join(args.output, 'test.parquet'), index=False)

    print(f"\n  Saved to: {args.output}/")
    print(f"    train.parquet: {len(df_train)} rows")
    print(f"    val.parquet:   {len(df_val)} rows")
    print(f"    test.parquet:  {len(df_test)} rows")
    print("  Done!")


if __name__ == '__main__':
    main()
