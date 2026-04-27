# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy", "tqdm"]
# ///
"""
风控建模全链路 — Step 2: 数据加载与格式化

用法:
    python scripts/load_data.py \
      --input data/raw_data.csv \
      --date-col apply_date \
      --target-col label \
      --org-col org_info \
      --key-cols user_id \
      --output data/loaded_data.parquet
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 数据加载与格式化')
    parser.add_argument('--input', required=True, help='输入数据文件路径（支持 parquet/csv/xlsx/pkl）')
    parser.add_argument('--date-col', required=True, help='日期列名')
    parser.add_argument('--target-col', required=True, help='标签列名')
    parser.add_argument('--org-col', default=None, help='机构列名（可选）')
    parser.add_argument('--key-cols', default=None, help='主键列名，逗号分隔（可选）')
    parser.add_argument('--drop-cols', default=None, help='需删除的列名，逗号分隔（可选）')
    parser.add_argument('--output', required=True, help='输出文件路径（parquet）')
    args = parser.parse_args()

    from references.func import get_dataset

    print("=" * 60)
    print("  Step 2: Data Loading & Formatting")
    print("=" * 60)

    key_cols = args.key_cols.split(',') if args.key_cols else None
    drop_cols = args.drop_cols.split(',') if args.drop_cols else None

    data = get_dataset(
        data_path=args.input,
        date_col=args.date_col,
        y_col=args.target_col,
        org_col=args.org_col,
        key_cols=key_cols,
        drop_cols=drop_cols,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.output.endswith('.parquet'):
        data.to_parquet(args.output, index=False)
    else:
        data.to_csv(args.output, index=False)

    print(f"\n  Saved to: {args.output}")
    print(f"  Shape: {data.shape}")
    print(f"  Bad rate: {data['new_target'].mean():.4f}")
    print("  Done!")


if __name__ == '__main__':
    main()
