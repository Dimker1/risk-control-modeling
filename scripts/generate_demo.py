# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy"]
# ///
"""
风控建模全链路 — Step 1: 合成数据生成（演示用）

用法:
    python scripts/generate_demo.py --output data/raw_data.csv --n-samples 20000 --seed 42
"""
import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def generate_demo_data(n_samples=20000, random_state=42):
    """生成贷前风控合成数据"""
    import pandas as pd
    import numpy as np

    np.random.seed(random_state)

    age = np.random.randint(18, 66, n_samples)
    income = np.random.lognormal(mean=9.5, sigma=0.6, size=n_samples).clip(3000, 80000)
    employment_years = (age - 18) * np.random.uniform(0.3, 0.8, n_samples).astype(int).clip(0, 30)
    employment_type = np.random.choice(['正式', '合同', '临时', '自由职业'], n_samples, p=[0.4, 0.3, 0.15, 0.15])

    credit_score = np.random.normal(650, 80, n_samples).clip(300, 850).astype(int)
    has_overdue = np.random.binomial(1, 0.15, n_samples)
    overdue_count = has_overdue * np.random.poisson(1.5, n_samples)
    loan_balance = np.random.exponential(50000, n_samples).clip(0, 500000)
    debt_to_income = loan_balance / (income * 12 + 1)

    login_frequency = np.random.poisson(15, n_samples).clip(0, 60)
    page_views = np.random.poisson(100, n_samples).clip(0, 500)
    session_duration = np.random.exponential(600, n_samples).clip(10, 3600)
    complaint_count = np.random.poisson(0.5, n_samples).clip(0, 10)

    gold_bar_amount = np.random.exponential(5000, n_samples).clip(100, 100000)
    purchase_frequency = np.random.poisson(3, n_samples).clip(0, 15)
    avg_holding_period = np.random.exponential(90, n_samples).clip(1, 365)
    redemption_rate = np.random.beta(2, 5, n_samples).clip(0, 1)

    # 标签生成
    risk_score = (
        (credit_score < 550).astype(float) * 2.0 +
        has_overdue * 1.5 +
        np.log1p(overdue_count) * 0.8 +
        (debt_to_income > 3).astype(float) * 1.2 +
        (income < 5000).astype(float) * 0.8 +
        (redemption_rate > 0.6).astype(float) * 0.6 +
        (complaint_count > 3).astype(float) * 0.5
    )
    risk_prob = 1 / (1 + np.exp(-risk_score + 3))
    label = (np.random.random(n_samples) < risk_prob).astype(int)

    # 日期和机构
    dates = pd.date_range('2023-01-01', periods=12, freq='MS')
    apply_date = np.random.choice(dates, n_samples)
    org_info = np.random.choice(['机构A', '机构B', '机构C'], n_samples, p=[0.5, 0.3, 0.2])

    df = pd.DataFrame({
        'user_id': [f'U{i:06d}' for i in range(n_samples)],
        'apply_date': apply_date,
        'org_info': org_info,
        'label': label,
        'age': age,
        'income': income,
        'employment_years': employment_years,
        'employment_type': employment_type,
        'credit_score': credit_score,
        'has_overdue': has_overdue,
        'overdue_count': overdue_count,
        'loan_balance': loan_balance,
        'debt_to_income': debt_to_income,
        'login_frequency': login_frequency,
        'page_views': page_views,
        'session_duration': session_duration,
        'complaint_count': complaint_count,
        'gold_bar_amount': gold_bar_amount,
        'purchase_frequency': purchase_frequency,
        'avg_holding_period': avg_holding_period,
        'redemption_rate': redemption_rate,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 生成合成演示数据')
    parser.add_argument('--output', required=True, help='输出文件路径（支持 csv/parquet）')
    parser.add_argument('--n-samples', type=int, default=20000, help='样本数量（默认 20000）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    args = parser.parse_args()

    print("=" * 60)
    print("  Step 1: Generate Demo Data")
    print("=" * 60)

    df = generate_demo_data(args.n_samples, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.output.endswith('.parquet'):
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"  Samples:  {len(df)}")
    print(f"  Features: {len(df.columns) - 4}")  # 排除 user_id, apply_date, org_info, label
    print(f"  Bad rate: {df['label'].mean():.4f}")
    print(f"  Saved to: {args.output}")
    print("  Done!")


if __name__ == '__main__':
    main()
