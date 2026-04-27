# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy", "scikit-learn", "joblib"]
# ///
"""
风控建模全链路 — Step 6: 评分卡建模

用法:
    python scripts/scorecard.py \
      --train data/split/train.parquet \
      --val data/split/val.parquet \
      --features data/feature_selection/selected_features.json \
      --output data/scorecard/ \
      --pdo 20 \
      --base-score 600 \
      --base-odds 50

输出:
    <output>/model.pkl          — 评分卡模型
    <output>/woe_encoder.pkl    — WOE编码器
    <output>/scorecard_report.json  — 评分卡报告
"""
import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 评分卡建模')
    parser.add_argument('--train', required=True, help='训练集路径（parquet）')
    parser.add_argument('--val', required=True, help='验证集路径（parquet）')
    parser.add_argument('--features', required=True, help='特征列表文件路径（JSON）')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--pdo', type=int, default=20, help='PDO（默认 20）')
    parser.add_argument('--base-score', type=int, default=600, help='基础分（默认 600）')
    parser.add_argument('--base-odds', type=int, default=50, help='基础 odds（默认 50）')
    args = parser.parse_args()

    import pandas as pd
    from references.modeling import ScorecardModel
    from references.evaluation import evaluate_model
    import config as cfg

    print("=" * 60)
    print("  Step 6: Scorecard Model")
    print("=" * 60)

    # 覆盖 config 中的评分卡参数
    cfg.SCORECARD_PDO = args.pdo
    cfg.SCORECARD_BASE_SCORE = args.base_score
    cfg.SCORECARD_BASE_ODDS = args.base_odds

    # 加载数据
    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)

    # 加载特征列表
    with open(args.features, 'r', encoding='utf-8') as f:
        features = json.load(f)

    # 过滤存在的特征
    features = [f for f in features if f in df_train.columns]
    print(f"  Features: {len(features)}")

    # 训练评分卡
    scorecard = ScorecardModel()
    scorecard.train(df_train, df_val, features)

    # 评估
    val_proba = scorecard.predict_proba(df_val)
    metrics, lift_df = evaluate_model(df_val['new_target'], val_proba, 'Scorecard')

    # 评分分布
    scores = scorecard.predict_score(df_val)
    print(f"  Score range: [{scores.min()}, {scores.max()}]")

    # 保存
    os.makedirs(args.output, exist_ok=True)
    scorecard.save(os.path.join(args.output, 'model.pkl'))
    scorecard.woe_encoder.save(os.path.join(args.output, 'woe_encoder.pkl'))

    # 保存报告
    report_path = os.path.join(args.output, 'scorecard_report.json')
    report = {
        'metrics': metrics,
        'pdo': args.pdo,
        'base_score': args.base_score,
        'base_odds': args.base_odds,
        'n_features': len(features),
        'score_range': {'min': int(scores.min()), 'max': int(scores.max())},
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Model saved to: {args.output}/model.pkl")
    print(f"  Report saved to: {report_path}")
    print("  Done!")


if __name__ == '__main__':
    main()
