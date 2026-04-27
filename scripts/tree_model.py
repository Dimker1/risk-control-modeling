# ///
# requires-python = ">=3.8"
# dependencies = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "optuna", "joblib"]
# ///
"""
风控建模全链路 — Step 7: 树模型建模

用法:
    python scripts/tree_model.py \
      --train data/split/train.parquet \
      --val data/split/val.parquet \
      --features data/feature_selection/selected_features.json \
      --output data/tree_model/ \
      --model-type lgb \
      --cross-validate \
      --optuna-tune

输出:
    <output>/model.pkl            — 树模型
    <output>/cv_report.json       — 交叉验证结果（如启用）
    <output>/optuna_params.json   — Optuna调参结果（如启用）
    <output>/feature_importance.csv — 特征重要性
"""
import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description='风控建模 — 树模型建模')
    parser.add_argument('--train', required=True, help='训练集路径（parquet）')
    parser.add_argument('--val', required=True, help='验证集路径（parquet）')
    parser.add_argument('--features', required=True, help='特征列表文件路径（JSON）')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--model-type', choices=['lgb', 'xgb'], default='lgb', help='模型类型（默认 lgb）')
    parser.add_argument('--cross-validate', action='store_true', help='运行交叉验证')
    parser.add_argument('--optuna-tune', action='store_true', help='运行 Optuna 超参数调优')
    parser.add_argument('--n-jobs', type=int, default=1, help='并行进程数（默认 1）')
    args = parser.parse_args()

    import pandas as pd
    from references.modeling import TreeModel, cross_validate, optuna_tune
    from references.evaluation import evaluate_model, evaluate_segment_stability
    import config as cfg

    print("=" * 60)
    print(f"  Step 7: Tree Model ({args.model_type.upper()})")
    print("=" * 60)

    # 加载数据
    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)

    # 加载特征列表
    with open(args.features, 'r', encoding='utf-8') as f:
        features = json.load(f)

    # 过滤存在的特征
    features = [feat for feat in features if feat in df_train.columns]
    print(f"  Features: {len(features)}")

    os.makedirs(args.output, exist_ok=True)

    # 训练树模型
    tree_model = TreeModel(model_type=args.model_type)
    tree_model.train(df_train, df_val, features)

    # 交叉验证
    if args.cross_validate:
        print("\n  Running cross-validation...")
        cv_results = cross_validate(df_train, features, model_type=args.model_type)
        cv_path = os.path.join(args.output, 'cv_report.json')
        cv_results.to_json(cv_path, orient='records', force_ascii=False, indent=2)
        print(f"  CV report saved to: {cv_path}")

    # Optuna 调参
    if args.optuna_tune:
        print("\n  Running Optuna hyperparameter tuning...")
        best_params = optuna_tune(df_train, df_val, features, model_type=args.model_type)
        if best_params:
            optuna_path = os.path.join(args.output, 'optuna_params.json')
            with open(optuna_path, 'w', encoding='utf-8') as f:
                json.dump(best_params, f, ensure_ascii=False, indent=2)
            print(f"  Optuna params saved to: {optuna_path}")

            # 用最优参数重新训练
            tree_model = TreeModel(model_type=args.model_type)
            tree_model.model = tree_model._create_model(best_params)
            tree_model.features = features
            X_train = df_train[features].copy()
            y_train = df_train['new_target']
            # 分类特征编码
            from sklearn.preprocessing import LabelEncoder
            cat_cols = [c for c in features if X_train[c].dtype == object or str(X_train[c].dtype) == 'category']
            tree_model.cat_encoders_ = {}
            for c in cat_cols:
                le = LabelEncoder()
                X_train[c] = le.fit_transform(X_train[c].astype(str))
                tree_model.cat_encoders_[c] = le
            tree_model.model.fit(X_train, y_train)

    # 评估
    val_proba = tree_model.predict_proba(df_val)
    metrics, lift_df = evaluate_model(df_val['new_target'], val_proba, args.model_type.upper())

    # 特征重要性
    imp = tree_model.feature_importance()
    imp_path = os.path.join(args.output, 'feature_importance.csv')
    imp.to_csv(imp_path, index=False, encoding='utf-8-sig')

    print(f"\n  Top 10 features:")
    for _, row in imp.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']}")

    # 分群稳定性
    if 'new_org' in df_val.columns:
        seg_results = evaluate_segment_stability(
            df_val, features, val_proba, segment_col='new_org'
        )
        if len(seg_results) > 0:
            print(f"\n  Segment stability:")
            for _, row in seg_results.iterrows():
                print(f"    {row['segment']}: AUC={row['auc']:.4f}, KS={row['ks']:.4f}")

    # 保存模型
    model_path = os.path.join(args.output, 'model.pkl')
    tree_model.save(model_path)

    # 保存评估指标
    metrics_path = os.path.join(args.output, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n  Model saved to: {model_path}")
    print(f"  Feature importance saved to: {imp_path}")
    print("  Done!")


if __name__ == '__main__':
    main()
