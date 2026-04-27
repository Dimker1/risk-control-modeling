"""
建模模块 — 评分卡建模 + 树模型建模
包含：WOE编码、评分卡、LightGBM/XGBoost训练、Optuna调参、交叉验证
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


# ============================================================
# WOE 编码
# ============================================================

class WOEEncoder:
    """WOE编码器：对特征进行决策树分箱并转换为WOE值"""

    def __init__(self, n_bins=10, min_bin_size=0.05):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.woe_maps_ = {}
        self.bin_maps_ = {}
        self.iv_values_ = {}
        self.label_encoders_ = {}

    def fit(self, df, features, target='new_target'):
        """拟合WOE编码"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
        print(f"\n[WOE Encoder] Fitting {len(features)} features...")

        for feat in features:
            try:
                X = df[[feat]].copy()
                y = df[target].copy()
                mask = X[feat].notna()
                X_valid, y_valid = X[mask], y[mask]

                if len(X_valid) < 50 or len(np.unique(y_valid)) < 2:
                    self.woe_maps_[feat] = {}
                    self.iv_values_[feat] = 0
                    continue

                # 分类特征：LabelEncoder
                is_categorical = X_valid[feat].dtype == object or str(X_valid[feat].dtype) == 'category'
                if is_categorical:
                    le = LabelEncoder()
                    X_numeric = le.fit_transform(X_valid[feat].astype(str)).reshape(-1, 1)
                    self.label_encoders_[feat] = le
                else:
                    X_numeric = X_valid.values.astype(float)

                # 决策树分箱
                dt = DecisionTreeClassifier(
                    max_leaf_nodes=self.n_bins,
                    min_samples_leaf=self.min_bin_size,
                    random_state=42
                )
                dt.fit(X_numeric, y_valid.values)
                bins = dt.apply(X_numeric)

                # 计算每箱WOE
                y_arr = y_valid.values
                total_good = (y_arr == 0).sum()
                total_bad = (y_arr == 1).sum()
                woe_map = {}
                iv_value = 0.0
                bin_detail = []

                for bin_id in sorted(np.unique(bins)):
                    bin_mask = bins == bin_id
                    bin_good = int(((y_arr == 0) & bin_mask).sum())
                    bin_bad = int(((y_arr == 1) & bin_mask).sum())

                    good_pct = max(bin_good / max(total_good, 1), 1e-6)
                    bad_pct = max(bin_bad / max(total_bad, 1), 1e-6)

                    woe = np.log(good_pct / bad_pct)
                    woe = np.clip(woe, -5, 5)
                    woe_map[bin_id] = woe
                    iv_value += (good_pct - bad_pct) * woe

                    # 分箱阈值
                    bin_data = X_numeric[bin_mask]
                    bin_detail.append({
                        'bin_id': bin_id,
                        'min': float(bin_data.min()),
                        'max': float(bin_data.max()),
                        'count': int(bin_mask.sum()),
                        'bad_rate': float(bin_bad / max(bin_mask.sum(), 1)),
                        'woe': woe
                    })

                self.woe_maps_[feat] = woe_map
                self.iv_values_[feat] = round(iv_value, 4)
                self.bin_maps_[feat] = bin_detail

            except Exception as e:
                print(f"  [WARN] WOE failed for {feat}: {e}")
                self.woe_maps_[feat] = {}
                self.iv_values_[feat] = 0

        valid_count = sum(1 for v in self.iv_values_.values() if v > 0)
        print(f"  Encoded {valid_count}/{len(features)} features with valid IV")
        return self

    def transform(self, df, features):
        """转换数据为WOE值"""
        df_woe = df.copy()

        for feat in features:
            if feat not in self.woe_maps_ or not self.woe_maps_[feat]:
                df_woe[f'{feat}_woe'] = 0
                continue

            X = df[[feat]].copy()
            mask = X[feat].notna()

            if mask.sum() == 0:
                df_woe[f'{feat}_woe'] = 0
                continue

            # 分类特征需要先LabelEncoder
            is_categorical = feat in self.label_encoders_
            if is_categorical:
                le = self.label_encoders_[feat]
                X_numeric = le.transform(X.loc[mask, feat].astype(str)).reshape(-1, 1).astype(float)
            else:
                X_numeric = X.loc[mask].values.astype(float)

            # 用决策树分箱
            bin_detail = self.bin_maps_.get(feat, [])
            if not bin_detail:
                df_woe[f'{feat}_woe'] = 0
                continue

            # 根据bin边界映射WOE值
            woe_col = pd.Series(0.0, index=df.index)
            valid_idx = df.index[mask]
            for bd in sorted(bin_detail, key=lambda x: x['min']):
                if is_categorical:
                    # 分类特征：精确匹配编码值
                    match_mask = (X_numeric.ravel() >= bd['min']) & (X_numeric.ravel() <= bd['max'])
                    woe_col.loc[valid_idx[match_mask.ravel()]] = bd['woe']
                else:
                    match_mask = (df.loc[mask, feat] >= bd['min']) & (df.loc[mask, feat] <= bd['max'])
                    woe_col.loc[valid_idx[match_mask.values]] = bd['woe']

            # 缺失值填充为0
            woe_col[df[feat].isna()] = 0
            woe_col = woe_col.clip(-5, 5)
            df_woe[f'{feat}_woe'] = woe_col

        return df_woe

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  WOE encoder saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# 评分卡模型
# ============================================================

class ScorecardModel:
    """标准评分卡模型 — WOE + LogisticRegression + 评分映射"""

    def __init__(self):
        self.model = None
        self.woe_encoder = None
        self.features = None
        self.score_map = None

    def train(self, df_train, df_test, features, target='new_target'):
        """训练评分卡模型"""
        print(f"\n{'=' * 60}")
        print("Scorecard Model Training")
        print(f"{'=' * 60}")

        self.features = features

        # WOE编码
        self.woe_encoder = WOEEncoder(
            n_bins=cfg.WOE_BINS,
            min_bin_size=cfg.WOE_MIN_BIN_SIZE
        )
        self.woe_encoder.fit(df_train, features, target)

        df_train_woe = self.woe_encoder.transform(df_train, features)
        df_test_woe = self.woe_encoder.transform(df_test, features)

        woe_features = [f'{f}_woe' for f in features if f'{f}_woe' in df_train_woe.columns]

        X_train = df_train_woe[woe_features]
        y_train = df_train_woe[target]
        X_test = df_test_woe[woe_features]
        y_test = df_test_woe[target]

        # LR训练
        self.model = LogisticRegression(
            C=cfg.LR_C, penalty=cfg.LR_PENALTY,
            max_iter=cfg.LR_MAX_ITER, random_state=cfg.RANDOM_SEED
        )
        self.model.fit(X_train, y_train)

        # 评分映射
        self._build_score_map(woe_features)

        # 评估
        train_auc = roc_auc_score(y_train, self.model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])

        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Test AUC:  {test_auc:.4f}")

        return self

    def _build_score_map(self, woe_features):
        """构建评分映射表"""
        pdo = cfg.SCORECARD_PDO
        base_score = cfg.SCORECARD_BASE_SCORE
        base_odds = cfg.SCORECARD_BASE_ODDS
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)

        self.score_map = {
            'factor': factor, 'offset': offset,
            'base_score': base_score, 'pdo': pdo,
            'feature_scores': {}
        }

        for i, feat in enumerate(woe_features):
            coef = self.model.coef_[0][i]
            orig_feat = feat.replace('_woe', '')
            self.score_map['feature_scores'][orig_feat] = {
                'coefficient': float(coef),
                'woe_bins': self.woe_encoder.bin_maps_.get(orig_feat, [])
            }

    def predict_proba(self, df):
        """预测概率"""
        df_woe = self.woe_encoder.transform(df, self.features)
        woe_features = [f'{f}_woe' for f in self.features if f'{f}_woe' in df_woe.columns]
        return self.model.predict_proba(df_woe[woe_features])[:, 1]

    def predict_score(self, df):
        """预测评分"""
        proba = self.predict_proba(df)
        factor = self.score_map['factor']
        offset = self.score_map['offset']
        odds = proba / (1 - proba + 1e-6)
        scores = offset + factor * np.log(odds + 1e-6)
        return scores.astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Scorecard model saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# 时序数据切分
# ============================================================

def time_based_split(df, test_size=None, val_size=None, target='new_target'):
    """时序切分（贷前场景必须用时间切分，不能用随机划分）"""
    if test_size is None:
        test_size = cfg.TEST_SIZE
    if val_size is None:
        val_size = cfg.VAL_SIZE

    n = len(df)
    test_n = int(n * test_size)
    val_n = int((n - test_n) * val_size)
    train_n = n - test_n - val_n

    df_train = df.iloc[:train_n].copy()
    df_val = df.iloc[train_n:train_n + val_n].copy()
    df_test = df.iloc[train_n + val_n:].copy()

    print(f"\n[Time-Based Split]")
    print(f"  Train: {len(df_train)} (bad_rate={df_train[target].mean():.2%})")
    print(f"  Val:   {len(df_val)} (bad_rate={df_val[target].mean():.2%})")
    print(f"  Test:  {len(df_test)} (bad_rate={df_test[target].mean():.2%})")

    return df_train, df_val, df_test


# ============================================================
# 树模型训练
# ============================================================

class TreeModel:
    """树模型封装：支持LightGBM和XGBoost"""

    def __init__(self, model_type='lgb'):
        self.model_type = model_type
        self.model = None
        self.features = None
        self.best_iteration = None

    def _create_model(self, params=None):
        """创建模型实例"""
        if params is None:
            params = cfg.LGB_PARAMS if self.model_type == 'lgb' else cfg.XGB_PARAMS

        if self.model_type == 'lgb':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        elif self.model_type == 'xgb':
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train(self, df_train, df_val, features, target='new_target'):
        """训练树模型"""
        print(f"\n{'=' * 60}")
        print(f"Tree Model Training ({self.model_type.upper()})")
        print(f"{'=' * 60}")

        self.features = features

        X_train = df_train[features].copy()
        y_train = df_train[target]
        X_val = df_val[features].copy()
        y_val = df_val[target]

        # 分类特征编码（LightGBM/XGBoost不支持object类型）
        from sklearn.preprocessing import LabelEncoder
        cat_cols = [c for c in features if X_train[c].dtype == object or str(X_train[c].dtype) == 'category']
        self.cat_encoders_ = {}
        for c in cat_cols:
            le = LabelEncoder()
            X_train[c] = le.fit_transform(X_train[c].astype(str))
            X_val[c] = X_val[c].map(lambda x: le.transform([str(x)])[0]
                                     if str(x) in le.classes_ else -1)
            self.cat_encoders_[c] = le

        self.model = self._create_model()

        fit_params = {}
        if self.model_type == 'lgb':
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                __import__('lightgbm').early_stopping(cfg.EARLY_STOPPING_ROUNDS, verbose=False),
                __import__('lightgbm').log_evaluation(period=0),
            ]
        elif self.model_type == 'xgb':
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        self.model.fit(X_train, y_train, **fit_params)

        # 评估
        train_proba = self.model.predict_proba(X_train)[:, 1]
        val_proba = self.model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"  Train AUC: {train_auc:.4f}")
        print(f"  Val AUC:   {val_auc:.4f}")

        if hasattr(self.model, 'best_iteration_'):
            self.best_iteration = self.model.best_iteration_
            print(f"  Best iteration: {self.best_iteration}")

        return self

    def predict_proba(self, df):
        """预测概率"""
        X = df[self.features].copy()
        # 分类特征编码
        if hasattr(self, 'cat_encoders_'):
            for c, le in self.cat_encoders_.items():
                if c in X.columns:
                    X[c] = X[c].map(lambda x: le.transform([str(x)])[0]
                                    if str(x) in le.classes_ else -1)
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, importance_type='gain'):
        """获取特征重要性"""
        if self.model_type == 'lgb':
            imp = self.model.booster_.feature_importance(importance_type=importance_type)
        else:
            imp = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.features,
            'importance': imp
        }).sort_values('importance', ascending=False).reset_index(drop=True)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Tree model saved to {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# 交叉验证
# ============================================================

def cross_validate(df, features, target='new_target', model_type='lgb',
                   n_folds=None, params=None):
    """时序交叉验证"""
    if n_folds is None:
        n_folds = cfg.CV_FOLDS

    print(f"\n[Cross Validation] {n_folds}-fold, model={model_type}")

    from sklearn.metrics import roc_curve

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[target])):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        model = TreeModel(model_type)
        if params:
            model.model = model._create_model(params)
        model.features = features

        X_train = df_train[features].copy()
        y_train = df_train[target]
        X_val = df_val[features].copy()
        y_val = df_val[target]

        # 分类特征编码
        from sklearn.preprocessing import LabelEncoder
        cat_cols = [c for c in features if X_train[c].dtype == object or str(X_train[c].dtype) == 'category']
        for c in cat_cols:
            le = LabelEncoder()
            X_train[c] = le.fit_transform(X_train[c].astype(str))
            X_val[c] = X_val[c].map(lambda x: le.transform([str(x)])[0]
                                     if str(x) in le.classes_ else -1)

        if model.model is None:
            model.model = model._create_model(params if params else None)
        model.model.fit(X_train, y_train)

        val_proba = model.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)

        # KS
        fpr, tpr, _ = roc_curve(y_val, val_proba)
        ks = max(tpr - fpr)

        cv_results.append({
            'fold': fold + 1,
            'train_size': len(df_train),
            'val_size': len(df_val),
            'val_auc': round(val_auc, 4),
            'val_ks': round(ks, 4),
            'val_gini': round(2 * val_auc - 1, 4)
        })
        print(f"  Fold {fold + 1}: AUC={val_auc:.4f}, KS={ks:.4f}, Gini={2*val_auc-1:.4f}")

    cv_df = pd.DataFrame(cv_results)
    print(f"\n  Mean AUC:  {cv_df['val_auc'].mean():.4f} ± {cv_df['val_auc'].std():.4f}")
    print(f"  Mean KS:   {cv_df['val_ks'].mean():.4f} ± {cv_df['val_ks'].std():.4f}")
    print(f"  Mean Gini: {cv_df['val_gini'].mean():.4f} ± {cv_df['val_gini'].std():.4f}")

    return cv_df


# ============================================================
# Optuna 调参
# ============================================================

def optuna_tune(df_train, df_val, features, target='new_target',
                model_type='lgb', n_trials=None):
    """Optuna超参数调优"""
    if n_trials is None:
        n_trials = cfg.OPTUNA_TRIALS

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [WARN] Optuna not installed, skipping tuning")
        return None

    print(f"\n[Optuna Tune] model={model_type}, trials={n_trials}")

    X_train = df_train[features]
    y_train = df_train[target]
    X_val = df_val[features]
    y_val = df_val[target]

    def objective(trial):
        if model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'verbose': -1,
            }
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
            }
            import xgboost as xgb
            model = xgb.XGBClassifier(**params)

        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, val_proba)

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params
