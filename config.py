"""
风控建模全链路 — 配置文件
适用于贷前场景（Pre-loan Credit Scoring）

参考 awesome-copilot/datanalysis-credit-risk 结构，
采用 references/ + scripts/ 分层架构
"""

import os
import multiprocessing

# ============================================================
# 1. 系统配置
# ============================================================
RANDOM_SEED = 42
CPU_COUNT = multiprocessing.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1)  # 多进程并行数，保留1核给系统

# ============================================================
# 2. 数据配置
# ============================================================

# 合成数据配置
FAKE_DATA_SIZE = 20000
FRAUD_RATIO = 0.05

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_data.csv')
OOT_DATA_PATH = os.path.join(DATA_DIR, 'oot_data.csv')

# 数据加载参数
DATE_COL = 'apply_date'       # 日期列名
Y_COL = 'label'               # 标签列名
ORG_COL = 'org_info'          # 机构列名（支持多机构分析）
KEY_COLS = ['user_id']         # 主键列名列表
MISS_VALS = [-1, -999, -1111]  # 异常值替换为NaN

# OOS机构配置（贷外样本机构列表）
OOS_ORGS = []

# ============================================================
# 3. 数据清洗配置
# ============================================================

# 异常月份过滤
MIN_YM_BAD_SAMPLE = 10     # 每月最低坏样本数
MIN_YM_SAMPLE = 500        # 每月最低总样本数

# 缺失率
MISSING_THRESHOLD = 0.6    # 整体缺失率阈值（>0.6 删除特征）

# ============================================================
# 4. 特征工程配置
# ============================================================
WOE_BINS = 10
WOE_MIN_BIN_SIZE = 0.05

# 合规排除特征
EXCLUDED_FEATURES = ['gender', 'race', 'nationality', 'religion', 'user_id']

# ============================================================
# 5. 特征筛选配置
# ============================================================

# IV筛选
IV_THRESHOLD = 0.02            # 全局IV阈值
ORG_IV_THRESHOLD = 0.02        # 单机构IV阈值
MAX_ORG_LOW_IV = 2             # 最大容忍低IV机构数

# PSI筛选
PSI_FEATURE_THRESHOLD = 0.10   # PSI阈值
MAX_MONTHS_RATIO = 1/3         # 最大不稳定月份比例
MAX_UNSTABLE_ORGS = 6          # 最大不稳定机构数

# Null Importance去噪
NULL_IMPORTANCE_N_ESTIMATORS = 100
NULL_IMPORTANCE_MAX_DEPTH = 5
NULL_IMPORTANCE_GAIN_THRESHOLD = 50

# 相关性筛选
CORRELATION_THRESHOLD = 0.8
FEATURE_IMPORTANCE_TOP_N = 20

# VIF筛选
VIF_THRESHOLD = 5.0

# ============================================================
# 6. 评分卡配置
# ============================================================
SCORECARD_PDO = 20
SCORECARD_BASE_SCORE = 600
SCORECARD_BASE_ODDS = 50

LR_C = 1.0
LR_PENALTY = 'l2'
LR_MAX_ITER = 1000

# ============================================================
# 7. 树模型配置
# ============================================================
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 5,
}

LGB_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.03,
    'max_depth': 4,
    'num_leaves': 10,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 5.0,
    'reg_lambda': 5.0,
    'min_split_gain': 0.1,
    'verbose': -1,
}

OPTUNA_TRIALS = 20

# ============================================================
# 8. 评估阈值
# ============================================================
AUC_THRESHOLD = 0.70
KS_THRESHOLD = 0.30
GINI_THRESHOLD = 0.40
LIFT_TOP10_THRESHOLD = 3.0
PSI_MODEL_THRESHOLD = 0.10

# ============================================================
# 9. 监控配置
# ============================================================
MONITOR_PSI_THRESHOLD = 0.10
MONITOR_MISSING_CHANGE = 0.05
MONITOR_AUC_DROP = 0.05
MONITOR_BAD_RATE_CHANGE = 0.01
MONITOR_BASELINE_PATH = os.path.join(DATA_DIR, 'monitor_baseline.json')

# ============================================================
# 10. 报告配置
# ============================================================
REPORT_DIR = os.path.join(DATA_DIR, 'reports')
CLEANING_REPORT_PATH = os.path.join(REPORT_DIR, '数据清洗报告.xlsx')
MODEL_REPORT_PATH = os.path.join(REPORT_DIR, '模型评估报告.json')
MONITOR_REPORT_PATH = os.path.join(DATA_DIR, 'monitor_report.json')

# ============================================================
# 11. 模型保存路径
# ============================================================
MODEL_DIR = os.path.join(DATA_DIR, 'models')
SCORECARD_MODEL_PATH = os.path.join(MODEL_DIR, 'scorecard.pkl')
TREE_MODEL_PATH = os.path.join(MODEL_DIR, 'tree_model.pkl')
WOE_ENCODER_PATH = os.path.join(MODEL_DIR, 'woe_encoder.pkl')
