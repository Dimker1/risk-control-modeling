# 风控建模全链路 AI Skill（Pre-loan Credit Scoring）

---

## 一、项目背景

在消费金融和信贷业务中，贷前风控模型是识别违约风险、控制不良率的核心工具。传统风控建模流程涉及数据清洗、特征工程、特征筛选、模型训练、模型评估等多个环节，各环节之间存在紧密的数据依赖和逻辑耦合。然而，当前行业面临几个痛点：

1. **流程碎片化**：建模各步骤散落在不同脚本和工具中，缺乏统一的串联机制，数据在步骤间的传递容易出错。
2. **机构维度缺失**：多数开源方案仅做全局层面的特征筛选，忽略了不同进件机构之间样本分布和特征表现的差异，导致模型在某些机构上表现不稳定。
3. **监控体系不完善**：模型上线后的特征漂移（PSI）和模型性能衰减缺乏自动化检测机制。
4. **AI 集成能力弱**：传统风控工具包以 Python API 形式存在，需要开发者手动编写调用代码，无法被 AI Agent 自动发现和执行。

基于以上背景，本项目以贷前信用评分（Pre-loan Credit Scoring）为场景，构建了一套覆盖「数据生成 → 特征筛选 → 建模 → 评估 → 监控」的全链路 AI Skill，并发布到 skills.sh 生态，可被 45+ 主流 AI 编程平台（Cursor、GitHub Copilot、Claude Code、OpenCode 等）自动识别和调用。

---

## 二、项目目的

本项目旨在实现以下目标：

1. **全链路自动化**：将风控建模的 8 个核心步骤封装为独立 CLI 工具，支持一键端到端执行，也可按步骤灵活调用。
2. **机构维度感知**：在缺失率筛选、IV 筛选、PSI 筛选等环节支持按机构（org）维度分析，识别「全局过关但个别机构不稳定」的特征，提升模型在各机构的鲁棒性。
3. **双模型对比**：同时支持评分卡模型（WOE + 逻辑回归）和树模型（LightGBM/XGBoost + Optuna 调参），满足不同业务场景的可解释性与预测力需求。
4. **监控闭环**：内置特征监控（PSI 漂移 + 缺失率变化）和模型监控（AUC 衰减 + 坏账率偏移），形成从建模到上线的完整闭环。
5. **AI Skill 标准化**：以 SKILL.md 作为 Agent 指令集（而非 API 文档），使 AI Agent 能按步骤引导用户完成建模流程，实现对话式交互。

---

## 三、实现方法

### 3.1 架构设计

项目采用 `references/ + scripts/` 分层架构，参考 awesome-copilot/datanalysis-credit-risk 的 Skill 规范：

```
risk-control-modeling/
├── SKILL.md                   # Agent 指令集（触发条件 + 逐步引导）
├── config.py                  # 全局配置（阈值、路径、模型参数）
├── references/                # 可复用的库模块
│   ├── func.py                # 基础函数：数据加载、机构分析、IV/PSI 计算、Excel 报告
│   ├── analysis.py            # 特征筛选：6 步串联（异常月份→缺失率→IV→PSI→Null Importance→相关性）
│   ├── modeling.py            # 建模：WOE 编码、评分卡、树模型、交叉验证、Optuna 调参
│   └── evaluation.py          # 评估 + 监控：KS/AUC/Gini/Lift/PR、特征监控、模型监控
└── scripts/                   # 8 个独立 CLI 工具（argparse + inline deps）
    ├── generate_demo.py       # Step 1: 合成数据生成
    ├── load_data.py           # Step 2: 数据加载与格式化
    ├── org_analysis.py        # Step 3: 机构样本分析
    ├── feature_selection.py   # Step 4: 特征筛选 6 步串联
    ├── split_data.py          # Step 5: 时序数据切分
    ├── scorecard.py           # Step 6: 评分卡建模
    ├── tree_model.py          # Step 7: 树模型建模（含 CV/Optuna）
    └── evaluate.py            # Step 8: 评估与监控（4 个子命令）
```

**分层原则**：
- `references/` 是纯 Python 库，提供可复用的函数和类，无命令行入口
- `scripts/` 是 CLI 入口层，薄封装，解析命令行参数后调用 references 函数
- `config.py` 是单一配置源，scripts 可通过 CLI 参数覆盖默认值

### 3.2 Pipeline 流程

```
Step 1 合成数据生成
  │  generate_demo.py --output raw_data.csv
  ▼
Step 2 数据加载与格式化
  │  load_data.py --input raw_data.csv --output loaded_data.parquet
  │  · 多格式读取（parquet/csv/xlsx/pkl）
  │  · 缺失值标记替换为 NaN
  │  · 去常量特征、去重、标签校验
  ▼
Step 3 机构样本分析
  │  org_analysis.py --input loaded_data.parquet --output org_analysis/
  │  · 按 [机构, 年月] 聚合统计坏账率
  │  · 标记 OOS（出箱）机构
  ▼
Step 4 特征筛选（6 步串联）
  │  feature_selection.py --input loaded_data.parquet --output fs/
  │  ┌─ 4a 异常月份过滤：删除坏样本<10 或总样本<500 的月-机构组合
  │  ├─ 4b 缺失率筛选：全局>60% 或 ≥2个机构超阈值 → 剔除
  │  ├─ 4c IV 筛选：全局<0.02 或 ≥2个机构 IV<0.02 → 剔除
  │  ├─ 4d PSI 筛选：按机构按月计算 PSI，不稳定月数≥1/3 且≥6个机构 → 剔除
  │  ├─ 4e Null Importance 去噪：LightGBM 置换标签≥5次，gain 差异<50 → 剔除
  │  └─ 4f 相关性筛选：|r|>0.8 的特征对，保留 IV 较高者
  │  输出：selected_features.json + filtered_data.parquet + selection_report.xlsx
  ▼
Step 5 时序数据切分
  │  split_data.py --input filtered_data.parquet --output split/
  │  · 按 new_date 排序后按时间顺序切分：70% train / 15% val / 15% test
  │  · 不使用随机切分（避免未来信息泄露）
  ▼
Step 6 评分卡建模
  │  scorecard.py --train train.parquet --val val.parquet --output scorecard/
  │  · 决策树分箱 → WOE 编码（clip [-5,5]）→ 逻辑回归 L2
  │  · 评分映射：PDO=20, Base=600, Odds=50
  ▼
Step 7 树模型建模
  │  tree_model.py --train train.parquet --val val.parquet --output tree_model/
  │  · LightGBM / XGBoost + Early Stopping
  │  · 可选：5-Fold 交叉验证 + Optuna 调参（20 trials）
  │  · 特征重要性导出 + 分群稳定性评估
  ▼
Step 8 评估与监控
  │  evaluate.py evaluate / save-baseline / monitor-features / monitor-model
  │  · 评估：AUC, KS, Gini, PR-AUC, Lift, Precision/Recall/F1
  │  · 监控：特征 PSI 漂移 + 缺失率变化，模型 AUC 衰减 + 坏账率偏移
  │  · 三级告警：HIGH / MEDIUM / LOW
  ▼
[完成]
```

### 3.3 核心算法

#### 3.3.1 决策树分箱 + WOE 编码

传统的等频/等距分箱对极端值和偏态分布不鲁棒。本项目采用 `DecisionTreeClassifier` 进行自适应分箱：
- `max_leaf_nodes=n_bins`（默认 10）：控制分箱数
- `min_samples_leaf=0.05`：防止单箱过小
- 分箱后计算 WOE = ln(好样本占比 / 坏样本占比)，clip 到 [-5, 5]
- IV = sum((好样本占比 - 坏样本占比) × WOE)

#### 3.3.2 Null Importance 去噪

通过置换标签来识别噪声特征：
1. 用原始标签训练 LightGBM，记录各特征 gain
2. 置换标签 ≥5 次，每次训练 LightGBM，记录 null gain
3. 取 null gain 的 50 分位数作为基线
4. 若特征的 gain - null_gain_50 < 阈值（50），则判定为噪声特征

#### 3.3.3 机构维度分析

在 IV、PSI、缺失率三个维度均支持按机构分析：
- **IV 按机构**：计算每个特征在每个机构的 IV，若全局 IV 过关但 ≥2 个机构 IV < 0.02，则剔除
- **PSI 按机构按月**：以每个机构的第一个月为基线，计算后续每月的 PSI；若不稳定月数 ≥1/3 且涉及 ≥6 个机构，则剔除
- **缺失率按机构**：若全局缺失率过关但 ≥2 个机构超阈值，则剔除

#### 3.3.4 时序切分

严格按时间顺序切分数据（非随机切分），避免未来信息泄露：
- 训练集：时间最早的前 70%
- 验证集：中间 15%
- 测试集：最后 15%（OOT 样本）

#### 3.3.5 Optuna 超参搜索

使用 TPE 采样器最大化 AUC，搜索空间：
- LightGBM：n_estimators[50,500], lr[0.01,0.3], max_depth[3,10], num_leaves[8,64], min_child_samples[20,200], subsample, colsample, reg_alpha/lambda[0.01,10]
- XGBoost：类似空间 + scale_pos_weight[1,20]

### 3.4 配置体系

所有阈值和参数集中在 `config.py`，以模块级常量定义：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| MISSING_THRESHOLD | 0.6 | 整体缺失率阈值 |
| IV_THRESHOLD | 0.02 | 全局 IV 阈值 |
| ORG_IV_THRESHOLD | 0.02 | 单机构 IV 阈值 |
| MAX_ORG_LOW_IV | 2 | 允许的低 IV 机构数上限 |
| PSI_FEATURE_THRESHOLD | 0.10 | 特征 PSI 阈值 |
| NULL_IMPORTANCE_GAIN_THRESHOLD | 50 | Null Importance gain 差异阈值 |
| CORRELATION_THRESHOLD | 0.8 | 相关系数阈值 |
| AUC_THRESHOLD | 0.70 | AUC 达标线 |
| KS_THRESHOLD | 0.30 | KS 达标线 |
| PDO | 20 | 评分卡翻倍比 |
| BASE_SCORE | 600 | 评分卡基础分 |
| OPTUNA_TRIALS | 20 | Optuna 搜索轮数 |
| CV_FOLDS | 5 | 交叉验证折数 |

CLI 参数可覆盖任意配置项。

---

## 四、实现效果

### 4.1 端到端验证结果

使用 20,000 条合成样本（欺诈率 ~5%，3 个机构，12 个月），全链路测试通过：

| 指标 | 评分卡模型 | 树模型（LightGBM） |
|------|-----------|-------------------|
| AUC | ~0.72 | **0.77** |
| KS | ~0.35 | **0.50** |
| Gini | ~0.44 | **0.54** |
| Top10 Lift | ~3.5x | **4.2x** |

模型评估均通过配置的达标线（AUC ≥ 0.70, KS ≥ 0.30）。

### 4.2 特征筛选效果

6 步筛选流程在测试数据上的典型表现：
- 初始特征数：15
- 异常月份过滤：剔除异常月-机构组合
- 缺失率筛选：剔除高缺失特征
- IV 筛选：剔除弱预测力特征
- PSI 筛选：剔除跨时间不稳定特征
- Null Importance：剔除噪声特征
- 相关性筛选：剔除高相关冗余特征
- 最终保留特征数：8-12 个

### 4.3 监控能力

- **特征监控**：检测 PSI 漂移和缺失率变化，三级告警（HIGH/MEDIUM/LOW）
- **模型监控**：检测评分分布 PSI、AUC 衰减、坏账率偏移
- **基线管理**：支持保存训练期基线，与线上新数据自动对比

### 4.4 报告输出

- **Excel 清洗报告**（selection_report.xlsx）：多 Sheet 结构，汇总每步筛选的条件、保留/剔除的特征列表
- **JSON 评估报告**：包含所有指标、阈值对比、分群稳定性
- **CSV 文件**：Lift 表、特征重要性、机构统计

### 4.5 AI Skill 发布

- 仓库地址：https://github.com/Dimker1/risk-control-modeling
- skills.sh 平台：按安装量排名，支持自动索引
- 兼容平台：45+ 主流 AI 编程工具（Cursor、GitHub Copilot、Claude Code、OpenCode 等）
- 安装后 Agent 自动发现 SKILL.md 中的触发条件，按步骤调用 scripts/ 中的 CLI 工具

---

## 五、安装方式

### 5.1 AI Skill 安装（推荐）

适用于 Cursor、GitHub Copilot、Claude Code 等 AI 编程平台：

```bash
npx skills add https://github.com/Dimker1/risk-control-modeling -y -g
```

安装后：
- Skill 位置：`~/.agents/skills/risk-control-modeling`（universal）
- Claude Code 符号链接：`~/.claude/skills/`
- AI Agent 自动识别 SKILL.md，按指令引导用户完成建模流程

### 5.2 Python 环境准备

```bash
# 创建/激活虚拟环境
conda create -n risk python=3.10 -y
conda activate risk

# 安装依赖
pip install pandas numpy scikit-learn lightgbm xgboost optuna joblib tqdm openpyxl pyarrow
```

### 5.3 从源码使用

```bash
git clone https://github.com/Dimker1/risk-control-modeling.git
cd risk-control-modeling

# 设置 Python 路径
ENV=~/anaconda3/envs/llm/bin/python
OUT=/tmp/risk_test

# Step 1: 生成合成数据
$ENV scripts/generate_demo.py --output $OUT/raw_data.csv --n-samples 20000 --seed 42

# Step 2: 加载数据
$ENV scripts/load_data.py --input $OUT/raw_data.csv \
  --date-col apply_date --target-col label --org-col org_info --key-cols user_id \
  --output $OUT/loaded_data.parquet

# Step 3: 机构分析
$ENV scripts/org_analysis.py --input $OUT/loaded_data.parquet --output $OUT/org_analysis/

# Step 4: 特征筛选
$ENV scripts/feature_selection.py --input $OUT/loaded_data.parquet --output $OUT/fs/ --skip-null-importance

# Step 5: 时序切分
$ENV scripts/split_data.py --input $OUT/fs/filtered_data.parquet --output $OUT/split/

# Step 6: 评分卡建模
$ENV scripts/scorecard.py --train $OUT/split/train.parquet --val $OUT/split/val.parquet \
  --features $OUT/fs/selected_features.json --output $OUT/scorecard/

# Step 7: 树模型建模
$ENV scripts/tree_model.py --train $OUT/split/train.parquet --val $OUT/split/val.parquet \
  --features $OUT/fs/selected_features.json --output $OUT/tree_model/ \
  --model-type lgb --cross-validate

# Step 8: 评估与监控
$ENV scripts/evaluate.py evaluate \
  --model $OUT/tree_model/model.pkl --test $OUT/split/test.parquet \
  --features $OUT/fs/selected_features.json --output $OUT/evaluation/ \
  --eval-segment-stability
```

### 5.4 监控子命令

```bash
# 保存监控基线
$ENV scripts/evaluate.py save-baseline \
  --train $OUT/split/train.parquet \
  --features $OUT/fs/selected_features.json \
  --output $OUT/evaluation/baseline.json

# 特征监控（用新数据对比基线）
$ENV scripts/evaluate.py monitor-features \
  --new-data $OUT/split/test.parquet \
  --baseline $OUT/evaluation/baseline.json \
  --output $OUT/monitoring/feature_alerts.json

# 模型监控
$ENV scripts/evaluate.py monitor-model \
  --new-data $OUT/split/test.parquet \
  --model $OUT/tree_model/model.pkl \
  --baseline $OUT/evaluation/baseline.json \
  --output $OUT/monitoring/model_alerts.json
```

### 5.5 在 Python 代码中调用

```python
import sys
sys.path.insert(0, '/path/to/risk-control-modeling')

import config as cfg
from references.func import get_dataset, calculate_iv, calculate_psi
from references.analysis import run_feature_selection
from references.modeling import ScorecardModel, TreeModel, time_based_split
from references.evaluation import evaluate_model, monitor_features, monitor_model

# 加载数据
df = get_dataset('data/raw.csv', date_col='apply_date', y_col='label', org_col='org_info')

# 特征筛选
df_filtered, report_steps = run_feature_selection(df)

# 时序切分
train, val, test = time_based_split(df_filtered, test_size=0.15, val_size=0.15)

# 树模型训练
model = TreeModel(model_type='lgb')
model.train(train, val, features, 'new_target')

# 评估
metrics, lift_df = evaluate_model(test['new_target'], model.predict_proba(test))
```

---

## 六、与其他方案对比

| 能力 | 本项目 | awesome-copilot/datanalysis-credit-risk |
|------|--------|---------------------------------------|
| 数据清洗 | ✅ 多格式 + 缺失值标记 | ✅ 基础清洗 |
| 机构维度分析 | ✅ IV/PSI/缺失率按机构 | ❌ |
| 异常月份过滤 | ✅ | ❌ |
| Null Importance 去噪 | ✅ | ❌ |
| 评分卡建模 | ✅ WOE + LR | ❌ |
| 树模型建模 | ✅ LGB/XGB + Optuna | ❌ |
| 交叉验证 | ✅ 5-Fold | ❌ |
| 模型评估 | ✅ AUC/KS/Gini/Lift/PR | ❌ 仅变量筛选 |
| 特征监控 | ✅ PSI + 缺失率告警 | ❌ |
| 模型监控 | ✅ AUC 衰减 + 坏账率 | ❌ |
| Excel 报告 | ✅ 多 Sheet 清洗报告 | ❌ |
| AI Skill 格式 | ✅ SKILL.md 指令集 | ✅ |
| CLI 工具 | ✅ 8 个独立脚本 | ❌ |

本项目是 datanalysis-credit-risk 的全链路超集，覆盖从数据到监控的完整闭环。

---

## 七、依赖清单

```
pandas
numpy
scikit-learn
lightgbm
xgboost
optuna
joblib
tqdm
openpyxl
pyarrow
```

---

## 八、踩坑记录

1. 时序切分前必须按 new_date 排序，否则按行号切分会出现 0 行
2. IV 计算用决策树分箱比等频/等距更稳定
3. WOE 编码需 clip 到 [-5, 5]，缺失值填充 0
4. PSI 计算需对齐分箱（以 expected 分布的分位数为基准）
5. Null Importance 至少 5 次置换取中位数
6. 机构维度需关注「全局过关但个别机构不稳定」的特征
7. 异常月份必须先剔除，否则影响后续统计
8. 相关性筛选保留 IV 较高者
9. WOE 编码和树模型对分类特征需先 LabelEncoder
10. 分群评估时 y_pred_proba 必须对齐 df.index
11. openpyxl 需安装才能导出 Excel 报告
12. parquet 格式需要 pyarrow 依赖
13. 5000 样本以下异常月份过滤可能把所有数据过滤掉（建议 20000+）
14. 评分卡 LR 训练会产生 sklearn RuntimeWarning（matmul overflow），不影响结果
15. feature_selection.py 的 Excel 报告默认写到 config.py 指定的 data/reports/ 目录
16. npx skills add 安装后，脚本需用完整路径运行

---

## 九、License

MIT
