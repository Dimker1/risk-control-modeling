---
name: risk-control-modeling
description: 风控建模全链路指导 — 当用户需要做贷前风控建模、信用评分卡、特征筛选、模型评估、模型监控时使用此技能。适用场景：信贷风控、反欺诈特征工程、评分卡开发、模型上线前评估、特征漂移监控。覆盖从原始数据到上线监控的完整流程。
compatibility: Python 3.8+, pandas, numpy, scikit-learn, lightgbm, xgboost, optuna, joblib, openpyxl
---

# 风控建模全链路 — 贷前场景（Pre-loan Credit Scoring）

你是一个风控建模专家。当用户需要进行贷前风控建模相关工作时，按照以下流程逐步引导用户完成。不要跳步，每一步确认结果后再进入下一步。

## 何时使用此技能

- 用户提到"风控建模"、"评分卡"、"信用评分"、"特征筛选"、"IV/PSI"、"WOE编码"
- 用户需要做贷前/信贷/反欺诈的模型开发
- 用户需要评估模型（KS/AUC/Lift）或做特征/模型监控
- 用户有带标签的时序数据，需要做二分类风控模型

## 整体流程

风控建模全链路共 8 个步骤，必须按顺序执行（后续步骤依赖前序结果）：

```
Step 1: 数据准备 → Step 2: 数据加载 → Step 3: 机构分析
→ Step 4: 特征筛选（6步串联） → Step 5: 数据切分
→ Step 6: 评分卡建模 → Step 7: 树模型建模 → Step 8: 评估与监控
```

用户可以只做其中某几步，但必须按顺序。例如可以只做 Step 1-4（特征筛选），或只做 Step 5-8（建模评估）。

---

## Step 1: 数据准备

确认用户的数据是否就绪。需要：

- **有标签列**：二分类标签（0=好/1=坏），坏样本率通常 1%-10%
- **有时间列**：申请日期/观测日期，贷前场景必须用时序切分
- **有机构列**（可选）：渠道/机构信息，用于分群稳定性分析
- **样本量**：建议至少 10000 条，坏样本至少 500 条

如果用户没有数据，运行 `scripts/generate_demo.py` 生成合成数据用于演示。

## Step 2: 数据加载与格式化

运行 `scripts/load_data.py`，传入以下参数：

```bash
python scripts/load_data.py \
  --input <数据文件路径> \
  --date-col <时间列名> \
  --target-col <标签列名> \
  --org-col <机构列名，可选> \
  --output <输出目录>
```

脚本会自动完成：
- 读取数据（支持 parquet/csv/xlsx/pkl）
- 替换异常值为 NaN（-1, -999, -1111）
- 过滤无效标签、去重、删除常数特征
- 标准化列名为 new_date / new_target / new_org
- 日期统一为 YYYYMMDD，新增 new_date_ym 列

输出：`<output>/loaded_data.parquet`

**验证**：检查输出日志中的样本数、坏样本率、特征数是否合理。坏样本率若 <0.5% 或 >30%，需提醒用户。

## Step 3: 机构样本分析

运行 `scripts/org_analysis.py`：

```bash
python scripts/org_analysis.py \
  --input <output>/loaded_data.parquet \
  --output <output>/org_analysis
```

输出每月每机构的样本数、坏样本数、坏样本率。检查：
- 是否有机构样本量过少（<500/月）
- 是否有机构坏样本率异常偏高/偏低
- 是否有月份数据断档

如果发现问题，提醒用户决定是否剔除该机构/月份。

## Step 4: 特征筛选（6步串联）

这是风控建模最核心的步骤。运行 `scripts/feature_selection.py`：

```bash
python scripts/feature_selection.py \
  --input <output>/loaded_data.parquet \
  --output <output>/feature_selection \
  --missing-threshold 0.6 \
  --iv-threshold 0.02 \
  --psi-threshold 0.10 \
  --run-null-importance \
  --correlation-threshold 0.8
```

6步筛选按顺序执行：

| 顺序 | 筛选方法 | 剔除逻辑 | 默认阈值 |
|------|----------|----------|----------|
| 4.1 | 异常月份过滤 | 坏样本<10或总量<500的月份剔除 | MIN_YM_BAD=10, MIN_YM_TOTAL=500 |
| 4.2 | 缺失率筛选 | 整体缺失率>0.6 或过多机构缺失率过高 | 0.6 |
| 4.3 | IV筛选 | 整体IV<0.02 或过多机构IV过低 | 0.02 |
| 4.4 | PSI筛选 | 按机构按月PSI>0.10 的特征剔除 | 0.10 |
| 4.5 | Null Importance | 标签置换后重要性不低于原始，则为噪声 | gain差>50 |
| 4.6 | 相关性筛选 | 相关系数>0.8 的特征对，保留IV较高者 | 0.8 |

输出：
- `<output>/feature_selection/selected_features.json` — 入选特征列表
- `<output>/feature_selection/selection_report.xlsx` — 每步筛选明细
- `<output>/feature_selection/filtered_data.parquet` — 筛选后的数据

**验证**：检查剩余特征数。通常筛选后保留 20-80 个特征。如果剩余 <10 或 >200，提醒用户调整阈值。

**注意**：Null Importance 耗时较长（约 3-10 分钟），用户可加 `--skip-null-importance` 跳过。

## Step 5: 数据切分

运行 `scripts/split_data.py`：

```bash
python scripts/split_data.py \
  --input <output>/feature_selection/filtered_data.parquet \
  --output <output>/split \
  --train-ratio 0.7 \
  --val-ratio 0.15
```

**必须使用时序切分**（按时间排序，前70%训练，中15%验证，后15%测试）。不能用随机划分！
贷前场景有时间依赖性，随机划分会导致数据泄露。

输出：`train.parquet`, `val.parquet`, `test.parquet`

## Step 6: 评分卡建模

运行 `scripts/scorecard.py`：

```bash
python scripts/scorecard.py \
  --train <output>/split/train.parquet \
  --val <output>/split/val.parquet \
  --features <output>/feature_selection/selected_features.json \
  --output <output>/scorecard \
  --pdo 20 \
  --base-score 600 \
  --base-odds 50
```

评分卡流程：WOE编码（决策树分箱）→ 逻辑回归 → 评分映射

输出：
- `<output>/scorecard/model.pkl` — 模型文件
- `<output>/scorecard/woe_bins.json` — WOE分箱映射
- `<output>/scorecard/scorecard_report.json` — 评分卡报告

## Step 7: 树模型建模

运行 `scripts/tree_model.py`：

```bash
python scripts/tree_model.py \
  --train <output>/split/train.parquet \
  --val <output>/split/val.parquet \
  --features <output>/feature_selection/selected_features.json \
  --output <output>/tree_model \
  --model-type lgb \
  --cross-validate \
  --optuna-tune
```

- `--model-type`：`lgb`（LightGBM）或 `xgb`（XGBoost）
- `--cross-validate`：时序5折交叉验证
- `--optuna-tune`：Optuna 超参数调优（可选，耗时约 5-15 分钟）

输出：
- `<output>/tree_model/model.pkl` — 模型文件
- `<output>/tree_model/cv_report.json` — 交叉验证结果
- `<output>/tree_model/optuna_params.json` — 调参结果（如启用）

## Step 8: 评估与监控

### 8.1 模型评估

运行 `scripts/evaluate.py`：

```bash
python scripts/evaluate.py \
  --model <output>/tree_model/model.pkl \
  --test <output>/split/test.parquet \
  --features <output>/feature_selection/selected_features.json \
  --output <output>/evaluation \
  --eval-segment-stability \
  --org-col new_org
```

核心指标：AUC / KS / Gini / PR-AUC / Lift@10 / Precision / Recall / F1

**达标线**：AUC >= 0.70，KS >= 0.30。未达标需提醒用户排查原因。

### 8.2 监控基线保存 + 漂移检测

```bash
# 保存训练集基线
python scripts/monitor.py save-baseline \
  --train <output>/split/train.parquet \
  --features <output>/feature_selection/selected_features.json \
  --output <output>/monitor/baseline.json

# 特征监控（新数据 vs 基线）
python scripts/monitor.py monitor-features \
  --new-data <新数据路径> \
  --baseline <output>/monitor/baseline.json \
  --output <output>/monitor/feature_monitor.json

# 模型监控
python scripts/monitor.py monitor-model \
  --new-data <新数据路径> \
  --model <output>/tree_model/model.pkl \
  --baseline <output>/monitor/baseline.json \
  --output <output>/monitor/model_monitor.json
```

---

## 关键踩坑（Gotchas）

这些是 agent 必须遵守的硬性规则，违反会导致结果错误：

1. **时序切分必须按时间排序**，绝对不能用随机划分（数据泄露，AUC虚高）
2. **IV计算用决策树分箱**比等频/等距更稳定，不要让用户改用等频分箱
3. **WOE编码必须 clip 到 [-5, 5]**，极端WOE值会导致评分溢出
4. **PSI计算必须对齐分箱**，用训练集分位数作为 breakpoints
5. **异常月份必须先剔除**，否则后续 IV/PSI 计算受极端月份干扰
6. **相关性筛选保留 IV 较高者**，不要简单删除，避免丢掉强特征
7. **机构维度分析不能跳过**，全局过关但个别机构不稳定的特征必须关注
8. **Null Importance 至少跑 5 次置换**取中位数，单次结果随机性太大
9. **监控基线必须保存**，否则上线后无法做漂移检测
10. **评分卡 PDO=20 意味着** odds 翻倍时评分变化 20 分，需根据业务调整
11. **分类特征做 WOE 前需 LabelEncoder**，否则 DecisionTree 无法处理
12. **LightGBM 不接受 object 列**，需先转为 category 类型
13. **PSI 对分类特征用值分布而非分箱**，不要对类别型用连续型分箱逻辑

## 配置参考

当用户需要自定义阈值时，以下是所有可调参数及默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MIN_YM_BAD_SAMPLE | 10 | 每月最低坏样本数 |
| MIN_YM_SAMPLE | 500 | 每月最低总样本数 |
| MISSING_THRESHOLD | 0.6 | 整体缺失率阈值 |
| IV_THRESHOLD | 0.02 | 全局IV阈值 |
| ORG_IV_THRESHOLD | 0.02 | 单机构IV阈值 |
| MAX_ORG_LOW_IV | 2 | 最大容忍低IV机构数 |
| PSI_FEATURE_THRESHOLD | 0.10 | 特征PSI阈值 |
| MAX_MONTHS_RATIO | 1/3 | 最大不稳定月份比例 |
| MAX_UNSTABLE_ORGS | 6 | 最大不稳定机构数 |
| NULL_IMPORTANCE_GAIN_THRESHOLD | 50 | Null Importance gain差异阈值 |
| CORRELATION_THRESHOLD | 0.8 | 相关系数阈值 |
| AUC_THRESHOLD | 0.70 | AUC达标线 |
| KS_THRESHOLD | 0.30 | KS达标线 |

## 参考文档

- `references/func.py` — 数据加载、机构分析、缺失率/IV/PSI 计算、Excel 报告导出的核心函数实现
- `references/analysis.py` — 特征筛选 6 步流程的完整实现
- `references/modeling.py` — WOE 编码、评分卡、树模型、交叉验证、Optuna 调参
- `references/evaluation.py` — 模型评估指标、特征监控、模型监控

如需深入了解某个步骤的算法细节或自定义行为，可读取对应文件。
