# 沪深A股多因子选股项目：从研究假设到 README 的完整路线

> 目标：完成一个可以用于量化金融实习求职的研究型项目，而不是只做“跑通回测”的教程项目。  
> 最终交付物应包括：**可复现代码、研究 notebook、因子检验、组合回测、稳健性分析、图表、README、简历 bullet 和面试讲述框架**。

---

## 0. 项目定位

### 0.1 核心研究问题

本项目围绕一个朴素但足够完整的量化研究问题展开：

> **A股横截面收益在多大程度上可以被公开可获得的股票特征预测？不同因子是否包含互补信息？简单的多因子组合在严格的样本外检验、交易成本和可交易性约束下是否仍然有效？**

项目不是为了证明“某个策略一定赚钱”，而是为了展示完整的 Quant Research 流程：

1. 提出经济/统计假设；
2. 获得并清洗可用数据；
3. 构造因子；
4. 防止未来函数和数据泄漏；
5. 检验单因子预测能力；
6. 研究因子相关性和互补性；
7. 构造多因子模型；
8. 将预测信号转换成组合；
9. 加入交易成本和交易限制；
10. 做样本外和稳健性检验；
11. 对结果进行解释，而不是只展示收益曲线。

---

## 1. 最终项目目录

```text
a-share-multifactor/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_data_universe.ipynb
│   ├── 02_label_and_preprocessing.ipynb
│   ├── 03_factor_construction.ipynb
│   ├── 04_factor_diagnostics.ipynb
│   ├── 05_single_factor_backtest.ipynb
│   ├── 06_neutralization_and_robustness.ipynb
│   ├── 07_multifactor_combination.ipynb
│   ├── 08_portfolio_backtest.ipynb
│   ├── 09_transaction_costs_and_constraints.ipynb
│   └── 10_out_of_sample_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── universe.py
│   │   └── preprocess.py
│   ├── factors/
│   │   ├── value.py
│   │   ├── quality.py
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   └── transform.py
│   ├── evaluation/
│   │   ├── ic.py
│   │   ├── quantile.py
│   │   └── performance.py
│   ├── portfolio/
│   │   ├── signal.py
│   │   ├── weighting.py
│   │   └── backtest.py
│   └── utils/
│       └── plotting.py
├── figures/
├── reports/
│   └── research_summary.md
└── tests/
```

原则：

- notebook 用于**研究、实验、解释**；
- `src/` 用于**复用逻辑**；
- 所有核心计算不要只留在 notebook 里；
- 图统一输出到 `figures/`；
- 中间因子面板统一输出到 `data/processed/`。

---

# Notebook 01：数据理解与股票池构建

文件：`notebooks/01_data_universe.ipynb`

## 研究问题

> 我到底在研究哪些股票？这些股票在当时是否真的可交易？

A股量化项目最容易被忽略的问题之一，是股票池定义过于理想化。

## 数据范围建议

为了兼顾样本量和实现难度，可以先选择：

- 市场：沪深 A 股；
- 频率：日频行情 + 季度/TTM 基本面；
- 研究区间：建议至少 5–8 年；
- 调仓频率：月频；
- 预测周期：未来 1 个月收益。

数据源可以来自：聚宽 / JoinQuant、米筐 / RiceQuant、Tushare Pro、AKShare；如果学校或实验室有 Wind、CSMAR、RESSET，则优先使用。

如果使用免费数据，README 中必须明确数据覆盖和潜在缺陷。

## 最低需要字段

### 行情数据

```text
date
stock_code
open
high
low
close
prev_close
volume
amount
adj_factor
```

### 股票状态

```text
list_date
delist_date
is_st
is_suspended
limit_up
limit_down
industry
```

### 基本面数据

```text
market_cap
total_assets
total_equity
net_profit_ttm
revenue_ttm
gross_profit_ttm
report_period
announcement_date
```

## 股票池规则

第一版建议：

1. 排除 ST / \*ST；
2. 排除停牌股票；
3. 排除上市不足 120 个交易日的新股；
4. 排除调仓日无法正常买入的股票；
5. 排除明显缺失核心因子数据的股票；
6. 可选：排除每日成交额最低的 5%–10%。

不要使用“今天的成分股名单”回填历史。

## 必做图表

1. 每月股票池数量时间序列；
2. 各行业股票数量分布；
3. 市值分布 / log 市值分布；
4. 每日/每月缺失率；
5. 收益率极端值分布。

## 输出

```text
data/processed/universe_monthly.parquet
data/processed/price_daily.parquet
figures/universe_size.png
```

## 这一阶段要能回答

- 为什么选月频调仓？
- 为什么过滤上市不足 120 天的新股？
- ST 和停牌为什么必须处理？
- 你的股票池有没有 survivorship bias？

---

# Notebook 02：标签构造与数据预处理

文件：`notebooks/02_label_and_preprocessing.ipynb`

## 研究问题

> 在时点 \(t\) 形成的信号，到底预测哪个未来收益？

必须先把 label 定义清楚，再做因子。

## 推荐标签

以月末 \(t\) 作为信号形成时点：

$$
R_{i,t+1}=\frac{P_{i,t+1}}{P_{i,t}}-1
$$

其中 \(R\_{i,t+1}\) 是下一个调仓周期的收益。

更严谨时，可以使用：

> 下一个交易日开盘建仓 → 下个月调仓日开盘平仓。

这样能避免假设“在月末收盘价看到所有信息后还能按同一收盘价成交”。

## Point-in-time 基本面

不能直接把财报所属季度的数据放在季度末使用。

正确原则：

> **只能在公告日之后使用该财务数据。**

例如 2025Q1 财报即使属于 3 月 31 日，也不能默认在 3 月 31 日的回测中已知。

需要有：

```text
report_period
announcement_date
```

并按 `announcement_date <= signal_date` 做 point-in-time 匹配。

## 收益率清洗

检查：

- 除权除息是否使用复权价格；
- 停牌导致的价格不更新；
- 涨跌停的成交可实现性；
- 极端收益是否是真实事件还是数据错误。

## 必做图表

1. 未来 1 个月收益分布；
2. 每年月收益波动变化；
3. 横截面收益均值与标准差时间序列；
4. 不同年份股票数量与缺失率。

## 输出

```text
data/processed/monthly_panel.parquet
```

推荐最终面板：

```text
date
stock_code
industry
market_cap
future_return_1m
...
```

---

# Notebook 03：因子构造

文件：`notebooks/03_factor_construction.ipynb`

第一版不要追求几十个因子。建议只做 **4 类 6–8 个因子**，重点是把研究流程做干净。

## 3.1 价值因子 Value

### Earnings-to-Price

$$
EP_{i,t}=\frac{\text{Net Profit TTM}_{i,t}}{\text{Market Cap}_{i,t}}
$$

直觉：盈利相对于市值更高的股票可能更便宜。

### Book-to-Price

$$
BP_{i,t}=\frac{\text{Book Equity}_{i,t}}{\text{Market Cap}_{i,t}}
$$

注意：

- 净利润为负时 EP 的解释会变弱；
- 极端值需要处理；
- 金融股的财务结构特殊，可后续做行业稳健性分析。

## 3.2 质量因子 Quality

### ROE

$$
ROE_{i,t}=\frac{\text{Net Profit TTM}_{i,t}}{\text{Average Equity}_{i,t}}
$$

### Gross Profitability

$$
GP_{i,t}=\frac{\text{Gross Profit TTM}_{i,t}}{\text{Total Assets}_{i,t}}
$$

直觉：盈利质量较高的公司未来基本面可能更稳定。

## 3.3 动量因子 Momentum

### 12-1 Momentum

$$
MOM_{12-1}=\frac{P_{t-21}}{P_{t-252}}-1
$$

含义：使用过去约 12 个月收益，排除最近约 1 个月，避免短期反转污染中期动量。

### 3个月动量

$$
MOM_{3M}=\frac{P_t}{P_{t-63}}-1
$$

后续比较不同 lookback。

## 3.4 低波动因子 Low Volatility

过去 60 个交易日：

$$
VOL_{60}=\sigma(r_{t-59},...,r_t)
$$

为了统一“因子值越大越好”：

$$
LOWVOL=-VOL_{60}
$$

## 3.5 可选：流动性

Amihud illiquidity：

$$
ILLIQ_i=\frac{1}{D}\sum_{d=1}^{D}\frac{|r_{i,d}|}{\text{Amount}_{i,d}}
$$

第一版可以先不加入多因子模型，只用于研究因子收益是否只是流动性风险补偿。

## 3.6 市值 Size：更适合作为控制变量

$$
SIZE=\ln(\text{Market Cap})
$$

第一版不一定把 SIZE 当作主 alpha，更建议把它作为中性化变量、风险暴露和稳健性检验变量。

## 必做因子预处理

每个调仓日做横截面处理。

### Step 1：去极值

推荐 winsorization：`1% / 99%`，或者 MAD 方法。

### Step 2：标准化

$$
z_{i,t}=\frac{x_{i,t}-\mu_t}{\sigma_t}
$$

### Step 3：方向统一

统一为：因子值越大 → 预期未来收益越高。

## 必做图表

每个因子至少：

1. 横截面分布；
2. 时间序列均值；
3. 时间序列标准差；
4. 缺失率；
5. 极值处理前后对比。

## 输出

```text
data/processed/factor_panel_raw.parquet
data/processed/factor_panel_zscore.parquet
```

---

# Notebook 04：单因子诊断

文件：`notebooks/04_factor_diagnostics.ipynb`

不要上来就回测策略。先问：

> 因子是否真的具有横截面预测能力？

## 4.1 Rank IC

每个月计算：

$$
IC_t=Spearman(Factor_{i,t},Return_{i,t+1})
$$

使用 Spearman 而不是只看 Pearson，因为多数选股模型更关心排序。

计算：

```text
Mean IC
IC Std
ICIR
Positive IC Ratio
t-stat
```

其中：

$$
ICIR=\frac{Mean(IC)}{Std(IC)}
$$

## 4.2 IC 时间序列

必须画：

- 每月 Rank IC；
- 12 个月滚动平均 Rank IC；
- 累计 IC。

研究：

- 因子是否长期稳定？
- 是否只在某几年有效？
- 是否出现结构性失效？

## 4.3 因子相关性

计算横截面因子相关性，并取时间平均：

```text
EP
BP
ROE
GP
MOM
LOWVOL
SIZE
```

绘制 **Factor Correlation Heatmap**。

## 4.4 因子自相关 / Turnover 预警

计算：

$$
Corr(Factor_t,Factor_{t-1})
$$

如果因子排序每个月变化非常大，未来组合换手率可能很高。

## 必做表格

| Factor | Mean Rank IC | ICIR | Positive IC % | IC t-stat |
| ------ | -----------: | ---: | ------------: | --------: |
| EP     |              |      |               |           |
| BP     |              |      |               |           |
| ROE    |              |      |               |           |
| GP     |              |      |               |           |
| MOM    |              |      |               |           |
| LOWVOL |              |      |               |           |

所有数字必须由真实结果填入。

---

# Notebook 05：单因子分层回测

文件：`notebooks/05_single_factor_backtest.ipynb`

## 研究问题

> 因子的排序能力能否形成具有经济意义的组合收益？

## Quintile Portfolio

每个月：

1. 按因子排序；
2. 分成 5 组；
3. Q1 最低；
4. Q5 最高；
5. 计算下一期组合收益。

第一版先使用等权，避免过早把研究复杂化。

## 核心结果

### 1. 分组收益是否单调

绘制 Q1–Q5 年化收益柱状图。

理想情况不只是 Q5 高，而是整体具有一定单调性。

### 2. Long-Short

$$
R_{LS}=R_{Q5}-R_{Q1}
$$

计算：

- Annualized Return；
- Annualized Volatility；
- Sharpe Ratio；
- Maximum Drawdown；
- Win Rate。

### 3. 累计收益曲线

至少包括 Q1、Q5 和 Q5-Q1。

---

# Notebook 06：中性化与稳健性

文件：`notebooks/06_neutralization_and_robustness.ipynb`

这是很能拉开“课程项目”和“研究项目”差距的一步。

## 6.1 市值中性化

对原始因子做横截面回归：

$$
f_i=\alpha+\beta\ln(MarketCap_i)+\epsilon_i
$$

使用残差：

$$
f_i^{neutral}=\epsilon_i
$$

重新计算 Rank IC、ICIR 和分层收益，研究原始因子是否只是 small-cap effect。

## 6.2 行业中性化

$$
f_i=\alpha+\sum_k\beta_k Industry_{ik}+\epsilon_i
$$

也可以同时控制 `industry dummies + log market cap`。

比较：

```text
Raw Factor
Size Neutral
Industry Neutral
Industry + Size Neutral
```

## 6.3 子样本分析

按年份或按 Train / Validation / Test 拆分，避免只展示全样本结果。

## 6.4 不同股票池

比较：

- 全 A；
- 剔除小市值尾部；
- 高流动性股票；
- 沪深300 / 中证500风格股票池。

## 6.5 不同参数

Momentum：`3M / 6M / 12-1M`

Volatility：`20D / 60D / 120D`

目的不是挑最好看的参数，而是观察结论是否对参数选择非常敏感。

---

# Notebook 07：多因子合成

文件：`notebooks/07_multifactor_combination.ipynb`

## 研究问题

> 不同因子是否包含互补信息？组合后能否得到比单因子更稳定的预测能力？

## 方法 A：Equal Weight

$$
Score=\frac{1}{K}\sum_{k=1}^{K}z(Factor_k)
$$

例如组合 Value、Quality、Momentum 和 Low Volatility。

这是最重要的 baseline，不要一开始就上复杂模型。

## 方法 B：IC Weight

$$
w_k=\frac{\overline{IC}_k}{\sum_j|\overline{IC}_j|}
$$

只能使用当前时点以前的数据计算权重，否则会产生 look-ahead bias。

## 方法 C：ICIR Weight

$$
w_k\propto ICIR_k
$$

同样使用 rolling historical window。

## 比较

至少比较：

```text
Best Single Factor
Equal Weight
IC Weight
ICIR Weight
```

指标：Mean Rank IC、ICIR、Top Quantile Return、Long-Short Sharpe、Turnover。

---

# Notebook 08：组合构建与回测

文件：`notebooks/08_portfolio_backtest.ipynb`

把预测信号正式转换成投资组合。

## Baseline Portfolio

每月：

1. 计算 multi-factor score；
2. 对股票排序；
3. 买入 Top 10% / Top 20%；
4. 等权；
5. 持有一个月；
6. 下个月再平衡。

## Benchmark

可以选择 CSI 300、CSI 500 或同股票池等权组合。若指数数据获取困难，至少使用同股票池等权作为 benchmark。

## 必须计算

### 年化波动率

$$
AnnualizedVol=Std(r_d)\sqrt{252}
$$

### Sharpe

$$
Sharpe=\frac{AnnualizedReturn-R_f}{AnnualizedVol}
$$

学生项目可明确说明暂设 \(R_f=0\)。

### 最大回撤

$$
MDD=\max_t\left(1-\frac{NAV_t}{\max_{s\le t}NAV_s}\right)
$$

### Turnover

$$
Turnover_t=\frac{1}{2}\sum_i|w_{i,t}-w_{i,t-1}|
$$

## 必做图

1. 策略 vs benchmark 累计收益；
2. 超额收益；
3. drawdown；
4. rolling 12M Sharpe；
5. monthly return heatmap；
6. turnover 时间序列；
7. 行业暴露；
8. 市值暴露。

---

# Notebook 09：交易成本与现实约束

文件：`notebooks/09_transaction_costs_and_constraints.ipynb`

这一 notebook 向面试官展示：

> 你知道“预测能力”和“可交易 alpha”不是一回事。

## 9.1 简化交易成本模型

$$
Cost_t=Turnover_t\times c
$$

测试多组假设：

```text
0 bps
10 bps
20 bps
50 bps
```

不要声称某个具体成本就是实际机构交易成本，称为 sensitivity analysis。

## 9.2 涨跌停

如果买入日股票涨停，可假设无法买入；如果卖出日跌停，可假设无法卖出或延迟处理。

第一版实现太复杂时，至少明确说明这是研究限制。

## 9.3 流动性约束

可以研究：只保留成交额较高的股票后，结果是否仍然存在。

进阶时加入：

```text
portfolio_weight × portfolio_AUM < x% × ADV20
```

## 9.4 换手控制

比较：

```text
Monthly rebalance
Every 2 months
Quarterly rebalance
```

观察 alpha decay 与 transaction cost 的 trade-off。

---

# Notebook 10：严格样本外检验

文件：`notebooks/10_out_of_sample_analysis.ipynb`

这是整个项目最后的研究检验。

## 推荐时间划分

```text
Train:      前 60%
Validation: 中间 20%
Test:       最后 20%
```

具体年份根据数据范围确定。

重要的是：

> **Test period 在最终模型选择完成前不能用于调参数。**

## 样本外重新检查

- Mean Rank IC；
- ICIR；
- Top quantile return；
- Sharpe；
- Max Drawdown；
- Turnover；
- transaction-cost-adjusted return。

## Walk-forward

如果时间允许，进一步做：

```text
过去 3 年估计因子权重
→ 下一年交易
→ 向前滚动
```

这比一次性的 train/test split 更贴近真实量化研究。

---

# 11. 最终必须出现的图表清单

最终 README 不需要放所有图，建议精选 8–10 张最有解释力的。

## 数据质量

1. 股票池数量时间序列。

## 因子研究

2. 因子 Rank IC summary；
3. Rank IC 时间序列；
4. Factor correlation heatmap；
5. Quintile return monotonicity。

## 组合结果

6. Strategy vs Benchmark cumulative NAV；
7. Excess return cumulative NAV；
8. Drawdown；
9. Rolling Sharpe；
10. Turnover / Transaction cost sensitivity。

如果 README 太长，只展示：IC summary、Quintile return、Cumulative NAV、Drawdown、Transaction cost sensitivity；其他图放到 `reports/research_summary.md`。

---

# 12. 建议的统一评估函数

不要每个 notebook 手写指标。

放到 `src/evaluation/`：

```python
calc_rank_ic()
calc_ic_summary()
calc_quantile_returns()
calc_annualized_return()
calc_annualized_vol()
calc_sharpe()
calc_max_drawdown()
calc_turnover()
```

因子处理放到 `src/factors/transform.py`：

```python
winsorize_cross_section()
zscore_cross_section()
neutralize_factor()
```

这能体现你的代码工程能力。

---

# 13. README 模板

```markdown
# A-Share Multi-Factor Equity Selection

## 1. Motivation

研究 A 股横截面收益是否能够被价值、质量、动量和低波动等公开股票特征预测，
以及不同因子的组合是否能在严格样本外检验和交易成本后保持预测能力。

## 2. Research Questions

1. 单因子是否具有稳定 Rank IC？
2. 因子收益是否具有单调性？
3. 中性化市值和行业后是否仍有效？
4. 不同因子是否包含互补信息？
5. 多因子是否优于最佳单因子？
6. 交易成本后结果是否仍有经济意义？

## 3. Data

市场：
频率：
时间范围：
数据源：

股票池过滤：

- ST
- 停牌
- 新股
- 流动性

## 4. Factors

### Value

EP / BP

### Quality

ROE / Gross Profitability

### Momentum

12-1 Momentum

### Risk

60D Low Volatility

## 5. Methodology

Winsorization
→ Z-score
→ Industry/Size Neutralization
→ Rank IC
→ Quantile Portfolio
→ Multi-factor Score
→ Portfolio Backtest
→ Transaction Cost
→ Out-of-Sample Test

## 6. Single-Factor Results

放真实 IC summary 表。

## 7. Multi-Factor Results

放真实结果。

## 8. Robustness

- Size neutralization
- Industry neutralization
- Subperiod
- Liquidity universe
- Parameter sensitivity

## 9. Transaction Costs

展示不同成本假设。

## 10. Limitations

例如：

- 免费数据可能存在历史覆盖问题；
- 简化成交模型；
- 未完整模拟冲击成本；
- 未使用真实机构组合风险模型。

## 11. Reproduction

pip install -r requirements.txt

## 12. Repository Structure

展示代码目录。
```

---

# 14. README 中最重要的研究结果表

## Factor Summary

| Factor | Mean Rank IC | ICIR | Positive IC | LS Sharpe |
| ------ | -----------: | ---: | ----------: | --------: |
| EP     |         实测 | 实测 |        实测 |      实测 |
| BP     |         实测 | 实测 |        实测 |      实测 |
| ROE    |         实测 | 实测 |        实测 |      实测 |
| MOM    |         实测 | 实测 |        实测 |      实测 |
| LOWVOL |         实测 | 实测 |        实测 |      实测 |

## Portfolio Summary

| Portfolio                 | Ann. Return | Ann. Vol | Sharpe | Max DD | Turnover |
| ------------------------- | ----------: | -------: | -----: | -----: | -------: |
| Benchmark                 |             |          |        |        |          |
| Best Single Factor        |             |          |        |        |          |
| Equal-weight Multi-factor |             |          |        |        |          |
| IC-weight Multi-factor    |             |          |        |        |          |

不要伪造数字。结果不好，也照实展示并解释。

---

# 15. 如何把项目写进简历

不要在项目没跑完之前写具体收益数字。

## 示例版本 A：偏 Quant Research

**A股多因子选股研究｜Python**

- 基于 A 股日频行情及 point-in-time 基本面数据构建月频横截面研究框架，设计价值、质量、动量及低波动等因子，并完成去极值、标准化、行业及市值中性化。
- 使用 Rank IC、ICIR、分层组合及多空收益评估单因子预测能力，比较等权、IC 加权等多因子合成方法，并分析因子相关性及样本期稳定性。
- 构建 Top-N / Top-Quantile 月度调仓组合，加入换手率、交易成本、停牌及流动性约束，使用严格时间切分进行样本外检验。
- 【完成项目后填写真实结果】：样本外 Rank IC = XX，策略 Sharpe = XX，最大回撤 = XX%，在 XX bps 成本假设下年化超额收益 = XX%。

## 示例版本 B：偏 Data / ML

**A股多因子横截面收益预测｜Python, pandas, NumPy**

- 构建股票 × 日期面板数据，完成行情、基本面及行业数据的时点对齐，避免财报公告日导致的未来信息泄漏。
- 实现可复用的因子清洗、横截面标准化、中性化、IC 评估和组合回测模块，并通过参数敏感性及滚动样本外测试分析策略稳健性。
- 【真实结果】。

---

# 16. 面试时的 3 分钟讲法

不要按 notebook 顺序念，按“问题 → 方法 → 发现 → 局限”讲。

## 1. Motivation

> 我想研究 A 股横截面收益在多大程度上可以被公开股票特征预测，而不是先选一个模型再套股票数据。

## 2. Method

> 我首先构造价值、质量、动量和低波动因子，通过 Rank IC 和分层组合测试单因子，然后控制市值和行业暴露，最后比较等权与历史 IC 加权的多因子组合。

## 3. Backtest

> 信号在月末形成，组合下一期持有一个月。我特别处理了财报公告日、ST、新股、停牌以及交易成本，避免明显的未来函数和不可交易假设。

## 4. Result

用真实结果回答：

- 哪个因子最稳定？
- 哪个因子全样本有效但样本外失效？
- 多因子是否提高 ICIR？
- 收益有多少被交易成本吃掉？

## 5. Limitation

主动说：

> 当前项目仍使用简化成交模型，没有模拟完整市场冲击，也没有机构级风险模型。

---

# 17. 面试官最可能追问的问题

## 数据

1. 复权价格怎么处理？
2. 财务数据如何防止 look-ahead bias？
3. 如何避免 survivorship bias？
4. 停牌和涨跌停如何处理？

## 因子

5. 为什么用 Rank IC 而不是 Pearson IC？
6. 为什么 momentum 要跳过最近一个月？
7. 为什么需要 winsorization？
8. z-score 是否会泄漏未来信息？
9. 为什么要市值中性化？
10. 为什么要行业中性化？

## 回测

11. 为什么月频？
12. 为什么等权？
13. turnover 怎么定义？
14. 交易成本怎么进入收益？
15. benchmark 为什么这样选？

## 统计

16. IC = 0.03 到底算不算好？
17. t-stat 是否考虑 IC 自相关？
18. 多重检验问题如何处理？
19. 如何判断是不是过拟合？
20. 为什么全样本表现好不代表有预测力？

## 投资

21. 低波动为什么可能有效？
22. value 为什么可能长期失效？
23. momentum 在 A 股有什么特殊风险？
24. 因子收益是否可能只是风险补偿？
25. alpha 和 factor exposure 的区别是什么？

---

# 18. 项目执行顺序

不要一次把所有 notebook 都做完再回头检查。

## Milestone 1：数据可靠

完成：`01 + 02`

验收标准：

- 能随便抽一个股票和日期，解释每个字段来自哪里；
- 基本面数据使用公告日；
- future return 没有错位；
- 股票池没有明显使用未来信息。

## Milestone 2：单因子研究完整

完成：`03 + 04 + 05 + 06`

验收标准：至少有 4 类因子，并能回答：

```text
哪个 IC 最稳定？
哪个分层最单调？
哪个受 size 影响最大？
哪个换手最高？
```

## Milestone 3：形成投资组合

完成：`07 + 08 + 09`

验收标准：

```text
为什么组合这些因子？
组合比单因子好在哪里？
交易成本影响多大？
```

## Milestone 4：研究可信

完成：`10 + README + research_summary.md`

README 中明确区分：`In-sample / Validation / Out-of-sample`，并列出项目限制。

---

# 19. 一个“及格项目”和“优秀项目”的区别

## 及格

```text
下载数据
→ 算因子
→ 排序
→ 回测
→ Sharpe 很高
```

## 优秀

```text
提出 hypothesis
→ point-in-time 数据
→ universe construction
→ factor definition
→ IC
→ quantile monotonicity
→ factor correlation
→ neutralization
→ robustness
→ portfolio construction
→ turnover
→ transaction cost
→ out-of-sample
→ limitations
```

真正要追求的是第二条路线。

---

# 20. 可选进阶：项目完成后再加机器学习

不要在多因子基础研究没完成前加入 LightGBM。

先建立 baseline：

$$
Score=\sum_k w_k Factor_k
$$

之后再问：

> 如果因子与未来收益存在非线性关系，机器学习能否改善样本外预测？

可以新增：

```text
11_ml_cross_sectional_prediction.ipynb
```

比较：

```text
Equal Weight
Linear Regression
Ridge
LightGBM
```

评价指标仍以 Rank IC、ICIR、Portfolio Return、Turnover、Out-of-Sample Performance 为主，而不是只看 MSE / R²。

这可以自然发展为第二个项目：**机器学习横截面收益预测**。

---

# 21. 推荐的最小版本

如果时间有限，不需要一开始全部实现。

最小可投简历版本：

```text
01 Data & Universe
02 Label
03 Factor Construction
04 IC Diagnostics
05 Quintile Backtest
07 Equal-weight Multi-factor
08 Portfolio Backtest
09 Transaction Costs
10 Out-of-Sample
README
```

因子只需要：

```text
EP
ROE
12-1 Momentum
60D Low Volatility
```

先把四个因子做扎实。

完成以后再增加：

```text
BP
Gross Profitability
Neutralization
IC Weight
Walk-forward
Machine Learning
```

---

# 22. 最后的项目自检清单

## Data

- [ ] 使用复权价格
- [ ] 财务数据按公告日可得
- [ ] 没有使用未来成分股
- [ ] ST / 停牌 / 新股有明确规则
- [ ] 缺失值处理有说明

## Factor

- [ ] 因子公式写清楚
- [ ] 因子方向统一
- [ ] 去极值
- [ ] 标准化
- [ ] IC / Rank IC
- [ ] 分层回测
- [ ] 因子相关性

## Portfolio

- [ ] 明确信号形成时间
- [ ] 明确交易时间
- [ ] benchmark
- [ ] turnover
- [ ] transaction cost
- [ ] drawdown

## Research Integrity

- [ ] Train / Validation / Test 时间切分
- [ ] 参数不是根据 test period 调出来的
- [ ] 展示失败或不稳定的结果
- [ ] 讨论 limitations
- [ ] 没有只挑最好看的图

## Engineering

- [ ] notebook 可以从头运行
- [ ] 核心函数放入 `src/`
- [ ] `requirements.txt`
- [ ] GitHub 不上传大体积原始数据
- [ ] README 给出复现说明

## Resume

- [ ] 每条 bullet 都是“问题/方法/结果”
- [ ] 只写真实指标
- [ ] 能解释每一个数字
- [ ] 能回答数据泄漏和交易成本问题

---

# 23. 最终目标

项目完成后，你应该能够不看代码回答：

> **我研究了什么？为什么值得研究？数据在当时是否真的可得？因子为什么可能有效？如何判断它有预测力？如何证明不是市值/行业暴露？如何从信号转成组合？交易成本后还剩多少？样本外是否仍有效？哪些结果失败了？**

如果这些问题都能回答，这个项目就已经不再是“量化入门练习”，而是一份可以在量化研究实习面试中深入展开的完整研究项目。
