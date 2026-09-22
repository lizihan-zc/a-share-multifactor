# 沪深A股多因子选股项目

## 核心研究问题

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

## 最终项目目录

```text
a-share-multifactor/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
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
│   │   ├── universe.py
│   │   ├── preprocess.py
│   │   ├── clean.py
│   │   └── label.py
│   ├── factors/
│   │   ├── size.py
│   │   ├── low_volatility.py
│   │   ├── neutralization.py
│   │   ├── construction.py
│   │   ├── diagnosis.py
│   │   └── ...
│   ├── evaluation/
│   │   ├── ic.py
│   │   ├── quantile.py
│   │   └── performance.py
│   ├── portfolio/
│   │   ├── single_factor.py
│   │   ├── signal.py
│   │   ├── weighting.py
│   │   └── backtest.py
│   └── utils/
│       └── ...
├── figures/
├── reports/
│   └── research_summary.md
└── tests/
```

原则：

- `notebooks/` 用于**研究、实验、解释**；
- `src/` 用于**复用逻辑**；
- 图统一输出到 `figures/`；
- 中间因子面板统一输出到 `data/processed/`。

---

# Notebook 01：数据理解与股票池构建

文件：`notebooks/01_data_universe.ipynb`

## 研究问题

> 我到底在研究哪些股票？这些股票在当时是否真的可交易？

## 数据范围

- 市场：沪深 A 股；
- 频率：日频行情 + 季度/TTM 基本面；
- 研究区间：2017-01-01 至 2025-12-31；
- 调仓频率：月频；
- 预测周期：未来 1 个月收益。
- 数据源：Tushare Pro。

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

用股票池资格字段标记特殊状态：

- ST / \*ST；
- 停牌股票；
- 上市不足 120 个交易日的新股；
- 调仓日无法正常买入的股票；
- 明显缺失核心因子数据的股票；
- 可选：每日成交额最低的 5%–10%。

## 输出

- 日频股票面板，每行是一只股票在一个交易日的价格、成交量、复权因子、市值和涨跌停状态；
- 月末股票面板，每行是一只股票在某个月末的行情、财务、行业、上市状态和股票池筛选结果。

将以上两者分别保存为

```text
data/processed/price_daily.parquet
data/processed/universe_monthly.parquet
```

注意二者都属于“不平衡面板”：并非每只股票在每个月都有记录，因为股票会新上市、退市、停牌或缺少数据。未来生成的完整研究面板应当是

```text
(date, stock_code)
+ 当时已知的因子
+ 行业和市值等控制变量
+ 是否可交易
+ future_return_1m
```

## 图表

- 每月股票池数量时间序列；
- 各行业股票数量分布；
- 市值分布 / log 市值分布；
- 每日/每月缺失率；
- 收益率极端值分布。

---

# Notebook 02：标签构造与数据清洗

文件：`notebooks/02_label_and_preprocessing.ipynb`

## 研究问题

> 在时点 $t$ 形成的信号，到底预测哪个未来收益？

必须先把 label（模型要预测的目标变量）定义清楚，再做因子。

## 标签定义

对于股票 $i$ 和月末调仓日 $t$，定义未来一个调仓周期收益：

$$
R_{i,t+1}=\frac{P^{adj}_{i,t+1}}{P^{adj}_{i,t}}-1
$$

并保存为 `future_return_1m` .

这里作出以下确定选择：

- **收益类型**：简单收益率，不使用对数收益率。
- **收益对象**：个股原始总收益，不减市场收益或行业收益。超额收益可在评估阶段另行构造，但不替换主标签。
- **持有周期**：从正式调仓日 $t$ 到下一次正式调仓日 $t+1$，通常约为一个自然月，但天数不固定。
- **价格时点**：两个调仓日的收盘价，即 close-to-close。
- **价格口径**：使用复权收盘价，`adjusted_close = close * adj_factor`。复权价的绝对尺度无关紧要，只要同一股票跨期口径一致。
- **计价单位**：标签以小数保存，例如 `0.05` 表示 5%，不保存成 `5`。
- **交易成本**：主标签不扣佣金、印花税、滑点和冲击成本；这些属于组合回测层，而不是股票收益预测目标。

该定义适合做因子 IC、分组收益和横截面预测研究，但不等同于完全可成交的策略收益：如果因子也使用了调仓日收盘信息，就不能声称在同一收盘价完成交易。正式组合回测应另用下一交易日开盘成交规则。

## 工作流

```
下载数据
↓
构造日频面板
↓
构造月度股票池(
	从交易日中提取月末调仓日
    ↓
	与股票信息直积得到月度骨架
		├── 相关字段: list_date,delist_date, 
        │   listing_trading_days
		└── 可用性字段: is_listed, has_price_record, 
            passes_listing_age
    ↓
	左连接月末调仓日行情
		└── 可用性字段: passes_liquidity, is_buyable
    ↓
	左连接 ST、停牌字段
		└── 可用性字段: is_st, is_suspended
    ↓
	左连接 point-in-time 历史行业和财务快照
		├── point-in-time: 报告期 ≤ 公布期 ≤ 可用期 ≤ 调仓日
		├── 相关字段: end_date, xxx_announcement_date, 
        │   financial_available_date
		└── 可用性字段:  has_core_data
	↓
    生成股票池资格字段和GP可用性字段
		├── 股票池资格字段: is_eligible
		└── GP可用性字段: has_gp_factor_data =
			   is_gross_profit_applicable 
               & has_gross_profit_data
)
↓
明确标签
↓
清洗数据
	├── 新增价格字段: adjusted_close
	├── 价格可用性字段: has_valid_ohlc,
	│	has_valid_label_price, has_invalid_label_price,
	│	has_valid_trading_value, has_valid_market_cap
	├── 财务时点可用性字段: has_future_financial_data, 
	│	has_point_in_time_financials
	├── 可用规则一致性字段: eligibility_rule_inconsistent
	├── 特殊状态字段: has_special_state, is_entry_blocked
	└── 起点清洗正确字段: is_clean_for_label
↓
构造标签(
	构造相邻调仓日映射 
	↓
    确定每条记录的终点日期
		└── 终点记录存在性字段: has_exit_record
	↓
    连接终点价格和停牌状态
	↓
    计算未来收益率的原始值
	↓
    区分有效、停牌、退市、缺行情和右截尾
		└── 问题标记和可用性字段: label_status,
            label_available, exit_blocked,
			is_in_label_sample = 
            is_clean_for_label & label_available
	↓
    保留合法的未来收益率
)
```

## 输出

```text
data/processed/price_daily_clean.parquet
data/processed/universe_monthly_clean.parquet
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

## 图表

- 未来 1 个月收益分布；
- 每年月收益波动变化；
- 横截面收益均值与标准差时间序列；
- 不同年份股票数量与缺失率。

---

# Notebook 03：因子构造

文件：`notebooks/03_factor_construction.ipynb`

## 3.1 价值因子 Value

### Earnings-to-Price

$$
EP_{i,t}=\frac{\text{Net Profit TTM}_{i,t}}{\text{Market Cap}_{i,t}}
$$

作用：衡量公司盈利相对于市场估值的高低。其他条件相同时，EP 越高通常表示股票价格相对于其盈利越便宜，因此常作为价值类选股信号；净利润为负时，其价值解释会减弱。

直觉：盈利相对于市值更高的股票可能更便宜。

### Book-to-Price

$$
BP_{i,t}=\frac{\text{Book Equity}_{i,t}}{\text{Market Cap}_{i,t}}
$$

作用：衡量股票市场价格相对于账面净资产的高低。其他条件相同时，BP 越高通常表示估值越低，可用于识别价值型股票；金融行业与其他行业的资产负债结构差异较大，研究时应关注行业可比性。

## 3.2 质量因子 Quality

### ROE

$$
ROE_{i,t}=\frac{\text{Net Profit TTM}_{i,t}}{\text{Average Equity}_{i,t}}
$$

作用：衡量公司利用股东投入资本创造利润的能力。ROE 越高通常表示资本使用效率和盈利质量越高，但负权益、异常杠杆和一次性损益可能削弱其解释力。

### Gross Profitability

$$
GP_{i,t}=\frac{\text{Gross Profit TTM}_{i,t}}{\text{Total Assets}_{i,t}}
$$

作用：衡量公司每单位资产创造毛利润的能力。GP 越高通常表示主营业务盈利能力和资产使用效率越强，可作为公司质量与基本面稳定性的信号。**该口径通常不适用于银行和非银金融公司。**

直觉：盈利质量较高的公司未来基本面可能更稳定。

## 3.3 动量因子 Momentum

### 12-1 Momentum

$$
MOM_{12-1}=\frac{P_{t-21}}{P_{t-252}}-1
$$

作用：衡量股票过去约一年、但不包含最近一个月的中期价格趋势。较高的因子值表示中期表现较强；剔除最近一个月可减少短期反转对中期动量信号的干扰。

### 3个月动量

$$
MOM_{3M}=\frac{P_t}{P_{t-63}}-1
$$

作用：衡量股票近期的价格趋势和相对强弱。因子值越高表示过去三个月表现越强，可与 12-1 动量比较不同回看周期的有效性和稳定性。

## 3.4 低波动因子 Low Volatility

过去 60 个交易日的波动性：

$$
VOL_{60}=\sigma(r_{t-59},...,r_t)
$$

同一方向使得因子值越大越好，定义低波动因子：

$$
LOWVOL=-VOL_{60}
$$

作用：衡量股票近期价格波动风险。取负号后，因子值越大表示波动越低，使其方向与“数值越大、预期表现越好”的统一约定一致，可用于研究低波动异象和风险暴露。

## 3.5 可选：流动性

Amihud illiquidity：该因子分两步计算。首先计算股票每天的价格冲击 ``daily_illiq[d] = abs(return[d]) / amount[d]``；然后在每个计算日，对最近 20 个连续交易日的 ``daily_illiq`` 取滚动平均：``ILLIQ_20[t] = mean(daily_illiq[t-19:t])``. 默认要求窗口内 20 个交易日全部连续
且数值有效，否则当日因子值为缺失。成交额 ``amount`` 的单位为元。

$$
ILLIQ_i=\frac{1}{D}\sum_{d=1}^{D}\frac{|r_{i,d}|}{\text{Amount}_{i,d}}
$$

作用：衡量单位成交额引起的价格变动幅度。因子值越高表示较小成交金额也可能造成较大价格变化，即流动性越差；可用于检验因子收益是否源于流动性风险补偿。

第一版可以先不加入多因子模型，只用于研究因子收益是否只是流动性风险补偿。

## 3.6 市值 Size：更适合作为控制变量

$$
SIZE=\ln(\text{Market Cap})
$$

作用：描述公司的市场规模，并压缩市值分布的右偏和数量级差异。本项目主要将 SIZE 用作中性化变量、风险暴露和稳健性检验变量，而不是默认视为主 Alpha 因子。

## 工作流

在每个调仓日横截面：

```
...
↓
清洗数据
↓
构造标签
↓
因子构造(
    计算因子的原始值
    ↓
    去极值
    ↓
    统一方向
    ↓
    (可选)中性化
        ├── 市值中性化
        ├── 行业中性化
        └── 市值和行业中性化
    ↓
    标准化
)
↓
...
```

## 输出

```text
data/processed/factor_panel_raw.parquet
data/processed/factor_panel_zscore.parquet
```

## 图表

对每个因子制作：

- 横截面分布；
- 时间序列均值；
- 时间序列标准差；
- 缺失率；
- 极值处理前后对比。

---

# Notebook 04：单因子诊断

文件：`notebooks/04_factor_diagnostics.ipynb`

## 研究问题

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

## 输出结果

| Factor | Mean Rank IC | ICIR | Positive IC % | IC t-stat |
| ------ | -----------: | ---: | ------------: | --------: |
| EP     |              |      |               |           |
| BP     |              |      |               |           |
| ROE    |              |      |               |           |
| GP     |              |      |               |           |
| MOM    |              |      |               |           |
| LOWVOL |              |      |               |           |

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

学生项目可明确说明暂设 $R_f=0$。

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
