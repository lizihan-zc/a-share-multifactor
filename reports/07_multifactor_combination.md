# Notebook 07：多因子合成运行记录

## 固定设定

- 输入：factor_panel_industry_size_neutral.parquet（行业、市值联合中性化并标准化）。
- 因子：EP、ROE、12-1 动量、60 日低波动；每类一个因子。
- 方法：四因子等权；36 个月滚动平均 Rank IC 加权。
- 每因子最少 36 个有效历史 IC，允许负权重，绝对权重之和为 1；历史不足留空。
- 输入信号日：2018-01-31 至 2025-12-31，96 个月。
- 首个可用滚动权重日：2021-10-29。
- 初始选择窗口：2018-10-31 至 2021-09-30。
- 固定单因子基准：LOWVOL 60D，按初始历史平均 IC 选择。
- 共同评价信号日：2021-10-29 至 2025-11-28，50 个月。
- 最后信号月 2025-12-31 保留信号，不评价尚未实现的收益。

## 真实结果

百分比字段以小数表示；ICIR 不年化；Sharpe 按月度收益年化。

```text
                          Mean Rank IC    ICIR  Positive IC %  Q5 Annualized Return  LS Annualized Return  LS Sharpe  LS Max Drawdown  Mean Target Turnover  Mean Q5 Label Coverage  Min Q5 Label Coverage  Months  Turnover Months
Method                                                                                                                                                                                                                             
Training-selected single        0.0911  0.7522           0.74                0.1731                0.1465     1.1750          -0.1037                0.2880                  0.9992                 0.9943      50               49
Equal weight                    0.0509  0.4323           0.70                0.1494                0.0606     0.5722          -0.1869                0.2381                  0.9993                 0.9957      50               49
Rolling IC weight               0.0856  0.6775           0.76                0.1726                0.1243     0.9720          -0.1471                0.2521                  0.9994                 0.9955      50               49
```

滚动 IC 加权相对等权：ICIR 变化 +0.245，Q5−Q1 Sharpe 变化 +0.40，平均目标换手变化 +1.40%。

## 解释边界

- 分组先于连接标签；缺失标签不改变持仓。组收益在有效标签持仓内重新归一化，属于可观察标签条件收益。
- 结果表同时报告 Q5 平均与最低标签权重覆盖率，完整覆盖明细见 multifactor_quantile_returns.parquet。
- 目标换手不含价格漂移；首次建仓不计入换手均值；Q5−Q1 是未计成本及做空限制的研究型收益。
- 延续 close-to-close 标签定义；尚未按下一交易日开盘价模拟真实交易。
- 因子中性化不保证分组后的组合行业、市值严格中性。
- 当前为初始历史选基准、向前滚动估计的研究比较，未隔离最终测试集；前序 notebook 已展示全样本结果。
- 不据本次结果调整因子、窗口或方向；Notebook 08/09/10 继续处理组合、成本、约束与严格样本外评估。

## 复现

从项目根目录的 Python 环境运行 notebooks/07_multifactor_combination.ipynb。
输入依赖 Notebook 06 生成的联合中性化面板。单元测试：python -m pytest tests/test_multifactor.py -q。
