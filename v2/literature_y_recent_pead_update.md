# 近期 PEAD / Post-announcement ACAR 文献索引更新

日期：2026-04-23

## 1. 底线判断

这条 Y 不是只能靠 Bernard and Thomas 1989 这种远古文献支撑。2020 年以后仍有不少论文直接使用 PEAD、post-announcement CAR/BHAR 或 earnings-announcement 后续漂移，且有高等级期刊锚点。

但要分清两层：

1. 近期 PEAD 文献大多仍是 **signed / directional drift**：按 earnings surprise 或公告窗口反应分组，观察公告后 CAR/BHAR 是否继续沿同一方向漂移。
2. 你现在跑出来更稳的是 **absolute post-announcement CAR**：`PEAD20_abs = abs(CAR[+2,+20])`、`PEAD60_abs = abs(CAR[+2,+60])`，衡量公告后还需要多大幅度的残余价格调整。

因此，正文不能写“本文首次提出 PEAD 类 Y”。更稳的写法是：

> 本文借鉴 PEAD 文献中用公告后 CAR/BHAR 衡量公告后价格调整的做法，但不关注盈余惊喜方向上的漂移收益，而使用公告后绝对累计异常收益衡量公告后残余价格调整幅度。

## 2. 可替换远古锚点的近期文献

| 文献 | 年份 / 期刊 | 用的 Y / 窗口 | 对本项目的作用 | 注意事项 |
|---|---|---|---|---|
| Liang and Zhang, "Post-earnings announcement drift and parameter uncertainty" | 2020, Review of Quantitative Finance and Accounting | PEAD / drift-period return | 说明 2020 后仍有人把 PEAD 当作核心 Y 研究，并从学习/参数不确定性解释公告后调整。 | 不是绝对 CAR，而是方向性 PEAD。 |
| He, "Credit rating, post-earnings-announcement drift, and arbitrage from transient institutions" | 2021, Journal of Business Finance and Accounting | PEAD | 用信用评级/信息不确定性解释 PEAD 横截面差异，适合支持“信息环境影响公告后调整”。 | 期刊层级弱于 RFS/JFQA/RAST，但逻辑贴近。 |
| Meursault, Liang, Routledge and Scanlon, "PEAD.txt: Post-Earnings-Announcement Drift Using Text" | 2023 issue, Journal of Financial and Quantitative Analysis | text-based PEAD | 很适合本项目：它说明公告文本/电话会文本中的信息也能产生公告后漂移，且近年 classic PEAD 接近 0 时 text-based PEAD 仍明显。 | 是文本 surprise 生成的 directional PEAD，不是 absolute CAR。 |
| Yang, Liu and Su, "Earnings communication conferences and post-earnings-announcement drift: Evidence from China" | 2023, Accounting and Finance | PEAD | 中国情境，研究业绩说明会/语调如何影响 PEAD。可作为中文市场/投资者解释不足的近期英文锚点。 | 发现召开业绩说明会反而可能增加 PEAD，需谨慎引用为“沟通质量”而非“沟通本身”。 |
| Barinov, Park and Yildizhan, "Firm complexity and post-earnings-announcement drift" | 2024, Review of Accounting Studies | SUE quintile / post-announcement CAR window | 高等级近期锚点。复杂企业 PEAD 更强，直接支持“信息处理成本越高，公告后价格吸收越慢”。 | 仍是 signed PEAD；但对“信息处理成本”理论非常好。 |
| Lan, Xie, Mi and Zhang, "Post earnings announcement drift: A simple earnings surprise measure, the medium effect of investor attention and investing strategy" | 2024, International Review of Financial Analysis | PEAD / investor attention / China market | 中国股票市场近期证据。用隔夜收益构造 earnings surprise，讨论投资者关注的中介作用。 | 偏投资策略/异常收益，不是披露治理论文。 |
| Hirshleifer, Peng and Wang, "News Diffusion in Social Networks and Stock Market Reactions" | 2025 issue, Review of Financial Studies | immediate reaction, post-announcement price drift, volatility decay | 顶级近期锚点。社会网络中心性提高即时价格反应、削弱公告后漂移，直接支持“更快信息扩散 -> 更弱公告后残余调整”。 | 不是数据资产披露；但机制非常贴合“信息传播效率”。 |
| Li, Wang and Huang, "The interactive quality of earnings communication conferences and post-earnings-announcement drift" | 2025, China Journal of Accounting Studies | BHAR/CAR after annual report disclosure, PEAD windows | 中国 A 股 + CSMAR/CNRDS + Word2Vec。很适合说明近年中国语境仍在用公告后 CAR/BHAR 做 PEAD。 | 期刊不是中文顶刊；可作为情境辅助，不宜当主文献锚点。 |

## 3. Absolute CAR / BHAR 的近期支撑

这些文献不一定使用“公告后 20/60 日 absolute CAR”，但能支撑 `abs(CAR)` / `abs(BHAR)` 作为非方向性市场反应或信息含量指标。

| 文献 | 年份 / 期刊 | 指标 | 对本项目的作用 |
|---|---|---|---|
| Capital market response to high quality annual reporting: evidence from UK annual report awards | 2022, Accounting and Business Research | earnings announcement / annual report release 周围的 `|CAR|` | 说明高质量报告研究可用绝对异常收益衡量公告信息含量或市场反应幅度。 |
| Barth, Berkovitch and Israeli, "Controlling the narrative: managers' topic-shifting behavior in conference calls" | 2026, Review of Accounting Studies | `BHAR_Absolute, Days [0,+2]` around conference calls | 高等级近期锚点。它明确把 absolute BHAR 用于“非方向性价格变化 / 新信息进入价格的总量”。 |

写作上可以这样接：

> 传统 PEAD 文献通常以 signed post-announcement CAR/BHAR 衡量市场是否沿盈余惊喜方向继续漂移。与此不同，本文采用 absolute post-announcement CAR，不判断漂移方向，而衡量公告后仍需吸收的信息量或残余价格调整幅度。该处理与近期披露研究中使用 absolute abnormal returns 捕捉非方向性信息含量的做法一致。

## 4. 建议进入索引的 Y 名称

中文变量名：

- 公告后绝对累计异常收益
- 公告后残余价格调整幅度

英文变量名：

- `Post-announcement ACAR`
- `Post-announcement absolute CAR`
- 如果使用 BHAR 公式，则写 `Post-announcement absolute BHAR`

不建议主变量名：

- `PEAD`：太容易被理解为 signed PEAD，会直接撞经典文献。
- `Absolute PEAD`：也容易让审稿人追问“PEAD 为什么可以取绝对值”。

建议变量解释：

> `Post-announcement ACAR` is defined as the absolute value of cumulative abnormal returns over the post-announcement window. It captures the magnitude of residual price adjustment after the earnings announcement, rather than directional drift conditional on the sign of earnings news.

## 5. 直接可放进 Y 索引的文献组合

若索引只能放 5-6 篇，不要再堆 1980s/1990s。建议如下：

1. Bernard and Thomas (1989) 或 Bernard and Thomas (1990)：只保留一个经典源头。
2. Meursault et al. (2023, JFQA)：近期文本 / 机器学习 PEAD。
3. Barinov et al. (2024, RAST)：近期会计高等级 PEAD + 信息处理成本。
4. Hirshleifer et al. (2025, RFS)：信息扩散削弱 post-announcement drift。
5. Yang et al. (2023, Accounting and Finance)：中国业绩说明会与 PEAD。
6. Li et al. (2025, China Journal of Accounting Studies)：中国 A 股、互动质量、PEAD。
7. Barth et al. (2026, RAST)：absolute BHAR 作为非方向性信息含量测度。

## 6. 对本项目的写法结论

保留当前实证判断：

- `PriceDelay` 仍适合做主 Y。
- `PEAD60_abs` 适合做“进一步结果 / 补充 Y”，因为它在两个 DU 测度下都显著为负。
- `PEAD20_abs` 可以作为短窗对照，但 60 日更稳。
- `PEAD_signed` 不支持，不要写成“降低传统 PEAD”。

推荐索引句：

> 在价格效率研究中，PEAD 文献长期使用公告后 CAR/BHAR 衡量市场对盈余信息的延迟吸收。近期研究仍持续使用这一框架，并将解释扩展到文本信息、企业复杂度、投资者关注、社会网络扩散和业绩说明会互动质量等场景。本文不以 signed PEAD 捕捉方向性漂移收益，而是使用公告后绝对累计异常收益衡量公告后残余价格调整幅度。若数据要素利用披露提升信息可理解性和可验证性，则公告后仍需通过价格继续修正的信息量应下降。

## 7. 检索来源

- Meursault et al. 2023 JFQA: https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/peadtxt-postearningsannouncement-drift-using-text/5EB217BB68B5FB054FE38541BAAC4679
- Barinov et al. 2024 RAST: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2360338
- Hirshleifer et al. 2025 RFS: https://academic.oup.com/rfs/article-abstract/38/3/883/7698199
- Yang et al. 2023 Accounting and Finance: https://ideas.repec.org/a/bla/acctfi/v63y2023i2p2145-2185.html
- Lan et al. 2024 IRFA: https://ideas.repec.org/a/eee/finana/v95y2024ipbs1057521924003922.html
- He 2021 JBFA: https://ideas.repec.org/a/bla/jbfnac/v48y2021i7-8p1434-1467.html
- Liang and Zhang 2020 RQFA: https://ideas.repec.org/a/kap/rqfnac/v55y2020i2d10.1007_s11156-019-00857-w.html
- Li et al. 2025 China Journal of Accounting Studies: https://www.tandfonline.com/doi/pdf/10.1080/21697213.2025.2473999
- Barth et al. 2026 RAST: https://link.springer.com/article/10.1007/s11142-026-09952-5
