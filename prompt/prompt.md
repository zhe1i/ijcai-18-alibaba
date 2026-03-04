你是一个资深推荐/广告算法工程师+科研工程化专家。请基于天池 IJCAI-18 阿里妈妈搜广转化率预测数据集（round1_ijcai_18_train_20180301.txt, round1_ijcai_18_test_a_20180301.txt, round1_ijcai_18_test_b_20180418.txt）实现一个完整、可复现、可在面试中解释的端到端项目。

【任务定义】
- 目标：pCVR 二分类预测，输出 P(is_trade=1 | user, item, context, shop, query-derived features)
- 训练集：478,138 行，27 列（含 is_trade）
- 测试集：test_a 18,371 行，26 列；test_b 42,888 行，26 列
- 标签极度不平衡：is_trade=1 约 1.89%
- 存在 -1 缺失哨兵值（例如 user_gender_id 缺失约 2.70%）
- 评估关注：logloss 为主，AUC 辅助；同时输出概率可靠性（校准）

【必须交付物（工程 + 文档 + 结果）】
1) 一个可运行的 GitHub 风格项目结构（本地即可）：
   - README.md：包含任务背景、数据说明、方案概述、训练/推理命令、实验结果表、消融与结论、面试可讲点
   - requirements.txt / environment.yml
   - src/ 目录：数据处理、特征、模型、训练、评估、推理、校准、工具
   - configs/ 目录：yaml/json 配置文件（路径、超参、fold 设置）
   - outputs/ 目录：日志、模型权重、预测文件、评估报告、图表
2) 训练与推理脚本：
   - python -m src.train --config configs/train.yaml
   - python -m src.predict --config configs/predict.yaml --split test_a/test_b
3) 生成提交文件：
   - 生成符合天池提交格式的 csv/txt（包含 instance_id 与预测概率）
4) 评估报告（自动生成）：
   - 时间切分的 AUC、logloss（overall + 按天/按场景）
   - Reliability diagram / ECE（校准前后）
   - 关键特征/模块消融对比表（至少 5 行）
5) 代码必须可复现：
   - 固定随机种子
   - 训练日志完整（超参、数据统计、fold、指标）
   - 关键中间产物可缓存（parsed 多值字段、freq 统计、token vocab/哈希设置）

【方案总览（必须实现，不能只写 README）】
A. Baseline（用于对照与诊断）
- 实现 LightGBM baseline（或 CatBoost 二选一）：
  - 支持类目/属性/预测意图的“匹配率/覆盖率/Jaccard”等手工特征
  - 支持平滑 CVR（Beta-Binomial smoothing）统计特征（按 user/item/shop/category 等）
  - 严格时序验证，输出 AUC/logloss

B. 主模型（面试主讲模型）
- 实现 “AutoInt backbone + Multi-value set attention pooling + Drift-aware MoE + calibration”
  1) Field Embedding：
     - 将高基数离散字段（如 user_id/item_id/shop_id/brand_id 等）映射为 embedding
     - 使用 hashing trick 或频次截断（<min_freq 归入 OOV）来控制词表大小
     - 对 -1 缺失值：既要保留 -1 作为独立类别 embedding，也要额外加入 is_missing 二值特征（或等效处理）
     - 连续/等级字段：可分桶当离散 embedding，或保留为数值输入（写清楚选择与原因）
  2) Multi-value 字段建模：
     - 对 item_category_list、item_property_list、predict_category_property 进行解析（';' 分割）
     - 对每个多值集合做 token embedding
     - 用 attention pooling 生成集合向量（可使用 context 条件化：例如用 item 主类目或 query/predict embedding 作为 query 向量）
     - 同时实现手工匹配特征：predict 与 item 的类目/属性命中率、覆盖率、Jaccard、是否命中主类目等（作为额外 dense features）
  3) AutoInt：
     - 将所有 field embedding（含集合向量）作为 tokens 输入多层 self-attention
     - 支持多头注意力、残差、layer norm、dropout
     - 输出拼接后接 MLP 得到 shared representation
  4) Drift-aware MoE（非多任务 MoE，不强制 MMoE）：
     - 至少 3 个 experts：Normal / Event(Drift) / Long-tail
     - gating network 输入必须包含：
       - 时间/日期相关特征（day/hour 或其 embedding）
       - drift_score：按天分布漂移度量（实现一种即可：day-level CVR z-score 或 PSI 或 day embedding + 统计量）
       - long-tail 信号：log_freq_user / log_freq_item / log_freq_shop（从训练集统计）
     - gating 输出 softmax 权重，组合 experts 的 logits 或概率
     - 防止 expert 饥饿：加入 load balancing auxiliary loss（或 entropy regularization），并在日志里打印各 expert 平均权重
  5) 不平衡处理（必须）：
     - 实现至少一种：pos_weight 的加权 BCE 或 focal loss
     - 可选：负样本下采样（若做则必须提供概率校正/校准方案）
  6) 概率校准（必须）：
     - 使用验证集做 temperature scaling 或 Platt scaling（优先 temperature scaling）
     - 输出校准前后 logloss、ECE、reliability diagram
     - 校准必须在每个 fold 内做，不能泄漏
  7) 评估：
     - 输出 overall AUC/logloss
     - 输出按天/按漂移场景（Normal vs Drift）分组的 logloss 与 gap
     - 输出消融实验结果

【验证策略（必须严格）】
- 使用 time-based cross validation：
  - 若有 day 字段：按 day 排序，滚动验证（例如最后 K 天做验证，或多折滚动）
  - 每折必须仅用过去训练，未来验证（防泄漏）
  - 在每折内报告 AUC/logloss；最终报告均值与方差
- 同时输出一个 “random split” 指标作为对照（用于展示泄漏风险），但以 time split 为准

【消融实验（必须自动运行或可一键切换）】
至少包含以下开关（通过 config 控制）并输出表格：
1) baseline LGBM（手工+统计特征）
2) AutoInt without MoE（仅 shared head）
3) AutoInt + Multi-value attention（无 MoE）
4) AutoInt + MoE（无校准）
5) AutoInt + MoE + 校准（完整模型）
可选再加：
6) 去掉 match features
7) 去掉 long-tail expert 或去掉 load balancing loss

【工程要求】
- 语言：Python 3.10+（可用 PyTorch）
- 训练必须支持 GPU（如可用），同时提供 CPU fallback
- 加速：对多值字段解析与词表/哈希映射做缓存（例如 parquet/pickle）
- 代码风格：清晰模块化；函数加注释；关键超参在 config
- 日志：使用 logging + 保存 metrics.json；建议接入 tensorboard（可选）
- 生成图：reliability diagram、按天 CVR 曲线、专家权重随天变化（可选但加分）

【数据字段处理提示（请你自行从表头读取，不要硬编码列名）】
- 程序自动读取 txt 表头，识别：
  - label 列 is_trade（train 才有）
  - instance_id（用于输出提交）
  - 多值字符串列：包含分号 ';' 的列（通常是 item_category_list、item_property_list、predict_category_property）
  - 其余列按类型推断：int/float/string
- -1 当缺失哨兵：生成 is_missing_x 特征或把 -1 当独立类别

【输出格式】
- test_a/test_b 预测文件：pred_test_a.csv / pred_test_b.csv
  - 至少包含两列：instance_id, predicted_score（0~1）
- 同时保存 oof（out-of-fold）预测：oof.csv（用于校准与分析）

【README 必须包含的“面试可讲点”】
- 为什么用 time-based CV、防泄漏
- 极不平衡下为何要 pos_weight/focal + 校准
- AutoInt 学特征交互的直觉
- 多值集合注意力池化为何优于平均池化（并给一个可解释例子）
- 为什么 MoE 适合普通日/漂移日混合分布；gating 输入为什么选这些
- Long-tail expert 如何缓解长尾过拟合
- 消融结果：哪部分带来提升，漂移日 gap 如何变小

【你需要做的第一步】
- 在代码最开头定义数据路径（用户将替换为本机路径），并提供一个 data/README 指示如何放置文件。
- 运行 train 脚本后，应能在 outputs/ 下看到模型权重、日志、报告、预测文件。

请生成完整代码与文件结构，并确保可以直接运行。
如果你需要做合理假设（例如 day 字段名），必须在代码里自动检测或在 config 里可配置，并写在 README。