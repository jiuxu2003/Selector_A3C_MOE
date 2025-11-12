# MoE模型图表说明文档

**文档版本**：v2.1  
**创建日期**：2025年11月4日  
**最后更新**：2025年11月7日  
**用途**：论文配图与技术报告

---

## 📊 图表总览

本目录包含**6个核心图表**，展示Selector+MoE模型在网络防御任务中的性能、架构特点和动态决策能力。

**图表生成脚本**：`../generate_figures.py`  
**数据收集脚本**：`../collect_activation_data.py`

**输出目录**：
- 中文版图表：`figures/zh/`
- 英文版图表：`figures/en/`

---

## 📝 图表标题（中英文对照）

**注意**：所有图的标题已从图中移除，标题信息仅在此文档中提供，符合论文格式要求。

| 图号 | 中文标题 | English Title |
|------|---------|---------------|
| **图1** | 九场景性能评估 - Selector + MoE模型 | Nine-Scenario Performance Evaluation - Selector + MoE Model |
| **图2** | Selector子网选择策略分析 | Selector Subnet Selection Strategy Analysis |
| **图3** | 三个专家的动作类型分布（B_line-50场景） | Action Type Distribution of Three Experts (B_line-50 Scenario) |
| **图4** | 三个Expert的训练曲线（子网奖励） | Training Curves of Three Experts (Subnet Rewards) |
| **图5** | Selector训练过程 - Epsilon衰减与Q表增长 | Selector Training Process - Epsilon Decay & Q-table Growth |
| **图6** | 专家激活模式与威胁演化（Meander-100场景） | Expert Activation Pattern with Threat Evolution (Meander-100 Scenario) |

### 🔍 附加对比图（vs_cardiff）

为配合新型红队智能体实验，`vs_cardiff/figures/` 目录新增两张成对的中英文对比图：

| 图号 | 中文标题 | English Title | 说明 |
|------|---------|---------------|------|
| **附图A** | DeceptiveRedAgent欺骗攻击下的防御表现对比 | Defensive Performance vs. DeceptiveRedAgent | 比较我们的模型与Cardiff冠军在30/50/100步场景下的平均奖励与标准差，显示我们模型稳定领先。 |
| **附图B** | SleepDeceptiveRedAgent潜伏攻击下的防御表现对比 | Defensive Performance vs. SleepDeceptiveRedAgent | 采用对数坐标展示平均奖励绝对值，凸显Cardiff在潜伏型欺骗面前损失指数级攀升，而我们模型保持稳定。 |

**文件位置**：
- 中文版：`vs_cardiff/figures/zh/deceptive_comparison.png`、`vs_cardiff/figures/zh/sleep_deceptive_comparison.png`
- 英文版：`vs_cardiff/figures/en/deceptive_comparison.png`、`vs_cardiff/figures/en/sleep_deceptive_comparison.png`

**阅读建议**：
- 附图A 对应文档核心发现“Deceptive场景优势倍数 3.1×→6.4×”。
- 附图B 支持“SleepDeceptive 场景优势 34.8×”及“Cardiff 指纹识别失效”结论。

### ⚙️ 关于威胁评估的说明

**决策系统**：我们的模型使用**Q-learning Selector**进行子网选择决策
- 基于Q表（47个状态）学习最优防御策略
- 评估时采用贪婪策略（epsilon=0）

**可视化威胁分数**：图6中的"威胁等级"来自**Hardcoded Selector**
- 用途：仅用于可视化，帮助理解攻击态势
- 计算：基于受损主机数、权限等级、子网重要性等明确规则
- 优势：具有直观的物理意义，便于分析和解释
- 注意：实际防御决策由Q-learning Selector做出，不依赖这些分数

---

## 图表详细说明

### 图1：九场景性能对比 (`fig1_nine_scenario_performance.png`)

**用途**：展示模型在9个测试场景下的综合性能，并与基线智能体对比

**关键信息**：
- **对比智能体**：
  - 我们的模型（MoE）：Selector + MoE混合专家模型
  - BlueReactAgent：CybORG内置的反应式智能体（使用Remove动作）
  - SleepAgent：CybORG内置的休眠智能体（始终执行Sleep动作）
- **场景分类**：
  - B_line场景（橙色背景）：激进型攻击，快速直达关键资产
  - Meander场景（紫色背景）：渐进式攻击，横向移动为主
  - Sleep场景（灰色背景）：零攻击场景，测试资源效率
- **步数**：30/50/100步，测试不同时间尺度
- **性能指标**：平均奖励（含标准差误差线）

**观察到的模式**：
- 我们的模型在所有场景下均优于基线智能体
- Sleep场景下，我们的模型接近SleepAgent的性能（近零损失）
- B_line和Meander场景下，我们的模型显著优于BlueReactAgent
- 步数增加，所有智能体的损失都有上升，但我们的模型保持相对稳定

**论文使用建议**：
- 放在实验结果第一部分，展示整体性能和对比优势
- 重点强调Sleep场景的资源效率（接近理论最优）
- 突出在攻击场景下的显著优势
- 说明不同时间尺度下的稳定性

**数据来源**：
- 我们的模型：`final_result/evaluation_results_*.pkl`
- 基线智能体：`baseline_agents_evaluation_*.pkl`（需先运行 `evaluate_baseline_agents.py`）

---

### 图2：子网选择频率 (`fig2_subnet_selection_frequency.png`)

**用途**：展示Selector在不同场景下的子网选择策略

**关键信息**：
- **三个子网**：
  - User子网（蓝色）：入口防御，优先级最高
  - Enterprise子网（绿色）：横向移动防御
  - Operational子网（红色）：关键资产防御
- **选择模式**：
  - B_line/Meander场景：User占65-75%，Enterprise占18-22%
  - Sleep场景：User占97-98%（资源节约策略）

**观察到的模式**：
- User子网始终占主导地位（入口防御优先）
- Sleep场景下智能资源分配
- 威胁驱动的动态选择机制

**论文使用建议**：
- 放在架构分析部分，解释Selector的决策逻辑
- 强调Sleep场景下的智能资源分配
- 展示威胁驱动的动态选择机制

**数据来源**：基于技术报告6.3节的统计数据

---

### 图3：专家动作分布对比 (`fig3_expert_action_distribution.png`)

**用途**：对比三个Expert的动作类型分布差异

**关键信息**：
- **User Expert（保守策略）**：Sleep占38%，重视资源效率
- **Enterprise Expert（平衡策略）**：Analyse占42%，重视情报收集
- **Operational Expert（激进策略）**：Remove占35%+Restore占25%，重视主动防御

**观察到的模式**：
- 三个Expert策略差异化明显
- 从保守到激进的梯度设计
- 体现了MoE架构的专家分工优势

**论文使用建议**：
- 放在模型架构部分，展示专家分工
- 突出三个Expert的策略差异化
- 配合文字说明"从保守到激进"的梯度设计

**数据来源**：基于技术报告6.4节的B_line-50场景统计

---

### 图4：三个Expert训练曲线 (`fig4_training_curves.png`)

**用途**：展示三个Expert的训练收敛过程

**关键信息**：
- **横轴**：训练步数（Steps）
- **纵轴**：子网平均奖励
- **训练量**：
  - User & Enterprise Expert：3M steps (30k episodes)
  - Operational Expert：1M steps (10k episodes，横坐标拉伸3倍对齐)
- **收敛特点**：
  - User Expert收敛最快（低威胁环境）
  - Enterprise Expert收敛适中
  - Operational Expert收敛较慢但最终性能最高（高威胁环境）

**观察到的模式**：
- 不同Expert在不同威胁环境下的训练特点
- Operational的训练时间较短（1M steps）但已充分收敛
- 所有Expert最终都达到稳定状态

**论文使用建议**：
- 放在训练过程部分
- 强调不同Expert在不同威胁环境下的训练特点
- 说明Operational的训练时间较短但已充分收敛

**数据来源**：
- `final_result/checkpoints/user_enterprise_expert_metadata.pkl`
- `final_result/checkpoints/operational_expert_metadata.pkl`

**技术细节**：
- 使用高斯平滑（window=50）减少波动
- y轴范围自动调整（5th-95th百分位+10%边界）

---

### 图5：Selector训练曲线 (`fig5_selector_training_curve.png`)

**用途**：展示Selector的Q-learning训练过程

**关键信息**：
- **上图**：Epsilon衰减曲线
  - 初始值：1.0（完全探索）
  - 最终值：0.1761（平衡探索与利用）
  - 训练阶段：探索期→过渡期→利用期
- **下图**：Q-table状态空间增长
  - 最终大小：47个状态
  - 饱和点：约6M steps（状态空间充分探索）
- **训练量**：84000 episodes (8.4M steps)

**观察到的模式**：
- Q-learning快速收敛特性
- 状态空间相对较小（47个状态），易于训练
- Epsilon衰减曲线平滑，避免过早收敛

**论文使用建议**：
- 放在Selector训练部分
- 强调Q-learning的快速收敛特性
- 说明状态空间相对较小（47个状态），易于训练

**数据来源**：`final_result/checkpoints/selector_qtable.pkl`

---

### 图6：专家激活热力图 (`fig6_expert_activation_heatmap.png`)

**用途**：展示单个episode中Expert选择与威胁演化的关系（✅ 真实数据）

**关键信息**：
- **上图**：热力图展示各时间步选择的Expert和威胁级别
  - 颜色深度代表威胁强度
  - 纵轴：User/Enterprise/Operational Expert
- **下图**：三个子网的威胁级别时间线
- **场景**：Meander-100，Episode 8
- **实际选择**：User=53步(53%), Enterprise=31步(31%), Operational=16步(16%)
- **特点**：三个子网均有显著激活，完整展示三层防御体系

**威胁等级（Threat Level）说明**：
- **定义**：威胁等级是威胁分数的归一化版本，范围为 [0, 1]
- **计算方式**：`威胁等级 = 威胁分数 / 30.0`（假设最大威胁分数为30）
- **用途**：用于热力图的颜色映射，便于可视化比较
- **来源**：由Hardcoded Selector的威胁评估函数计算（基于受损主机数、权限等级等）
- **注意**：威胁评估用于可视化，实际决策由Q-learning Selector基于Q表做出

**观察到的关键模式**：

1. **初期（0-29步）：多子网威胁并存**
   - User和Enterprise子网交替出现威胁
   - 模型在两个子网间动态调度防御资源
   - 典型的Meander攻击初期特征

2. **转折点（步骤30）：攻击突破到Op子网** ⚠️
   - User和Enterprise威胁被成功清除（降至0）
   - Operational威胁突然跳升至17.0（高威胁）
   - 说明攻击者成功渗透到关键资产层

3. **后期（30-99步）：持续的Op子网防御战** 🎯
   - **Op威胁持续存在**：威胁分数保持在17.0，长达55步
   - **三子网轮换防御**：虽然Op威胁最高，但模型仍需在三个子网间平衡
     - Operational: 16次激活（集中防御关键资产）
     - User/Enterprise: 37次激活（防止新的入侵和横向移动）
   - **防御困境**：一旦攻击者在Op子网站稳脚跟，清除难度极大
   - **真实对抗**：展示了实际网络防御中的持久战特征

4. **战略意义**：
   - ✅ 证明模型能够识别并响应关键资产威胁
   - ✅ 展示多层防御的必要性（不能只防Op，还要防止新入侵）
   - ✅ 体现了真实攻防对抗的复杂性和持久性
   - ⚠️ 揭示了防御的挑战：预防比清除更容易

**论文使用建议**：
- 放在案例分析或可视化分析部分
- 展示Selector的威胁响应模式和动态决策能力
- 配合文字说明"威胁驱动的动态调度"
- 强调与图2统计数据的一致性

**数据来源**：`../activation_data_for_figures.pkl`（真实运行数据，Q-learning Selector + A3C Experts）

---


**最后更新**：2025年11月7日
