# Phase-06 全向修改计划书：BL融合与层次风险平价 (HRP) 头寸管理

## 1. 现存系统缺陷剖析（尖锐审计）
* **马科维茨的脆弱性**：传统的均值-方差优化（MVO）或者 Black-Litterman 融合，极度依赖于历史协方差矩阵的逆矩阵。协方差矩阵只要有微小的噪声，在求逆过程中就会被成百上千倍地放大，导致分配出的头寸频繁出现“极多/极空”的抖动，产生巨大的换手成本和实盘滑点。
* **过度集中的风险**：当资产相关性在危机期间迅速向 1 靠拢时，传统的风险平价会因无法识别层级结构而导致资产集中暴露在某一特定板块（如科技股大崩盘）。

## 2. 全向改造计划与设计架构
### A. 层次风险平价算法 (Hierarchical Risk Parity, HRP)
* 引入 Marcos Lopez de Prado 提出的 **Hierarchical Risk Parity (HRP)** 头寸配比模型。该模型完全绕过了协方差矩阵求逆（Matrix Inversion）这一不适定问题。
* **算法三部曲**：
  1. **层级聚类 (Tree Clustering)**：利用资产相关性矩阵计算距离度量，通过层次凝聚聚类（Hierarchical Agglomerative Clustering）将资产组织成树状层级。
  2. **矩阵准对角化 (Quasi-Diagonalization)**：重排协方差矩阵，使强相关资产相邻，将相似风险块集中在对角线周围。
  3. **递归分配 (Recursive Bisection)**：自顶向下分割树状图，根据各子集的风险（方差）逆比例递归地分配头寸。

### B. 基于校准置信度的 Black-Litterman 融合 (Calibrated BL Fusion)
* 如果策略需要融合主动观点，利用 Phase-5 概率校准引擎输出的物理概率，转换为 Black-Litterman 中的主观观点矩阵 $P$ 和观点收益率 $Q$。
* 观点的置信度矩阵 $\Omega$ 的对角元素，由校准后的物理概率的方差（Brier Score 的变体）进行自适应填充。

### C. 仓位平滑与调仓换手约束 (Turnover Constraint Control)
* 为了防止日频策略在微小波动中频繁调仓，在 `convex_optimizer.py` 中引入 **L1范数换手率惩罚项 (Turnover Penalty)**。
* 设定最小调仓触发阈值（如当前资产预期权重与实际权重差异小于 2% 时不予调仓），极大降低摩擦成本。

## 3. 具体修改步骤与代码骨架
1. 引入 `scipy.cluster.hierarchy` 库。
2. 新增 `hrp_optimizer.py`，实现 `HierarchicalRiskParity` 算法。
3. 修改 `step6_position_sizing.py`，将 HRP 与校准后的 BL 融合。

## 4. 2026-08-03 实施核验与补全

- **正式主链已实现**：Black-Litterman、稳健协方差、凸优化、仓位上限与额外风险平价数学模块。
- **仅旁路原型**：HRP 在 `enhance_dlc/phase_06_enhance.py`，尚未成为正式可选优化器。
- **未完成**：组合层风险预算、成交容量、行业约束的统一可行性证明；协方差估计误差压力测试。
- **补全验收**：权重和、上下限、行业/流动性/换手约束必须逐日输出机器可验证证据。
