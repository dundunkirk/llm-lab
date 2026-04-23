# XGBoost

## 1. 泰勒展开：把复杂函数局部多项式化

### 1.1 核心思想

若函数 $f(x)$ 在 $x=a$ 附近足够光滑，则：

$$
f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\cdots
$$

- 一阶项描述局部斜率；二阶项描述局部弯曲。
- 展开点取 $a=0$ 即为麦克劳林展开。

### 1.2 二阶展开在优化中的意义

$$
f(x+\Delta x) \approx f(x)+f'(x)\Delta x+\frac{1}{2}f''(x)(\Delta x)^2
$$

把复杂目标函数局部近似成二次函数，二次函数有闭式最优解，这是 XGBoost 的核心工具。

### 1.3 多元形式

$$
f(\mathbf{x}+\Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top H(\mathbf{x})\Delta\mathbf{x}
$$

其中 $\nabla f$ 是梯度，$H$ 是 Hessian 矩阵，分别描述下降方向与曲率。

## 2. 从 GBDT 到 XGBoost：加法模型

### 2.1 Boosting 基本想法

Boosting 是串行集成：后一棵树修正前一棵树的误差。整体模型为：

$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i)
$$

> 对比 Bagging（如随机森林）：各树独立并行，最后投票；Boosting 各树串行，逐步修正。

第 $t$ 轮增量更新：

$$
\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)
$$

新树 $f_t$ 不从零预测，而是在已有结果上做增量修正。

### 2.2 GBDT 为什么拟合残差？

#### 直觉

残差 $r_i = y_i - \hat{y}_i^{(t-1)}$ 就是"当前还差多少"，新树把差补上，预测就朝真实值靠近：

$$
\hat{y}_i^{(t)} = \underbrace{\hat{y}_i^{(t-1)}}_{\text{已有预测}} + \underbrace{f_t(x_i)}_{\text{补差}} \;\longrightarrow\; y_i
$$

#### 数学本质：函数空间的梯度下降

记号说明：
- $y_i$：**固定的真实标签**，训练过程中不变
- $\hat{y}_i^{(t-1)} = F_{t-1}(x_i)$：前 $t-1$ 棵树的累计预测值
- $f_t(x_i)$：第 $t$ 棵新树，是我们要求的"增量函数"

目标是找 $f_t$，使损失 $l(y_i,\, F_{t-1}(x_i) + f_t(x_i))$ 最小。

GBDT 把预测函数 $F(x)$ 本身看作优化变量，在**函数空间**做梯度下降——每步用一棵新树近似损失的负梯度：

$$
F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)
$$

每个样本的负梯度（即"最速下降方向"）为：

$$
\tilde{r}_i^{(t)} = -\frac{\partial\, l(y_i,\,\hat{y}_i)}{\partial\, \hat{y}_i}\Bigg|_{\hat{y}_i = F_{t-1}(x_i)}
$$

新树 $h_t$ 拟合这组 $\tilde{r}_i^{(t)}$，使函数沿减小损失的方向迈一步。

#### MSE 下负梯度 = 残差

设 $l = \tfrac{1}{2}(y_i - \hat{y}_i)^2$：

$$
\frac{\partial l}{\partial \hat{y}_i} = \hat{y}_i - y_i
\quad\Longrightarrow\quad
\tilde{r}_i = -(\hat{y}_i - y_i) = y_i - \hat{y}_i
$$

**负梯度恰好等于残差**，这是 MSE 的特殊结构，不是巧合。对其他损失函数（MAE、log-loss 等），负梯度不等于普通残差，但 GBDT 同样拟合它，统称**伪残差**。

## 3. XGBoost 的目标函数

XGBoost 在损失之外显式加入正则项：

$$
\mathrm{Obj} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

第 $t$ 轮只优化新树 $f_t$，之前各树贡献为常数：

$$
\mathrm{Obj}^{(t)} = \sum_{i=1}^{n} l\bigl(y_i,\,\hat{y}_i^{(t-1)}+f_t(x_i)\bigr)+\Omega(f_t)+\mathrm{const}
$$

## 4. 对损失做二阶泰勒展开

直接优化上式很难，XGBoost 对 $f_t(x_i)$ 做二阶泰勒展开：

$$
l\bigl(y_i,\hat{y}_i^{(t-1)}+f_t(x_i)\bigr)
\approx
l\bigl(y_i,\hat{y}_i^{(t-1)}\bigr)
+ g_i f_t(x_i)
+ \frac{1}{2} h_i f_t(x_i)^2
$$

其中 $g_i,\, h_i$ 是损失对当前预测值的一、二阶导数：

$$
g_i = \frac{\partial\, l(y_i,\hat{y}_i^{(t-1)})}{\partial\,\hat{y}_i^{(t-1)}},
\qquad
h_i = \frac{\partial^2 l(y_i,\hat{y}_i^{(t-1)})}{\partial\,(\hat{y}_i^{(t-1)})^2}
$$

去掉与 $f_t$ 无关的常数，第 $t$ 轮目标化简为：

$$
\mathrm{Obj}^{(t)} \approx \sum_{i=1}^{n}\left[g_i f_t(x_i)+\frac{1}{2}h_i f_t(x_i)^2\right]+\Omega(f_t)
$$

**MSE 下的具体值**：取 $l=(y_i-\hat{y}_i)^2$，则 $g_i = 2(\hat{y}_i^{(t-1)}-y_i)$，$h_i = 2$。

## 5. 树结构写入目标函数

XGBoost 把一棵树表示为：

$$
f_t(x)=w_{q(x)}
$$

- $q(x)$：样本 $x$ 落到的叶子编号；
- $w_j$：第 $j$ 个叶子的输出值；
- $T$：叶子总数。

正则项惩罚复杂度：

$$
\Omega(f)=\gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

$\gamma T$ 抑制叶子过多，$\lambda\sum w_j^2$ 抑制叶子权重过大。

将树代入目标，把同一叶子 $j$ 内的样本聚合：

$$
G_j = \sum_{i\in I_j} g_i,\qquad H_j = \sum_{i\in I_j} h_i
$$

$G_j$ 是叶子 $j$ 内所有样本一阶导之和，$H_j$ 是二阶导之和，$I_j$ 为该叶子的样本集合。目标函数变成按叶子求和：

$$
\mathrm{Obj}^{(t)} = \sum_{j=1}^{T}\left[G_j w_j + \frac{1}{2}(H_j+\lambda)w_j^2\right] + \gamma T
$$

每个叶子现在是一个独立的一元二次优化问题。

## 6. 叶子最优权重与结构分数

对每个叶子的二次目标对 $w_j$ 求导并令其为零：

$$
\frac{\partial}{\partial w_j}\left[G_j w_j + \frac{1}{2}(H_j+\lambda)w_j^2\right] = G_j + (H_j+\lambda)w_j = 0
$$

解得叶子最优权重：

$$
w_j^* = -\frac{G_j}{H_j+\lambda}
$$

代回目标函数，得固定树结构下的最优得分：

$$
\mathrm{Obj}^* = -\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T
$$

这个值越小，树结构越好。$\lambda$ 在分母起到正则作用——梯度信息越弱或 $\lambda$ 越大，叶子权重越趋向 0。

## 7. 分裂增益

XGBoost 用**贪心算法**逐层建树：枚举所有特征和分裂点，选 Gain 最大的一个。

将叶子分裂成左右两个叶子的收益为：

$$
\mathrm{Gain}=\frac{1}{2}\left(\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right)-\gamma
$$

- 前两项：分裂后左右子叶的得分；
- 第三项：分裂前原叶子的得分；
- 减 $\gamma$：增加一个叶子的复杂度代价。

Gain $> 0$ 才值得分裂；否则停止，这是 XGBoost 正则化驱动剪枝的核心。

## 8. 为什么 XGBoost 比普通 GBDT 更强

| | GBDT | XGBoost |
|---|---|---|
| 拟合目标 | 负梯度（一阶） | 负梯度 + 二阶曲率 |
| 正则化 | 无或手动 | 显式内置于目标函数 |
| 叶子权重 | 启发式 | 闭式最优解 |
| 分裂准则 | 基于误差减少 | 基于 Gain 公式，可剪枝 |

核心优势：二阶信息让近似更准，使学习步长更稳；正则化统一进目标，避免单独调参。

## 9. 总结

$$
\underbrace{\text{泰勒展开}}_{\text{局部二次近似}}
+
\underbrace{\text{Boosting}}_{\text{逐轮加树修正误差}}
+
\underbrace{\text{正则化}}_{\text{控制复杂度}}
\;\Longrightarrow\;
\text{XGBoost}
$$

每轮新增一棵树，对目标在当前预测处做二阶泰勒展开，再通过闭式解求最优叶子权重、用 Gain 公式搜索最优分裂结构。

> XGBoost 的数学核心：在 boosting 框架下，用二阶泰勒展开把每一步优化变成带正则的可解二次问题。

## 参考

- XGBoost 官方文档：Introduction to Boosted Trees
- [通俗理解 XGBoost — CSDN](https://blog.csdn.net/v_JULY_v/article/details/81410574)
- [如何通俗地解释泰勒公式？— 知乎](https://www.zhihu.com/question/21149770)
- [\[损失函数设计\] 为什么多分类用交叉熵而非 MSE](https://www.bilibili.com/video/BV13NHfewE3o)
- [Lasso 与岭回归 — 博客园](https://www.cnblogs.com/wuliytTaotao/p/10837533.html)
