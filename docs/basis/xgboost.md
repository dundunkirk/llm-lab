## 1. 泰勒展开：把复杂函数在某一点附近“局部多项式化”

### 1.1 核心思想
泰勒展开的目的，是在展开点附近，用一个更容易计算的多项式近似原函数。

若函数 $f(x)$ 在 $x=a$ 附近足够光滑，则它在 $a$ 处的泰勒展开为：

$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(x-a)^n
$$

前几项写开就是：

$$
f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)^2+\frac{f^{(3)}(a)}{3!}(x-a)^3+\cdots
$$

其中：

- $0$ 阶导数就是函数本身：$f^{(0)}(a)=f(a)$；
- 一阶项描述局部斜率；
- 二阶项描述局部弯曲；
- 更高阶项继续修正局部形状。

若展开点是 $a=0$，就得到 **麦克劳林展开**：

$$
f(x)=\sum_{n=0}^{\infty}\frac{f^{(n)}(0)}{n!}x^n
$$
### 1.2 二阶泰勒展开在机器学习里的意义
在优化问题里，二阶泰勒展开尤其重要，因为它把复杂目标函数局部近似成“线性项 + 二次项”的形式：

$$
f(x+\Delta x) \approx f(x)+f'(x)\Delta x+\frac{1}{2}f''(x)(\Delta x)^2
$$

这相当于把原函数局部变成一个二次函数，而二次函数通常更容易优化。

### 1.3 多元函数的二阶形式
对多元函数 $f(\mathbf{x})$，二阶展开可写成：

$$
f(\mathbf{x}+\Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top H(\mathbf{x})\Delta\mathbf{x}
$$

其中：

- $\nabla f(\mathbf{x})$ 是梯度；
- $H(\mathbf{x})$ 是 Hessian 矩阵；
- 一阶项给出下降方向；
- 二阶项反映曲率。


## 2. 从 GBDT 到 XGBoost：逐步加树的加法模型

### 2.1 Boosting 的基本想法
Boosting 属于串行集成学习：后一轮模型依赖前一轮的预测结果。和 Bagging 不同，Boosting 不是“各树独立投票”，而是“前一棵树先预测，后一棵树继续修正前面的误差”。

对 GBDT / XGBoost 而言，整体模型写成：

$$
\hat y_i = \sum_{k=1}^{K} f_k(x_i)
$$

其中每个 $f_k$ 都是一棵回归树。

第 $t$ 轮时：

$$
\hat y_i^{(t)} = \hat y_i^{(t-1)} + f_t(x_i)
$$

也就是说，新树 $f_t$ 的作用不是从零开始预测，而是在旧预测基础上做增量修正。

### 2.2 GBDT 为什么拟合残差？

#### 直觉：残差就是"还差多少"

最直接的理解：残差 $r_i = y_i - \hat{y}_i^{(t-1)}$ 告诉新树"当前预测还差多少"，把差补上，加法模型就朝真实值靠近：

$$
\hat{y}_i^{(t)} = \underbrace{\hat{y}_i^{(t-1)}}_{\text{已有预测}} + \underbrace{f_t(x_i)}_{\text{拟合残差}} \;\longrightarrow\; y_i
$$

但这只说了"是什么"，没说"为什么这样做是合理的"。

#### 数学本质：函数空间的梯度下降

更深层的原因来自**函数空间梯度下降**的视角。

普通参数优化里，梯度下降更新的是参数向量：

$$
\theta \leftarrow \theta - \eta\,\frac{\partial L}{\partial \theta}
$$

GBDT 的思路不同：把整个预测函数 $F(x)$ 本身看成"要优化的变量"，在函数空间沿着损失的负梯度方向走一步——每一步用一棵新树来近似这个负梯度：

$$
F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)
$$

每个样本 $i$ 上的"下降方向"为：

$$
\tilde{r}_i^{(t)} = -\frac{\partial\, l\bigl(y_i,\,\hat{y}_i\bigr)}{\partial\, \hat{y}_i}\Bigg|_{\hat{y}_i\,=\,F_{t-1}(x_i)}
$$

新树 $h_t$ 就去拟合这组 $\tilde{r}_i^{(t)}$，让函数在当前位置沿最速下降方向更新。

#### 以 MSE 为例：负梯度恰好等于残差

设 $l(y_i, \hat{y}_i) = \tfrac{1}{2}(y_i - \hat{y}_i)^2$，对预测值求导：

$$
\frac{\partial\, l}{\partial\,\hat{y}_i} = -\,(y_i - \hat{y}_i) = \hat{y}_i - y_i
$$

取负号：

$$
\tilde{r}_i = -\frac{\partial\, l}{\partial\,\hat{y}_i} = y_i - \hat{y}_i
$$

**负梯度 $=$ 残差**。这不是巧合，而是 MSE 的结构决定的。

#### 推广到一般损失：伪残差

当损失函数不是 MSE 时（如 MAE、对数损失），负梯度不再等于普通残差，但 GBDT 仍用它来指导新树的拟合目标，此时称为**伪残差（pseudo-residual）**。几个常见例子：

| 损失函数 | $l(y,\hat{y})$ | 负梯度（伪残差） |
|---|---|---|
| MSE | $\tfrac{1}{2}(y-\hat{y})^2$ | $y - \hat{y}$（即真实残差）|
| MAE | $\|y-\hat{y}\|$ | $\mathrm{sign}(y-\hat{y})$ |
| Log-loss（二分类） | $-y\ln\hat{p}-(1-y)\ln(1-\hat{p})$ | $y - \hat{p}$ |

**核心结论**：GBDT 拟合的不是"残差"，而是**当前损失的负梯度**；只是对 MSE，两者恰好相同，所以历史上留下了"拟合残差"的说法。


## 3. XGBoost 的目标函数

XGBoost 在训练时，不仅考虑训练误差，还考虑模型复杂度。目标函数写成：

$$
\mathrm{Obj} = \sum_{i=1}^{n} l(y_i, \hat y_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

其中：

- $l(y_i,\hat y_i)$ 是样本损失；
- $\Omega(f_k)$ 是第 $k$ 棵树的正则项。

第 $t$ 轮只关心新增树 $f_t$，于是：

$$
\mathrm{Obj}^{(t)}
=
\sum_{i=1}^{n} l\bigl(y_i,\hat y_i^{(t-1)}+f_t(x_i)\bigr)+\Omega(f_t)+\mathrm{const}
$$

这里的 `const` 表示和当前树无关的常数项。


## 4. XGBoost 的关键：对损失做二阶泰勒展开

### 4.1 为什么要展开？
因为很多损失函数都不容易直接优化。XGBoost 的关键技巧，是把

$$
l\bigl(y_i,\hat y_i^{(t-1)}+f_t(x_i)\bigr)
$$

看成关于预测值 $\hat y_i^{(t-1)}$ 的函数，然后对增量 $f_t(x_i)$ 做二阶泰勒展开：

$$
l\bigl(y_i,\hat y_i^{(t-1)}+f_t(x_i)\bigr)
\approx
l\bigl(y_i,\hat y_i^{(t-1)}\bigr)
+
 g_i f_t(x_i)
+
\frac{1}{2} h_i f_t(x_i)^2
$$

其中：

$$
g_i = \partial_{\hat y_i^{(t-1)}} l\bigl(y_i,\hat y_i^{(t-1)}\bigr)
$$

$$
h_i = \partial^2_{\hat y_i^{(t-1)}} l\bigl(y_i,\hat y_i^{(t-1)}\bigr)
$$

于是，第 $t$ 轮优化目标就变成：

$$
\mathrm{Obj}^{(t)} \approx
\sum_{i=1}^{n}
\left[
 g_i f_t(x_i)+\frac{1}{2}h_i f_t(x_i)^2
\right]
+\Omega(f_t)
$$

去掉常数后，只剩下与当前树有关的部分。

### 4.2 平方误差下的特殊情形
若
$$
l(y_i,\hat y_i)=(y_i-\hat y_i)^2
$$

则：
$$
g_i = 2(\hat y_i^{(t-1)}-y_i)
$$
$$
h_i = 2
$$
注：
- $\hat y_i^{(t-1)}$ 视为 $x$
- $f_t(x_i)$ 视为增量 $\Delta x$


因此一阶项与“残差”直接相关，二阶项给出稳定的曲率信息。

这一点也解释了：

- GBDT 常被理解为“拟合残差”；
- XGBoost 则是把这种思想推广到一般损失，并显式利用二阶信息。


## 5. 树模型如何写进目标函数

XGBoost 把一棵树写成：

$$
f_t(x)=w_{q(x)}
$$

其中：

- $q(x)$ 表示样本落到哪一个叶子；
- $w_j$ 表示第 $j$ 个叶子的分数；
- $T$ 表示叶子数。

正则项定义为：

$$
\Omega(f)=\gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
$$

含义很清楚：

- $\gamma T$ 惩罚叶子数过多，抑制树过深、过复杂；
- $\lambda \sum w_j^2$ 惩罚叶子权重过大，抑制过拟合。

把树形式代入目标函数后，可以把同一叶子上的样本聚合起来。定义：

$$
G_j = \sum_{i\in I_j} g_i,
\qquad
H_j = \sum_{i\in I_j} h_i
$$

其中 $I_j$ 表示落到第 $j$ 个叶子的样本集合。

则目标函数变成：

$$
\mathrm{Obj}^{(t)} = \sum_{j=1}^{T}
\left[
G_j w_j + \frac{1}{2}(H_j+\lambda) w_j^2
\right] + \gamma T
$$

这时每个叶子都变成一个独立的一元二次优化问题。


## 6. 叶子最优权重与结构分数

对每个叶子上的二次函数求最优值，可得叶子最优权重：

$$
w_j^\ast = -\frac{G_j}{H_j+\lambda}
$$

将其代回目标函数，得到固定树结构下的最优目标值：

$$
\mathrm{Obj}^{\ast} = -\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j+\lambda}+\gamma T
$$

这个式子很重要，因为它告诉人：

- 一个叶子“好不好”，不再只看误差；
- 还要同时看梯度、Hessian 和正则项；
- XGBoost 选树结构，本质是在让这个目标尽量更小。


## 7. 分裂增益（Split Gain）

若把一个叶子分裂成左右两个叶子，则分裂带来的收益为：

$$
\mathrm{Gain}=
\frac{1}{2}
\left(
\frac{G_L^2}{H_L+\lambda}
+
\frac{G_R^2}{H_R+\lambda}
-
\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right)
-\gamma
$$

解释：

- 前两项是左右子叶分裂后的收益；
- 第三项是分裂前原叶子的收益；
- 最后减去 $\gamma$，表示增加一个分裂会带来复杂度惩罚。

因此：

- 若 Gain 足够大，就值得分裂；
- 若 Gain 不大，说明“继续分”不划算，应停止或剪枝。

这也是 XGBoost 里“正则化驱动分裂与剪枝”的核心来源。


## 8. 为什么 XGBoost 往往比普通 GBDT 更强

可以把它概括成 4 点：

1. **目标函数更完整**：不仅最小化损失，还显式加入模型复杂度惩罚；
2. **利用二阶信息**：不仅用梯度，还用 Hessian，使近似和优化更稳；
3. **叶子权重有闭式解**：每次分裂、每个叶子的最优分数都能直接算；
4. **树结构选择有明确评分公式**：Gain 直接衡量“分还是不分”。

所以，XGBoost 不是简单地“多棵树叠起来”，而是：

> 在 boosting 框架下，用二阶泰勒展开把复杂损失局部二次化，再结合正则化，把“树结构搜索 + 叶子权重求解”统一到一个可计算的目标函数里。


## 9. 最后的理解框架

可以把整件事浓缩成一条主线：

1. **泰勒展开** 提供了一种“局部二次近似”的工具；
2. **Boosting** 提供了一种“逐轮加模型修正误差”的框架；
3. **XGBoost** 则把两者结合起来：
   - 每轮新增一棵树；
   - 对目标函数在当前预测处做二阶泰勒展开；
   - 再通过正则化和闭式解，选择最优叶子权重与分裂结构。

一句话说：

> XGBoost 的数学核心，不是“树很多”，而是“在 boosting 框架下，用二阶泰勒展开把每一步优化变成一个带正则的可解二次问题”。


## 参考

- XGBoost 官方文档：Introduction to Boosted Trees
- OpenStax Calculus Volume 2：Taylor and Maclaurin Series
- [通俗理解kaggle比赛大杀器xgboost\_xgboost目标函数-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/81410574)
- [如何通俗地解释泰勒公式？](https://www.zhihu.com/question/21149770)
- [\[损失函数设计\] 为什么多分类问题损失函数用交叉熵损失，而不是 MSE](https://www.bilibili.com/video/BV13NHfewE3o?spm_id_from=333.788.videopod.sections&vd_source=977765d761bdef8d6cb7b4e570bb9270)
- [线性回归——lasso回归和岭回归（ridge regression） - wuliytTaotao - 博客园](https://www.cnblogs.com/wuliytTaotao/p/10837533.html)
