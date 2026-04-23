# Micrograd

Karpathy 的 `micrograd` 用最少的代码实现了自动求导的核心机制，揭示了神经网络"能学会"的底层原因：损失函数对参数可微 → 链式法则把误差传回每个参数 → 负梯度更新把损失压低。


## 训练闭环

神经网络训练本质上是不断重复这四步：

$$x \xrightarrow{\text{forward}} \hat{y} \xrightarrow{\text{loss}} \mathcal{L} \xrightarrow{\text{backward}} \frac{\partial \mathcal{L}}{\partial \theta} \xrightarrow{\text{update}} \theta$$

- **前向传播**：根据当前参数计算预测值
- **损失函数**：衡量预测与真实值的差距
- **反向传播**：用链式法则计算每个参数对损失的贡献
- **参数更新**：沿负梯度方向调整参数

$$\theta \leftarrow \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$$


## 计算图

计算图把复杂表达式拆成简单运算节点，再把依赖关系连起来。例如：

$$a = bc,\quad d = a + e,\quad L = d^2$$

构成一张有向图：$b,c \to a \to d \to L$，$e \to d$。

意义在于：每个局部操作只需知道自己的导数，全局梯度由链式法则自动组合出来。神经网络是一个巨大的复合函数 $\mathcal{L} = f(\theta)$，直接手推导数不现实，但拆成计算图后每个节点只负责自己那一小步。


## 梯度

### 偏导数：单个参数的敏感度

先从最简单的情况出发。如果损失只依赖一个参数 $\theta$，偏导数的定义是：

$$\frac{\partial \mathcal{L}}{\partial \theta} = \lim_{\epsilon \to 0} \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta)}{\epsilon}$$

这个值回答的问题是：**把 $\theta$ 增大一个无穷小量，损失会变多少？**

- 偏导数为正 → 增大 $\theta$ 会让损失上升，应该减小 $\theta$
- 偏导数为负 → 增大 $\theta$ 会让损失下降，应该增大 $\theta$
- 偏导数绝对值大 → 损失对这个参数非常敏感

注意：偏导数衡量的是**敏感度**，不是参数的重要性。参数值为 100，偏导数可能是 0；参数值为 0.01，偏导数可能很大。

### 梯度：所有参数的敏感度向量

当参数有多个时，梯度就是所有偏导数组成的向量：

$$\nabla_\theta \mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \theta_1}, \frac{\partial \mathcal{L}}{\partial \theta_2}, \ldots, \frac{\partial \mathcal{L}}{\partial \theta_n}\right)$$

梯度的几何意义：它指向损失（$\mathcal{L}$）**上升最快**的方向，其模长表示上升的速率。

### 链式法则：梯度如何跨层传递

神经网络是复合函数。设 $z = f(y),\ y = g(x)$，链式法则给出：

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

在计算图中，这意味着每个节点只需要知道两件事：
1. 自己的输出对输入的局部导数（$\frac{dy}{dx}$）
2. 从后续节点传来的梯度（$\frac{dz}{dy}$）

两者相乘，就得到损失对这个节点输入的梯度，再继续往前传。反向传播就是在计算图上系统地应用这个过程。

**具体例子**：沿用上面的计算图 $a = bc,\ d = a + e,\ L = d^2$，求 $\frac{\partial L}{\partial b}$：

$$\frac{\partial L}{\partial d} = 2d, \quad \frac{\partial d}{\partial a} = 1, \quad \frac{\partial a}{\partial b} = c$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial d} \cdot \frac{\partial d}{\partial a} \cdot \frac{\partial a}{\partial b} = 2d \cdot 1 \cdot c = 2dc$$

每一步只用了局部导数，链式法则把它们串联起来。

### 为什么沿负梯度更新，损失会下降

#### 第一步：方向选择问题

设当前参数为 $\theta \in \mathbb{R}^n$，我们要找一个更新方向 $\mathbf{d} \in \mathbb{R}^n$，使得沿该方向走一小步后损失下降。对 $\mathcal{L}(\theta + \epsilon\,\mathbf{d})$ 做一阶泰勒展开（$\epsilon > 0$ 为步长）：

$$\mathcal{L}(\theta + \epsilon\,\mathbf{d}) = \mathcal{L}(\theta) + \epsilon\,\nabla_\theta \mathcal{L} \cdot \mathbf{d} + O(\epsilon^2)$$

当 $\epsilon$ 足够小时，$O(\epsilon^2)$ 项可以忽略，损失的变化量近似为：

$$\Delta\mathcal{L} \approx \epsilon\,\nabla_\theta \mathcal{L} \cdot \mathbf{d}$$

要让损失下降（$\Delta\mathcal{L} < 0$），需要点积 $\nabla_\theta \mathcal{L} \cdot \mathbf{d} < 0$，即方向 $\mathbf{d}$ 与梯度**夹角大于 90°**。

#### 第二步：哪个方向下降最快

在所有满足 $\|\mathbf{d}\| = 1$ 的方向中，哪个让 $\Delta\mathcal{L}$ 最小（下降最多）？

由 Cauchy-Schwarz 不等式：

$$\nabla_\theta \mathcal{L} \cdot \mathbf{d} \geq -\|\nabla_\theta \mathcal{L}\|\,\|\mathbf{d}\| = -\|\nabla_\theta \mathcal{L}\|$$

等号在 $\mathbf{d} = -\dfrac{\nabla_\theta \mathcal{L}}{\|\nabla_\theta \mathcal{L}\|}$ 时成立，即**负梯度方向是下降最快的方向**。

#### 第三步：梯度下降的更新规则

将步长 $\epsilon$ 和方向 $\mathbf{d}$ 合并，令学习率 $\eta > 0$，更新量为：

$$\Delta\theta = -\eta\,\nabla_\theta \mathcal{L}$$

代入损失变化：

$$\Delta\mathcal{L} \approx \nabla_\theta \mathcal{L} \cdot (-\eta\,\nabla_\theta \mathcal{L}) = -\eta\,\|\nabla_\theta \mathcal{L}\|^2$$

因为 $\|\nabla_\theta \mathcal{L}\|^2 \geq 0$，当梯度不为零时：

$$\mathcal{L}(\theta - \eta\,\nabla_\theta \mathcal{L}) \approx \mathcal{L}(\theta) - \eta\,\|\nabla_\theta \mathcal{L}\|^2 < \mathcal{L}(\theta)$$

损失严格下降。

#### 第四步：近似成立的前提

以上结论依赖一阶泰勒近似，该近似在 $\eta$ 足够小时才成立。若 $\eta$ 过大，二阶项 $O(\eta^2)$ 不可忽略，可能导致损失不降反升。这就是学习率需要调参的根本原因：**太小收敛慢，太大破坏近似保证，可能发散**。


## micrograd 的实现

每个节点只需保存四件事：

- `data`：当前数值
- `grad`：当前梯度
- 前驱节点（由哪些节点算出来的）
- `_backward()`：局部反向规则

**前向阶段**：按定义把数值算出来，得到最终 loss。

**反向阶段**：将 loss 的梯度初始化为 1（因为 $\frac{d\mathcal{L}}{d\mathcal{L}} = 1$），然后按**拓扑顺序**反向调用每个节点的 `_backward()`。顺序至关重要：一个节点的梯度依赖后面的节点先把梯度传回来。


### 算子粒度：你可以自由决定

实现一个算子（op）时，可以选择不同的粒度。以 `tanh` 为例：

- **方式一**：直接把 `tanh` 作为一个原子算子实现，内部一步完成
- **方式二**：只实现 `exp`，然后用基本运算在外部拼出 $\tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$

两种方式的前向结果相同，梯度传播也相同，数学上完全等价。

**核心原则**：只要一个操作能写出"输入 → 输出"的映射，并且能写出局部梯度，它就可以作为计算图中的节点参与链式求导。算子内部有多少中间步骤、粒度粗还是细，不影响正确性。计算粒度的选择权完全在实现者手里。

```python
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'
o.backward()


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# ----
e = (2*n).exp()
o = (e - 1) / (e + 1)
# ----
o.label = 'o'
o.backward()
```


## 注意事项

- **梯度会累积**：多次 backward 的梯度会叠加，训练循环需要 `zero_grad()` 清零
- **不是每步都下降**：学习率过大、非凸性、小批量噪声都可能让某一步损失上升，关注整体趋势即可
- **自动求导不是"猜"**：每个局部操作都有明确的导数公式，系统组合出全局梯度
