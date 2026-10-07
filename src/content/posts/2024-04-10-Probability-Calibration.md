---
title: Probability Calibration
published: 2024-04-10
description: 介绍分类模型中的概率校准方法，包括校准曲线的绘制以及逻辑回归、贝叶斯、随机森林等模型在校准表现上的差异与原因分析。
category: Machine Learning
tags:
- machine learning
- calibration
draft: false
---
## 0. 前言

分类时，我们不仅希望模型预测类别，也希望它输出概率。但有些模型不能直接输出概率，或者输出的概率只是相对值，这时就需要校准。


良好校准指的是：在预测概率接近 0.8 的大量样本中，正例比例也接近 0.8。这是群体频率性质，不能对单个样本验证“有 80% 概率为正例”。

## 1. 校准曲线

将预测值按升序排列，再划分为若干 bin。每个 bin 中的平均预测概率作为横坐标，正样本所占比例作为纵坐标，由此绘制校准曲线。

![校准曲线.png](https://s2.loli.net/2024/04/10/APQRTWiVeMFEYgU.png)

图中的条形图可以这样理解：

- 逻辑回归使用对数损失，在模型假设较合适时往往具有较好的校准表现；但交叉熵训练本身不保证在任意数据分布上都校准，仍应通过留出数据上的校准曲线检查。

- 贝叶斯看起来更倾向于输出接近 0 或 1 的值，主要原因可能是它假设特征相互独立，但这一解释仍**存疑**。

- 随机森林的输出明显集中在 0.1 或 0.9 附近。Niculescu-Mizil and Caruana [[3]](https://scikit-learn.org/stable/modules/calibration.html#id14) 认为，随机森林是 bagging 模型，最终输出是多棵树预测值的平均。要让一个样本的输出严格为 0，就需要所有基决策树都把它预测为 0；单棵树具有高方差，可能引入噪声，因此这通常不可能，最终输出通常也不会是 0 或 1。
    - 什么是高方差？

        通常所说的高方差，是指模型相对复杂，完美匹配了训练数据，却只学到了一个局部模型，距离“平均模型”较远，也就是过拟合。由于数据本身也可能是局部的，这样的模型对数据变化很敏感。

        > 为什么单个树的方差很高？
        >
        >
        > 单棵决策树的高方差与其结构和学习方式有关。下面列出几个因素（ChatGPT 语气）：
        >
        > 1. **不剪枝**：决策树会持续分裂节点，直到满足停止条件。缺少适当剪枝时，树会继续生长，尝试连同噪声和异常值一起完美拟合。这种过拟合会使模型对训练数据的微小变化非常敏感，增加方差。
        > 2. **贪婪搜索**：决策树通常在每个节点选择局部最优的特征来分裂，而不考虑全局最优解。这种选择可能让树过于复杂，增加模型方差。
        > 3. **数据噪声和异常值**：决策树根据数据特征分裂，噪声和异常值可能对树的结构产生不成比例的影响，让树沿错误方向分裂，增加方差。
        > 4. **特征选择的随机性**：构建决策树时，通常从特征集中选择一个特征来分裂节点。若没有引入适当的随机性，很小的数据变化也可能改变所选特征，形成完全不同的树结构，增加方差。
        > 5. **树的深度**：树越深，越可能学到数据中的噪声和偶然规律，从而产生高方差。
        > 6. **数据表示的选择**：决策树对数据表示非常敏感。特征若缺少适当的预处理和特征工程，树可能对某些特定表示过度拟合，增加方差。
        >
        > 这些高方差的树因随机性而彼此不同，在一些数据点上预测错误，在另一些点上正确。集成时，预测误差会相互抵消；随机森林通过平均多棵树的结果，降低整体方差。
- SVM 的原始输出通常是决策分数，不是概率；不能直接说它的输出大多在 0.5 左右。若需要概率，应使用单独的概率估计或校准方法。


## 2. Calibrating a classifier

### 2.1 函数介绍

可以使用 [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) 校准分类器。校准器必须使用基模型训练时未见过的预测结果；交叉验证可生成这样的折外预测。对于已训练模型，需使用独立的校准集。以下按当前文档中的参数名 `estimator` 说明：

- 当 `ensemble=True`：
    - 将数据分成 k 组 `(train_set, test_set)`
    - 每折复制一份 `estimator`，在训练折拟合基模型，再用验证折的预测拟合校准器。校准方法可选 sigmoid、isotonic 等，具体以安装版本为准。
    - 拟合后的校准器保存在 `calibrated_classifiers_` 属性中，每项都是一个校准后的分类器，其 [predict_proba](https://scikit-learn.org/stable/glossary.html#term-predict_proba) 方法输出校准后的概率。
    - CCV 类本身的 [predict_proba](https://scikit-learn.org/stable/glossary.html#term-predict_proba) 方法会取 `k` 个估计器的预测概率平均值，这些估计器保存在 `calibrated_classifiers_` 列表中。
    - [predict](https://scikit-learn.org/stable/glossary.html#term-predict) 返回概率最高的类别。
- 当 `ensemble=False`：对未训练的普通基模型，用折外预测拟合一个校准器，推理时配合在全部训练数据上重新拟合的基模型；这也是合理选择。对于 `FrozenEstimator` 包装的已训练模型，则只用独立校准集拟合校准器，不重新训练基模型。当前版本 `ensemble` 默认是 `"auto"`，前者通常走 `True` 分支，后者走 `False` 分支。

### 2.2 校准方法
#### 2.2.1 sigmoid 方法


$$
p(y_i = 1 | f_i) = \frac{1}{1 + \exp(A f_i + B)}
$$


A 和 B 是实数，在拟合回归器时通过最大似然确定。

该方法适用于以下情况：
- 校准误差对称
    - 即二分类中每个类别的模型输出都服从正态分布，且方差相同
    - **在高度不平衡的分类问题中，两类输出的方差不同，这可能成为问题**
- 样本量较小
- 未校准模型的置信度偏低，且高分和低分输出的校准误差相近

#### 2.2.2 isotonic 方法

该方法拟合一个非参数的 isotonic 回归器，输出阶梯状的非递减函数，参见 [sklearn.isotonic](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.isotonic)。它最小化下面的目标：
> 该方法拟合的是**非递减**函数，而非严格递增函数：若原分数 $f_A<f_B$，校准后只保证 $\hat f_A\le\hat f_B$，两者可能映射到同一个概率。


$$
\sum_{i=1}^{n} (y_i - \hat{f}_i)^2
$$


约束为：


$$
\hat{f}_i \geq \hat{f}_j
$$


当：


$$
f_i \geq f_j
$$


$y_i$ 是样本的真实标签，$\hat{f}_i$ 是校准后分类器对该样本的输出，即校准概率。

总体而言，当数据足够多（大于约 1000 个样本）、能够避免过拟合时，isotonic 的表现会与 sigmoid 相当或更好。校准样本太少 `(<<1000)` 时不建议使用 isotonic，因为它容易过拟合。


## References

[1] [Sklearn-calibration](https://scikit-learn.org/stable/modules/calibration.html)

[2] [模型校准(Calibration of Models)技术](https://zhuanlan.zhihu.com/p/502959226)
