
### 2.1 经验误差与过拟合

错误率(error rate)： ，其中 为样本个数， 为分类错误样本个数

精度(accuracy)：精度=1-错误率

误差(error)：学习器的实际预测输出与样本的真实输出之间的差异

训练误差(training error)：学习器在训练集上的误差，又称为 经验误差(empirical error)

泛化误差(generalization)：学习器在新样本上的误差


过拟合(overfitting)是由于模型的学习能力相对于数据来说过于强大，学习能力太强；反过来说， 欠拟合(underfitting)是因为模型的学习能力相对于数据来说过于低下，学习能力太差。



### 评估方法
测试集来测试学习器对新样本的判别能力，然后以测试集上的测试误差作为泛化误差的近似。通常我们假设测试样本也是从样本真实分布中独立同分布采样而得，但需注意的是，测试集应该尽可能与训练集互斥，即测试样本尽量不在训练集中出现，未在训练过程中使用过。


####  留出法 
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

训练，测试集的划分要尽可能保持数据分布的一致性，避免因数据划分过程引入额外的偏差而对最终结果产生影响。

即使在给定训练/测试集的样本比例后，仍存在多种划分方式对初始数据集进行分割。

将大约2/3-4/5的样本用于训练，剩余样本用于测试。

#### 交叉验证法
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

与留出法相似，将数据集D划分为k个子集同样存在多种划分方式，为减少因样本划分不同而引入的差别，k折交叉验证通常要随机使用不同的划分重复p次，最终的评估结果是这p次k折交叉验证结果的均值。

#### 自助法
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

自助法在数据集较小，难以有效划分训练/测试集时很有用，此外，自助法能从初始数据集中产生多个不同的训练集，这对集成学习等方法有很大的好处，然而自助法产生的数据集改变了初始数据集的分布，这会引入估计偏差，因此，在初始数据量足够时，留出法和交叉验证法更常用一些。

### 性能度量
衡量模型泛化能力的评价标准，性能度量反映了任务需求，在对比不同模型的能力时，使用不同的性能度量往往会导致不同的评判结果，这意味着模型的好坏是相对的，什么样的模型是好的，不仅取决于算法和数据，还决定于任务需求。

![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

#### 错误率与精度
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

#### 查准率，查准率与F1
![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

以查准率为纵轴，查全率为横轴作图，就得到了查准率-查全率曲线，简称"P-R"曲线![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

   #### roc与auc
   ![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)

![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)
