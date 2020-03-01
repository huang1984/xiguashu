
## 算法原理
[![3gC38K.png](assets/3gC38K.png)](https://imgchr.com/i/3gC38K)

### 算法优缺点

**优点**：准确性高，对异常值和噪声有较高的容忍度。

**缺点**：计算量较大，对内存的需求也较大。每次对一个未标记样本进行分类时，都需要全部计算一遍距离。

### 算法参数
参数选择需要根据数据来决定。k值越大，模型的偏差越大，对噪声数据越不敏感，当k值很大时，可能造成模型欠拟合；k值越小，模型的方差就会越大，当k值太小，就会造成模型过拟合。

### 算法的变种
[![3gCYKe.png](assets/3gCYKe.png)](https://imgchr.com/i/3gCYKe)

## 使用k近邻算法进行分类
在scikit-learn里用sklern.neighbors.KNeighborsClassifier类。

### 生成已标记的数据集


```python
from sklearn.datasets.samples_generator import make_blobs
#生成数据
centers=[[-2,2],[2,2],[0,4]]
x,y=make_blobs(n_samples=60,centers=centers,random_state=0,cluster_std=0.6)

```

![%E5%9B%BE%E7%89%87.png](attachment:%E5%9B%BE%E7%89%87.png)


```python
#画出数据
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(16,10),dpi=144)
c=np.array(centers)
plt.scatter(x[:,0],x[:,1],c=y,s=100,cmap="cool")#画出样本
plt.scatter(c[:,0],c[:,1],s=100,marker="*",c="orange")#画出中心点
```




    <matplotlib.collections.PathCollection at 0x1bac6c3beb8>




![png](output_8_1.png)


### 使用KNeighborsClassifier对算法进行训练，选择k=5



```python
from sklearn.neighbors import KNeighborsClassifier
#模型预测
k=5
clf=KNeighborsClassifier(n_neighbors=k)
clf.fit(x,y)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')




```python
# 进行预测
x_sample=[0,2]
x_sample = np.array(x_sample).reshape(1, -1)
y_sample=clf.predict(x_sample)
neighbors=clf.kneighbors(x_sample,return_distance=False)
```


```python
# 画出示意图
plt.figure(figsize=(16, 10))
plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='cool')    # 样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心点
plt.scatter(x_sample[0][0], x_sample[0][1], marker="x", 
            s=100, cmap='cool')    # 待预测的点

for i in neighbors[0]:
    # 预测点与距离最近的 5 个样本的连线
    plt.plot([x[i][0], x_sample[0][0]], [x[i][1], x_sample[0][1]], 
             'k--', linewidth=0.6)
```


![png](output_12_0.png)


## 使用k近邻算法进行回归拟合
分类问题的预测值是离散的，也可以用k近邻算法在连续区间内对数值进行预测，进行回归拟合。sklearn.neighbors.KNeighborsRegressor

### 生成数据集，在余弦曲线的基础上加入噪声


```python
import numpy as np
n_dots=40
x=5*np.random.rand(n_dots,1)
y=np.cos(x).ravel()

#添加一些噪声
y+=0.2*np.random.rand(n_dots)-0.1
```

### 使用函数来训练模型


```python
from sklearn.neighbors import KNeighborsRegressor
k=5
knn=KNeighborsRegressor(k)
knn.fit(x,y)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=None, n_neighbors=5, p=2,
              weights='uniform')




```python
#生成足够密集的点进行预测
t=np.linspace(0,5,500)[:,np.newaxis]
y_pred=knn.predict(t)
knn.score(x,y)
```




    0.9895623287942961



### 把预测点连接起来，构成拟合曲线



```python
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(x,y,c="g",label="data",s=100)
plt.plot(t,y_pred,c="k",label="prediction",lw=4)
plt.axis("tight")
plt.title("KNeighborsRegressor(K=%i)"%k)
plt.show()
```


![png](output_21_0.png)


## 糖尿病预测


```python
#加载数据
import pandas as pd
data=pd.read_csv("diabetes.csv")
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>116</td>
      <td>74</td>
      <td>0</td>
      <td>0</td>
      <td>25.6</td>
      <td>0.201</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>78</td>
      <td>50</td>
      <td>32</td>
      <td>88</td>
      <td>31.0</td>
      <td>0.248</td>
      <td>26</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>35.3</td>
      <td>0.134</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>197</td>
      <td>70</td>
      <td>45</td>
      <td>543</td>
      <td>30.5</td>
      <td>0.158</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>125</td>
      <td>96</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.232</td>
      <td>54</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
    Pregnancies                 768 non-null int64
    Glucose                     768 non-null int64
    BloodPressure               768 non-null int64
    SkinThickness               768 non-null int64
    Insulin                     768 non-null int64
    BMI                         768 non-null float64
    DiabetesPedigreeFunction    768 non-null float64
    Age                         768 non-null int64
    Outcome                     768 non-null int64
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB


```python
data.groupby("Outcome").size()
```




    Outcome
    0    500
    1    268
    dtype: int64




```python
import pandas as pd
data=pd.read_csv("diabetes.csv")
x=data.iloc[:,0:8]
y=data.iloc[:,8]
print(x.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
print(x_test.shape)
```

    (768, 8)
    (768,)
    (231, 8)



```python
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
models=[]
models.append(("knn",
              KNeighborsClassifier(n_neighbors=5)))
models.append(("knn with weights",
              KNeighborsClassifier(n_neighbors=5,weights="distance")))
models.append(("Radius",
              RadiusNeighborsClassifier(n_neighbors=5,radius=500.0)))
results=[]
for name,model in models:
    model.fit(x_train,y_train)
    results.append((name,model.score(x_test,y_test)))
for i in range(len(results)):
    print("name:{};score:{}".format(results[i][0],results[i][1]))
```

    name:knn;score:0.6883116883116883
    name:knn with weights;score:0.6926406926406926
    name:Radius;score:0.645021645021645


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results=[]
for name,model in models:
    kfold=KFold(n_splits=10)
    cv_result=cross_val_score(model,x,y,cv=kfold)
    results.append((name,cv_result))
for i in range(len(results)):
    print ("name:{};cross val score:{}".format(results[i][0],results[i][1].mean()))
```

    name:knn;cross val score:0.7265550239234451
    name:knn with weights;cross val score:0.7265550239234451
    name:Radius;cross val score:0.6497265892002735


### 模型训练及分析
用普通k均值算法模型对数据集进行训练，并查看对训练样本的拟合情况以及测试样本的预测准确性情况。


```python
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
train_score=knn.score(x_train,y_train)
test_score=knn.score(x_test,y_test)
print(train_score)
print(test_score)
```

    0.839851024208566
    0.683982683982684


训练样本的拟合情况不佳，模型的准确性欠佳。画出学习曲线

### 特征选择及数据可视化


```python
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=2)
x_new = selector.fit_transform(x, y)
x_new[0:5]
```




    array([[148. ,  33.6],
           [ 85. ,  26.6],
           [183. ,  23.3],
           [ 89. ,  28.1],
           [137. ,  43.1]])




```python
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, x_new, y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross val score: {}".format(
        results[i][0],results[i][1].mean()))
```

    name: knn; cross val score: 0.7369104579630894
    name: knn with weights; cross val score: 0.7199419002050581
    name: Radius; cross val score: 0.6510252904989747



```python
# 画出数据
plt.figure(figsize=(10, 6))
plt.ylabel("BMI")
plt.xlabel("Glucose")
plt.scatter(x_new[y==0][:, 0],
           x_new[y==0][:, 1],
            c='r', s=20, marker='o');         # 画出样本
plt.scatter(x_new[y==1][:, 0], 
            x_new[y==1][:, 1], 
            c='g', s=20, marker='^');         # 画出样本
```


![png](output_37_0.png)



## KNN面试题

### 1. 请简单描述一下KNN算法原理，以及其优缺点
见开头

### 2. 在KNN算法中如何计算距离？为什么用欧式不用曼哈顿
曼哈顿距离、欧式距离和闵可夫斯基距离。

设特征空间X是n维实数向量空间Rn, xi,xj∈X,xi=(x(1)i,x(2)i,...,x(n)i)T, xj=(x(1)j,x(2)j,...,x(n)j)T, xi,xj的Lp距离定义为：
　　

Lp(xi,xj)=(∑nl=1|x(l)i−x(l)j|p)1p

这里 p≥1 

当p=1时，称为曼哈顿距离(Manhattan distance), 公式为：
　　
L1(xi,xj)=∑nl=1|x(l)i−x(l)j|

当p=2时，称为欧式距离(Euclidean distance)，即
　　
L2(xi,xj)=(∑nl=1|x(l)i−x(l)j|2)12

当p=∞时，它是各个坐标距离的最大值，计算公式为：
　　
L∞(xi,xj)=maxl|x(l)i−x(l)j|


欧式距离用于计算两点或多点之间的距离。

缺点：它将样本的不同属性（即各指标或各变量量纲）之间的差别等同看待，这一点有时不能满足实际要求。比如年龄和学历对工资的影响，将年龄和学历同等看待；收入单位为元和收入单位为万元同等看待。

标准化欧式距离，将属性进行标准化处理，区间设置在[0,1]之间，减少量纲的影响。

曼哈顿距离：欧式几何空间两点之间的距离在两个坐标轴的投影。

曼哈顿距离和欧式距离一般用途不同，无相互替代性。在kmeans和knn算法，我们一般用欧式距离，有时也用曼哈顿距离。

### 3. 如何选取超参数k值

1、通过领域知识得到。不同领域内，遇到不同的问题，超参数一般不同。

2、经验数值。一般机器学习算法库中，会封装一些默认的超参数，这些默认的超参数一般都是经验数值； kNN算法这scikit-learn库中，k值默认为5，5就是在经验上比较好的数值

3、通过试验搜索得到将不同的超参数输入模型，选取准确度最高的超参数；试验搜索也称为网格搜索：对不同的超参数，使用对个for语句，逐层搜索。


### 4. knn算法的时间复杂是多少，在面对高维数据时如何处理呢？
时间复杂度o(n*k)：n为样本数量，k为单个样本特征的维度。如果不考虑特征维度的粒度为o(n)

降维
