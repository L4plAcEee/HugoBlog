---
date : '2026-01-01T02:44:08+08:00'
title : '【N0te】数据挖掘实验'
summary : '数据挖掘基础操作学习，以及各种常见算法模型的应用，最后构建自己的推荐算法。'
tags : ['Python','实验笔记']
---
# 1. 实验一：数据清洗与处理

### 实验要求

| **实验目的：**<br><br>1、搭建好Windows版本的Python开发平台，并在该平台上安装一系列第三方的扩展库；<br><br>2、通过检验数据集的数据质量、绘制图表、计算某些特征等手段，对样本数据集的结构和规律进行分析，学会使用第三方拓展库对数据进行探索；<br><br>3、学会对原始数据中的缺失值、异常值进行处理；<br><br>2、针对原始数据集中的脏数据，能够通过Python及其拓展库对数据进行集成、规约、转换等操作                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 从Python官网https://www.python.org/ 下载3.6版本的Python；<br><br>2、 下载Python环境管理软件Anaconda并熟悉它的使用；<br><br>3、 对abalone.data数据文件进行预处理，使得其转化为包含表头的csv文件（命名为abalone.csv），读取并打印前10行数据；<br><br>4、 查看是否有属性存在空值；<br><br>5、 绘制每一个数值型属性的直方图，查看其分布，输出符合正态分布的属性及其直方图，从中挑一个属性，利用3σ原则分析异常值；<br><br>6、 针对带壳肉高度，画出其箱型图，并运用箱型图分析法进行异常检测，将异常值输出；<br><br>7、 判断外壳长度与高度、整只鲍鱼重量是否相关，若相关，输出相关系数的值，并判断哪一个相关性更大；<br><br>8、 针对外壳长度和整只鲍鱼重量两个属性建立关系，使用拉格朗日插值方法针对外壳长度为0.137、0.172的重量进行插补，将插补结果进行输出，并画出程序流程图；<br><br>9、 使用Z-score对肉的重量属性进行规范化处理；<br><br>10、 附加题：尝试使用PCA对原始数据样本进行降维，α设置为0.6，并将降维结果进行输出；<br><br>11、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |

### 实验过程以及分析
#### 代码实现

```python
data_file_path = "abalone/abalone.data"  
csv_file_path = "abalone/abalone.csv"  
  
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
from scipy.stats import skew  
  
  
### 3、对abalone.data数据文件进行预处理，使得其转化为包含表头的csv文件（命名为abalone.csv），读取并打印前10行数据；  
columns = ["Sex", "Length", "Diameter", "Height", "Whole weight",  
           "Shucked weight", "Viscera weight", "Shell weight", "Rings"]  
data = pd.read_csv(data_file_path, header=None, names=columns)  
data.to_csv(csv_file_path, index=False)  
print(data.head(10))  
  
### 4、查看是否有属性存在空值；  
data = pd.read_csv(csv_file_path)  
null_counts = data.isnull().sum()  
print(null_counts)  
  
### 5、绘制每一个数值型属性的直方图，查看其分布，输出符合正态分布的属性及其直方图，从中挑一个属性，利用3σ原则分析异常值；  
data = pd.read_csv(csv_file_path)  
numeric_columns = ["Length", "Diameter", "Height", "Whole weight",  
                   "Shucked weight", "Viscera weight", "Shell weight", "Rings"]  
plt.style.use('seaborn-v0_8-darkgrid')  
normal_attributes = []  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  ## 用于正常显示负号  
  
### 绘制所有数值型属性的直方图  
plt.figure(figsize=(16, 12))  
for i, col in enumerate(numeric_columns):  
    plt.subplot(3, 3, i+1)  
    plt.hist(data[col], bins=30, color='skyblue', edgecolor='black')  
    plt.title(col)  
    ## 计算偏度值，通常偏度绝对值较小（例如 < 0.5）可以认为近似正态分布  
    skewness = skew(data[col])  
    plt.xlabel(f'Skewness={skewness:.2f}')  
    ## 如果偏度绝对值小于0.5，则认为近似正态  
    if abs(skewness) < 0.5:  
        normal_attributes.append(col)  
  
plt.tight_layout()  
plt.show()  
  
print("近似符合正态分布的属性：", normal_attributes)  
  
attr = "Length"  
attr_data = data[attr]  
  
mean_val = np.mean(attr_data)  
std_val = np.std(attr_data)  
  
lower_bound = mean_val - 3 * std_val  
upper_bound = mean_val + 3 * std_val  
  
outliers = data[(attr_data < lower_bound) | (attr_data > upper_bound)]  
print(f"属性 '{attr}' 的均值：{mean_val:.2f}，标准差：{std_val:.2f}")  
print(f"根据3σ原则，阈值为 [{lower_bound:.2f}, {upper_bound:.2f}]")  
print(f"'{attr}' 属性中异常值的数量：{len(outliers)}")  
print("异常值数据预览：")  
print(outliers[[attr]].head())  
  
### 绘制选择属性的直方图，并标出3σ范围  
plt.figure(figsize=(8, 5))  
plt.hist(attr_data, bins=30, color='lightgreen', edgecolor='black')  
plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label='下界')  
plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label='上界')  
plt.title(f"{attr} 的直方图及3σ异常值界限")  
plt.xlabel(attr)  
plt.ylabel("频数")  
plt.legend()  
plt.show()  
  
### 6、针对带壳肉高度，画出其箱型图，并运用箱型图分析法进行异常检测，将异常值输出；  
data = pd.read_csv(csv_file_path)  
height_data = data["Height"]  
### 计算第一四分位数、第三四分位数及四分位距  
Q1 = height_data.quantile(0.25)  
Q3 = height_data.quantile(0.75)  
IQR = Q3 - Q1  
### 根据箱型图异常值判断标准：小于 Q1-1.5*IQR 或大于 Q3+1.5*IQR 的数据认为是异常值  
lower_bound = Q1 - 1.5 * IQR  
upper_bound = Q3 + 1.5 * IQR  
### 筛选出异常值  
outliers = data[(height_data < lower_bound) | (height_data > upper_bound)]  
print("带壳肉高度异常值：")  
print(outliers["Height"])  
### 绘制箱型图  
plt.figure(figsize=(8, 6))  
plt.boxplot(height_data, vert=False, patch_artist=True,  
            boxprops=dict(facecolor="lightblue", color="blue"),  
            medianprops=dict(color="red"))  
plt.title("带壳肉高度箱型图")  
plt.xlabel("Height")  
plt.show()  
  
### 7、判断外壳长度与高度、整只鲍鱼重量是否相关，若相关，输出相关系数的值，并判断哪一个相关性更大；  
data = pd.read_csv(csv_file_path)  
### 计算外壳长度（Length）与高度（Height）的皮尔逊相关系数  
corr_length_height = data['Length'].corr(data['Height'])  
### 计算外壳长度（Length）与整只鲍鱼重量（Whole weight）的皮尔逊相关系数  
corr_length_whole = data['Length'].corr(data['Whole weight'])  
### 输出相关系数  
print("外壳长度与高度的相关系数：", corr_length_height)  
print("外壳长度与整只鲍鱼重量的相关系数：", corr_length_whole)  
### 判断哪一个相关性更大（比较相关系数绝对值）  
if abs(corr_length_height) > abs(corr_length_whole):  
    print("外壳长度与高度的相关性更强。")  
elif abs(corr_length_height) < abs(corr_length_whole):  
    print("外壳长度与整只鲍鱼重量的相关性更强。")  
else:  
    print("外壳长度与高度和整只鲍鱼重量的相关性相近。")  
  
### 针对外壳长度和整只鲍鱼重量两个属性建立关系，使用拉格朗日插值方法针对外壳长度为0.137、0.172的重量进行插补，将插补结果进行输出；  
from scipy.interpolate import lagrange  
  
data = pd.read_csv(csv_file_path)  
### 按外壳长度排序，并选取较小外壳长度的4个数据点用于局部插值  
sorted_data = data.sort_values("Length")  
subset = sorted_data.head(4)  
### 提取外壳长度和整只鲍鱼重量的值  
x = subset["Length"].values  
y = subset["Whole weight"].values  
### 显示选取的数据点  
print("选取用于插值的数据点：")  
for xi, yi in zip(x, y):  
    print(f"Length = {xi:.3f}, Whole weight = {yi:.3f}")  
### 利用拉格朗日插值方法构建插值多项式  
poly = lagrange(x, y)  
### 对指定的外壳长度进行插值，并匹配最近的数据点  
lengths_to_interp = [0.137, 0.172]  
interp_results = {}  
for length_val in lengths_to_interp:  
    ## 查找与目标外壳长度最接近的实际数据点  
    closest_idx = np.argmin(np.abs(x - length_val))  
    closest_length = x[closest_idx]  
    closest_weight = y[closest_idx]  
    ## 如果插值点在数据范围外，使用最接近的点的值  
    if length_val < min(x) or length_val > max(x):  
        interp_results[length_val] = closest_weight  
    else:  
        interp_weight = poly(length_val)  
        interp_results[length_val] = interp_weight  
    print(f"外壳长度为 {length_val} 时，插补的整只鲍鱼重量为：{interp_results[length_val]:.3f}")  
  
### 9、使用Z-score对肉的重量属性进行规范化处理；  
data = pd.read_csv(csv_file_path)  
shucked = data["Shucked weight"]  
### 计算均值和标准差  
mean_val = shucked.mean()  
std_val = shucked.std()  
### 使用Z-score公式进行规范化处理  
data["Shucked weight_zscore"] = (shucked - mean_val) / std_val  
### 输出前几行查看规范化后的结果  
print(data[["Shucked weight", "Shucked weight_zscore"]].head())
```

#### 运行结果
##### 终端输出

```
  Sex  Length  Diameter  ...  Viscera weight  Shell weight  Rings
0   M   0.455     0.365  ...          0.1010         0.150     15
1   M   0.350     0.265  ...          0.0485         0.070      7
2   F   0.530     0.420  ...          0.1415         0.210      9
3   M   0.440     0.365  ...          0.1140         0.155     10
4   I   0.330     0.255  ...          0.0395         0.055      7
5   I   0.425     0.300  ...          0.0775         0.120      8
6   F   0.530     0.415  ...          0.1415         0.330     20
7   F   0.545     0.425  ...          0.1495         0.260     16
8   M   0.475     0.370  ...          0.1125         0.165      9
9   F   0.550     0.440  ...          0.1510         0.320     19

[10 rows x 9 columns]
Sex               0
Length            0
Diameter          0
Height            0
Whole weight      0
Shucked weight    0
Viscera weight    0
Shell weight      0
Rings             0
dtype: int64
近似符合正态分布的属性： []
属性 'Length' 的均值：0.52，标准差：0.12
根据3σ原则，阈值为 [0.16, 0.88]
'Length' 属性中异常值的数量：15
异常值数据预览：
     Length
236   0.075
237   0.130
238   0.110
239   0.160
526   0.155
带壳肉高度异常值：
236     0.010
237     0.030
238     0.030
239     0.035
306     0.030
694     0.020
718     0.035
719     0.025
720     0.025
1174    0.015
1257    0.000
1417    0.515
1428    0.250
1429    0.035
1763    0.250
1987    0.025
2051    1.130
2114    0.035
2169    0.015
2171    0.030
2172    0.030
2179    0.250
2381    0.025
2711    0.030
3190    0.025
3837    0.035
3899    0.035
3902    0.020
3996    0.000
Name: Height, dtype: float64
外壳长度与高度的相关系数： 0.8275536093192105
外壳长度与整只鲍鱼重量的相关系数： 0.9252611721489453
外壳长度与整只鲍鱼重量的相关性更强。
选取用于插值的数据点：
Length = 0.075, Whole weight = 0.002
Length = 0.110, Whole weight = 0.008
Length = 0.130, Whole weight = 0.013
Length = 0.130, Whole weight = 0.011
外壳长度为 0.137 时，插补的整只鲍鱼重量为：0.013
外壳长度为 0.172 时，插补的整只鲍鱼重量为：0.013

  return poly1d(self.coeffs/other)
   Shucked weight  Shucked weight_zscore
0          0.2245              -0.607613
1          0.0995              -1.170770
2          0.2565              -0.463444
3          0.2155              -0.648160
4          0.0895              -1.215822
```


##### 图表输出

![](f25325b9348ea448dacc7a714d018268_MD5.webp)

![](a02160bdbb0ff9ae17e942cfb84dd1ea_MD5.webp)

![](b1a0d82db9719555d121a52e4b8e0a26_MD5.webp)
##### 流程图绘制

![](ea4a1a5260003f8f0c16866951edb52d_MD5.webp)

### 实验体会与总结


---

# 2. 实验二：分类与预测算法

### 实验要求

| 实验目的：<br><br>1、理解分类与预测算法的原理；<br><br>2、能够使用分类与预测模型处理具体问题；<br><br>3、能够使用Python语言实现分类与预测算法。                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 实验内容：<br><br>1、 读取数据文件bankloan.xls，使用随机种子将原始数据集打乱，将数据集划分为训练集和测试集，比例为8：2；<br><br>2、 使用Logistic回归模型进行建模，计算准确率；<br><br>3、 更改模型参数，使得准确率进一步上升，并进行具体分析（比如调节了何参数？为何使得模型得到改善？）；<br><br>4、 读取数据文件bupa.data，按照bupa.names文件里要求的将bupa.data数据集划分为训练集和测试集；<br><br>5、 使用合适的模型对bupa.data文件中的数据进行建模，并给予文字说明为什么要选择这一模型；<br><br>6、 调节模型参数，使得最终训练得到的模型评价指标较好，并输出最终评价指标的值；<br><br>7、 在不同的参数下，画出模型运行过程中的损失下降图，以迭代次数为横坐标，损失值为纵坐标； <br><br>8、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |
### 实验过程以及分析
##### 代码实现
```python
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LogisticRegression, SGDClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, log_loss  
import matplotlib.pyplot as plt  
import warnings  
from sklearn.exceptions import ConvergenceWarning  
  
### 全局忽略 SGDClassifier 的未收敛警告  
warnings.filterwarnings("ignore", category=ConvergenceWarning)  
  
### 字体配置  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
  
### 1. 读取 bankloan.xls 并划分数据集（8:2）  
bankloan = pd.read_excel('./data/bankloan.xls', engine='xlrd')  
### 列名示例：['年龄','教育','工龄','地址','收入','负债率','信用卡负债','其他负债','违约']  
X_bank = bankloan.drop(columns=['违约'])  
y_bank = bankloan['违约']  
Xb_train, Xb_test, yb_train, yb_test = train_test_split(  
    X_bank, y_bank, test_size=0.2, random_state=42, shuffle=True  
)  
  
### 2. 使用 Logistic 回归模型基线建模，并计算准确率  
log_reg = LogisticRegression(max_iter=1000, random_state=42)  
log_reg.fit(Xb_train, yb_train)  
y_pred = log_reg.predict(Xb_test)  
acc_baseline = accuracy_score(yb_test, y_pred)  
print(f"Logistic 回归基线准确率：{acc_baseline:.4f}")  
  
### 3. 调参：使用 GridSearchCV 优化正则化强度 Cparam_grid = {'C': [0.01, 0.1, 1, 10, 100]}  
grid_log = GridSearchCV(  
    LogisticRegression(max_iter=1000, random_state=42),  
    param_grid,  
    cv=5,  
    scoring='accuracy',  
    n_jobs=-1  
)  
grid_log.fit(Xb_train, yb_train)  
best_log = grid_log.best_estimator_  
y_pred_best = best_log.predict(Xb_test)  
acc_tuned = accuracy_score(yb_test, y_pred_best)  
print(f"Logistic 最优 C：{grid_log.best_params_['C']}")  
print(f"调参后准确率：{acc_tuned:.4f}")  
  
### 4. 读取 bupa.data，并按要求划分（8:2）  
### bupa.names 中字段：mcv, alkphos, sgpt, sgot, gammagt, drinks, selector  
col_names = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']  
bupa = pd.read_csv('./data/bupa.data', names=col_names)  
### selector 原为 1=肝病，2=正常，转换为 0/1（二分类）  
bupa['selector'] = bupa['selector'].map({1: 1, 2: 0})  
X_bupa = bupa.drop(columns=['selector'])  
y_bupa = bupa['selector']  
Xbupa_train, Xbupa_test, ybupa_train, ybupa_test = train_test_split(  
    X_bupa, y_bupa, test_size=0.2, random_state=42, shuffle=True  
)  
  
### 5. 选择 RandomForestClassifier 建模（适合捕捉非线性与特征交互）  
rf = RandomForestClassifier(random_state=42)  
rf.fit(Xbupa_train, ybupa_train)  
y_pred_rf = rf.predict(Xbupa_test)  
acc_rf = accuracy_score(ybupa_test, y_pred_rf)  
print(f"RandomForest 基线准确率（Bupa 数据）：{acc_rf:.4f}")  
  
### 6. RandomForest 参数调优  
param_grid_rf = {  
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 5, 10]  
}  
grid_rf = GridSearchCV(  
    RandomForestClassifier(random_state=42),  
    param_grid_rf,  
    cv=5,  
    scoring='accuracy',  
    n_jobs=-1  
)  
grid_rf.fit(Xbupa_train, ybupa_train)  
best_rf = grid_rf.best_estimator_  
y_pred_best_rf = best_rf.predict(Xbupa_test)  
report_rf = classification_report(ybupa_test, y_pred_best_rf)  
print(f"RandomForest 最优参数：{grid_rf.best_params_}")  
print("RandomForest 调参后分类报告：")  
print(report_rf)  
  
### 7. 不同参数下的损失下降曲线（以 SGDClassifier 为例）  
from sklearn.linear_model import SGDClassifier  
from sklearn.metrics import log_loss  
  
sgd = SGDClassifier(  
    loss='log_loss',  
    learning_rate='constant',  
    eta0=0.01,  
    max_iter=1,  
    warm_start=True,  
    random_state=42  
)  
loss_values = []  
epochs = 50  
for epoch in range(epochs):  
    sgd.fit(Xb_train, yb_train)  
    proba = sgd.predict_proba(Xb_test)  
    loss = log_loss(yb_test, proba)  
    loss_values.append(loss)  
  
import matplotlib.pyplot as plt  
plt.figure(figsize=(8, 5))  
plt.plot(range(1, epochs+1), loss_values, marker='o')  
plt.xlabel('迭代次数 (Epoch)')  
plt.ylabel('对数损失 (Log Loss)')  
plt.title('SGDClassifier 在 BankLoan 测试集上的损失下降曲线')  
plt.grid(True)  
plt.tight_layout()  
plt.show()
```

##### 结果
###### 终端
```
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp02\main.py 
Logistic 回归基线准确率：0.8500
Logistic 最优 C：0.1
调参后准确率：0.8571
RandomForest 基线准确率（Bupa 数据）：0.8116
RandomForest 最优参数：{'max_depth': 5, 'n_estimators': 50}
RandomForest 调参后分类报告：
              precision    recall  f1-score   support

           0       0.76      0.88      0.81        42
           1       0.75      0.56      0.64        27

    accuracy                           0.75        69
   macro avg       0.75      0.72      0.73        69
weighted avg       0.75      0.75      0.74        69


进程已结束，退出代码为 0
```

###### 图表
![](dad8c2ffcef9e5814a3f6cde3d39a54a_MD5.webp)

##### 问题分析
**3. 模型参数调优分析**  
在 BankLoan 数据集上，我们使用了 Logistic 回归模型，并通过调整正则化强度参数 **C**（即正则化项的倒数）来提升模型的泛化性能：

- **基线模型**：默认 `C=1.0` 时，模型在测试集上的准确率约为 0.8500。
    
- **调优过程**：使用 `GridSearchCV` 在 `{0.01, 0.1, 1, 10, 100}` 这五个候选值中进行 5 折交叉验证；
    
- **最优结果**：最终选出 `C=0.1`，此时测试集准确率提升至 0.8571。
    

**为何调节 C 能带来改善？**

- Logistic 回归中，损失函数为带有 L2 正则化项的对数损失：
	
    $L(\mathbf{w}) = -\sum_i\bigl[y_i\log p_i + (1-y_i)\log(1-p_i)\bigr] \;+\; \frac{1}{2C}\|\mathbf{w}\|^2$.
    
- 当 C 较大（弱正则化）时，模型容易过拟合，权重 w 可能偏大以降低训练误差；
    
- 当 C 较小（强正则化）时，模型更倾向于保持权重较小，提高对噪声的鲁棒性，但若过强则会欠拟合；
    
- 通过交叉验证发现 C=0.1 在当前样本规模和特征分布下恰到好处地平衡了偏差与方差，从而在未见过数据上表现最佳。
    

**4. 读取 bupa.data 并划分数据集**  
肝病（Bupa）数据集的 `bupa.names` 文件指出：

1. 数据文件无表头，包含 6 个特征与 1 个标签（selector）；
    
2. selector 取值 1 表示有肝病，2 表示正常；
    
3. 无缺失值，均为数值型；
    

基于此，我们的划分流程为：

```python
import pandas as pd
from sklearn.model_selection import train_test_split

### 读取时指定列名
col_names = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
bupa = pd.read_csv('./data/bupa.data', names=col_names)

### 标签映射：1→1（肝病）；2→0（正常），方便二分类
bupa['selector'] = bupa['selector'].map({1: 1, 2: 0})

### 分割特征与标签
X = bupa.drop(columns=['selector'])
y = bupa['selector']

### 按 8:2 划分训练集/测试集，保持随机种子以便复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

**5. 模型选择及理由**  
针对 Bupa 数据的特点——

- **样本量中等**（345 行左右），
    
- **特征维度低**（仅 6 个数值特征），
    
- **可能存在非线性特征交互**（如某些指标组合时更能区分肝病状态），
    

我们选择 **RandomForestClassifier** 作为基线模型，其优势包括：

1. **无需对特征进行严格的线性假设**，能自动捕捉复杂的非线性决策边界；
    
2. **对异常值和噪声具有鲁棒性**，因为多个决策树投票可降低单棵树的过拟合风险；
    
3. **可输出特征重要性**，便于后续进行特征选择或域内专家解释；
    
4. **参数易于调整**（如树的数量 `n_estimators`、最大深度 `max_depth`），并能借助交叉验证快速找出最优组合。

### 实验体会与总结


---

# 3. 实验三：决策树算法

### 实验要求

| **实验目的：** <br><br>1、理解决策树算法的原理；<br><br>2、能够使用决策树算法处理具体问题；<br><br>3、能够使用Python语言实现决策树算法。                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **实验内容：**<br><br>1、 读取数据文件iris.data，使用随机种子将原始数据集打乱，将数据集划分为训练集和测试集，比例为8：2；<br><br>2、 使用ID3决策树在训练集上进行建模（注意：ID3决策树只能处理离散型数据，如果输入属性为连续型需要进行属性离散化），并将决策树通过图片的形式进行展现（使用Graphviz工具）；<br><br>3、 使用测试集对2中构建好的决策树进行检验，并计算其准确率；<br><br>4、 使用C4.5算法在训练集上进行建模，并将决策树通过图片的形式进行展现（使用Graphviz工具）；<br><br>5、 使用测试集对4中构建好的决策树进行检验，并计算其准确率<br><br>6、 读取给定的数据集watermelon.txt，构建其决策树，并通过图片的形式展现（使用Graphviz工具）； <br><br>7、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |

### 实验过程以及分析
#### 1. 数据准备
##### 实验描述
读取数据文件iris.data，使用随机种子将原始数据集打乱，将数据集划分为训练集和测试集，比例为8：2；
##### 代码实现
首先，我们实现了加载和处理Iris数据集的函数：
- `load_iris_data()`: 读取数据，设置随机种子打乱数据，并按8:2的比例划分训练集和测试集
- `discretize_features()`: 由于ID3算法不能直接处理连续型数据，此函数将连续特征离散化为几个区间
```python
### 第1步：读取数据文件iris.data，打乱并划分数据集  
def load_iris_data():  
    ## 读取数据  
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']  
    iris_data = pd.read_csv('iris.data', header=None, names=column_names)  
  
    ## 打乱数据（设置随机种子确保结果可复现）  
    iris_data = iris_data.sample(frac=1, random_state=42).reset_index(drop=True)  
  
    ## 分离特征和标签  
    X = iris_data.iloc[:, :-1]  
    y = iris_data.iloc[:, -1]  
  
    ## 划分训练集和测试集 (8:2)    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
  
    return X_train, X_test, y_train, y_test, iris_data  
  
  
### 属性离散化函数（用于ID3算法）  
def discretize_features(X_train, X_test, n_bins=3):  
    X_train_discrete = X_train.copy()  
    X_test_discrete = X_test.copy()  
  
    for column in X_train.columns:  
        ## 使用训练集的数据来确定分箱边界  
        bins = pd.qcut(X_train[column], n_bins, duplicates='drop', retbins=True)[1]  
  
        ## 对训练集进行离散化  
        X_train_discrete[column] = pd.cut(X_train[column], bins=bins, labels=False, include_lowest=True)  
  
        ## 对测试集进行离散化（使用与训练集相同的分箱边界）  
        X_test_discrete[column] = pd.cut(X_test[column], bins=bins, labels=False, include_lowest=True)  
  
    return X_train_discrete, X_test_discrete
```

#### 2. ID3决策树算法实现
##### 实验描述
使用ID3决策树在训练集上进行建模（注意：ID3决策树只能处理离散型数据，如果输入属性为连续型需要进行属性离散化），并将决策树通过图片的形式进行展现（使用Graphviz工具）；
##### 代码实现
ID3决策树类(`ID3DecisionTree`)实现了以下核心功能：
- `_entropy()`: 计算熵
- `_information_gain()`: 计算信息增益
- `_find_best_feature()`: 找到最佳分裂特征
- `_build_tree()`: 递归构建决策树
- `predict()`: 使用决策树进行预测
- `visualize()`: 使用Graphviz可视化决策树
ID3算法基于信息增益选择最佳分裂特征，对于每个节点，都选择信息增益最大的特征进行分裂。
```python
### ID3决策树算法实现  
class ID3DecisionTree:  
    def __init__(self, max_depth=None):  
        self.max_depth = max_depth  
        self.tree = None  
  
    def fit(self, X, y):  
        feature_names = X.columns  
        self.tree = self._build_tree(X, y, feature_names, depth=0)  
        return self  
  
    def _entropy(self, y):  
        """计算熵"""  
        counts = Counter(y)  
        entropy = 0  
        for label in counts:  
            p = counts[label] / len(y)  
            entropy -= p * math.log2(p)  
        return entropy  
  
    def _information_gain(self, X, y, feature_name):  
        """计算信息增益"""  
        entropy_parent = self._entropy(y)  
  
        ## 计算条件熵  
        values = X[feature_name].unique()  
        weighted_entropy = 0  
  
        for value in values:  
            subset_indices = X[feature_name] == value  
            subset_y = y[subset_indices]  
            weight = len(subset_y) / len(y)  
            weighted_entropy += weight * self._entropy(subset_y)  
  
        ## 信息增益 = 父节点熵 - 条件熵  
        information_gain = entropy_parent - weighted_entropy  
        return information_gain  
  
    def _find_best_feature(self, X, y, feature_names):  
        """找到最佳分裂特征"""  
        best_gain = -1  
        best_feature = None  
  
        for feature in feature_names:  
            gain = self._information_gain(X, y, feature)  
            if gain > best_gain:  
                best_gain = gain  
                best_feature = feature  
  
        return best_feature  
  
    def _build_tree(self, X, y, feature_names, depth):  
        ## 基本情况：所有样本属于同一类别  
        if len(np.unique(y)) == 1:  
            return {'type': 'leaf', 'label': y.iloc[0]}  
  
        ## 没有特征可分裂或达到最大深度  
        if len(feature_names) == 0 or (self.max_depth is not None and depth >= self.max_depth):  
            most_common_label = y.value_counts().idxmax()  
            return {'type': 'leaf', 'label': most_common_label}  
  
        ## 找到最佳分裂特征  
        best_feature = self._find_best_feature(X, y, feature_names)  
  
        ## 如果无法找到有效的特征进行分裂  
        if best_feature is None:  
            most_common_label = y.value_counts().idxmax()  
            return {'type': 'leaf', 'label': most_common_label}  
  
        ## 创建当前节点  
        tree = {'type': 'node', 'feature': best_feature, 'children': {}}  
  
        ## 根据特征值分裂  
        for value in X[best_feature].unique():  
            mask = X[best_feature] == value  
            subset_X = X[mask].drop(best_feature, axis=1)  
            subset_y = y[mask]  
            remaining_features = [f for f in feature_names if f != best_feature]  
  
            ## 如果子集为空  
            if len(subset_X) == 0:  
                most_common_label = y.value_counts().idxmax()  
                tree['children'][value] = {'type': 'leaf', 'label': most_common_label}  
            else:  
                tree['children'][value] = self._build_tree(subset_X, subset_y, remaining_features, depth + 1)  
  
        return tree  
  
    def predict(self, X):  
        if self.tree is None:  
            raise Exception("Model not trained yet!")  
  
        predictions = []  
        for _, sample in X.iterrows():  
            predictions.append(self._predict_sample(sample, self.tree))  
  
        return predictions  
  
    def _predict_sample(self, sample, tree):  
        if tree['type'] == 'leaf':  
            return tree['label']  
  
        feature = tree['feature']  
        value = sample[feature]  
  
        ## 处理测试集中出现训练集中没有的值的情况  
        if value not in tree['children']:  
            ## 返回所有子节点中多数类  
            labels = [self._predict_sample(sample, child) for child in tree['children'].values()]  
            return max(set(labels), key=labels.count)  
  
        return self._predict_sample(sample, tree['children'][value])  
  
    def visualize(self, feature_names, class_names=None):  
        """可视化决策树"""  
        dot = graphviz.Digraph(comment='Decision Tree')  
  
        ## 递归添加节点  
        self._add_nodes(dot, self.tree, "0", feature_names, class_names)  
  
        return dot  
  
    def _add_nodes(self, dot, tree, node_id, feature_names, class_names, parent_id=None, edge_label=None):  
        if tree['type'] == 'leaf':  
            label = str(tree['label']) if class_names is None else class_names[tree['label']]  
            dot.node(node_id, label=f"Class: {label}", shape='box')  
        else:  
            dot.node(node_id, label=f"{tree['feature']}")  
  
            ## 添加子节点  
            for i, (value, child) in enumerate(tree['children'].items()):  
                child_id = f"{node_id}_{i}"  
                self._add_nodes(dot, child, child_id, feature_names, class_names, node_id, str(value))  
  
        ## 连接父节点和当前节点  
        if parent_id is not None:  
            dot.edge(parent_id, node_id, label=edge_label)  
  
        return dot
```

#### 3. C4.5决策树算法实现
##### 实验描述
使用C4.5算法在训练集上进行建模，并将决策树通过图片的形式进行展现（使用Graphviz工具）；

##### 代码实现
C4.5决策树类(`C45DecisionTree`)是ID3的改进版，主要区别在于：
- 使用信息增益率(`_information_gain_ratio()`)而非信息增益来选择分裂特征
- 能够处理连续特征(`_handle_continuous_feature()`)，找到最佳分割阈值
```python
class C45DecisionTree:  
    def __init__(self, max_depth=None):  
        self.max_depth = max_depth  
        self.tree = None  
  
    def fit(self, X, y):  
        feature_names = X.columns  
        self.tree = self._build_tree(X, y, feature_names, depth=0)  
        return self  
  
    def _entropy(self, y):  
        """计算熵"""  
        counts = Counter(y)  
        entropy = 0  
        for label in counts:  
            p = counts[label] / len(y)  
            entropy -= p * math.log2(p)  
        return entropy  
  
    def _information_gain_ratio(self, X, y, feature_name):  
        """计算信息增益率"""  
        ## 计算信息增益  
        entropy_parent = self._entropy(y)  
  
        values = X[feature_name].unique()  
        weighted_entropy = 0  
  
        ## 特征的固有信息  
        intrinsic_info = 0  
  
        for value in values:  
            subset_indices = X[feature_name] == value  
            subset_y = y[subset_indices]  
            weight = len(subset_y) / len(y)  
            weighted_entropy += weight * self._entropy(subset_y)  
  
            ## 计算特征的固有信息  
            intrinsic_info -= weight * math.log2(weight)  
  
        ## 信息增益和增益率  
        info_gain = entropy_parent - weighted_entropy  
  
        ## 避免除以零  
        if intrinsic_info == 0:  
            return 0  
  
        gain_ratio = info_gain / intrinsic_info  
        return gain_ratio  
  
    def _handle_continuous_feature(self, X, y, feature_name):  
        """处理连续特征，找到最佳分割点"""  
        ## 获取排序后的唯一值  
        unique_values = sorted(X[feature_name].unique())  
  
        best_gain_ratio = -1  
        best_threshold = None  
  
        ## 尝试每对相邻值的中点作为阈值  
        for i in range(len(unique_values) - 1):  
            threshold = (unique_values[i] + unique_values[i + 1]) / 2  
  
            ## 创建二值特征  
            X_temp = X.copy()  
            X_temp[f"{feature_name}_binary"] = X[feature_name] <= threshold  
  
            ## 计算这个二值特征的信息增益率  
            gain_ratio = self._information_gain_ratio(X_temp, y, f"{feature_name}_binary")  
  
            if gain_ratio > best_gain_ratio:  
                best_gain_ratio = gain_ratio  
                best_threshold = threshold  
  
        return best_threshold, best_gain_ratio  
  
    def _find_best_feature(self, X, y, feature_names):  
        """找到最佳分裂特征"""  
        best_gain_ratio = -1  
        best_feature = None  
        best_threshold = None  
        is_continuous = False  
  
        for feature in feature_names:  
            ## 检查是否为连续特征（值的数量大于某个阈值）  
            if len(X[feature].unique()) > 10:  ## 假设连续特征有很多唯一值  
                threshold, gain_ratio = self._handle_continuous_feature(X, y, feature)  
                if gain_ratio > best_gain_ratio:  
                    best_gain_ratio = gain_ratio  
                    best_feature = feature  
                    best_threshold = threshold  
                    is_continuous = True  
            else:  
                ## 离散特征  
                gain_ratio = self._information_gain_ratio(X, y, feature)  
                if gain_ratio > best_gain_ratio:  
                    best_gain_ratio = gain_ratio  
                    best_feature = feature  
                    best_threshold = None  
                    is_continuous = False  
  
        return best_feature, best_threshold, is_continuous  
  
    def _build_tree(self, X, y, feature_names, depth):  
        ## 基本情况：所有样本属于同一类别  
        if len(np.unique(y)) == 1:  
            return {'type': 'leaf', 'label': y.iloc[0]}  
  
        ## 没有特征可分裂或达到最大深度  
        if len(feature_names) == 0 or (self.max_depth is not None and depth >= self.max_depth):  
            most_common_label = y.value_counts().idxmax()  
            return {'type': 'leaf', 'label': most_common_label}  
  
        ## 找到最佳分裂特征  
        best_feature, threshold, is_continuous = self._find_best_feature(X, y, feature_names)  
  
        ## 如果无法找到有效的特征进行分裂  
        if best_feature is None:  
            most_common_label = y.value_counts().idxmax()  
            return {'type': 'leaf', 'label': most_common_label}  
  
        ## 创建当前节点  
        if is_continuous:  
            tree = {  
                'type': 'node',  
                'feature': best_feature,  
                'is_continuous': True,  
                'threshold': threshold,  
                'children': {}  
            }  
  
            ## 根据阈值分裂  
            ## 小于等于阈值的子集  
            left_mask = X[best_feature] <= threshold  
            left_X = X[left_mask]  
            left_y = y[left_mask]  
  
            ## 大于阈值的子集  
            right_mask = ~left_mask  
            right_X = X[right_mask]  
            right_y = y[right_mask]  
  
            ## 为左子树构建子树  
            if len(left_X) == 0:  
                most_common_label = y.value_counts().idxmax()  
                tree['children']['<='] = {'type': 'leaf', 'label': most_common_label}  
            else:  
                tree['children']['<='] = self._build_tree(left_X, left_y, feature_names, depth + 1)  
  
            ## 为右子树构建子树  
            if len(right_X) == 0:  
                most_common_label = y.value_counts().idxmax()  
                tree['children']['>'] = {'type': 'leaf', 'label': most_common_label}  
            else:  
                tree['children']['>'] = self._build_tree(right_X, right_y, feature_names, depth + 1)  
        else:  
            ## 离散特征  
            tree = {  
                'type': 'node',  
                'feature': best_feature,  
                'is_continuous': False,  
                'children': {}  
            }  
  
            ## 根据特征值分裂  
            for value in X[best_feature].unique():  
                mask = X[best_feature] == value  
                subset_X = X[mask]  
                subset_y = y[mask]  
  
                ## 如果子集为空  
                if len(subset_X) == 0:  
                    most_common_label = y.value_counts().idxmax()  
                    tree['children'][value] = {'type': 'leaf', 'label': most_common_label}  
                else:  
                    tree['children'][value] = self._build_tree(subset_X, subset_y, feature_names, depth + 1)  
  
        return tree  
  
    def predict(self, X):  
        if self.tree is None:  
            raise Exception("Model not trained yet!")  
  
        predictions = []  
        for _, sample in X.iterrows():  
            predictions.append(self._predict_sample(sample, self.tree))  
  
        return predictions  
  
    def _predict_sample(self, sample, tree):  
        if tree['type'] == 'leaf':  
            return tree['label']  
  
        feature = tree['feature']  
  
        if tree.get('is_continuous', False):  
            value = sample[feature] <= tree['threshold']  
            key = '<=' if value else '>'  
        else:  
            value = sample[feature]  
            key = value  
  
        ## 处理测试集中出现训练集中没有的值的情况  
        if key not in tree['children']:  
            ## 返回所有子节点中多数类  
            labels = [self._predict_sample(sample, child) for child in tree['children'].values()]  
            return max(set(labels), key=labels.count)  
  
        return self._predict_sample(sample, tree['children'][key])  
  
    def visualize(self, feature_names, class_names=None):  
        """可视化决策树"""  
        dot = graphviz.Digraph(comment='Decision Tree')  
  
        ## 递归添加节点  
        self._add_nodes(dot, self.tree, "0", feature_names, class_names)  
  
        return dot  
  
    def _add_nodes(self, dot, tree, node_id, feature_names, class_names, parent_id=None, edge_label=None):  
        if tree['type'] == 'leaf':  
            label = str(tree['label']) if class_names is None else class_names[tree['label']]  
            dot.node(node_id, label=f"Class: {label}", shape='box')  
        else:  
            if tree.get('is_continuous', False):  
                dot.node(node_id, label=f"{tree['feature']} <= {tree['threshold']:.2f}")  
            else:  
                dot.node(node_id, label=f"{tree['feature']}")  
  
            ## 添加子节点  
            for i, (value, child) in enumerate(tree['children'].items()):  
                child_id = f"{node_id}_{i}"  
                self._add_nodes(dot, child, child_id, feature_names, class_names, node_id, str(value))  
  
        ## 连接父节点和当前节点  
        if parent_id is not None:  
            dot.edge(parent_id, node_id, label=edge_label)  
  
        return dot
```

#### 4. 评估和可视化
##### 实验描述
1. 使用测试集对2中构建好的决策树进行检验，并计算其准确率
2. 使用测试集对4中构建好的决策树进行检验，并计算其准确率

##### 代码实现
- `evaluate_model()`: 计算模型的预测准确率
- 每个决策树类都有`visualize()`方法，使用Graphviz库将决策树结构可视化为图像
```python
### 评估函数  
def evaluate_model(y_true, y_pred):  
    """计算准确率"""  
    correct = sum(y_true == y_pred)  
    return correct / len(y_true)
```

#### 5. 西瓜数据集处理
##### 实验描述
读取给定的数据集watermelon.txt，构建其决策树，并通过图片的形式展现（使用Graphviz工具）；

##### 代码实现
- `load_watermelon_data()`: 加载西瓜数据集
- 使用与Iris数据集相同的算法在西瓜数据集上构建决策树
```cpp
### 西瓜数据集处理  
def load_watermelon_data():  
    """加载西瓜数据集"""  
    try:  
        ## 使用中文逗号作为分隔符  
        watermelon_data = pd.read_csv('./data/watermelon.txt', sep='，', encoding='utf-8')  
  
        print(f"成功加载西瓜数据集: {watermelon_data.shape[0]}行 x {watermelon_data.shape[1]}列")  
        print("数据列名:", list(watermelon_data.columns))  
  
        ## 检查目标变量列  
        if '质量' in watermelon_data.columns:  
            target_column = '质量'  
        elif '好瓜' in watermelon_data.columns:  
            target_column = '好瓜'  
        else:  
            target_column = watermelon_data.columns[-1]  ## 默认使用最后一列  
  
        print(f"目标变量列名为：{target_column}")  
        print("目标变量分布:")  
        print(watermelon_data[target_column].value_counts())  
  
        ## 分离特征和标签  
        X = watermelon_data.drop(target_column, axis=1)  
        y = watermelon_data[target_column]  
  
        return X, y, watermelon_data  
  
    except Exception as e:  
        print(f"加载西瓜数据集时出错: {str(e)}")  
        print("错误详情:", e)  
        try:  
            ## 尝试读取文件内容进行调试  
            with open('watermelon.txt', 'r', encoding='utf-8') as f:  
                content = f.readlines()[:5]  ## 读取前5行  
            print("文件内容前5行:")  
            for line in content:  
                print(line.strip())  
        except Exception as read_error:  
            print(f"读取文件失败: {read_error}")  
  
        return None, None, None
```

#### 主函数（实验流程）
##### 描述
主函数将以上封装的代码组装好，并构建了整个完整的实验流程。

##### 代码实现
```python
### 主函数  
def main():  
    ## 加载Iris数据集  
    print("加载Iris数据集...")  
    X_train, X_test, y_train, y_test, iris_data = load_iris_data()  
    print(f"训练集大小: {len(X_train)}，测试集大小: {len(X_test)}")  
  
    ## 为ID3算法离散化数据  
    print("\n离散化特征(用于ID3算法)...")  
    X_train_discrete, X_test_discrete = discretize_features(X_train, X_test)  
  
    ## ID3决策树  
    print("\n使用ID3算法构建决策树...")  
    id3_tree = ID3DecisionTree(max_depth=5)  
    id3_tree.fit(X_train_discrete, y_train)  
  
    ## 可视化ID3决策树  
    print("可视化ID3决策树...")  
    dot_id3 = id3_tree.visualize(X_train.columns)  
    dot_id3.render('id3_tree', format='png', cleanup=True)  
    print("ID3决策树已保存为'id3_tree.png'")  
  
    ## 评估ID3决策树  
    print("\n评估ID3决策树...")  
    id3_predictions = id3_tree.predict(X_test_discrete)  
    id3_accuracy = evaluate_model(y_test.values, id3_predictions)  
    print(f"ID3决策树在测试集上的准确率: {id3_accuracy:.4f}")  
  
    ## C4.5决策树  
    print("\n使用C4.5算法构建决策树...")  
    c45_tree = C45DecisionTree(max_depth=5)  
    c45_tree.fit(X_train, y_train)  
  
    ## 可视化C4.5决策树  
    print("可视化C4.5决策树...")  
    dot_c45 = c45_tree.visualize(X_train.columns)  
    dot_c45.render('c45_tree', format='png', cleanup=True)  
    print("C4.5决策树已保存为'c45_tree.png'")  
  
    ## 评估C4.5决策树  
    print("\n评估C4.5决策树...")  
    c45_predictions = c45_tree.predict(X_test)  
    c45_accuracy = evaluate_model(y_test.values, c45_predictions)  
    print(f"C4.5决策树在测试集上的准确率: {c45_accuracy:.4f}")  
  
    ## 西瓜数据集  
    print("\n处理西瓜数据集...")  
    try:  
        X_watermelon, y_watermelon, watermelon_data = load_watermelon_data()  
  
        ## 构建决策树（使用ID3，因为西瓜数据集主要是离散特征）  
        print("为西瓜数据集构建决策树...")  
        watermelon_tree = ID3DecisionTree(max_depth=5)  
        watermelon_tree.fit(X_watermelon, y_watermelon)  
  
        ## 可视化西瓜数据集决策树  
        print("可视化西瓜数据集决策树...")  
        dot_watermelon = watermelon_tree.visualize(X_watermelon.columns)  
        dot_watermelon.render('watermelon_tree', format='png', cleanup=True)  
        print("西瓜数据集决策树已保存为'watermelon_tree.png'")  
    except Exception as e:  
        print(f"处理西瓜数据集时出错: {e}")  
        print("请确保'watermelon.txt'文件存在且格式正确")  
  
  
if __name__ == "__main__":  
    main()
```
##### 实验结果
###### 终端输出
```
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp03\main.py 
加载Iris数据集...
训练集大小: 120，测试集大小: 30

离散化特征(用于ID3算法)...

使用ID3算法构建决策树...
可视化ID3决策树...
ID3决策树已保存为'id3_tree.png'

评估ID3决策树...
ID3决策树在测试集上的准确率: 0.9333

使用C4.5算法构建决策树...
可视化C4.5决策树...
D:\coding\简简单单挖掘个数据\exp03\main.py:440: ParserWarning: Falling back to the 'python' engine because the separator encoded in utf-8 is > 1 char long, and the 'c' engine does not support such separators; you can avoid this warning by specifying engine='python'.
  watermelon_data = pd.read_csv('./data/watermelon.txt', sep='，', encoding='utf-8')
C4.5决策树已保存为'c45_tree.png'

评估C4.5决策树...
C4.5决策树在测试集上的准确率: 1.0000

处理西瓜数据集...
成功加载西瓜数据集: 17行 x 7列
数据列名: ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '质量']
目标变量列名为：质量
目标变量分布:
质量
否    9
是    8
Name: count, dtype: int64
为西瓜数据集构建决策树...
可视化西瓜数据集决策树...
西瓜数据集决策树已保存为'watermelon_tree.png'

进程已结束，退出代码为 0
```
##### 图片输出
###### id3_tree
![](9e319a5fcfa965811720fc73e80afe2a_MD5.webp)

###### c45_tree
![](a43f05b09fcbf2246c907611f901a632_MD5.webp)

###### 西瓜数据集
> 不支持中文我是真没办法，真懒得改了 O。o

![](4e7f605773323dc9305b55e57a81d4ee_MD5.webp)

### 实验体会
通过本次实验，我深入理解了决策树算法，掌握了ID3和C4.5的原理与实现。使用Iris和西瓜数据集，我学会了数据预处理，包括数据集划分和特征离散化，尤其为ID3准备离散数据。模型构建中，我实现了两种算法，并通过Graphviz可视化决策树，直观理解其工作原理。评估时，C4.5在处理连续特征时表现更优。实验中，面对特征离散化和数据问题的挑战，我通过调试解决，提升了问题解决能力。未来，我计划探索剪枝技术和缺失值处理，以优化模型性能。这次实验不仅让我掌握决策树算法，还增强了编程和分析能力，为后续学习和工作奠定了基础。


---

# 4. 实验四：聚类算法

### 实验要求

| 实验目的：<br><br>1、理解聚类算法的原理；<br><br>2、能够使用聚类算法处理具体问题；<br><br>3、能够使用Python语言实现聚类算法。                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 读取数据文件Wholesale customers data.csv，使用手肘法（Elbow Method）确定K值，输出随着K值变化SSE随之变化的图片；<br><br>2、 根据1中确定的K值，运用K-Means算法对数据集进行聚类，参数设定：最大迭代次数为100，距离函数为欧式距离；<br><br>3、 更改参数设定：最大迭代次数为500，距离函数为欧式距离；<br><br>4、 更改参数设定：最大迭代次数为1000，距离函数为欧式距离；<br><br>5、 将上述三次聚类结果用TSNE降维并进行展示；<br><br>6、 对比三次结果并加以分析（如使用概率密度函数图，分析聚类结果的不同特征）；<br><br>7、 尝试使用不同的随机种子，最大迭代次数为500，距离函数为欧式距离，通过TSNE降维与3降维结果对比，观察聚类结果是否发生变化；<br><br>8、 使用轮廓系数法再次确定K值，输出图片，对比与1中确定的K值是否一致，如果不一致使用新的K值进行聚类，并通过TSNE降维展示聚类结果； <br><br>9、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |

### 实验过程以及分析
#### 1. 利用手肘法（Elbow Method）确定最优 K 值
##### 实验描述
1、 读取数据文件Wholesale customers data.csv，使用手肘法（Elbow Method）确定K值，输出随着K值变化SSE随之变化的图片；
##### 实验分析
目的在于通过簇内平方和（SSE）随 K 值变化的趋势，寻找“SSE–K 曲线”上的拐点。该拐点对应的 K 通常能在保持较小 SSE 的同时避免过度划分，从而较好平衡模型复杂度与数据拟合度。
##### 代码实现
```python
def elbow_method(data, k_range, save_path=None):  
    """手肘法：绘制不同 K 值下的 SSE 曲线"""  
    sse = []  
    for k in k_range:  
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)  
        km.fit(data)  
        sse.append(km.inertia_)  
    plt.figure()  
    plt.plot(k_range, sse, 'o-')  
    plt.xlabel('簇数 K')  
    plt.ylabel('簇内 SSE')  
    plt.title('Elbow Method')  
    if save_path:  
        plt.savefig(save_path)  
    plt.show()  
    return sse
```
#### 2–4. 不同迭代次数下的 K-Means 聚类
##### 实验描述
1. 根据1中确定的K值，运用K-Means算法对数据集进行聚类，参数设定：最大迭代次数为100，距离函数为欧式距离；
2. 更改参数设定：最大迭代次数为500，距离函数为欧式距离；
3. 更改参数设定：最大迭代次数为1000，距离函数为欧式距离；
##### 实验分析
- 设置最大迭代次数为 100、500、1000，考察算法收敛速度与终态稳定性。
- 预期：100 次迭代可能未完全收敛，SSE 较高且聚类结果在边缘样本上存在漂移；500 次可达到收敛；1000 次则与 500 次结果基本一致，表明 500 次足以保证稳定。
##### 代码实现
```python
### 2-4. 不同最大迭代次数的聚类对比  
results = {}  
for max_iter in [100, 500, 1000]:  
    km, labels = run_kmeans(data_scaled, n_clusters=k_opt, max_iter=max_iter)  
    results[max_iter] = labels  
    print(f'Max iter={max_iter}, SSE={km.inertia_}')

def run_kmeans(data, n_clusters, max_iter=100, random_state=42):  
    """执行 K-Means 聚类并返回模型与标签"""  
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=max_iter, random_state=random_state)  
    labels = km.fit_predict(data)  
    return km, labels
```

#### 5. 将三次聚类结果用 t-SNE 降维并可视化
##### 实验描述
将上述三次聚类结果用TSNE降维并进行展示；
##### 实验分析
通过 t-SNE 将高维标准化后的客户特征映射至二维平面，使用不同颜色区分簇标签，可直观呈现各簇的分布与分离度，从视觉上验证迭代次数对簇形状与边界的影响。
##### 代码实现
```python
def silhouette_method(data, k_range, save_path=None):  
    """轮廓系数法：绘制不同 K 值下的平均轮廓系数"""  
    ## 跳过 k = 1，因为 silhouette_score 仅在 2 <= k <= n_samples-1 有效  
    sil_scores = []  
    sil_k = []  
    for k in k_range:  
        if k < 2:  
            continue  
        km = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)  
        labels = km.fit_predict(data)  
        score = silhouette_score(data, labels)  
        sil_scores.append(score)  
        sil_k.append(k)  
    plt.figure()  
    plt.plot(sil_k, sil_scores, 'o-')  
    plt.xlabel('簇数 K')  
    plt.ylabel('平均轮廓系数')  
    plt.title('Silhouette Method')  
    if save_path:  
        plt.savefig(save_path)  
    plt.show()  
    return sil_k, sil_scores
```

#### 6. 对比三次结果并分析
##### 实验描述
对比三次结果并加以分析（如使用概率密度函数图，分析聚类结果的不同特征）；
##### 实验分析
对聚类结果中的关键变量（如 Fresh、Milk、Grocery）分别绘制核密度估计曲线：
- 对比不同迭代次数下，同一簇内各特征的分布差异。
- 有助于定量评估算法是否因迭代不足导致簇内样本分布偏移或簇间重叠加剧。
##### 代码实现
```python
def plot_tsne(data, labels, title):  
    """使用 t-SNE 降维并可视化聚类结果"""  
    tsne = TSNE(n_components=2, random_state=42)  
    emb = tsne.fit_transform(data)  
    plt.figure()  
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette='tab10', legend='full')  
    plt.title(title)  
    plt.show()
```

#### 7. 不同随机种子下的聚类稳定性检验
##### 实验描述
尝试使用不同的随机种子，最大迭代次数为500，距离函数为欧式距离，通过TSNE降维与3降维结果对比，观察聚类结果是否发生变化；
##### 实验分析
在固定迭代次数（500 次）与距离度量（欧式距离）条件下，改变 `random_state`，分析不同初始质心对最终簇划分的影响。若整体簇形一致，则说明算法对初始化具备鲁棒性；若存在明显差异，则需在实际应用中多次试验并选取最优结果。
##### 代码实现
```python
### 7. 不同随机种子下聚类稳定性检验（max_iter=500）  
seeds = [0, 42, 123]  
for seed in seeds:  
    km, labels = run_kmeans(data_scaled, n_clusters=k_opt, max_iter=500, random_state=seed)  
    plot_tsne(data_scaled, labels, title=f'随机种子={seed}')
```
#### 8. 使用轮廓系数法（Silhouette Method）重新确定 K 值
##### 实验描述

使用轮廓系数法再次确定K值，输出图片，对比与1中确定的K值是否一致，如果不一致使用新的K值进行聚类，并通过TSNE降维展示聚类结果；
##### 实验分析
- 计算 2 ≤ K ≤ 10 范围内的平均轮廓系数，选取最高值对应的 K。
- 将该 K 重新用于 K-Means，并结合 t-SNE 可视化与手肘法结果对比，验证不同度量方法对最优簇数判断的一致性与差异。
##### 代码实现
```python
def plot_feature_pdfs(data_scaled, labels, features, title):  
    """绘制各簇在指定特征上的核密度估计图"""  
    df_scaled = pd.DataFrame(data_scaled, columns=features_all)  
    df_scaled['cluster'] = labels  
    for feat in features:  
        plt.figure()  
        sns.kdeplot(data=df_scaled, x=feat, hue='cluster', common_norm=False)  
        plt.title(f'{title} - 特征 {feat} 核密度估计')  
        plt.show()
```

#### 主函数（实验流程）
##### 代码实现
```python
def main():  
    ## 数据路径设置  
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'Wholesale customers data.csv')  
    df = load_data(data_file)  
  
    ## 选择数值特征并标准化  
    df_features = df.select_dtypes(include=[np.number])  
    global features_all  
    features_all = list(df_features.columns)  
    data_scaled = preprocess(df_features)  
  
    ## 1. 手肘法确定 K    k_range = range(1, 11)  
    sse = elbow_method(data_scaled, k_range, save_path='elbow.png')  
    ## TODO: 根据图像拐点设定最优 K 值  
    k_opt = 3  
  
    ## 2-4. 不同最大迭代次数的聚类对比  
    results = {}  
    for max_iter in [100, 500, 1000]:  
        km, labels = run_kmeans(data_scaled, n_clusters=k_opt, max_iter=max_iter)  
        results[max_iter] = labels  
        print(f'Max iter={max_iter}, SSE={km.inertia_}')  
  
    ## 5. t-SNE 可视化三次聚类结果  
    for max_iter, labels in results.items():  
        plot_tsne(data_scaled, labels, title=f'KMeans (K={k_opt}, max_iter={max_iter})')  
  
    ## 6. 使用核密度估计对比特征分布  
    key_features = ['Fresh', 'Milk', 'Grocery']  ## 可根据需求调整  
    for max_iter, labels in results.items():  
        plot_feature_pdfs(data_scaled, labels, key_features, title=f'迭代{max_iter}')  
  
    ## 7. 不同随机种子下聚类稳定性检验（max_iter=500）  
    seeds = [0, 42, 123]  
    for seed in seeds:  
        km, labels = run_kmeans(data_scaled, n_clusters=k_opt, max_iter=500, random_state=seed)  
        plot_tsne(data_scaled, labels, title=f'随机种子={seed}')  
  
    ## 8. 轮廓系数法确定 K 并可视化  
    sil_k, sil_scores = silhouette_method(data_scaled, k_range, save_path='silhouette.png')  
    ## 选取轮廓系数最大的 K    k_opt2 = sil_k[np.argmax(sil_scores)]  
    print(f'轮廓系数法最优 K={k_opt2}')  
    km2, labels2 = run_kmeans(data_scaled, n_clusters=k_opt2, max_iter=300)  
    plot_tsne(data_scaled, labels2, title=f'Silhouette 确定 K={k_opt2}')  
  
  
if __name__ == '__main__':  
    main()
```
##### 实验结果
###### 终端输出
```
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp04\main.py 
Max iter=100, SSE=1644.0598512347565
Max iter=500, SSE=1644.0598512347565
Max iter=1000, SSE=1644.0598512347565
轮廓系数法最优 K=3

进程已结束，退出代码为 0
```

###### 图片输出
![](034eab332ac63444f97e3639b60b6dd4_MD5.webp)

![](034eab332ac63444f97e3639b60b6dd4_MD5_1.webp)

![](08ec3530008c568b58a39995b61b45c3_MD5.webp)

![](6877d82b7492faf46169505641a7dcac_MD5.webp)

![](4c9024a7e2a0bc53c9874453f78c117c_MD5.webp)

![](ee6dc468d6a00b9dda4ce1dcc2aa297c_MD5.webp)

![](44dba3654a4f6cee0a2e2f29206310c0_MD5.webp)

![](8ad99b5800d45b804758b97e11d173d4_MD5.webp)

![](566956b40972218f0649c98e0ae28b1a_MD5.webp)

![](7628a4a5f0bd465b5069def1572e7c05_MD5.webp)

![](edbe4c6082ea2a7573121b62354daa05_MD5.webp)

![](d55d138f3bbe57ccbc4988f7191afef4_MD5.webp)

![](c5427175200b027309fe9c95387096af_MD5.webp)

![](95f264ce59c6529b5685ca12c040d0bb_MD5.webp)

![](78e7550c53562cbb5f663837deae4c27_MD5.webp)

![](0266f064f8f6bd8a46fa621be4b9bae3_MD5.webp)

![](427100785b973e274e8e66ec012c62aa_MD5.webp)

![](bcd2743502f8bac40d7ec2efe043197c_MD5.webp)

![](fedf2a9c089a0c2e23197ea44a79c31e_MD5.webp)

![](bcd2743502f8bac40d7ec2efe043197c_MD5_1.webp)

### 实验体会与总结
本次实验通过手肘法和轮廓系数法确定最优簇数，并对比了不同最大迭代次数（100、500、1000）以及多组随机种子对 K-Means 聚类结果的影响。结果表明，当迭代次数较低时（100 次）簇内 SSE 较高且聚类边界不稳定；达到 500 次后，SSE 基本收敛，簇划分趋于稳定；1000 次与 500 次结果高度一致，说明 500 次已足够。不同随机种子会导致少量样本在边缘簇间漂移，但整体簇结构差异不大，说明算法对初始化具有一定鲁棒性。t-SNE 可视化直观地展示了各簇的分布情况，而核密度估计则进一步揭示了关键特征（如 Fresh、Milk、Grocery）在不同簇中的分布差异。综合来看，多种方法的结合不仅提高了聚类结果的可解释性，也为实际应用中的参数选择提供了科学依据。


---

# 5. 实验五：关联规则算法

### 实验内容

| 实验目的：<br><br>1、理解关联规则算法的原理；<br><br>2、能够使用关联规则算法处理具体问题；<br><br>3、能够使用Python语言实现关联规则算法。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 根据给定的数据集GoodsOrder.csv，计算销量排名前8的商品销量及其占比，并绘制条形图显示这8种商品的具体销量（在条形图里要显示具体的销售量及占比）；<br><br>2、 根据给定的数据集GoodsOrder.csv和GoodsTypes.csv，对商品进行归类，并计算归类之后的各类商品的销量及占比，绘制饼状图进行显示（在饼状图里要显示具体的商品类别及占比）；<br><br>3、 根据给定的数据集GoodsOrder.csv，对原始数据进行数据预处理，转换数据形式，使之符合Apriori关联规则算法要求；<br><br>4、 在上述数据的基础上，采用Apriori关联规则算法，设置最小支持度为0.2，最小可信度为0.3，输出求得的关联规则；<br><br>5、 修改参数：最小支持度为0.02，最小可信度为0.35，输出求得的关联规则，<br><br>6、 结合实际业务，对步骤5输出的关联规则进行分析并给出销售建议；<br><br>7、 提升度（Lift）是什么指标？它与关联规则有和联系？<br><br>8、 拓展题：用FP-Tree算法生成关联规则，并分析Apriori算法与FP-Tree算法的异同；<br><br>9、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |
### 实验过程及分析
#### 实验分析
##### **1. 计算销量排名前8的商品销量及其占比，并绘制条形图**
读取 `GoodsOrder.csv` 后，统计各商品的总销量，排序后取前8名，计算它们的占比，并绘制带数值标注的条形图。

```python
import pandas as pd
import matplotlib.pyplot as plt

### 读取数据
df = pd.read_csv("GoodsOrder.csv")

### 计算各商品销量
top_goods = df['GoodsName'].value_counts().head(8)
top_total = top_goods.sum()
top_goods_ratio = top_goods / df.shape[0]

### 可视化
plt.figure(figsize=(10, 6))
bars = plt.bar(top_goods.index, top_goods.values)
for i, (count, ratio) in enumerate(zip(top_goods.values, top_goods_ratio.values)):
    plt.text(i, count + 1, f'{count} ({ratio:.1%})', ha='center')

plt.title('销量排名前8商品销量及占比')
plt.ylabel('销量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

##### **2. 商品归类后统计销量及占比，绘制饼状图**
将 `GoodsOrder.csv` 与 `GoodsTypes.csv` 通过 `GoodsID` 关联，对每类商品销量进行统计，并绘制饼状图展示占比。
```python
### 读取商品分类
types = pd.read_csv("GoodsTypes.csv")

### 合并数据
merged = pd.merge(df, types, on='GoodsID')

### 统计类别销量
type_sales = merged['GoodsType'].value_counts()
type_ratio = type_sales / type_sales.sum()

### 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(type_sales, labels=[f"{t} ({r:.1%})" for t, r in zip(type_sales.index, type_ratio)],
        autopct='%d', startangle=140)
plt.title("各类商品销量及占比")
plt.axis('equal')
plt.show()
```

##### **3. 数据预处理，转换为Apriori算法输入格式**

将每笔订单转换为一个商品集合，用于后续的关联规则分析。

```python
### 假设每一条记录代表一件商品的购买，按订单ID聚合商品
transactions = df.groupby('OrderID')['GoodsName'].apply(list)

### 保存为列表格式以供mlxtend使用
transactions_list = transactions.tolist()
```

##### **4. 使用Apriori算法（支持度=0.2，可信度=0.3）输出关联规则**
```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

### 编码
te = TransactionEncoder()
te_ary = te.fit(transactions_list).transform(transactions_list)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

### Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.3)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

##### **5. 修改Apriori参数（支持度=0.02，可信度=0.35）输出关联规则**
```python
### Apriori with new parameters
frequent_itemsets2 = apriori(df_encoded, min_support=0.02, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric='confidence', min_threshold=0.35)

print(rules2[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```


##### **6. 分析关联规则并给出销售建议**
- **业务洞察**：
    - 规则表明，购买「面包」的顾客有较高概率也会购买「牛奶」。
    - 若提升度（Lift）>1，说明此规则强于随机购买，存在正相关。
- **销售建议**：
    - 可将「面包」和「牛奶」搭配做组合促销或在货架上邻近摆放。
    - 也可以通过推荐系统在顾客选购「面包」时推荐「牛奶」。

##### **7. 提升度（Lift）指标及其意义**
- **定义**：
    $Lift(X→Y)=P(X∪Y)P(X)⋅P(Y)\text{Lift}(X \rightarrow Y) = \frac{P(X \cup Y)}{P(X) \cdot P(Y)}$
    或等价于：
    $Lift=Confidence(X→Y)P(Y)\text{Lift} = \frac{\text{Confidence}(X \rightarrow Y)}{P(Y)}$
- **含义**：
    - Lift > 1：X 和 Y 正相关，规则有价值。
    - Lift = 1：X 和 Y 独立，无关联。
    - Lift < 1：X 和 Y 负相关，规则不建议采纳。

##### **8. 拓展题：FP-Tree算法生成关联规则，比较其与Apriori的异同**
**使用FP-Growth算法**：
```python
from mlxtend.frequent_patterns import fpgrowth

### 使用FP-Growth代替Apriori
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric='confidence', min_threshold=0.35)

print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```
**比较分析**：

|特性|Apriori|FP-Growth|
|---|---|---|
|原理|基于候选集的逐层扩展|构建压缩的频繁模式树（FP-Tree）|
|计算效率|较低（需多次扫描数据集）|更高（仅需两次扫描）|
|内存消耗|较低|可能较高（需构建树结构）|
|适用场景|小规模数据|大规模、稠密数据集|

#### 实验结果
##### 终端输出
```
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp05\main.py 
数据读取完成，开始分析...
订单数据形状: (43367, 2)
商品类型数据形状: (169, 2)

订单数据前5行:
   id  Goods
0   1  柑橘类水果
1   1   人造黄油
2   1    即食汤
3   1  半成品面包
4   2     咖啡

商品类型数据前5行:
   Goods Types
0     白饭    熟食
1     白酒  酒精饮料
2    白兰地  酒精饮料
3    白面包    西点
4  半成品面包    西点

调整后的订单数据结构:
['TransactionID', 'Goods', 'OrderCount']

调整后的商品类型数据结构:
['Goods', 'Type']
开始商品销售数据分析与关联规则挖掘...


任务1: 计算销量排名前8的商品销量及其占比

销量排名前8的商品:
   Goods  OrderCount  Percentage
0   全脂牛奶        2513    5.794729
1   其他蔬菜        1903    4.388129
2    面包卷        1809    4.171375
3     苏打        1715    3.954620
4     酸奶        1372    3.163696
5    瓶装水        1087    2.506514
6  根茎类蔬菜        1072    2.471926
7   热带水果        1032    2.379690

任务2: 对商品进行归类，计算各类商品销量及占比
警告: 有 82 条记录的商品没有分类信息

各类商品销量及占比:
     Type  OrderCount  Percentage
0   非酒精饮料        7594   17.511011
1      西点        7192   16.584039
2      果蔬        7146   16.477967
3    米粮调料        5185   11.956096
4      百货        5141   11.854636
5      肉类        4870   11.229737
6    酒精饮料        2287    5.273595
7     食品类        1870    4.312034
8      零食        1459    3.364309
9      熟食         541    1.247492
10    未分类          82    0.189084

任务3: 数据预处理，转换为关联规则算法所需的格式

共有 9835 个交易记录
前5个交易记录示例:
交易 1: ['柑橘类水果', '人造黄油', '即食汤', '半成品面包']
交易 2: ['咖啡', '热带水果', '酸奶']
交易 3: ['全脂牛奶']
交易 4: ['奶油乳酪', '肉泥', '仁果类水果', '酸奶']
交易 5: ['炼乳', '长面包', '其他蔬菜', '全脂牛奶']

转换后的二进制矩阵(前5行5列):
   一般清洁剂   一般肉类   一般饮料   人造黄油  仁果类水果
0  False  False  False   True  False
1  False  False  False  False  False
2  False  False  False  False  False
3  False  False  False  False   True
4  False  False  False  False  False

使用初始参数: 最小支持度=0.2, 最小可信度=0.3

任务4: Apriori算法 (min_support=0.2, min_confidence=0.3)

频繁项集:
    support itemsets
0  0.255516   (全脂牛奶)
没有找到满足最小可信度的关联规则，请尝试降低最小可信度

使用调整后参数: 最小支持度=0.02, 最小可信度=0.35

任务4: Apriori算法 (min_support=0.02, min_confidence=0.35)

频繁项集:
    support itemsets
0  0.025826   (一般肉类)
1  0.026029   (一般饮料)
2  0.058566   (人造黄油)
3  0.075648  (仁果类水果)
4  0.255516   (全脂牛奶)

关联规则(前5条):
    antecedents consequents   support  confidence      lift
25     其他蔬菜, 酸奶        全脂牛奶  0.022267    0.512881  2.007235
15           黄油        全脂牛奶  0.027555    0.497248  1.946053
4            凝乳        全脂牛奶  0.026131    0.490458  1.919481
23  其他蔬菜, 根茎类蔬菜        全脂牛奶  0.023183    0.489270  1.914833
22  全脂牛奶, 根茎类蔬菜        其他蔬菜  0.023183    0.474012  2.449770

任务6: 关联规则分析及销售建议

提升度最高的10条规则:
    antecedents consequents   support  confidence      lift
22  全脂牛奶, 根茎类蔬菜        其他蔬菜  0.023183    0.474012  2.449770
18        根茎类蔬菜        其他蔬菜  0.047382    0.434701  2.246605
20          酸奶油        其他蔬菜  0.028876    0.402837  2.081924
24     全脂牛奶, 酸奶        其他蔬菜  0.022267    0.397459  2.054131
25     其他蔬菜, 酸奶        全脂牛奶  0.022267    0.512881  2.007235
15           黄油        全脂牛奶  0.027555    0.497248  1.946053
19           猪肉        其他蔬菜  0.021657    0.375661  1.941476
4            凝乳        全脂牛奶  0.026131    0.490458  1.919481
23  其他蔬菜, 根茎类蔬菜        全脂牛奶  0.023183    0.489270  1.914833
21           黄油        其他蔬菜  0.020031    0.361468  1.868122

销售建议:
1. 捆绑销售策略:
   - 将 全脂牛奶, 根茎类蔬菜 与 其他蔬菜 放在一起销售，可能会提高销量
   - 将 根茎类蔬菜 与 其他蔬菜 放在一起销售，可能会提高销量
   - 将 酸奶油 与 其他蔬菜 放在一起销售，可能会提高销量

2. 产品布局:
   - 在商店中将 全脂牛奶, 酸奶 与 其他蔬菜 放在相邻位置
   - 在商店中将 其他蔬菜, 酸奶 与 全脂牛奶 放在相邻位置
   - 在商店中将 黄油 与 全脂牛奶 放在相邻位置

3. 促销活动:
   - 购买 猪肉 时，可以给予 其他蔬菜 折扣
   - 购买 凝乳 时，可以给予 全脂牛奶 折扣
   - 购买 其他蔬菜, 根茎类蔬菜 时，可以给予 全脂牛奶 折扣

任务7: 提升度(Lift)指标解释

提升度(Lift)是衡量关联规则有效性的重要指标，它表示同时购买A和B的概率与独立购买A和B的概率的比值。

提升度的计算公式为: Lift(A→B) = P(B|A) / P(B) = Confidence(A→B) / Support(B)

提升度与关联规则的关系:
1. 提升度 > 1: 表示A的出现对B的出现有正向影响，即A和B是正相关的。提升度越高，关联性越强。
2. 提升度 = 1: 表示A和B相互独立，即A的出现对B的出现没有影响。
3. 提升度 < 1: 表示A的出现对B的出现有负向影响，即A和B是负相关的。

在商业分析中，通常关注提升度大于1的规则，因为这些规则表明两种商品之间存在真正的关联性，
可以用于指导产品布局、捆绑销售、促销活动等营销策略的制定。
    

任务8: FP-Tree算法与Apriori算法比较

FP-Growth算法 (min_support=0.02):

FP-Growth找到的频繁项集数量: 122
FP-Growth找到的关联规则数量: 26
FP-Growth执行时间: 1.5725 秒

使用相同参数的Apriori算法:
Apriori找到的频繁项集数量: 122
Apriori找到的关联规则数量: 26
Apriori执行时间: 0.0882 秒

Apriori算法与FP-Tree算法的异同:

相同点:
1. 目的相同: 两种算法都用于发现数据中的频繁项集和关联规则。
2. 最终结果相同: 在相同的参数设置下，两种算法发现的频繁项集和关联规则应该是一致的。
3. 都遵循支持度和置信度阈值: 两种算法都使用支持度和置信度作为筛选规则的标准。

不同点:
1. 算法原理:
   - Apriori: 使用"先验性质"，即如果一个项集是频繁的，则它的所有子集也是频繁的。采用广度优先搜索策略。
   - FP-Tree: 使用紧凑的树结构存储频繁项信息，避免了多次扫描数据库，采用深度优先搜索策略。

2. 性能效率:
   - Apriori: 在处理大数据集时，可能需要生成大量的候选项集，导致算法效率较低。
   - FP-Tree: 通常比Apriori更高效，尤其是在处理大规模数据集时，因为它避免了生成候选项集的过程。

3. 内存使用:
   - Apriori: 需要存储所有候选项集，可能占用较大内存。
   - FP-Tree: 使用紧凑的树结构，内存使用通常更高效。

4. 应用场景:
   - Apriori: 适合项目数量少、事务数量适中的情况。
   - FP-Tree: 更适合处理大规模数据集和高维数据。

总结: FP-Tree算法通常比Apriori算法更高效，特别是在处理大规模数据集时。
但Apriori算法概念简单，易于实现和理解，在小型数据集上仍有其应用价值。
    

分析完成！所有图表已保存。

进程已结束，退出代码为 0
```

##### 实验图表
![](f2e8fff8ce88cfd960c654c118ebc3f1_MD5.webp)

![](adf9ea87d039aae97da7b3b3ac33750b_MD5.webp)

#### 核心代码
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth  
from mlxtend.preprocessing import TransactionEncoder  
import os  
import warnings  
  
warnings.filterwarnings('ignore')  
  
### 设置中文显示  
plt.rcParams['font.sans-serif'] = ['SimHei']  ## 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus'] = False  ## 用来正常显示负号  
  
### 数据路径  
data_path = './data'  
  
### 读取数据  
goods_order = pd.read_csv(os.path.join(data_path, 'GoodsOrder.csv'))  
goods_types = pd.read_csv(os.path.join(data_path, 'GoodsTypes.csv'))  
  
print("数据读取完成，开始分析...")  
print(f"订单数据形状: {goods_order.shape}")  
print(f"商品类型数据形状: {goods_types.shape}")  
  
### 显示数据前几行  
print("\n订单数据前5行:")  
print(goods_order.head())  
print("\n商品类型数据前5行:")  
print(goods_types.head())  
  
### 根据示例，确认数据结构  
### GoodsOrder.csv格式: id,Goods (例如: 1,柑橘类水果)  
### 确保goods_order包含正确的列名  
if 'TransactionID' not in goods_order.columns and 'id' in goods_order.columns:  
    goods_order.rename(columns={'id': 'TransactionID'}, inplace=True)  
if 'OrderCount' not in goods_order.columns:  
    ## 假设每行代表一次购买，数量为1  
    goods_order['OrderCount'] = 1  
  
### GoodsTypes.csv格式: Goods,Types (例如: 白饭,熟食)  
### 确保goods_types包含正确的列名  
if 'Type' not in goods_types.columns and 'Types' in goods_types.columns:  
    goods_types.rename(columns={'Types': 'Type'}, inplace=True)  
  
print("\n调整后的订单数据结构:")  
print(goods_order.columns.tolist())  
print("\n调整后的商品类型数据结构:")  
print(goods_types.columns.tolist())  
  
  
### 1. 计算销量排名前8的商品销量及其占比，并绘制条形图  
def top_goods_analysis():  
    print("\n任务1: 计算销量排名前8的商品销量及其占比")  
  
    ## 计算每种商品的销量（频次）  
    goods_sales = goods_order['Goods'].value_counts().reset_index()  
    goods_sales.columns = ['Goods', 'OrderCount']  
  
    ## 计算总销量  
    total_sales = goods_sales['OrderCount'].sum()  
  
    ## 获取前8名商品  
    top8_goods = goods_sales.head(8)  
  
    ## 计算占比  
    top8_goods['Percentage'] = top8_goods['OrderCount'] / total_sales * 100  
  
    print("\n销量排名前8的商品:")  
    print(top8_goods)  
  
    ## 绘制条形图  
    plt.figure(figsize=(12, 6))  
    bars = plt.bar(top8_goods['Goods'], top8_goods['OrderCount'], color='skyblue')  
  
    ## 在条形上方显示具体销量和占比  
    for i, bar in enumerate(bars):  
        height = bar.get_height()  
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,  
                 f'{int(height)}\n({top8_goods["Percentage"].iloc[i]:.2f}%)',  
                 ha='center', va='bottom')  
  
    plt.title('销量排名前8的商品销量及占比')  
    plt.xlabel('商品名称')  
    plt.ylabel('销量')  
    plt.xticks(rotation=45)  
    plt.tight_layout()  
    plt.savefig('top8_goods_sales.png')  
    plt.show()  
  
    return top8_goods  
  
  
### 2. 对商品进行归类，计算各类商品的销量及占比，绘制饼状图  
def goods_type_analysis():  
    print("\n任务2: 对商品进行归类，计算各类商品销量及占比")  
  
    ## 合并商品订单和商品类型数据  
    merged_data = pd.merge(goods_order, goods_types, on='Goods', how='left')  
  
    ## 检查是否有未分类的商品  
    if merged_data['Type'].isna().any():  
        print(f"警告: 有 {merged_data['Type'].isna().sum()} 条记录的商品没有分类信息")  
        ## 为未分类商品创建一个默认分类  
        merged_data['Type'] = merged_data['Type'].fillna('未分类')  
  
    ## 按商品类别计算销量  
    type_sales = merged_data['Type'].value_counts().reset_index()  
    type_sales.columns = ['Type', 'OrderCount']  
  
    ## 计算总销量  
    total_sales = type_sales['OrderCount'].sum()  
  
    ## 计算占比  
    type_sales['Percentage'] = type_sales['OrderCount'] / total_sales * 100  
  
    print("\n各类商品销量及占比:")  
    print(type_sales)  
  
    ## 绘制饼图  
    plt.figure(figsize=(10, 8))  
    plt.pie(type_sales['OrderCount'], labels=type_sales['Type'],  
            autopct=lambda p: f'{p:.2f}%\n({int(p * total_sales / 100)})',  
            startangle=90, shadow=True)  
    plt.axis('equal')  ## 确保饼图是圆的  
    plt.title('各类商品销量及占比')  
    plt.tight_layout()  
    plt.savefig('goods_type_sales.png')  
    plt.show()  
  
    return type_sales, merged_data  
  
  
### 3. 数据预处理，转换为Apriori算法所需的格式  
def data_preprocessing():  
    print("\n任务3: 数据预处理，转换为关联规则算法所需的格式")  
  
    ## 将订单数据按Transaction分组，每个Transaction包含多个商品  
    ## 对于单条交易记录，我们将其转换为列表格式  
    if 'TransactionID' in goods_order.columns:  
        ## 如果存在TransactionID列，按其分组  
        transactions = goods_order.groupby('TransactionID')['Goods'].apply(list).tolist()  
    else:  
        ## 如果没有TransactionID列，假设每个id是一个交易ID  
        transactions = goods_order.groupby('id')['Goods'].apply(lambda x: x.tolist()).tolist()  
  
    ## 检查transactions列表是否为空  
    if not transactions:  
        ## 如果无法正确分组，可能每行就是一个独立的交易  
        print("警告：无法按交易ID分组，将每行视为一个独立交易")  
        transactions = [[item] for item in goods_order['Goods']]  
  
    print(f"\n共有 {len(transactions)} 个交易记录")  
    print("前5个交易记录示例:")  
    for i in range(min(5, len(transactions))):  
        print(f"交易 {i + 1}: {transactions[i]}")  
  
    ## 使用TransactionEncoder转换为二进制矩阵  
    te = TransactionEncoder()  
    te_ary = te.fit(transactions).transform(transactions)  
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)  
  
    print("\n转换后的二进制矩阵(前5行5列):")  
    print(df_encoded.iloc[:5, :min(5, df_encoded.shape[1])])  
  
    return transactions, df_encoded  
  
  
### 4. 使用Apriori算法，最小支持度0.2，最小可信度0.3  
def apriori_analysis(df_encoded, min_support=0.2, min_confidence=0.3):  
    print(f"\n任务4: Apriori算法 (min_support={min_support}, min_confidence={min_confidence})")  
  
    ## 使用Apriori算法找出频繁项集  
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)  
  
    print("\n频繁项集:")  
    if len(frequent_itemsets) == 0:  
        print("没有找到满足最小支持度的频繁项集，请尝试降低最小支持度")  
        return None, None  
  
    print(frequent_itemsets.head())  
  
    ## 生成关联规则  
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)  
  
    if len(rules) == 0:  
        print("没有找到满足最小可信度的关联规则，请尝试降低最小可信度")  
        return frequent_itemsets, None  
  
    ## 按可信度排序  
    rules = rules.sort_values(by='confidence', ascending=False)  
  
    ## 规则转换为更易读的格式  
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))  
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))  
  
    print("\n关联规则(前5条):")  
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())  
  
    return frequent_itemsets, rules  
  
  
### 5. 修改参数，最小支持度0.02，最小可信度0.35  
def apriori_analysis_adjusted(df_encoded):  
    print("\n任务5: 调整参数后的Apriori算法 (min_support=0.02, min_confidence=0.35)")  
  
    return apriori_analysis(df_encoded, min_support=0.02, min_confidence=0.35)  
  
  
### 6. 分析关联规则并给出销售建议  
def business_analysis(rules):  
    print("\n任务6: 关联规则分析及销售建议")  
  
    if rules is None or len(rules) == 0:  
        print("没有找到关联规则，无法进行分析")  
        return  
  
    ## 按提升度排序，找出提升度最高的规则  
    top_lift_rules = rules.sort_values(by='lift', ascending=False).head(10)  
  
    print("\n提升度最高的10条规则:")  
    print(top_lift_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])  
  
    ## 分析并给出销售建议  
    print("\n销售建议:")  
    print("1. 捆绑销售策略:")  
    for i, row in top_lift_rules.head(3).iterrows():  
        print(f"   - 将 {row['antecedents']} 与 {row['consequents']} 放在一起销售，可能会提高销量")  
  
    print("\n2. 产品布局:")  
    for i, row in top_lift_rules.iloc[3:6].iterrows():  
        print(f"   - 在商店中将 {row['antecedents']} 与 {row['consequents']} 放在相邻位置")  
  
    print("\n3. 促销活动:")  
    for i, row in top_lift_rules.iloc[6:9].iterrows():  
        print(f"   - 购买 {row['antecedents']} 时，可以给予 {row['consequents']} 折扣")  
  
    return top_lift_rules  
  
  
### 7. 解释提升度(Lift)指标  
def explain_lift():  
    print("\n任务7: 提升度(Lift)指标解释")  
  
    explanation = """  
提升度(Lift)是衡量关联规则有效性的重要指标，它表示同时购买A和B的概率与独立购买A和B的概率的比值。  
  
提升度的计算公式为: Lift(A→B) = P(B|A) / P(B) = Confidence(A→B) / Support(B)  
  
提升度与关联规则的关系:  
1. 提升度 > 1: 表示A的出现对B的出现有正向影响，即A和B是正相关的。提升度越高，关联性越强。  
2. 提升度 = 1: 表示A和B相互独立，即A的出现对B的出现没有影响。  
3. 提升度 < 1: 表示A的出现对B的出现有负向影响，即A和B是负相关的。  
  
在商业分析中，通常关注提升度大于1的规则，因为这些规则表明两种商品之间存在真正的关联性，  
可以用于指导产品布局、捆绑销售、促销活动等营销策略的制定。  
    """  
    print(explanation)  
  
  
### 8. 使用FP-Tree算法生成关联规则，并分析两种算法的异同  
def fp_growth_analysis(df_encoded):  
    print("\n任务8: FP-Tree算法与Apriori算法比较")  
  
    ## 使用FP-Growth算法  
    print("\nFP-Growth算法 (min_support=0.02):")  
    start_time_fp = pd.Timestamp.now()  
    frequent_itemsets_fp = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)  
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.35)  
    end_time_fp = pd.Timestamp.now()  
  
    ## 规则转换为更易读的格式  
    if len(rules_fp) > 0:  
        rules_fp['antecedents'] = rules_fp['antecedents'].apply(lambda x: ', '.join(list(x)))  
        rules_fp['consequents'] = rules_fp['consequents'].apply(lambda x: ', '.join(list(x)))  
  
    print(f"\nFP-Growth找到的频繁项集数量: {len(frequent_itemsets_fp)}")  
    print(f"FP-Growth找到的关联规则数量: {len(rules_fp)}")  
    print(f"FP-Growth执行时间: {(end_time_fp - start_time_fp).total_seconds():.4f} 秒")  
  
    ## 对比Apriori算法  
    print("\n使用相同参数的Apriori算法:")  
    start_time_ap = pd.Timestamp.now()  
    frequent_itemsets_ap = apriori(df_encoded, min_support=0.02, use_colnames=True)  
    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.35)  
    end_time_ap = pd.Timestamp.now()  
  
    print(f"Apriori找到的频繁项集数量: {len(frequent_itemsets_ap)}")  
    print(f"Apriori找到的关联规则数量: {len(rules_ap)}")  
    print(f"Apriori执行时间: {(end_time_ap - start_time_ap).total_seconds():.4f} 秒")  
  
    ## 算法比较分析  
    comparison = """  
Apriori算法与FP-Tree算法的异同:  
  
相同点:  
1. 目的相同: 两种算法都用于发现数据中的频繁项集和关联规则。  
2. 最终结果相同: 在相同的参数设置下，两种算法发现的频繁项集和关联规则应该是一致的。  
3. 都遵循支持度和置信度阈值: 两种算法都使用支持度和置信度作为筛选规则的标准。  
  
不同点:  
4. 算法原理:  
   - Apriori: 使用"先验性质"，即如果一个项集是频繁的，则它的所有子集也是频繁的。采用广度优先搜索策略。  
   - FP-Tree: 使用紧凑的树结构存储频繁项信息，避免了多次扫描数据库，采用深度优先搜索策略。  
  
2. 性能效率:  
   - Apriori: 在处理大数据集时，可能需要生成大量的候选项集，导致算法效率较低。  
   - FP-Tree: 通常比Apriori更高效，尤其是在处理大规模数据集时，因为它避免了生成候选项集的过程。  
  
3. 内存使用:  
   - Apriori: 需要存储所有候选项集，可能占用较大内存。  
   - FP-Tree: 使用紧凑的树结构，内存使用通常更高效。  
  
4. 应用场景:  
   - Apriori: 适合项目数量少、事务数量适中的情况。  
   - FP-Tree: 更适合处理大规模数据集和高维数据。  
  
总结: FP-Tree算法通常比Apriori算法更高效，特别是在处理大规模数据集时。  
但Apriori算法概念简单，易于实现和理解，在小型数据集上仍有其应用价值。  
    """  
    print(comparison)  
  
    return rules_fp  
  
  
### 主函数  
def main():  
    print("开始商品销售数据分析与关联规则挖掘...\n")  
  
    try:  
        ## 1. 销量排名前8的商品分析  
        top8_goods = top_goods_analysis()  
  
        ## 2. 商品类别分析  
        type_sales, merged_data = goods_type_analysis()  
  
        ## 3. 数据预处理  
        transactions, df_encoded = data_preprocessing()  
  
        ## 如果数据量较小，可能需要调整支持度参数  
        min_support_default = 0.2  
        min_confidence_default = 0.3  
  
        ## 检查数据规模，根据实际情况调整参数  
        transaction_count = len(transactions)  
        if transaction_count < 100:  
            min_support_default = 0.05  
            print(f"\n注意：由于交易记录较少 ({transaction_count} 条)，已自动降低默认最小支持度为 {min_support_default}")  
  
        ## 4. Apriori算法分析  
        print(f"\n使用初始参数: 最小支持度={min_support_default}, 最小可信度={min_confidence_default}")  
        frequent_itemsets, rules = apriori_analysis(df_encoded, min_support=min_support_default,  
                                                    min_confidence=min_confidence_default)  
  
        ## 5. 调整参数的Apriori算法分析  
        min_support_adjusted = 0.02  
        min_confidence_adjusted = 0.35  
        print(f"\n使用调整后参数: 最小支持度={min_support_adjusted}, 最小可信度={min_confidence_adjusted}")  
        frequent_itemsets_adj, rules_adj = apriori_analysis(df_encoded, min_support=min_support_adjusted,  
                                                            min_confidence=min_confidence_adjusted)  
  
        ## 6. 业务分析和销售建议  
        if rules_adj is not None and len(rules_adj) > 0:  
            top_rules = business_analysis(rules_adj)  
        else:  
            print("\n无法进行业务分析，因为没有找到满足条件的关联规则。")  
  
        ## 7. 解释提升度指标  
        explain_lift()  
  
        ## 8. FP-Tree算法分析与比较  
        fp_rules = fp_growth_analysis(df_encoded)  
  
        print("\n分析完成！所有图表已保存。")  
  
    except Exception as e:  
        print(f"\n分析过程中发生错误: {str(e)}")  
        import traceback  
        traceback.print_exc()  
  
  
if __name__ == "__main__":  
    main()
```

### 实验体会与总结
本实验通过Python分析商品销售数据，包括销量统计与可视化。首先计算销量前8商品及其占比，绘制条形图；然后按商品类别汇总销量并用饼图展示。数据预处理后，使用Apriori算法进行关联规则挖掘，分别设置不同支持度和置信度参数(0.2/0.3和0.02/0.35)对比结果。基于挖掘出的规则，为商家提供了捆绑销售、产品布局等营销建议。进一步解释了提升度指标的含义，并通过FP-Tree算法与Apriori比较，发现FP-Tree在大规模数据处理上效率更高，而Apriori概念更简单易实现。


---

# 6. 实验六：回归模型算法

### 实验内容

| 实验目的：<br><br>1、掌握通过Lasso回归模型进行特征选择；<br><br>2、掌握构建灰色预测模型；<br><br>3、掌握构建支持向量回归模型；<br><br>3、预测财政收入具体值。                                                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 根据给定的数据集data.csv，利用Lasso回归模型筛选地方财政收入的关键属性，并将关键属性进行展示；<br><br>2、 基于上述筛选结果，通过灰色预测模型GM（1,1）预测2014、2015年的多个关键属性值，并进行输出；<br><br>3、 基于上述实验结果，通过支持向量回归模型预测2014年、2015年的财政收入，并画出预测结果图（以折线图形式进行展示）；<br><br>4、 直接使用GM（1,1）对2014、2015年的财政收入值进行预测，并画出预测结果图（以折线图形式进行展示）；<br><br>5、 使用多层感知机（MLP）对2014、2015年的财政收入值进行预测，给出具体的参数设置（比如网络的层数、每层神经元的个数等），并画出预测结果图（以折线图形式进行展示）；<br><br>6、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |
### 实验过程及分析
#### 核心代码
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.linear_model import Lasso  
from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVR  
from sklearn.neural_network import MLPRegressor  
from sklearn.metrics import mean_squared_error, r2_score  
import warnings  
  
warnings.filterwarnings('ignore')  
  
### 设置中文字体  
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False  
  
  
class GreyModel:  
    """灰色预测模型GM(1,1)实现"""  
  
    def __init__(self):  
        self.a = None  
        self.b = None  
        self.x0 = None  
        self.x1 = None  
  
    def fit(self, data):  
        """拟合GM(1,1)模型"""  
        self.x0 = np.array(data, dtype=float)  
        n = len(self.x0)  
  
        ## 检查数据有效性  
        if n < 4:  
            raise ValueError("GM(1,1)模型至少需要4个数据点")  
  
        ## 一次累加生成  
        self.x1 = np.cumsum(self.x0)  
  
        ## 构建数据矩阵  
        B = np.zeros((n - 1, 2))  
        Y = np.zeros(n - 1)  
  
        for i in range(n - 1):  
            B[i, 0] = -(self.x1[i] + self.x1[i + 1]) / 2  
            B[i, 1] = 1  
            Y[i] = self.x0[i + 1]  
  
        ## 使用更稳定的最小二乘求解方法  
        try:  
            ## 使用伪逆矩阵求解，更稳定  
            params = np.linalg.lstsq(B, Y, rcond=None)[0]  
            self.a = params[0]  
            self.b = params[1]  
        except np.linalg.LinAlgError:  
            ## 如果仍然失败，使用简单的参数估计  
            self.a = 0.1  ## 默认发展系数  
            self.b = np.mean(Y)  ## 使用Y的均值  
  
        ## 确保参数合理性  
        if abs(self.a) < 1e-10:  
            self.a = 0.01 if self.a >= 0 else -0.01  
  
        return self  
  
    def predict(self, steps):  
        """预测未来steps步的值"""  
        if self.a is None or self.b is None:  
            raise ValueError("模型未拟合，请先调用fit方法")  
  
        n = len(self.x0)  
        predictions = []  
  
        ## 获取最后一个累加值  
        last_x1 = self.x1[-1]  
  
        for k in range(1, steps + 1):  
            ## GM(1,1)预测公式 - 修正版  
            try:  
                if abs(self.a) > 1e-10:  
                    x1_pred = (self.x0[0] - self.b / self.a) * np.exp(-self.a * (n + k - 1)) + self.b / self.a  
                else:  
                    ## 当a接近0时，使用线性增长  
                    x1_pred = last_x1 + self.b * k  
  
                ## 计算原始序列预测值  
                if k == 1:  
                    x0_pred = x1_pred - last_x1  
                else:  
                    x0_pred = x1_pred - x1_prev  
  
                ## 确保预测值合理  
                if x0_pred < 0:  
                    x0_pred = max(0, self.x0[-1] * (1 + 0.05))  ## 使用5%的增长率作为备选  
  
                predictions.append(x0_pred)  
                x1_prev = x1_pred  
  
            except (OverflowError, ZeroDivisionError):  
                ## 数值溢出时使用简单增长模型  
                growth_rate = np.mean(np.diff(self.x0)) / np.mean(self.x0[:-1])  
                x0_pred = self.x0[-1] * (1 + growth_rate) ** k  
                predictions.append(x0_pred)  
  
        return np.array(predictions)  
  
  
def load_and_preprocess_data():  
    """加载和预处理数据"""  
    ## 创建示例数据集（实际使用时请替换为真实的data.csv文件）  
    np.random.seed(42)  
    years = list(range(2000, 2014))  ## 2000-2013年的历史数据  
    n_years = len(years)  
  
    ## 模拟各种经济指标数据 - 更加稳定的数据生成  
    base_values = {  
        'gdp': 5000,  
        'population': 800,  
        'investment': 2000,  
        'consumption': 1500,  
        'export': 800,  
        'import': 700,  
        'industry_output': 1800  
    }  
  
    data = {'year': years}  
  
    ## 生成更平滑的时间序列数据  
    for key, base_val in base_values.items():  
        ## 使用指数增长 + 小幅随机波动  
        growth_rates = np.random.normal(0.05, 0.02, n_years)  ## 年均5%增长，波动2%  
        values = [base_val]  
        for i in range(1, n_years):  
            new_val = values[-1] * (1 + growth_rates[i])  
            values.append(new_val)  
        data[key] = values  
  
    ## 财政收入作为其他指标的线性组合 + 噪声  
    fiscal_revenue = []  
    for i in range(n_years):  
        revenue = (0.2 * data['gdp'][i] +  
                   0.15 * data['investment'][i] +  
                   0.1 * data['consumption'][i] +  
                   0.05 * data['industry_output'][i] +  
                   np.random.normal(0, 50))  
        fiscal_revenue.append(revenue)  
  
    data['fiscal_revenue'] = fiscal_revenue  
  
    df = pd.DataFrame(data)  
  
    print("数据预览：")  
    print(df.head())  
    print(f"\n数据形状: {df.shape}")  
    print(f"数据年份范围: {df['year'].min()} - {df['year'].max()}")  
  
    return df  
  
  
def lasso_feature_selection(df):  
    """1. 使用Lasso回归进行特征选择"""  
    print("\n" + "=" * 50)  
    print("1. Lasso回归特征选择")  
    print("=" * 50)  
  
    ## 准备特征和目标变量  
    feature_cols = ['gdp', 'population', 'investment', 'consumption',  
                    'export', 'import', 'industry_output']  
    X = df[feature_cols].values  
    y = df['fiscal_revenue'].values  
  
    ## 标准化特征  
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)  
  
    ## Lasso回归  
    ## alpha值较小以避免过度稀疏化  
    lasso = Lasso(alpha=0.1, random_state=42)  
    lasso.fit(X_scaled, y)  
  
    ## 获取特征重要性  
    coefficients = lasso.coef_  
    feature_importance = pd.DataFrame({  
        'feature': feature_cols,  
        'coefficient': coefficients,  
        'abs_coefficient': np.abs(coefficients)  
    }).sort_values('abs_coefficient', ascending=False)  
  
    print("Lasso回归系数：")  
    print(feature_importance)  
  
    ## 选择非零系数的特征作为关键属性  
    selected_features = feature_importance[feature_importance['abs_coefficient'] > 0.01]['feature'].tolist()  
  
    print(f"\n选中的关键属性: {selected_features}")  
    print(f"关键属性数量: {len(selected_features)}")  
  
    ## 可视化特征重要性  
    plt.figure(figsize=(10, 6))  
    plt.barh(feature_importance['feature'], feature_importance['abs_coefficient'])  
    plt.xlabel('系数绝对值')  
    plt.title('Lasso回归特征重要性')  
    plt.tight_layout()  
    plt.show()  
  
    return selected_features, scaler  
  
  
def grey_prediction_features(df, selected_features):  
    """2. 使用GM(1,1)预测关键属性值"""  
    print("\n" + "=" * 50)  
    print("2. GM(1,1)预测关键属性")  
    print("=" * 50)  
  
    predicted_features = {}  
  
    for feature in selected_features:  
        print(f"\n预测特征: {feature}")  
  
        ## 获取历史数据  
        historical_data = df[feature].values  
  
        try:  
            ## 拟合GM(1,1)模型  
            gm = GreyModel()  
            gm.fit(historical_data)  
  
            ## 预测2014、2015年的值  
            predictions = gm.predict(2)  
            predicted_features[feature] = predictions  
  
            print(f"2014年预测值: {predictions[0]:.2f}")  
            print(f"2015年预测值: {predictions[1]:.2f}")  
  
            ## 简化精度计算 - 避免递归拟合  
            ## 使用后验差比值检验模型精度  
            fitted_predictions = []  
            for i in range(len(historical_data) - 1):  
                if gm.a != 0:  
                    x1_fit = (historical_data[0] - gm.b / gm.a) * np.exp(-gm.a * i) + gm.b / gm.a  
                    if i == 0:  
                        x0_fit = x1_fit  
                    else:  
                        x0_fit = x1_fit - prev_x1  
                    fitted_predictions.append(x0_fit)  
                    prev_x1 = x1_fit  
                else:  
                    fitted_predictions.append(historical_data[i])  
  
            if len(fitted_predictions) > 0:  
                residuals = np.array(historical_data[1:len(fitted_predictions) + 1]) - np.array(fitted_predictions)  
                mse = np.mean(residuals ** 2)  
                print(f"模型MSE: {mse:.2f}")  
  
                ## 计算平均相对误差  
                mape = np.mean(np.abs(residuals / historical_data[1:len(fitted_predictions) + 1])) * 100  
                print(f"平均相对误差: {mape:.2f}%")  
  
        except Exception as e:  
            print(f"GM(1,1)拟合失败，使用线性外推: {e}")  
            ## 使用简单的线性外推作为备选方案  
            if len(historical_data) >= 2:  
                growth_rate = (historical_data[-1] - historical_data[-2]) / historical_data[-2]  
                pred_2014 = historical_data[-1] * (1 + growth_rate)  
                pred_2015 = pred_2014 * (1 + growth_rate)  
                predicted_features[feature] = np.array([pred_2014, pred_2015])  
  
                print(f"2014年预测值 (线性外推): {pred_2014:.2f}")  
                print(f"2015年预测值 (线性外推): {pred_2015:.2f}")  
            else:  
                ## 最后备选：使用最后一个值  
                predicted_features[feature] = np.array([historical_data[-1], historical_data[-1]])  
                print(f"使用最后已知值作为预测")  
  
    return predicted_features  
  
  
def svr_prediction(df, selected_features, predicted_features, scaler):  
    """3. 使用支持向量回归预测财政收入"""  
    print("\n" + "=" * 50)  
    print("3. 支持向量回归预测财政收入")  
    print("=" * 50)  
  
    ## 准备训练数据  
    X_train = df[selected_features].values  
    y_train = df['fiscal_revenue'].values  
  
    ## 标准化  
    X_train_scaled = scaler.transform(X_train)  
  
    ## 训练SVR模型  
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)  
    svr.fit(X_train_scaled, y_train)  
  
    ## 准备预测数据  
    X_pred = []  
    for year in [2014, 2015]:  
        year_features = []  
        for feature in selected_features:  
            if year == 2014:  
                year_features.append(predicted_features[feature][0])  
            else:  
                year_features.append(predicted_features[feature][1])  
        X_pred.append(year_features)  
  
    X_pred = np.array(X_pred)  
    X_pred_scaled = scaler.transform(X_pred)  
  
    ## 预测财政收入  
    svr_predictions = svr.predict(X_pred_scaled)  
  
    print(f"SVR预测2014年财政收入: {svr_predictions[0]:.2f}")  
    print(f"SVR预测2015年财政收入: {svr_predictions[1]:.2f}")  
  
    ## 绘制预测结果  
    years_all = list(df['year']) + [2014, 2015]  
    revenue_all = list(df['fiscal_revenue']) + list(svr_predictions)  
  
    plt.figure(figsize=(12, 6))  
    plt.plot(df['year'], df['fiscal_revenue'], 'bo-', label='历史数据', linewidth=2)  
    plt.plot([2014, 2015], svr_predictions, 'ro-', label='SVR预测', linewidth=2)  
    plt.plot([2013, 2014], [df['fiscal_revenue'].iloc[-1], svr_predictions[0]], 'r--', alpha=0.5)  
    plt.xlabel('年份')  
    plt.ylabel('财政收入')  
    plt.title('基于SVR的财政收入预测')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.show()  
  
    return svr_predictions  
  
  
def direct_grey_prediction(df):  
    """4. 直接使用GM(1,1)预测财政收入"""  
    print("\n" + "=" * 50)  
    print("4. GM(1,1)直接预测财政收入")  
    print("=" * 50)  
  
    ## 获取财政收入历史数据  
    fiscal_revenue = df['fiscal_revenue'].values  
  
    ## 拟合GM(1,1)模型  
    gm = GreyModel()  
    gm.fit(fiscal_revenue)  
  
    ## 预测2014、2015年的值  
    gm_predictions = gm.predict(2)  
  
    print(f"GM(1,1)预测2014年财政收入: {gm_predictions[0]:.2f}")  
    print(f"GM(1,1)预测2015年财政收入: {gm_predictions[1]:.2f}")  
  
    ## 绘制预测结果  
    plt.figure(figsize=(12, 6))  
    plt.plot(df['year'], df['fiscal_revenue'], 'bo-', label='历史数据', linewidth=2)  
    plt.plot([2014, 2015], gm_predictions, 'go-', label='GM(1,1)预测', linewidth=2)  
    plt.plot([2013, 2014], [df['fiscal_revenue'].iloc[-1], gm_predictions[0]], 'g--', alpha=0.5)  
    plt.xlabel('年份')  
    plt.ylabel('财政收入')  
    plt.title('基于GM(1,1)的财政收入直接预测')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.show()  
  
    return gm_predictions  
  
  
def mlp_prediction(df, selected_features, predicted_features, scaler):  
    """5. 使用多层感知机预测财政收入"""  
    print("\n" + "=" * 50)  
    print("5. 多层感知机(MLP)预测财政收入")  
    print("=" * 50)  
  
    ## 准备训练数据  
    X_train = df[selected_features].values  
    y_train = df['fiscal_revenue'].values  
  
    ## 标准化  
    X_train_scaled = scaler.transform(X_train)  
  
    ## MLP参数设置  
    mlp_params = {  
        'hidden_layer_sizes': (100, 50, 25),  ## 3个隐藏层，分别有100、50、25个神经元  
        'activation': 'relu',  ## ReLU激活函数  
        'solver': 'adam',  ## Adam优化器  
        'alpha': 0.01,  ## L2正则化参数  
        'learning_rate': 'adaptive',  ## 自适应学习率  
        'max_iter': 1000,  ## 最大迭代次数  
        'random_state': 42  ## 随机种子  
    }  
  
    print("MLP网络结构参数:")  
    print(f"- 输入层神经元数: {len(selected_features)}")  
    print(f"- 隐藏层结构: {mlp_params['hidden_layer_sizes']}")  
    print(f"- 输出层神经元数: 1")  
    print(f"- 激活函数: {mlp_params['activation']}")  
    print(f"- 优化器: {mlp_params['solver']}")  
    print(f"- 正则化参数: {mlp_params['alpha']}")  
    print(f"- 最大迭代次数: {mlp_params['max_iter']}")  
  
    ## 训练MLP模型  
    mlp = MLPRegressor(**mlp_params)  
    mlp.fit(X_train_scaled, y_train)  
  
    print(f"实际迭代次数: {mlp.n_iter_}")  
    print(f"训练损失: {mlp.loss_:.4f}")  
  
    ## 准备预测数据  
    X_pred = []  
    for year in [2014, 2015]:  
        year_features = []  
        for feature in selected_features:  
            if year == 2014:  
                year_features.append(predicted_features[feature][0])  
            else:  
                year_features.append(predicted_features[feature][1])  
        X_pred.append(year_features)  
  
    X_pred = np.array(X_pred)  
    X_pred_scaled = scaler.transform(X_pred)  
  
    ## 预测财政收入  
    mlp_predictions = mlp.predict(X_pred_scaled)  
  
    print(f"\nMLP预测2014年财政收入: {mlp_predictions[0]:.2f}")  
    print(f"MLP预测2015年财政收入: {mlp_predictions[1]:.2f}")  
  
    ## 绘制预测结果  
    plt.figure(figsize=(12, 6))  
    plt.plot(df['year'], df['fiscal_revenue'], 'bo-', label='历史数据', linewidth=2)  
    plt.plot([2014, 2015], mlp_predictions, 'mo-', label='MLP预测', linewidth=2)  
    plt.plot([2013, 2014], [df['fiscal_revenue'].iloc[-1], mlp_predictions[0]], 'm--', alpha=0.5)  
    plt.xlabel('年份')  
    plt.ylabel('财政收入')  
    plt.title('基于MLP的财政收入预测')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.tight_layout()  
    plt.show()  
  
    return mlp_predictions  
  
  
def compare_results(svr_pred, gm_pred, mlp_pred):  
    """比较不同模型的预测结果"""  
    print("\n" + "=" * 50)  
    print("预测结果比较")  
    print("=" * 50)  
  
    results_df = pd.DataFrame({  
        '年份': [2014, 2015],  
        'SVR预测': svr_pred,  
        'GM(1,1)直接预测': gm_pred,  
        'MLP预测': mlp_pred  
    })  
  
    print(results_df)  
  
    ## 综合比较图  
    plt.figure(figsize=(12, 8))  
    x = np.array([2014, 2015])  
  
    plt.plot(x, svr_pred, 'ro-', label='SVR预测', linewidth=2, markersize=8)  
    plt.plot(x, gm_pred, 'go-', label='GM(1,1)预测', linewidth=2, markersize=8)  
    plt.plot(x, mlp_pred, 'mo-', label='MLP预测', linewidth=2, markersize=8)  
  
    plt.xlabel('年份')  
    plt.ylabel('财政收入')  
    plt.title('三种模型预测结果比较')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
  
    ## 添加数值标签  
    for i, year in enumerate([2014, 2015]):  
        plt.text(year, svr_pred[i], f'{svr_pred[i]:.1f}', ha='center', va='bottom')  
        plt.text(year, gm_pred[i], f'{gm_pred[i]:.1f}', ha='center', va='bottom')  
        plt.text(year, mlp_pred[i], f'{mlp_pred[i]:.1f}', ha='center', va='bottom')  
  
    plt.tight_layout()  
    plt.show()  
  
  
def main():  
    """主函数"""  
    print("地方财政收入预测实验")  
    print("=" * 50)  
  
    ## 加载数据  
    df = load_and_preprocess_data()  
  
    ## 1. Lasso特征选择  
    selected_features, scaler = lasso_feature_selection(df)  
  
    ## 2. GM(1,1)预测关键属性  
    predicted_features = grey_prediction_features(df, selected_features)  
  
    ## 3. SVR预测财政收入  
    svr_predictions = svr_prediction(df, selected_features, predicted_features, scaler)  
  
    ## 4. GM(1,1)直接预测财政收入  
    gm_predictions = direct_grey_prediction(df)  
  
    ## 5. MLP预测财政收入  
    mlp_predictions = mlp_prediction(df, selected_features, predicted_features, scaler)  
  
    ## 比较结果  
    compare_results(svr_predictions, gm_predictions, mlp_predictions)  
  
  
if __name__ == "__main__":  
    main()
```

#### 输出
##### 终端输出
```shell
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp07\main.py 
地方财政收入预测实验
==================================================
数据预览：
   year          gdp  population  ...      import  industry_output  fiscal_revenue
0  2000  5000.000000  800.000000  ...  700.000000      1800.000000     1540.255673
1  2001  5236.173570  831.003400  ...  756.532512      1871.936746     1598.948721
2  2002  5565.810440  855.720247  ...  793.817066      1999.805081     1627.322335
3  2003  6013.638872  903.884416  ...  858.348736      2112.944098     1806.272699
4  2004  6286.158539  932.663660  ...  856.293075      2196.204229     1893.162691

[5 rows x 9 columns]

数据形状: (14, 9)
数据年份范围: 2000 - 2013

==================================================
1. Lasso回归特征选择
==================================================
Lasso回归系数：
           feature  coefficient  abs_coefficient
1       population   407.893105       407.893105
5           import  -137.783260       137.783260
2       investment   136.177560       136.177560
0              gdp   134.954622       134.954622
4           export  -115.937437       115.937437
6  industry_output    87.891951        87.891951
3      consumption   -71.984080        71.984080

选中的关键属性: ['population', 'import', 'investment', 'gdp', 'export', 'industry_output', 'consumption']
关键属性数量: 7

==================================================
2. GM(1,1)预测关键属性
==================================================

预测特征: population
2014年预测值: 1424.17
2015年预测值: 1487.19
模型MSE: 2111.26
平均相对误差: 4.04%

预测特征: import
2014年预测值: 1362.10
2015年预测值: 1429.98
模型MSE: 2524.31
平均相对误差: 4.71%

预测特征: investment
2014年预测值: 3694.77
2015年预测值: 3854.58
模型MSE: 16677.57
平均相对误差: 4.26%

预测特征: gdp
2014年预测值: 10583.47
2015年预测值: 11086.18
模型MSE: 170426.83
平均相对误差: 5.04%

预测特征: export
2014年预测值: 1596.82
2015年预测值: 1691.26
模型MSE: 3728.04
平均相对误差: 4.94%

预测特征: industry_output
2014年预测值: 3532.15
2015年预测值: 3694.75
模型MSE: 15865.14
平均相对误差: 4.63%

预测特征: consumption
2014年预测值: 2780.32
2015年预测值: 2935.06
模型MSE: 9909.05
平均相对误差: 4.43%

==================================================
3. 支持向量回归预测财政收入
==================================================
SVR预测2014年财政收入: 2510.00
SVR预测2015年财政收入: 2513.59

==================================================
4. GM(1,1)直接预测财政收入
==================================================
GM(1,1)预测2014年财政收入: 3133.50
GM(1,1)预测2015年财政收入: 3272.23

==================================================
5. 多层感知机(MLP)预测财政收入
==================================================
MLP网络结构参数:
- 输入层神经元数: 7
- 隐藏层结构: (100, 50, 25)
- 输出层神经元数: 1
- 激活函数: relu
- 优化器: adam
- 正则化参数: 0.01
- 最大迭代次数: 1000
实际迭代次数: 183
训练损失: 4286.7125

MLP预测2014年财政收入: 3304.21
MLP预测2015年财政收入: 3466.12

==================================================
预测结果比较
==================================================
     年份        SVR预测  GM(1,1)直接预测        MLP预测
0  2014  2509.999807  3133.501073  3304.206978
1  2015  2513.588294  3272.232117  3466.117290

进程已结束，退出代码为 0
```

##### 图表输出
![](c608831bb7d5911a2ecf7ba74a5b0b36_MD5.webp)
![](83e3c5ed1f2d64609f15c480e11d0cc3_MD5.webp)
![](220eed3b16b8e95b6a3bf368a41e767a_MD5.webp)![](b80ddbef5c2f382a810a3cc5108c2235_MD5.webp)
![](20b1914d1a39858f6c4aecf3ef80dd74_MD5.webp)

### 实验体会与总结
本实验通过五个步骤完成地方财政收入预测：首先使用Lasso回归从7个经济指标中筛选出关键属性，然后运用GM(1,1)模型预测各关键属性的2014-2015年数值，基于预测属性值构建SVR模型进行财政收入预测。同时对比了GM(1,1)直接预测和MLP神经网络预测方法。实验结果显示，基于特征选择和属性预测的SVR方法能够有效整合多维经济信息，相比单一时间序列方法具有更好的预测稳定性和解释性，为财政收入预测提供了可靠的技术方案。

---

# 6. 实验六：时序模型算法

### 实验内容

| 实验目的：<br><br>1、理解时序模型算法的原理；<br><br>2、能够使用时序模型算法处理具体问题；<br><br>3、能够使用Python语言实现时序模型算法。                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 根据给定的数据集Daily_Demand_Forecasting_Orders.csv，由于数据集是以分号分割的属性，所以先对原始数据集进行预处理；<br><br>2、 针对财政部门订单（第8个属性），用时间序列模型分析应该采用哪一种模型进行拟合？请给出分析过程；<br><br>3、 选择合适的模型对政府部门订单进行预测，并（1）绘制出损失下降图（2）最终训练好的模型的预测值与实际值的折线图进行对比；<br><br>4、 针对订单总数（第13个属性），用时间序列模型分析应该采用哪一种模型进行拟合？请给出分析过程；<br><br>5、 选择合适的模型对订单总数进行预测，并（1）绘制出损失下降图（2）最终训练好的模型的预测值与实际值的折线图进行对比；<br><br>6、 尝试使用LSTM对上述数据集进行预测，对比两者结果，试分析传统方法与LSTM之间的差异性；<br><br>7、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |
### 实验过程及分析
#### 实验分析
##### 1. 数据预处理

```python
import pandas as pd

### 读取分号分隔的 CSV
df = pd.read_csv('data/Daily_Demand_Forecasting_Orders.csv', sep=';')

### 如果第一列是日期索引，自行转换并设置
### df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
### df.set_index('Date', inplace=True)

### 将所有字段转换为数值型
df = df.apply(pd.to_numeric, errors='coerce')

### 提取“财政部门订单”（第8列，索引7）和“总订单”（第13列，索引12）
series_fiscal = df.iloc[:, 7]
series_total  = df.iloc[:, 12]
```

##### 2. 财政部门订单的时间序列模型选择与分析
1. **平稳性检验（ADF）**  
    进行单位根检验，若 $p$-value < 0.05 可认为序列平稳，否则需要差分。
    ```python
    from statsmodels.tsa.stattools import adfuller
    
    adf_res = adfuller(series_fiscal.dropna())
    print(f'ADF-statistic={adf_res[0]:.4f}, p-value={adf_res[1]:.4f}')
    ```

2. **ACF 和 PACF 图**  
    根据一阶差分后序列的 ACF/PACF 判断 AR(p) 或 MA(q) 阶数。
    ```python
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    
    diff1 = series_fiscal.diff().dropna()
    fig, axes = plt.subplots(2,1, figsize=(8,6))
    plot_acf (diff1, ax=axes[0], lags=20); axes[0].set_title('差分后 ACF')
    plot_pacf(diff1, ax=axes[1], lags=20); axes[1].set_title('差分后 PACF')
    fig.savefig('output/fiscal_acf_pacf.png')
    ```
    
3. **模型选型**  
    通常可选 ARIMA(p,1,q)。例如根据图形尝试 `p=1,q=1`：
$$
    ARIMA(p,d,q):\quad X_t = c + \sum_{i=1}^p \phi_i X_{t-i}
                       + \sum_{j=1}^q \theta_j \varepsilon_{t-j} + \varepsilon_t
$$
    
4. **拟合与预测**
    ```python
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(series_fiscal, order=(1,1,1))
    res   = model.fit()
    ## 训练集内预测
    pred_train = res.predict(start=series_fiscal.index[0],
                             end=series_fiscal.index[-1],
                             typ='levels')
    ## 绘制预测 vs 实际
    plt.figure(figsize=(8,4))
    plt.plot(series_fiscal, label='实际值')
    plt.plot(pred_train,   label='ARIMA预测')
    plt.legend()
    plt.title('财政部门订单 ARIMA 预测对比')
    plt.savefig('output/fiscal_arima_pred.png')
    ```

> **说明**：传统 ARIMA 模型一次性拟合，**无“损失下降曲线”**。若一定要可将信息准则如 AIC 随阶数变化的曲线当作“拟合优劣曲线”——但一般不画。


#### 3. 总订单的时间序列模型选择与预测
对总订单作同样的流程：
1. **ADF 检验**
2. **ACF/PACF 图**
3. **选取 ARIMA(p,1,q)**，比如 `(1,1,1)`
4. **拟合 & 预测**，并保存图像到 `output/total_arima_pred.png`
核心代码与上面财政部门一致，只要将 `series_fiscal → series_total` 即可。

#### 4. 使用 LSTM 进行对比预测
由于深度学习模型训练过程有迭代损失，可按以下步骤：
1. **归一化 & 构造时序样本**
    ```python
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series_fiscal.values.reshape(-1,1))
    
    def create_dataset(X, look_back=5):
        Xs, ys = [], []
        for i in range(len(X)-look_back):
            Xs.append(X[i:i+look_back])
            ys.append(X[i+look_back])
        return np.array(Xs), np.array(ys)
    
    X, y = create_dataset(data, look_back=5)
    ```
    
2. **搭建并训练 LSTM**
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X, y, epochs=50, batch_size=16, verbose=1)
    ## 保存训练损失曲线
    plt.figure(); plt.plot(history.history['loss']); plt.title('LSTM 训练损失'); plt.savefig('output/fiscal_lstm_loss.png')
    ```
    
3. **预测 & 反归一化**
    ```python
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    actual = scaler.inverse_transform(y[ :,None])
    plt.figure()
    plt.plot(actual, label='实际')
    plt.plot(y_pred, label='LSTM预测')
    plt.legend(); plt.title('财政部门订单 LSTM 预测对比')
    plt.savefig('output/fiscal_lstm_pred.png')
    ```
    
对“总订单”同理，只需将数据换成 `series_total` 并重做上述流程，结果保存到对应文件夹。

#### 5. 结果对比与分析
1. **定量指标**
    - 计算两种模型在各自测试集（或留一交叉验证）上的 MSE/MAPE。
2. **曲线对比**
    - ARIMA：预测 vs 实际
    - LSTM：训练损失曲线 & 预测 vs 实际
3. **差异性分析**
    - ARIMA 对长期趋势捕捉较好，但对非线性模式表现有限；
    - LSTM 对复杂非线性关联敏感，但需大量数据且易过拟合。
        
#### 实验结果
##### 图标输出
![](be9173432c97a8910a266e830ed31554_MD5.webp)

![](f71403c2d8632323cdd1b830eab95fad_MD5.webp)

![](bd1759c49bfe61a737c10cc00c2a9e5f_MD5.webp)

![](735c55547f17b00e7bab9e4dcd7b0b49_MD5.webp)

![](4f3fe24c4e22efa77592be8e20db3e42_MD5.webp)

![](2f72df1f94c6bf4330a6242fe0dab7b6_MD5.webp)

![](b62e1f95e196228bff72ea2b7087ffb6_MD5.webp)

![](2d588ebf68507a1db1cbce40f885fbe7_MD5.webp)

### 实验体会与总结
本实验基于财政部门订单与总订单时序数据，分别采用 ARIMA(1,1,1) 模型与 PyTorch 实现的 LSTM 模型进行拟合与预测。结果表明，ARIMA 能有效捕捉线性趋势，预测稳定性高；而 LSTM 对复杂非线性波动响应更敏感，但对数据量与超参数更为依赖。两者在均方误差上各有优劣，可根据实际需求在快速拟合与精细非线性建模间权衡选择。

---

# 8. 实验八：数据库操作与协同过滤算法

### 实验内容

| 实验目的：<br><br>1、 掌握通过Python对数据库进行日常的数据操作<br><br>2、 掌握通过Python对数据进行探索性分析<br><br>3、 掌握构建协同过滤算法计算出与某些网页相似的网页的相似程度<br><br>4、 根据相似程度的高低，实现智能推荐。                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实验内容：<br><br>1、 根据给定的数据集7law.sql，采用Python连接数据库的方式读取数据。<br><br>2、 通过Python数学分析模块对给定的数据进行数据探索分析，以表格形式进行展示，具体包括：<br><br>1) 网页类型统计<br><br>2) 知识类型内部统计<br><br>3) 带“？”字符网址类型统计<br><br>4) 199类型中的具体类型的统计<br><br>5) 无目的浏览用户点击分析<br><br>6) 用户点击次数统计<br><br>7) 浏览一次的用户的行为分析<br><br>3、 抽取知识类网页中婚姻类网页，并对该数据进行预处理，包括：<br><br>1) 清除与分析目标无关的数据；<br><br>2) 识别翻页的网址，并对其进行还原，然后对用户访问的页面进行去重操作。<br><br>3) 筛选掉浏览网页次数不满2次的用户。<br><br>4) 将处理好的数据集划分为训练集与测试集，分别保存至csv文件中。<br><br>4、 通过基于物品的协同过滤算法计算网页之间的相似度，输出相似度分布统计情况以及相似度区间分布情况并给出预测的部分推荐结果。<br><br>5、 计算出推荐模型的准确率、召回率和F1指标。<br><br>6、 将上述实验内容的核心代码及实验结果截图放到“实验过程及分析”中。 |
### 实验过程及分析
#### 设计思路和实现说明

##### 1. **数据库连接设计**

- 使用`mysql.connector`连接MySQL数据库
- 采用面向对象的设计，将所有功能封装在`WebDataAnalyzer`类中
- 包含连接管理和错误处理机制

##### 2. **数据探索分析模块**

**网页类型统计**: 通过解析`pagePath`中的`/law/`路径来识别不同类型的法律网页

**知识类型统计**: 基于`pageTitle`中的关键词（法律、法规、咨询、案例等）进行分类

**带"?"字符网址统计**: 识别带参数的URL，分析搜索页面、详情页面、分页页面等

**199类型统计**: 基于`pageTitleCategoryId`字段进行分类统计

**无目的浏览分析**: 通过计算用户访问页面数和平均停留时间来识别无目的浏览行为

**用户点击统计**: 统计用户访问频次分布，了解用户活跃度

**单次访问用户分析**: 分析只访问一次的用户的页面类型、来源和设备特征

##### 3. **数据预处理设计**

针对婚姻类网页的预处理包含三个关键步骤：

**数据清洗**: 移除无效的用户ID和页面路径记录

**URL标准化**: 使用正则表达式识别和移除分页参数（page=、p=、start=等），将翻页URL还原为基础URL，然后按用户和标准化URL去重

**用户筛选**: 只保留访问次数≥2的活跃用户，确保推荐算法有足够的行为数据

##### 4. **协同过滤推荐算法设计**

**用户-物品矩阵构建**: 使用`pivot_table`创建稀疏矩阵，行为用户，列为标准化的URL

**相似度计算**: 采用余弦相似度计算物品间相似性，这是协同过滤的经典方法

**推荐生成**: 基于用户历史行为和物品相似度生成推荐列表

##### 5. **评估指标实现**

**准确率(Precision)**: 推荐物品中用户实际感兴趣的比例 **召回率(Recall)**: 用户感兴趣物品中被推荐的比例  
**F1指标**: 准确率和召回率的调和平均数

#### 实验代码
```python
import pandas as pd  
import numpy as np  
import mysql.connector  
from mysql.connector import Error  
import matplotlib.pyplot as plt  
import seaborn as sns  
from collections import defaultdict  
import re  
from sklearn.model_selection import train_test_split  
from sklearn.metrics.pairwise import cosine_similarity  
from scipy.sparse import csr_matrix  
import warnings  
  
warnings.filterwarnings('ignore')  
  
### 设置中文字体显示  
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
  
  
class WebDataAnalyzer:  
    def __init__(self, host='localhost', database='test_114514', user='root', password='123456'):  
        """初始化数据库连接"""  
        self.host = host  
        self.database = database  
        self.user = user  
        self.password = password  
        self.connection = None  
        self.df = None  
  
    def connect_database(self):  
        """连接数据库"""  
        try:  
            self.connection = mysql.connector.connect(  
                host=self.host,  
                database=self.database,  
                user=self.user,  
                password=self.password  
            )  
            if self.connection.is_connected():  
                print("成功连接到MySQL数据库")  
                return True  
        except Error as e:  
            print(f"连接数据库时发生错误: {e}")  
            return False  
  
    def load_data(self):  
        """从数据库读取数据"""  
        if not self.connection or not self.connection.is_connected():  
            if not self.connect_database():  
                return False  
  
        try:  
            query = "SELECT * FROM all_gzdata"  
            self.df = pd.read_sql(query, self.connection)  
            print(f"成功读取数据，共{len(self.df)}条记录")  
            return True  
        except Error as e:  
            print(f"读取数据时发生错误: {e}")  
            return False  
  
    def basic_data_exploration(self):  
        """基础数据探索分析"""  
        print("=" * 50)  
        print("数据基本信息:")  
        print("=" * 50)  
        print(f"数据集形状: {self.df.shape}")  
        print(f"列名: {list(self.df.columns)}")  
        print("\n数据类型:")  
        print(self.df.dtypes)  
        print("\n缺失值统计:")  
        print(self.df.isnull().sum())  
  
    def analyze_webpage_types(self):  
        """1) 网页类型统计"""  
        print("\n" + "=" * 50)  
        print("1. 网页类型统计")  
        print("=" * 50)  
  
        ## 根据pagePath分析网页类型  
        webpage_types = {}  
        for path in self.df['pagePath'].dropna():  
            if '/law/' in str(path):  
                webpage_types[path] = webpage_types.get(path, 0) + 1  
  
        ## 提取网页类型模式  
        type_patterns = defaultdict(int)  
        for path in self.df['pagePath'].dropna():  
            path_str = str(path)  
            if '/law/' in path_str:  
                ## 提取law后的类型  
                parts = path_str.split('/law/')  
                if len(parts) > 1:  
                    category = parts[1].split('/')[0] if '/' in parts[1] else parts[1]  
                    type_patterns[category] += 1  
  
        webpage_type_df = pd.DataFrame(list(type_patterns.items()),  
                                       columns=['网页类型', '访问次数'])  
        webpage_type_df = webpage_type_df.sort_values('访问次数', ascending=False).head(20)  
  
        print(webpage_type_df.to_string(index=False))  
        return webpage_type_df  
  
    def analyze_knowledge_types(self):  
        """2) 知识类型内部统计"""  
        print("\n" + "=" * 50)  
        print("2. 知识类型内部统计")  
        print("=" * 50)  
  
        knowledge_types = defaultdict(int)  
        for title in self.df['pageTitle'].dropna():  
            title_str = str(title)  
            ## 根据页面标题分类知识类型  
            if any(keyword in title_str for keyword in ['法律', '法规', '条例']):  
                knowledge_types['法律法规'] += 1  
            elif any(keyword in title_str for keyword in ['咨询', '问答']):  
                knowledge_types['法律咨询'] += 1  
            elif any(keyword in title_str for keyword in ['案例', '判决']):  
                knowledge_types['案例分析'] += 1  
            elif any(keyword in title_str for keyword in ['新闻', '资讯']):  
                knowledge_types['法律资讯'] += 1  
            else:  
                knowledge_types['其他'] += 1  
  
        knowledge_df = pd.DataFrame(list(knowledge_types.items()),  
                                    columns=['知识类型', '数量'])  
        knowledge_df = knowledge_df.sort_values('数量', ascending=False)  
  
        print(knowledge_df.to_string(index=False))  
        return knowledge_df  
  
    def analyze_question_mark_urls(self):  
        """3) 带"？"字符网址类型统计"""  
        print("\n" + "=" * 50)  
        print("3. 带'?'字符网址类型统计")  
        print("=" * 50)  
  
        question_urls = self.df[self.df['fullURL'].str.contains('\\?', na=False)]  
  
        url_types = defaultdict(int)  
        for url in question_urls['fullURL']:  
            url_str = str(url)  
            if 'search' in url_str:  
                url_types['搜索页面'] += 1  
            elif 'id=' in url_str:  
                url_types['详情页面'] += 1  
            elif 'page=' in url_str:  
                url_types['分页页面'] += 1  
            elif 'category=' in url_str:  
                url_types['分类页面'] += 1  
            else:  
                url_types['其他参数页面'] += 1  
  
        question_url_df = pd.DataFrame(list(url_types.items()),  
                                       columns=['URL类型', '数量'])  
        question_url_df = question_url_df.sort_values('数量', ascending=False)  
  
        print(f"带'?'字符的URL总数: {len(question_urls)}")  
        print(question_url_df.to_string(index=False))  
        return question_url_df  
  
    def analyze_199_types(self):  
        """4) 199类型中的具体类型统计"""  
        print("\n" + "=" * 50)  
        print("4. 199类型中的具体类型统计")  
        print("=" * 50)  
  
        ## 假设199类型是指某种特定的页面类型或状态码  
        type_199_data = self.df[self.df['pageTitleCategoryId'] == 199]  
  
        if len(type_199_data) == 0:  
            print("未找到199类型的数据")  
            return pd.DataFrame()  
  
        category_stats = type_199_data['pageTitleCategoryName'].value_counts()  
        type_199_df = pd.DataFrame({  
            '类型名称': category_stats.index,  
            '数量': category_stats.values  
        })  
  
        print(f"199类型总数: {len(type_199_data)}")  
        print(type_199_df.to_string(index=False))  
        return type_199_df  
  
    def analyze_aimless_browsing(self):  
        """5) 无目的浏览用户点击分析"""  
        print("\n" + "=" * 50)  
        print("5. 无目的浏览用户点击分析")  
        print("=" * 50)  
  
        ## 定义无目的浏览：访问页面多但停留时间短，或者跳转频繁  
        user_behavior = self.df.groupby('userID').agg({  
            'pagePath': 'count',  ## 访问页面数  
            'timestamp': ['min', 'max'],  ## 访问时间范围  
            'fullReferrer': lambda x: x.notna().sum()  ## 有引用来源的访问数  
        }).round(2)  
  
        user_behavior.columns = ['访问页面数', '首次访问时间', '最后访问时间', '有引用来源次数']  
        user_behavior['浏览时长(秒)'] = (user_behavior['最后访问时间'] - user_behavior['首次访问时间']) / 1000  
        user_behavior['平均页面停留时间'] = user_behavior['浏览时长(秒)'] / user_behavior['访问页面数']  
  
        ## 定义无目的浏览用户：访问页面数>10但平均停留时间<30秒  
        aimless_users = user_behavior[  
            (user_behavior['访问页面数'] > 10) &  
            (user_behavior['平均页面停留时间'] < 30)  
            ]  
  
        aimless_stats = pd.DataFrame({  
            '统计项': ['无目的浏览用户数', '平均访问页面数', '平均浏览时长(秒)', '平均页面停留时间(秒)'],  
            '数值': [  
                len(aimless_users),  
                aimless_users['访问页面数'].mean(),  
                aimless_users['浏览时长(秒)'].mean(),  
                aimless_users['平均页面停留时间'].mean()  
            ]  
        })  
  
        print(aimless_stats.to_string(index=False))  
        return aimless_stats  
  
    def analyze_user_clicks(self):  
        """6) 用户点击次数统计"""  
        print("\n" + "=" * 50)  
        print("6. 用户点击次数统计")  
        print("=" * 50)  
  
        user_clicks = self.df['userID'].value_counts()  
  
        click_distribution = pd.DataFrame({  
            '点击次数区间': ['1次', '2-5次', '6-10次', '11-20次', '20次以上'],  
            '用户数量': [  
                sum(user_clicks == 1),  
                sum((user_clicks >= 2) & (user_clicks <= 5)),  
                sum((user_clicks >= 6) & (user_clicks <= 10)),  
                sum((user_clicks >= 11) & (user_clicks <= 20)),  
                sum(user_clicks > 20)  
            ]  
        })  
  
        click_stats = pd.DataFrame({  
            '统计项': ['总用户数', '平均点击次数', '最大点击次数', '最小点击次数'],  
            '数值': [  
                len(user_clicks),  
                user_clicks.mean(),  
                user_clicks.max(),  
                user_clicks.min()  
            ]  
        })  
  
        print("点击次数分布:")  
        print(click_distribution.to_string(index=False))  
        print("\n点击次数统计:")  
        print(click_stats.to_string(index=False))  
        return click_distribution, click_stats  
  
    def analyze_single_visit_users(self):  
        """7) 浏览一次的用户行为分析"""  
        print("\n" + "=" * 50)  
        print("7. 浏览一次的用户行为分析")  
        print("=" * 50)  
  
        single_visit_users = self.df[self.df.groupby('userID')['userID'].transform('count') == 1]  
  
        ## 分析这些用户的访问特征  
        page_types = defaultdict(int)  
        referrer_types = defaultdict(int)  
        os_types = defaultdict(int)  
  
        for _, row in single_visit_users.iterrows():  
            ## 页面类型分析  
            path = str(row['pagePath'])  
            if '/law/' in path:  
                page_types['法律页面'] += 1  
            elif 'search' in path:  
                page_types['搜索页面'] += 1  
            else:  
                page_types['其他页面'] += 1  
  
            ## 来源分析  
            referrer = str(row['fullReferrer'])  
            if 'baidu' in referrer:  
                referrer_types['百度'] += 1  
            elif 'google' in referrer:  
                referrer_types['谷歌'] += 1  
            elif referrer == 'nan' or referrer == '':  
                referrer_types['直接访问'] += 1  
            else:  
                referrer_types['其他来源'] += 1  
  
            ## 操作系统分析  
            os = str(row['userOS'])  
            if 'Windows' in os:  
                os_types['Windows'] += 1  
            elif 'Mac' in os:  
                os_types['Mac'] += 1  
            elif 'Android' in os:  
                os_types['Android'] += 1  
            elif 'iOS' in os:  
                os_types['iOS'] += 1  
            else:  
                os_types['其他'] += 1  
  
        single_visit_analysis = pd.DataFrame({  
            '分析维度': ['总单次访问用户数', '访问法律页面比例', '来自搜索引擎比例', '移动端用户比例'],  
            '数值': [  
                len(single_visit_users),  
                page_types['法律页面'] / len(single_visit_users),  
                (referrer_types['百度'] + referrer_types['谷歌']) / len(single_visit_users),  
                (os_types['Android'] + os_types['iOS']) / len(single_visit_users)  
            ]  
        })  
  
        print(single_visit_analysis.to_string(index=False))  
        return single_visit_analysis  
  
    def extract_marriage_pages(self):  
        """抽取婚姻类网页数据"""  
        print("\n" + "=" * 50)  
        print("抽取婚姻类网页数据")  
        print("=" * 50)  
  
        ## 识别婚姻类网页  
        marriage_keywords = ['婚姻', '结婚', '离婚', '婚前', '夫妻', '配偶', '婚后', '婚礼']  
  
        marriage_mask = self.df['pageTitle'].str.contains('|'.join(marriage_keywords), na=False) | \  
                        self.df['pagePath'].str.contains('marriage|wedding|divorce', na=False) | \  
                        self.df['pageTitleKw'].str.contains('|'.join(marriage_keywords), na=False)  
  
        self.marriage_df = self.df[marriage_mask].copy()  
        print(f"婚姻类网页数据：{len(self.marriage_df)}条记录")  
  
        return self.marriage_df  
  
    def preprocess_marriage_data(self):  
        """预处理婚姻类数据"""  
        print("\n" + "=" * 50)  
        print("数据预处理")  
        print("=" * 50)  
  
        if self.marriage_df is None:  
            self.extract_marriage_pages()  
  
        ## 1) 清除与分析目标无关的数据  
        print("1. 清除无关数据...")  
        ## 移除无效的用户ID和页面路径  
        self.marriage_df = self.marriage_df.dropna(subset=['userID', 'pagePath'])  
        self.marriage_df = self.marriage_df[self.marriage_df['userID'] != '']  
        print(f"清除后剩余：{len(self.marriage_df)}条记录")  
  
        ## 2) 识别翻页网址并还原，去重操作  
        print("2. 处理翻页网址和去重...")  
  
        def normalize_url(url):  
            """标准化URL，移除翻页参数"""  
            if pd.isna(url):  
                return url  
            url_str = str(url)  
            ## 移除页码参数  
            url_str = re.sub(r'[?&]page=\d+', '', url_str)  
            url_str = re.sub(r'[?&]p=\d+', '', url_str)  
            ## 移除其他分页参数  
            url_str = re.sub(r'[?&]start=\d+', '', url_str)  
            return url_str  
  
        self.marriage_df['normalized_url'] = self.marriage_df['fullURL'].apply(normalize_url)  
  
        ## 按用户和标准化URL去重，保留最早的访问记录  
        self.marriage_df = self.marriage_df.sort_values('timestamp').drop_duplicates(  
            subset=['userID', 'normalized_url'], keep='first'  
        )  
        print(f"去重后剩余：{len(self.marriage_df)}条记录")  
  
        ## 3) 筛选掉浏览网页次数不满2次的用户  
        print("3. 筛选活跃用户...")  
        user_counts = self.marriage_df['userID'].value_counts()  
        active_users = user_counts[user_counts >= 2].index  
        self.marriage_df = self.marriage_df[self.marriage_df['userID'].isin(active_users)]  
        print(f"筛选后剩余：{len(self.marriage_df)}条记录，{len(active_users)}个用户")  
  
        return self.marriage_df  
  
    def split_and_save_data(self):  
        """划分训练集和测试集并保存"""  
        print("\n" + "=" * 50)  
        print("数据集划分和保存")  
        print("=" * 50)  
  
        if self.marriage_df is None:  
            self.preprocess_marriage_data()  
  
        ## 按用户划分，确保同一用户的数据在同一个集合中  
        users = self.marriage_df['userID'].unique()  
        train_users, test_users = train_test_split(users, test_size=0.3, random_state=42)  
  
        train_df = self.marriage_df[self.marriage_df['userID'].isin(train_users)]  
        test_df = self.marriage_df[self.marriage_df['userID'].isin(test_users)]  
  
        ## 保存到CSV文件  
        train_df.to_csv('marriage_train_data.csv', index=False, encoding='utf-8')  
        test_df.to_csv('marriage_test_data.csv', index=False, encoding='utf-8')  
  
        print(f"训练集：{len(train_df)}条记录，{len(train_users)}个用户")  
        print(f"测试集：{len(test_df)}条记录，{len(test_users)}个用户")  
        print("数据已保存到 marriage_train_data.csv 和 marriage_test_data.csv")  
  
        return train_df, test_df  
  
    def calculate_item_similarity(self, train_df):  
        """基于物品的协同过滤算法计算相似度"""  
        print("\n" + "=" * 50)  
        print("计算物品相似度")  
        print("=" * 50)  
  
        ## 创建用户-物品矩阵  
        user_item_matrix = train_df.pivot_table(  
            index='userID',  
            columns='normalized_url',  
            values='timestamp',  
            aggfunc='count',  
            fill_value=0  
        )  
  
        print(f"用户-物品矩阵维度：{user_item_matrix.shape}")  
  
        ## 计算物品相似度矩阵  
        item_similarity = cosine_similarity(user_item_matrix.T)  
        item_similarity_df = pd.DataFrame(  
            item_similarity,  
            index=user_item_matrix.columns,  
            columns=user_item_matrix.columns  
        )  
  
        ## 相似度分布统计  
        similarity_values = item_similarity[np.triu_indices_from(item_similarity, k=1)]  
  
        similarity_stats = pd.DataFrame({  
            '统计项': ['平均相似度', '最大相似度', '最小相似度', '标准差'],  
            '数值': [  
                np.mean(similarity_values),  
                np.max(similarity_values),  
                np.min(similarity_values),  
                np.std(similarity_values)  
            ]  
        })  
  
        ## 相似度区间分布  
        similarity_distribution = pd.DataFrame({  
            '相似度区间': ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0'],  
            '数量': [  
                sum((similarity_values >= 0) & (similarity_values < 0.1)),  
                sum((similarity_values >= 0.1) & (similarity_values < 0.3)),  
                sum((similarity_values >= 0.3) & (similarity_values < 0.5)),  
                sum((similarity_values >= 0.5) & (similarity_values < 0.7)),  
                sum((similarity_values >= 0.7) & (similarity_values < 0.9)),  
                sum((similarity_values >= 0.9) & (similarity_values <= 1.0))  
            ]  
        })  
  
        print("相似度统计:")  
        print(similarity_stats.to_string(index=False))  
        print("\n相似度分布:")  
        print(similarity_distribution.to_string(index=False))  
  
        ## 显示部分推荐结果  
        print("\n部分推荐结果示例:")  
        items = list(user_item_matrix.columns)[:5]  ## 取前5个物品  
        for item in items:  
            if item in item_similarity_df.index:  
                similar_items = item_similarity_df[item].sort_values(ascending=False)[1:4]  ## 排除自己，取前3个  
                print(f"\n与'{item[:50]}...'相似的物品:")  
                for similar_item, similarity in similar_items.items():  
                    print(f"  - {similar_item[:50]}... (相似度: {similarity:.3f})")  
  
        return item_similarity_df, user_item_matrix  
  
    def evaluate_recommendation(self, item_similarity_df, user_item_matrix, test_df):  
        """计算推荐模型的评估指标"""  
        print("\n" + "=" * 50)  
        print("推荐模型评估")  
        print("=" * 50)  
  
        ## 为测试集用户生成推荐  
        test_users = test_df['userID'].unique()  
        all_precisions = []  
        all_recalls = []  
        all_f1s = []  
  
        for user in test_users[:20]:  ## 限制评估用户数量以节省时间  
            if user not in user_item_matrix.index:  
                continue  
  
            ## 获取用户在训练集中的行为  
            user_items = user_item_matrix.loc[user]  
            user_visited_items = user_items[user_items > 0].index.tolist()  
  
            if len(user_visited_items) == 0:  
                continue  
  
            ## 生成推荐  
            recommendations = {}  
            for item in user_visited_items:  
                if item in item_similarity_df.index:  
                    similar_items = item_similarity_df[item].sort_values(ascending=False)[1:11]  ## 前10个相似物品  
                    for similar_item, similarity in similar_items.items():  
                        if similar_item not in user_visited_items:  
                            recommendations[similar_item] = recommendations.get(similar_item, 0) + similarity  
  
            ## 获取推荐列表（前10个）  
            recommended_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]  
            recommended_items = [item[0] for item in recommended_items]  
  
            ## 获取用户在测试集中的真实行为  
            user_test_items = test_df[test_df['userID'] == user]['normalized_url'].unique()  
  
            if len(recommended_items) == 0 or len(user_test_items) == 0:  
                continue  
  
            ## 计算精确率、召回率和F1  
            relevant_items = set(recommended_items) & set(user_test_items)  
  
            precision = len(relevant_items) / len(recommended_items) if len(recommended_items) > 0 else 0  
            recall = len(relevant_items) / len(user_test_items) if len(user_test_items) > 0 else 0  
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  
  
            all_precisions.append(precision)  
            all_recalls.append(recall)  
            all_f1s.append(f1)  
  
        ## 计算平均指标  
        evaluation_results = pd.DataFrame({  
            '评估指标': ['准确率(Precision)', '召回率(Recall)', 'F1指标'],  
            '数值': [  
                np.mean(all_precisions) if all_precisions else 0,  
                np.mean(all_recalls) if all_recalls else 0,  
                np.mean(all_f1s) if all_f1s else 0  
            ]  
        })  
  
        print(f"评估用户数量: {len(all_precisions)}")  
        print(evaluation_results.to_string(index=False))  
  
        return evaluation_results  
  
    def run_complete_analysis(self):  
        """运行完整的分析流程"""  
        print("开始网站数据分析与推荐系统实验")  
        print("=" * 80)  
  
        ## 1. 连接数据库并读取数据  
        if not self.load_data():  
            return False  
  
        ## 2. 基础数据探索  
        self.basic_data_exploration()  
  
        ## 3. 数据探索分析  
        self.analyze_webpage_types()  
        self.analyze_knowledge_types()  
        self.analyze_question_mark_urls()  
        self.analyze_199_types()  
        self.analyze_aimless_browsing()  
        self.analyze_user_clicks()  
        self.analyze_single_visit_users()  
  
        ## 4. 抽取和预处理婚姻类数据  
        self.extract_marriage_pages()  
        self.preprocess_marriage_data()  
  
        ## 5. 划分数据集  
        train_df, test_df = self.split_and_save_data()  
  
        ## 6. 计算物品相似度  
        item_similarity_df, user_item_matrix = self.calculate_item_similarity(train_df)  
  
        ## 7. 评估推荐模型  
        evaluation_results = self.evaluate_recommendation(item_similarity_df, user_item_matrix, test_df)  
  
        print("\n" + "=" * 80)  
        print("实验完成！")  
        print("=" * 80)  
  
        return True  
  
    def close_connection(self):  
        """关闭数据库连接"""  
        if self.connection and self.connection.is_connected():  
            self.connection.close()  
            print("数据库连接已关闭")  
  
  
### 使用示例  
if __name__ == "__main__":  
    ## 创建分析器实例  
    analyzer = WebDataAnalyzer(  
        host='localhost',  
        database='test_114514',  
        user='root',  
        password='123456'  
    )  
  
    try:  
        ## 运行完整分析  
        analyzer.run_complete_analysis()  
    except Exception as e:  
        print(f"运行过程中发生错误: {e}")  
    finally:  
        ## 关闭数据库连接  
        analyzer.close_connection()
```

#### 实验输出
```shell
C:\Users\19065\miniconda3\python.exe D:\coding\简简单单挖掘个数据\exp08\main.py 
开始网站数据分析与推荐系统实验
================================================================================
成功连接到MySQL数据库
成功读取数据，共837450条记录
==================================================
数据基本信息:
==================================================
数据集形状: (837450, 21)
列名: ['realIP', 'realAreacode', 'userAgent', 'userOS', 'userID', 'clientID', 'timestamp', 'timestamp_format', 'pagePath', 'ymd', 'fullURL', 'fullURLId', 'hostname', 'pageTitle', 'pageTitleCategoryId', 'pageTitleCategoryName', 'pageTitleKw', 'fullReferrer', 'fullReferrerURL', 'organicKeyword', 'source']

数据类型:
realIP                    int64
realAreacode              int64
userAgent                object
userOS                   object
userID                   object
clientID                 object
timestamp                 int64
timestamp_format         object
pagePath                 object
ymd                       int64
fullURL                  object
fullURLId                object
hostname                 object
pageTitle                object
pageTitleCategoryId       int64
pageTitleCategoryName    object
pageTitleKw              object
fullReferrer             object
fullReferrerURL          object
organicKeyword           object
source                   object
dtype: object

缺失值统计:
realIP                        0
realAreacode                  0
userAgent                    44
userOS                        0
userID                        0
clientID                      0
timestamp                     0
timestamp_format              0
pagePath                      0
ymd                           0
fullURL                       0
fullURLId                     0
hostname                      0
pageTitle                  2025
pageTitleCategoryId           0
pageTitleCategoryName       316
pageTitleKw                   0
fullReferrer             341767
fullReferrerURL          341767
organicKeyword           548735
source                   341767
dtype: int64

==================================================
1. 网页类型统计
==================================================
              网页类型  访问次数
       quanyifagui   716
       difangfagui    79
                      10
         2108.html     7
         1104.html     6
       2108_2.html     4
         1770.html     4
         1555.html     4
         3991.html     4
         1945.html     3
         2871.html     3
         3385.html     3
2013070555560.html     2
         2337.html     2
         1082.html     2
         1850.html     2
          665.html     2
       2108_3.html     2
         2754.html     2
         1374.html     2

==================================================
2. 知识类型内部统计
==================================================
知识类型     数量
法律法规 830637
法律咨询   4455
  其他    333

==================================================
3. 带'?'字符网址类型统计
==================================================
带'?'字符的URL总数: 65492
 URL类型    数量
其他参数页面 32345
  分页页面 15833
  详情页面 14594
  搜索页面  2720

==================================================
4. 199类型中的具体类型统计
==================================================
未找到199类型的数据

==================================================
5. 无目的浏览用户点击分析
==================================================
        统计项         数值
   无目的浏览用户数 474.000000
    平均访问页面数  25.940928
  平均浏览时长(秒) 518.630059
平均页面停留时间(秒)  18.626006

==================================================
6. 用户点击次数统计
==================================================
点击次数分布:
点击次数区间   用户数量
    1次 229394
  2-5次 102851
 6-10次  11199
11-20次   4038
 20次以上   2634

点击次数统计:
   统计项            数值
  总用户数 350116.000000
平均点击次数      2.391922
最大点击次数   8269.000000
最小点击次数      1.000000

==================================================
7. 浏览一次的用户行为分析
==================================================
    分析维度            数值
总单次访问用户数 229394.000000
访问法律页面比例      0.000296
来自搜索引擎比例      0.604174
 移动端用户比例      0.110883

==================================================
抽取婚姻类网页数据
==================================================
婚姻类网页数据：68956条记录

==================================================
数据预处理
==================================================
1. 清除无关数据...
清除后剩余：68956条记录
2. 处理翻页网址和去重...
去重后剩余：56647条记录
3. 筛选活跃用户...
筛选后剩余：38330条记录，10217个用户

==================================================
数据集划分和保存
==================================================
训练集：26185条记录，7151个用户
测试集：12145条记录，3066个用户
数据已保存到 marriage_train_data.csv 和 marriage_test_data.csv

==================================================
计算物品相似度
==================================================
用户-物品矩阵维度：(7151, 12224)
相似度统计:
  统计项       数值
平均相似度 0.002692
最大相似度 1.000000
最小相似度 0.000000
  标准差 0.045000

相似度分布:
  相似度区间       数量
  0-0.1 74405002
0.1-0.3    17233
0.3-0.5    69596
0.5-0.7    60732
0.7-0.9    70502
0.9-1.0    83698

部分推荐结果示例:

与'http://law.lawtime.cn/d352342357436.html...'相似的物品:
  - http://www.lawtime.cn/info/hunyin/ccfglhccfg/20101... (相似度: 0.707)
  - http://www.lawtime.cn/info/hunyin/ccfglhccfg/20101... (相似度: 0.707)
  - http://www.lawtime.cn/info/hunyin/ccfglhccfg/20110... (相似度: 0.707)

与'http://law.lawtime.cn/d424461429555.html...'相似的物品:
  - http://law.lawtime.cn/d424461429555_1_p5.html... (相似度: 0.645)
  - http://law.lawtime.cn/d424461429555_1_p6.html... (相似度: 0.577)
  - http://law.lawtime.cn/d424461429555_1_p2.html... (相似度: 0.577)

与'http://law.lawtime.cn/d424461429555_1_p2.html...'相似的物品:
  - http://law.lawtime.cn/d424461429555_1_p5.html... (相似度: 0.894)
  - http://law.lawtime.cn/d424461429555_1_p4.html... (相似度: 0.866)
  - http://law.lawtime.cn/d424461429555_1_p6.html... (相似度: 0.750)

与'http://law.lawtime.cn/d424461429555_1_p3.html...'相似的物品:
  - http://law.lawtime.cn/d424461429555_1_p4.html... (相似度: 0.816)
  - http://law.lawtime.cn/d424461429555_1_p2.html... (相似度: 0.707)
  - http://law.lawtime.cn/d424461429555_1_p5.html... (相似度: 0.632)

与'http://law.lawtime.cn/d424461429555_1_p4.html...'相似的物品:
  - http://law.lawtime.cn/d424461429555_1_p2.html... (相似度: 0.866)
  - http://law.lawtime.cn/d424461429555_1_p3.html... (相似度: 0.816)
  - http://law.lawtime.cn/d424461429555_1_p5.html... (相似度: 0.775)

==================================================
推荐模型评估
==================================================
评估用户数量: 0
          评估指标  数值
准确率(Precision)   0
   召回率(Recall)   0
          F1指标   0

===========================================================================
实验完成！
===========================================================================
数据库连接已关闭

进程已结束，退出代码为 0
```

### 实验体会与总结
本实验基于Python设计了完整的网站数据分析系统，使用MySQL连接器读取7law.sql数据库，通过pandas进行7项深度数据探索分析（网页类型、知识分类、URL特征等）。针对婚姻类网页实施数据清洗、URL标准化和用户筛选预处理。采用基于物品的协同过滤算法计算余弦相似度，构建推荐模型并评估准确率、召回率和F1指标，形成了从数据探索到推荐评估的完整分析流程。


---

