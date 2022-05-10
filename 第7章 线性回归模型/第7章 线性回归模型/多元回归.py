#导入模块
import statsmodels.api as sm
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# #这种方法会导致自动把California作为对照组
# #导入数据
# Profit = pd.read_excel('Predict to Profit.xlsx')
# print(Profit.head(5))
# #将数据集拆分为训练集和测试集
# train,test = model_selection.train_test_split(Profit,test_size=0.2,random_state=1234)  #test_size是测试集占20%
#
# #根据train数据集建模
# model = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)',data = train).fit() #State是非数值型离散变量，建模时将其设置为哑变量，套用category分类
# print('模型的偏回归系数分别为：\n',model.params)
#
# #删除test数据集中的Profit变量，用剩下的自变量进行预测
# test_X = test.drop(labels = 'Profit',axis = 1)
# pred = model.predict(exog = test_X)
# print('对比预测值和实际值的差异：\n',pd.DataFrame({'Prediction':pred,'Real':test.Profit}))


#导入数据
Profit = pd.read_excel('Predict to Profit.xlsx')

#生成由State变量衍生的哑变量
dummies = pd.get_dummies(Profit.State)

#将哑变量与原始数据集水平合并
Profit_New = pd.concat([Profit,dummies],axis=1)

#删除State变量和California变量（因为State变量已经被分解成哑变量，New York变量需要作为参照组）(这时New York就是基准线了)
Profit_New.drop(labels=['State','New York'],axis=1,inplace=True)

#拆分数据集Profit_New
train,test = model_selection.train_test_split(Profit_New,test_size=0.2,random_state=1234)

#建模
model2 = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + Florida + California',data = train).fit()

print('模型的偏回归系数分别为:\n',model2.params)

#显著性检验
print()
"""F检验"""
#统计变量个数和观测个数
p = model2.df_model   #p是SSR的自由度，n-p-1是SSE的自由度
n = train.shape[0]
#查表
from scipy.stats import f
#计算F分布的理论值
F_Theroy = f.ppf(q = 0.95,dfn = p,dfd = n-p-1)
print('F分布的理论值为:',F_Theroy)
print('F分布的实际值为:',model2.fvalue)


"""t检验"""
print(model2.summary())

