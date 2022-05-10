#导入第三方模块
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#导入数据集
income = pd.read_csv('Salary_Data.csv')

#绘制散点图
sns.lmplot(x='YearsExperience',y='Salary',data = income,ci=None)

#显示图形
plt.show()

#导入第三方模块
import statsmodels.api as sm

#利用收入数据集，构建回归模型
fit = sm.formula.ols('Salary ~ YearsExperience',data=income).fit()

#返回模型的参数值
print(fit.params)