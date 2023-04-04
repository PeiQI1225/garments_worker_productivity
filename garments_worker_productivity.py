import pandas as pd

#设置value的显示长度为200，默认为50
pd.set_option('max_colwidth',200)
#显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
#显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
# %matplotlib inline

import warnings

# 不显示警告
warnings.filterwarnings('ignore')

# 读取数据，查看前五行数据
garments = pd.read_csv(r'D:\study\data\garment_worker\garments_worker_productivity.csv')

# print(garments['quarter'].value_counts())
# print(garments.describe(include='all'))

# print(round(garments.wip.isnull().sum()/len(garments.wip),2))

# 删除wip列
garments.drop(columns=['wip'], inplace=True)

garments.date= pd.to_datetime(garments.date)
garments.set_index(['date'], inplace=True)
# print(garments.index)

garments['month'] = garments.index.month_name().values
# print(garments.head())

# 将over_time转化为小时
garments['over_time'] = garments['over_time'] / 60
# print(garments.head())

# 使用astype方法将team转为object
garments['team'] = garments['team'].astype('object')


# print(garments['department'].unique())
garments['department'] = garments.department.apply(lambda x: 'finishing' if 'finishing' in x else 'sewing')
# print(garments['department'].value_counts())
garments['actual_productivity'] = garments.actual_productivity.apply(lambda x: 1 if x>1 else x)
# print(garments.head())
team = garments.groupby(['department', 'team'])['team'].count()
# print(team)
# print(garments.groupby(['department','team'])[['targeted_productivity','actual_productivity']].agg({'mean','max','min'}))

# # 绘制嵌套饼图
# fig, ax = plt.subplots(figsize=(5,5))
# size = 0.3
# vals = team.unstack().values
# # 设置每一块的颜色
# cmap = sns.diverging_palette(220,10,as_cmap=True)
# outer_colors = cmap([30,220])
# inner_colors = cmap(np.hstack([np.arange(60,120,5),np.arange(130,190,5)]))
# # 外圈部门数量
# ax.pie(vals.sum(axis=1), radius=1, labels = ['精加工车间','缝纫部'], colors=outer_colors,
#        wedgeprops=dict(width=size, edgecolor='w'),textprops={'fontsize': 12},autopct='%.1f%%',pctdistance=0.85
#       )
# # 内圈团队数量
# ax.pie(vals.flatten(), radius=1-size,colors=inner_colors,labels=['']*12+[x for x in range(1,13)],
#        labeldistance=.8,wedgeprops=dict(width=size, edgecolor='w')
#       )
# ax.set(aspect="equal")
# plt.title('各部门员工人数占比', fontsize=15)
# plt.show()

team_c = round(garments.groupby(['department', 'team'])['no_of_workers'].mean())
# print(team_c)

# # 绘制嵌套饼图
# fig, ax = plt.subplots(figsize=(5,5))
# size = 0.3
# vals = team_c.unstack().values
# # 设置每一块的颜色
# cmap = sns.diverging_palette(220,10,as_cmap=True)
# outer_colors = cmap([30,220])
# inner_colors = cmap(np.hstack([np.arange(60,120,5),np.arange(130,190,5)]))
# # 外圈部门数量
# ax.pie(vals.sum(axis=1), radius=1, labels = ['精加工车间','缝纫部'], colors=outer_colors,
#        wedgeprops=dict(width=size, edgecolor='w'),textprops={'fontsize': 12},autopct='%.1f%%',pctdistance=0.85
#       )
# # 内圈团队数量
# ax.pie(vals.flatten(), radius=1-size,colors=inner_colors,labels=['']*12+[x for x in range(1,13)],
#        labeldistance=.8,wedgeprops=dict(width=size, edgecolor='w')
#       )
# ax.set(aspect="equal")
# plt.title('各部门员工人数占比', fontsize=15)
# plt.show()

# fig, ax =plt.subplots(figsize=(8,4))
# dpt_g = garments.groupby('department')['actual_productivity']
# sns.distplot(dpt_g.get_group('finishing'),label='精加工车间')
# sns.distplot(dpt_g.get_group('sewing'), label='缝纫部')
# plt.title('各部门生产率分布情况',size=15)
# plt.legend()
# plt.xlabel('生产率', size=12)
# plt.ylabel('频次', size=12)
# plt.show()

# fig, ax =plt.subplots(figsize=(12,4))
# sns.boxplot(x='team',y='actual_productivity',data=garments,hue='department')
# plt.title('各团队生产率分散情况',size=15)
# plt.xlabel('团队编号', size=12)
# plt.ylabel('生产率', size=12)
# plt.legend(loc='lower left')
# plt.show()

# fig, ax =plt.subplots(3,1,figsize=(8,10))
# # 设置颜色
# my_colors = ["#F1B5B9", "#AAC8D1"]
# sns.set_palette( my_colors )
# sns.boxplot(x='day',y='actual_productivity',data=garments,hue='department', order=['Monday','Tuesday','Wednesday','Thursday','Saturday','Sunday'], ax=ax[0])
# sns.boxplot(x='quarter',y='actual_productivity',data=garments,hue='department', ax=ax[1])
# sns.boxplot(x='month',y='actual_productivity',data=garments,hue='department', ax=ax[2])
# ax[0].set_title('不同星期对员工生产率的影响',size=15)
# ax[1].set_title('每月各周对员工生产率的影响',size=15)
# ax[2].set_title('不同月份对员工生产率的影响',size=15)
# ax[0].legend(loc='lower right')
# ax[1].legend(loc='lower right')
# ax[2].legend(loc='lower right')
# # 设置默认的间距
# plt.tight_layout()
# plt.show()

# 计算相关系数
# corr = garments.corr()
# fig = plt.figure(figsize=(7,7))
# cmap = sns.diverging_palette(220,10,as_cmap=True)
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# sns.heatmap(corr, annot=True, square=True,linewidths=1.5, cmap=cmap, mask=mask, center=0, cbar_kws={"shrink": .5})
# plt.title("特征间的相关性",fontsize=15)
# plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
garments['department'] = le.fit_transform(garments['department'])
garments = pd.get_dummies(garments)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X=garments.drop(['actual_productivity'],1)
y=garments['actual_productivity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# 数据标准化
s = StandardScaler()
X_train_sm = s.fit_transform(X_train)
X_test_sm = s.transform(X_test)
X_train_sm = X_train
X_test_sm = X_test

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
mlr = LinearRegression()
mlr.fit(X_train_sm, y_train)
y_pred_sm = mlr.predict(X_test_sm)
# print(f'R2 为 {round(r2_score(y_test,y_pred_sm),4)}')
# print('均方误差 (MSE): %.4f' % mean_squared_error(y_test, y_pred_sm))
# 可视化
# plt.figure(figsize=(6, 4))
# plt.scatter(x=y_test,y=y_pred_sm, alpha=.5)
# plt.xlabel('实际值')
# plt.ylabel('预测值')
# plt.title('实际值预测值对比',size=15)
# plt.show()

# 使用回归决策树
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
dtreg = DecisionTreeRegressor(max_depth=10,random_state=0)
dtreg.fit(X_train_sm, y_train)
y_pred_sm = dtreg.predict(X_test_sm)
# print(f'R2 为 {round(r2_score(y_test,y_pred_sm),4)}')
# print('均方误差 (MSE): %.4f' % mean_squared_error(y_test, y_pred_sm))
#
# # 可视化
# plt.figure(figsize=(6, 4))
# plt.scatter(x=y_test, y=y_pred_sm, alpha=.5)
# plt.xlabel('实际值')
# plt.ylabel('预测值')
# plt.title('实际值预测值对比',size=15)
# plt.show()

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(random_state=1024)
mlp.fit(X_train_sm, y_train)
y_pred_sm = mlp.predict(X_test_sm)
y_pred_sm = pd.DataFrame(y_pred_sm)[0].apply(lambda x: 1 if x > 1 else x).values
y_pred_sm = pd.DataFrame(y_pred_sm)[0].apply(lambda x: 0 if x < 0 else x).values
print(f'R2 为 {round(r2_score(y_test,y_pred_sm),4)}')
print('均方误差 (MSE): %.4f' % mean_squared_error(y_test, y_pred_sm))
# 可视化
plt.figure(figsize=(6, 4))
plt.scatter(x=y_test,y=y_pred_sm, alpha=.5)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值预测值对比',size=15)
plt.show()
