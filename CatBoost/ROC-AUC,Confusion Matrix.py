import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


#1.读取数据，数据处理划分训练数据，测试数据
data = pd.read_csv('Features.csv')
X =data.dropna(how='any', axis='rows') #去掉空值
y = X.pop('main')  #标签把原来XY中y去掉，赋值给y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #训练测试按7:3分
cat_features = None #[0,1,6] 离散值单独拎出来处理。如果是x1特征列为离散值，则表达式为cat_feature = [0]
train_pool = Pool(X_train, y_train, cat_features=cat_features)#XY数据是从csv表里读进来的，要把数据包装成catboost喜欢的数据格式
test_pool = Pool(X_test, y_test, cat_features=cat_features)

#2.定义模型
model = CatBoostClassifier(task_type='GPU', learning_rate=0.03,
                           depth=4,
                           l2_leaf_reg=3, iterations=100,
                         scale_pos_weight=1) #定义模型时，采用GPU加速，学习率空值学习精度0.03，精度太大，学的快，但是不容易找到好的规律，小的话，找的慢，找的细

#3.训练模型，得到最优结果
model.fit(train_pool, eval_set=test_pool, silent=True)#silent即中间结果不输出
# model.fit(X,y,silent=True)

##特征变量统计：float feature
import matplotlib.pyplot as plt
feature = 'C_S'
res = model.calc_feature_statistics(X_train, y_train, feature,plot=True)

#####模型评估
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import plotly.graph_objs as go
import plotly.express as px

######1.评估指标及结果：准确率、查准率、召回率、F1分值
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_f1_score = f1_score(y_train, y_pred_train)
test_f1_score = f1_score(y_test, y_pred_test)
print('train_f1_score:{:.5}'.format(train_f1_score))
print('test_f1_score:{:.5}\n'.format(test_f1_score))

train_pre_score = precision_score(y_train, y_pred_train)
test_pre_score = precision_score(y_test, y_pred_test)
print('train_pre_score:{:.5}'.format(train_pre_score))
print('test_pre_score:{:.5}\n'.format(test_pre_score))

train_rec_score = recall_score(y_train, y_pred_train)
test_rec_score = recall_score(y_test, y_pred_test)
print('train_rec_score:{:.5}'.format(train_rec_score))
print('test_rec_score:{:.5}\n'.format(test_rec_score))


######2.查看是否过拟合：训练集和测试集的分数,训练集分数和测试集分数基本相当，没有出现过拟合现象
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('train_acc_score:{:.4f}'.format(train_score))
print('test_acc_score:{:.4f}'.format(test_score))

######3.混淆矩阵
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

y_pred_test = model.predict(X_test) #获得预测结果
y_pred_true = y_test.copy() #获得真实标签
cm = confusion_matrix(y_pred_true, y_pred_test,labels=[0,1])
ax=sn.heatmap(cm, annot=True,cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predict')
ax.set_ylabel('True')
plt.savefig('CM.jpg',dpi=500)
plt.show()



######4.ROC曲线FPR-TPR/FPR-FNR
from catboost.utils import get_roc_curve
import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

curve = get_roc_curve(model, test_pool)
(fpr, tpr, thresholds) = curve
roc_auc = sklearn.metrics.auc(fpr, tpr)
print('AUC:{:.4f}'.format(roc_auc))
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label="ROC curve(area=%.4f)" % roc_auc, alpha=0.5)#绘制AUC曲线
# plt.xlim(-0.05,1.05)
# plt.ylim(-0.05,1.05) #x,y坐标范围
plt.legend(loc='lower right')#设置显示标签的位置
plt.xlabel('False Positive Rate') #x轴对应标签
plt.ylabel('True Positive Rate') #y轴对应标签
plt.grid(visible=True, ls=':') #绘制网格底板,ls表示line style
plt.title(u'Decision ROC Curve and AUC') #打印标题
plt.plot([0,1], [0,1], color='navy',lw=lw, linestyle='--', alpha=0.5)
plt.savefig('AUC.jpg', dpi=600)
plt.show()
