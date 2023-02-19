import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split


#1.读取数据，数据处理划分训练数据，测试数据
data = pd.read_csv('Features.csv')
X =data.dropna(how='any', axis='rows') #去掉空值
y = X.pop('main')  #标签把原来XY中y去掉，赋值给y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6) #训练测试按7:3分
cat_features = None #[0,1,6] 离散值单独拎出来处理。如果是x1特征列为离散值，则表达式为cat_feature = [0]
train_pool = Pool(X_train, y_train, cat_features=cat_features)#XY数据是从csv表里读进来的，要把数据包装成catboost喜欢的数据格式
test_pool = Pool(X_test, y_test, cat_features=cat_features)

#2.定义模型
model = CatBoostClassifier(task_type='GPU', learning_rate=0.03, depth=4,
                           l2_leaf_reg=3, iterations=100,
                           bagging_temperature=1, random_strength=1, scale_pos_weight=1) #定义模型时，采用GPU加速，学习率空值学习精度0.03，精度太大，学的快，但是不容易找到好的规律，小的话，找的慢，找的细

#3.训练模型，得到最优结果
model.fit(train_pool, eval_set=test_pool, silent=True)#silent即中间结果不输出

#4.分析训练好的模型在测试数据上的效果
bestloss = model.get_best_score() #最优loss
acc = model.score(X_test, y_test) #准确率
model.plot_tree(tree_idx=0, pool=test_pool) #对称树
for i, j in zip(X.columns, model.feature_importances_): #不同特征值对最后结果影响大小
    print('{}:{:.6f}%'.format(i, j)) #将重要性比例打印出来

#5.保存训练模型，也就是保存模型找到的规律文件
model.save_model('class.model') #保存模型（找到的规律文件）

# 6.使用保存好的模型来实际预测将来
XY_noclass = pd.read_csv('实验数据特征.csv')
load_model = CatBoostClassifier().load_model('class.model') #加载保存好的模型到变量load_model
jieguo = load_model.predict(XY_noclass) #直接输出结果1，0
jieguogailv = load_model.predict_proba(XY_noclass)#输出1，0分类的概率


df = pd.DataFrame(data=jieguo, columns=['if_main'])
df.to_csv('实验结果2', index=False)