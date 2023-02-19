import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from catboost import *
shap.initjs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#1.读取数据，数据处理划分训练数据，测试数据
data = pd.read_csv('Features.csv')
X = data.dropna(how='any', axis='rows') #去掉空值
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


###################################SHAP解释###################################################
##################################### Tree  Explainer###################################
tree_explainer = shap.TreeExplainer(model)
tree_shap_values = tree_explainer.shap_values(X)
print(tree_shap_values.shape)

#1.单特征图
#(1)局部：对单个解释
#force_plot 形象表达了博弈学的‘对抗’概念。用于解释每个样本的预测结果，也可用于解释多个样本的预测结果.
i=346
shap.force_plot(tree_explainer.expected_value, tree_shap_values[i], X.iloc[i], matplotlib=True,show=False,link="identity", plot_cmap="RdBu")
plt.show()
# y_base=tree_explainer.expected_value(i)

#water fall 瀑布图：对特征的贡献(重要度)进行了排名
shap.plots._waterfall.waterfall_legacy(tree_explainer.expected_value, tree_shap_values[i], feature_names=X.columns)
plt.savefig('water.jpg', dpi=500)

# #(2)全局：整体
# # decision_plot
# shap.decision_plot(tree_explainer.expected_value, tree_shap_values, X, return_objects=True)
# plt.savefig('decision.jpg',dpi=500)
# # summary_plot
# shap.summary_plot(tree_shap_values,X)
# shap.summary_plot(tree_shap_values, X, plot_type='bar',feature_names=X.columns)
# plt.savefig('water.jpg', dpi=500)


# #2.特征组合图
# # dependence_plot 依赖图 单变量影响图
# import matplotlib.pyplot as plt
# shap.dependence_plot('C_S', tree_shap_values, X)
# shap.dependence_plot('L_R', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('A_I', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('N_R', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('N_S', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('D_R', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('C_R', tree_shap_values, X, interaction_index='C_S')
# shap.dependence_plot('M_U', tree_shap_values, X, interaction_index='C_S')
# plt.savefig('water.jpg', dpi=5000)
