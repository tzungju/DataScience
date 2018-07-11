from sklearn.datasets import load_iris
from sklearn import tree

# 載入資料集
iris = load_iris()

# 建立模型及訓練
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 畫出 Decision Tree，存成 iris.pdf，並產生規則文字檔(iris)
import graphviz 
# 畫出 Decision Tree 單色圖，存成 iris.pdf
#dot_data = tree.export_graphviz(clf, out_file=None) 
# 畫出 Decision Tree 彩色圖，存成 iris.pdf
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("iris") 

# 預測
print("第2筆資料 ...")
pred_data = iris.data[:1, :]
print('pred_data', pred_data)
print('預測值  =', clf.predict(pred_data)[0])
print('預測名稱=', iris.target_names[clf.predict(pred_data)[0]])
# 顯示各類的機率
print('各類的機率=', clf.predict_proba(pred_data))
print("")

print("第52筆資料 ...")
pred_data = iris.data[51:52, :]
print('pred_data', pred_data)
print('預測值  =', clf.predict(pred_data)[0])
print('預測名稱=', iris.target_names[clf.predict(pred_data)[0]])
# 顯示各類的機率
print('各類的機率=', clf.predict_proba(pred_data))
print("")

# 計算準確率
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data[:,0:4], iris.target, test_size=0.2)
print('準確率=',clf.score(X_test, y_test))
