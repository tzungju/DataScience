# install graphviz and put <installation path>\bin into path
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = load_iris()
X = iris.data[:, 1:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)


export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=iris.feature_names[1:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


# convert .dot to .png from DOS
# dot -Tpng iris_tree.dot -o iris_tree.png


print(tree_clf.predict_proba([[5, 1.5, 5]]))
print(tree_clf.predict([[5, 1.5, 5]]))

labels=['Setosa', 'Versicolour', 'Virginica']
print(labels[tree_clf.predict([[5, 1.5, 5]])[0]])

