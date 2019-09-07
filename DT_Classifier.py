'''
利用iris数据建立决策树
并可视化化决策树，保存到
《iris.pdf》中
'''


from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
# import pydotplus

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 结果可视化
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

