from sklearn import tree

fertures = [[140,1],[130,1],[150,0],[170,0]]

labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()

clf.fit(fertures,labels)

predict = clf.predict([[150,0]])

print(predict)