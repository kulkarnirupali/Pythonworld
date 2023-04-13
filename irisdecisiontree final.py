import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()
print("Features names of iris data set ")
print(iris.feature_names)

print("Target names of the data set")
print(iris.target_names)

# indices of removed elements
test_index = [1,51,101]

# training data with removed elements
train_target = np.delete(iris.target,test_index)
train_data = np.delete(iris.data,test_index,axis=0)

# testing data of testing on training data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

# From decision tree classifier
classifier = tree.DecisionTreeClassifier()

# Apply training data to form tree
classifier.fit(train_data,train_target)

print("Values that we removed for testing ")
print(test_target)

print("Result of testing ")
print(classifier.predict(test_data))