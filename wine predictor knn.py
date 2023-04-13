import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

wine = datasets.load_wine()
df = pd.DataFrame(wine["data"],columns=wine["feature_names"])
df["target"] = wine["target"]
df.head()
print(df.head())
df.shape
print(df.shape)

#### trainig

x = df
y = x.pop("target")
x.head()
print(x.head())
y.head()
print(y.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=55)
x_train.shape
x_test.shape

knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
print(knn.score(x_test,y_test))

## Using diff values of k

k_range= range(1,25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    scores.append(knn.score(x_test,y_test))

plt.figure()
plt.xlabel("k count")
plt.ylabel("Model Accuracy")
plt.scatter(k_range,scores)
plt.grid()
plt.xticks([0,5,10,15,20,30])
plt.show()


### Make predictions

predictions = knn.predict(x_test)
print(predictions)
cm = confusion_matrix(y_test,predictions)
print(cm)