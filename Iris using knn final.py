from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


def euc(a,b):
    return distance.euclidean(a,b)


class MarvellousKNN():
    def fit(self,trainingdata,trainingtarget):
        self.trainingdata = trainingdata
        self.trainingtarget = trainingtarget

    def predict(self,testdata):
        prediction = []
        for row in testdata:
            lebel = self.closet(row)
            prediction.append(lebel)
        return prediction


    def closet(self,row):
        bestdistance = euc(row,self.trainingdata[0])
        bestindex = 0
        for i in range(1,len(self.trainingdata)):
            dist = euc(row,self.trainingdata[i])
            if dist < bestdistance:
                bestdistance = dist
                bestindex = i

        return self.trainingtarget[bestindex]

def Marvellousneighbors():
    border = "-" * 50

    iris = load_iris()
    data = iris.data
    target = iris.target
    print(border)
    print("Actual data set ")
    print(border)

    for i in range(len(iris.target)):
        print("ID : %d ,label %s,Feature : %s "% (i,iris.data[i],iris.target[i]))
    print("Size of actual dataset is %d" % (i+1))

    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.5)
    print(border)
    print("training data set")
    print(border)

    for i in range(len(data_test)):
        print("ID : %d ,label %s,Feature : %s "% (i,data_test[i],target_test[i]))
    print("Size of actual dataset is %d" % (i + 1))
    print(border)

    classifier = MarvellousKNN()
    classifier.fit(data_train,target_train)
    predictions = classifier.predict(data_test)
    Accuracy = accuracy_score(target_test,predictions)
    return Accuracy

def main():
    Accuracy = Marvellousneighbors()
    print("Accuracy of classification algorithm with K neighbor classifier is ",Accuracy*100,"%")

if __name__=="__main__":
    main()

