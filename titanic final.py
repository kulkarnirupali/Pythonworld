import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousTitanicLogistic():
    # step1 : Load the data
    titanic_data = pd.read_csv('MarvellousTitanicDataset.csv')
    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passengers are "+str(len(titanic_data)))

    # Step 2 = Analyze data
    print(" Visualization : Survived and non survived passangers")
    figure()
    target = "Survived"
    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystem : Survived and non survived passangers")
    show()

    print(" Visualization : Survived and non survived passangers based on the Passanger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data,x=target,hue = "Sex").set_title("Marvellous Infosystem : Survived and non survived passangers based on Gender")
    show()

    print(" Visualization : Survived and non survived passangers based on the Passanger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data,x=target,hue = "Pclass").set_title("Marvellous Infosystem : Survived and non survived passangers based on Class")
    show()

    print(" Visualization : Survived and non survived passangers based on the AGE")
    figure()

    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystem : Survived and non survived passanger based on Age")
    show()

    print(" Visualization : Survived and non survived passangers based on the Fare")
    figure()

    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystem : Survived and non survived passanger based on Age")
    show()

    # Step 3 = Data Cleaning
    titanic_data.drop("zero",axis=1,inplace=True)

    print("First 5 entries from loaded dataset after removing zexo column")
    print(titanic_data.head(5))

    print("Values of sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))

    print("Values of Pass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))

    print("Values of Pass column after concatenating new column")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
    print(titanic_data.head(5))

    print("Values of data set after removing irrelevent column")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head(5))

    x = titanic_data.drop("Survived",axis=1)
    y = titanic_data["Survived"]

    # Step 4 = Data Training
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.5)

    logmodel = LogisticRegression()
    logmodel.fit(xtrain,ytrain)

    # step 5 = Data testing
    prediction = logmodel.predict(xtest)

    # step 6 = Calculate Accuracy
    print("Classification report of Logistic Regression is :")
    print(classification_report(ytest,prediction))

    print("Confusion Matrix  of Logistic Regression is :")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistic Regression is :")
    print(accuracy_score(ytest,prediction))

def main():
    print("Supervised Machine Learning")
    print("Logistic Regression on Titanic data set ")
    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()





