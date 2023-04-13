import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MarvellousHeadBrainPredictor():

    # Load data
    data = pd.read_csv('MarvellousHeadBrain (1) (1).csv')

    print("Size of dataset ",data.shape)

    x = data['Head Size(cm^3)'].values
    y = data['Brain Weight(grams)'].values

    x = x.reshape(-1,1)
    n = len(x)

    reg = LinearRegression()
    reg = reg.fit(x,y)
    y_pred = reg.predict(x)
    r2 = reg.score(x,y)
    print(r2)

def main():
    print("Supervised Machine Learning ")
    print("Linear Regression On Head and Brain size data set ")
    MarvellousHeadBrainPredictor()

if __name__=="__main__":
    main()
