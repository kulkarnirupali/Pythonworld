##################################################################
# Required Python Packages
##################################################################

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


##################################################################
# File path
#################################################################

INPUT_PATH = "breast-cancer-wisconsin.data"
OUTPUT_PATH = "breast-cancer-wisconsin.csv"

###############################################################
# Header
###############################################################

HEADERS = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]

###############################################################
# FUNCTION NAME = read_data
# DESCRIPTION = Read the data into pandas data frame
# Inpt = path of the CSV file
# output = Gives the data
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def read_data(path):
    data = pd.read_csv(path)
    return data

################################################################
# FUNCTION NAME = get_header
# DESCRIPTION = dataset header
# Inpt = dataset
# output = Return the header
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def get_header(dataset):
    return dataset.columns.values

#################################################################

################################################################
# FUNCTION NAME = add_header
# DESCRIPTION = Add the header to the dataset
# Inpt = dataset
# output = Update dataset
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def add_headers(dataset,headers):
    dataset.columns = headers
    return dataset

#################################################################

################################################################
# FUNCTION NAME = data_file_to_csv
# Inpt = Nothing
# output = Write data into the CSV
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def data_file_to_csv():
    # Header
    headers = ["CodeNumber","ClumpThickness","UniformityCellSize","UniformityCellShape","MarginalAdhesion","SingleEpithelialCellSize","BareNuclei","BlandChromatin","NormalNucleoli","Mitoses","CancerType"]
    # Laod the dataset into Pandas data frame
    dataset = read_data(INPUT_PATH)
    # Add the header to the added dataset
    dataset = add_headers(dataset,headers)
    # Save the loaded dataset into csv format
    dataset.to_csv(OUTPUT_PATH,index=False)
    print("File Saved Successfully ..........!")

##################################################################

################################################################
# FUNCTION NAME = split_dataset
# Description = split the dataset with train_percentage
# Input = Dataset with related information
# output = dataset after splitting
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def split_dataset(dataset,train_percentage,feature_headers,target_headers):
    # Split dataset into train and test dataset
    train_x,test_x,train_y,test_y = train_test_split(dataset[feature_headers],dataset[target_headers],train_size=train_percentage)
    return train_x,test_x,train_y,test_y

##################################################################


################################################################
# FUNCTION NAME = handel_missing_values
# Description = Filter missing values from the dataset
# Input = Dataset with missing values
# output = dataset by removing missing values
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def handle_missing_values(dataset,missing_values_header,missing_label):
    return dataset[dataset[missing_values_header] != missing_label]
################################################################


################################################################
# FUNCTION NAME = Random_forest_classifier
# Description = To train the random forest classifier with features and target data
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def ramdom_forst_clssifier(features,target):
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf
################################################################


################################################################
# FUNCTION NAME = dataset_statistics
# Description = Basic statistics of the dataset
# Input = Dataset
# output =  Description of dataset
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def dataset_statistics(dataset):
    print(dataset.describe())

#################################################################

################################################################
# FUNCTION NAME = Main
# Description = Main function from where execution starts
# Author = Piyush Manohar KHairnar
# Data = 01/02/2022
################################################################

def main():
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)
    # get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    # Filter Missing Values
    dataset = handle_missing_values(dataset,HEADERS[6],'?')
    train_x,test_x,train_y,test_y = split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])

    # Train and test dataset size details
    print("Train_x Shape::",train_x.shape)
    print("Train_y Shape::",train_y.shape)
    print("test_x Shape ::",test_x.shape)
    print("test_x Shape ::",test_y.shape)

    # Create Random forest classifier instance
    trained_model = ramdom_forst_clssifier(train_x,train_y)
    print("Trained Model ::",trained_model)
    predictions = trained_model.predict(test_x)


    for i in range(0,205):
        print("Actual outcome :: {} and Predicted outcome :: {} ".format(list(test_y)[i],predictions[i]))

    print("Train Accuracy ::",accuracy_score(train_y,trained_model.predict(train_x)))
    print("Test Accuracy ::",accuracy_score(test_y,predictions))
    print("Confusion matrix",confusion_matrix(test_y,predictions))

###################################################################
# Application starter
###################################################################

if __name__=="__main__":
    main()

