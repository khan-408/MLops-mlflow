import pandas as pd
import numpy as np
import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split


def get_data():
    URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    #reading the data
    try:
        df = pd.read_csv(URL,sep=';')

        return df

    except Exception as e:
        raise e

def evaluate_func(y_test,y_pred):
    #evaluation
    # mae = mean_absolute_error(y_test,y_pred)
    # mse = mean_squared_error(y_test,y_pred)
    # rmse = np.sqrt(mse)
    # r2 = r2_score(y_test,y_pred)*100
    # return mae,mse,rmse,r2
    accu  = accuracy_score(y_test,y_pred)*100
    return accu

def main(n_estimators,max_depth):
    
    df = get_data()
    #train Test Split
    train,test = train_test_split(df,random_state=42)
    # Dependent adn independent variable
    X_train = train.drop(['quality'],axis=1)
    y_train = train['quality']
    X_test = test.drop(['quality'],axis=1)
    y_test = test['quality']
    #model Training
    # model = ElasticNet()
    model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    model.fit(X_train, y_train)
    #prediction
    y_pred = model.predict(X_test)
    #evaluating the model
    accu = evaluate_func(y_test=y_test,y_pred=y_pred)

    # print(f'mae: {mae}, mse: {mse}, rmse: {rmse},r2_score: {r2}.')
    print(f'accuracy: {accu}.')

if __name__=='__main__':






     
    args = argparse.ArgumentParser()
    args.add_argument('--n_estimators','-n',default=50,type=int)
    args.add_argument('--max_depth','-m', default=5,type=int)
    parse_args = args.parse_args()
    try: 
        main(n_estimators=parse_args.n_estimators, max_depth=parse_args.max_depth)
    except Exception as e:
        raise e