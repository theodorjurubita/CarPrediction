################ Data preparation ################

import pandas as pd
from pymongo import MongoClient;
pd.options.mode.chained_assignment = None  # default='warn'

mongoUrl = 'mongodb://localhost:27017/Cars'
client = MongoClient(mongoUrl)
cars = client.Cars
info = cars.Info

# import seaborn as sns
# import matplotlib.pyplot as plt

sf = pd.DataFrame.from_dict(info.find())
sfSeria3 = sf[sf.Model == 'BMW 3']
sfSeria5 = sf[sf.Model == 'BMW 5']
sfSeria7 = sf[sf.Model == 'BMW 7']

# sf = pd.read_csv('data/toateSeriileCurate.csv')
#print(sfSeria3)
#print(sfSeria5)
#print(sfSeria7)

def createModelColumn(X):
    model = X.split()
    if (model[1].startswith('3')):
        return 1
    elif (model[1].startswith('5')):
        return 2
    elif (model[1].startswith('7')):
        return 3


def createYearColumn(X):
    return X


def createHorsePowerColumn(X):
    return X

def createFuelColumn(X):
    if (X == 'Benzina'):
        return 1
    elif (X == 'Diesel'):
        return 2


def createTransmisionColumn(X):
    if (X.lower() == 'automata'):
        return 1
    elif (X.lower() == 'manuala'):
        return 2


def createMilageColumn(X):
    milage = X
    return milage


def createPriceColumn(Y):
    price = Y
    return price

def fitTheColumns(dataFrame):
    dataFrame['ModelCategory'] = dataFrame['Model'].apply(createModelColumn)
    dataFrame['YearCategory'] = dataFrame['Year'].apply(createYearColumn)
    dataFrame['HorsePowerCategory'] = dataFrame['HorsePower'].apply(createHorsePowerColumn)
    dataFrame['FuelCategory'] = dataFrame['Fuel'].apply(createFuelColumn)
    dataFrame['Km'] = dataFrame['Mileage'].apply(createMilageColumn)
    dataFrame['TransmissionCategory'] = dataFrame['Transmission'].apply(createTransmisionColumn)
    dataFrame['PriceCategory'] = dataFrame['Price'].apply(createPriceColumn)

    dataFrame = dataFrame.drop(['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage', 'Transmission', 'Price'], axis=1)

    dataFrame = dataFrame.rename(index=str, columns={'ModelCategory': 'Model', "YearCategory": "Year",
                                       "HorsePowerCategory": "HorsePower", "FuelCategory": "Fuel", 'Km': 'Mileage',
                                       'TransmissionCategory': 'Transmission', 'PriceCategory': 'Price'})
    return dataFrame
# print(sf.head())

################# Training Classifier ################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import statistics
import numpy as np
import category_encoders as ce

import time

mean = [2]
R2score = [2]
RMSE = [2]
Error = [2]

def trainTheData(dataFrame):
    X = dataFrame[['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage', 'Transmission']]
    Y = dataFrame['Price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=int(time.time()))

    regressor = GradientBoostingRegressor()
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_test)

    model = int(dataFrame['Model'][0])
    print(f"------------------------Actual vs Predicted values for BMW Seria {model*2+1}--------------------------")
    df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred.astype(int)})
    df1 = df.head(25)
    print(df1)

    theMean = statistics.mean(Y_test)
    theRootMeanSquaredError = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))
    r2Score = metrics.r2_score(Y_test, y_pred)
    percentageOfError = (theRootMeanSquaredError * 100) / theMean

    mean.insert(model-1, theMean)
    R2score.insert(model-1, r2Score)
    RMSE.insert(model-1, theRootMeanSquaredError)
    Error.insert(model-1, percentageOfError)

    print('Mean', theMean)
    print('R2 score:', r2Score)
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Error:', theRootMeanSquaredError)
    print('Percentage of error: ', percentageOfError, ' %')
    return regressor

dataFrameSeria3 = fitTheColumns(sfSeria3)
dataFrameSeria5 = fitTheColumns(sfSeria5)
dataFrameSeria7 = fitTheColumns(sfSeria7)

seria3Regressor = trainTheData(dataFrameSeria3)
seria5Regressor = trainTheData(dataFrameSeria5)
seria7Regressor = trainTheData(dataFrameSeria7)

#print(100 * seria3Regressor.feature_importances_/seria3Regressor.feature_importances_.max())


################# API ################
from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)


class PricePrediction(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('Model')
        parser.add_argument('Year')
        parser.add_argument('HorsePower')
        parser.add_argument('Fuel')
        parser.add_argument('Mileage')
        parser.add_argument('Transmission')

        args = parser.parse_args()
        print(args)
        result = 0
        if(args['Model'].startswith('BMW 3')):
            result = seria3Regressor.predict(
                [[
                    createModelColumn(args['Model']),
                    int(args['Year']),
                    int(args['HorsePower']),
                    createFuelColumn(args['Fuel']),
                    int(args['Mileage']),
                    createTransmisionColumn(args['Transmission'])
                ]]
            )
        elif(args['Model'].startswith('BMW 5')):
            result = seria5Regressor.predict(
                [[
                    createModelColumn(args['Model']),
                    int(args['Year']),
                    int(args['HorsePower']),
                    createFuelColumn(args['Fuel']),
                    int(args['Mileage']),
                    createTransmisionColumn(args['Transmission'])
                ]]
            )
        elif(args['Model'].startswith('BMW 7')):
            result = seria7Regressor.predict(
                [[
                    createModelColumn(args['Model']),
                    int(args['Year']),
                    int(args['HorsePower']),
                    createFuelColumn(args['Fuel']),
                    int(args['Mileage']),
                    createTransmisionColumn(args['Transmission'])
                ]]
            )

        print("-----------------------RESULT-------------------------")
        print(result)

        responseBody = {
            "Price": f"{result[0]}"
        }

        return responseBody, 200

class DataModelStatistics(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('Model')
        args = parser.parse_args()
        index = createModelColumn(args['Model']) - 1

        responseBody = {
            "Mean": f"{mean[index]}",
            "RMSE": f"{RMSE[index]}",
            "R2Score": f"{R2score[index]}",
            "Error": f"{Error[index]}",
        }

        return responseBody, 200

api.add_resource(PricePrediction, "/price")
api.add_resource(DataModelStatistics, "/statistics")

app.run(debug=True, port=5000)
