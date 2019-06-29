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
    # horsePower = X;
    #
    # if (horsePower < 184):
    #     return 1
    # elif (horsePower >= 184 and horsePower < 245):
    #     return 2
    # elif (horsePower >= 245 and horsePower < 286):
    #     return 3
    # elif (horsePower >= 286):
    #     return 4


def createFuelColumn(X):
    if (X == 'Benzina'):
        return 1
    elif (X == 'Diesel'):
        return 2


def createTransmisionColumn(X):
    if (X == 'automata'):
        return 1
    elif (X == 'manuala'):
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

    print('Mean', theMean)
    print('R2 score:', metrics.r2_score(Y_test, y_pred))
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
    print('Root Mean Squared Error:', theRootMeanSquaredError)
    print('Percentage of error: ', (theRootMeanSquaredError * 100) / theMean, ' %')
    return regressor

dataFrameSeria3 = fitTheColumns(sfSeria3)
dataFrameSeria5 = fitTheColumns(sfSeria5)
dataFrameSeria7 = fitTheColumns(sfSeria7)

seria3Regressor = trainTheData(dataFrameSeria3)
seria5Regressor = trainTheData(dataFrameSeria5)
seria7Regressor = trainTheData(dataFrameSeria7)


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
                    (args['Year']),
                    args['HorsePower'],
                    createFuelColumn(args['Fuel']),
                    args['Mileage'],
                    createTransmisionColumn(args['Transmission'])
                ]]
            )
        elif(args['Model'].startswith('BMW 5')):
            result = seria5Regressor.predict(
                [[
                    createModelColumn(args['Model']),
                    (args['Year']),
                    args['HorsePower'],
                    createFuelColumn(args['Fuel']),
                    args['Mileage'],
                    createTransmisionColumn(args['Transmission'])
                ]]
            )
        elif(args['Model'].startswith('BMW 7')):
            result = seria7Regressor.predict(
                [[
                    createModelColumn(args['Model']),
                    (args['Year']),
                    args['HorsePower'],
                    createFuelColumn(args['Fuel']),
                    args['Mileage'],
                    createTransmisionColumn(args['Transmission'])
                ]]
            )

        print("-----------------------RESULT-------------------------")
        print(result)

        responseBody = {
            "Price": f"{result[0]}"
        }

        return responseBody, 200


api.add_resource(PricePrediction, "/price")

app.run(debug=True, port=5000)
