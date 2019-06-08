################ Data preparation ################

import pandas as pd

# import seaborn as sns
# import matplotlib.pyplot as plt

sf = pd.read_csv('data/toateSeriileCurate.csv')


# sf = pd.read_csv('data/toateSeriileCurate.csv')
# sf = pd.read_csv('data/toateSeriileCurate.csv')
# sf = pd.read_csv('data/seria7Curata.csv')


def createModelColum(X):
    model = X.split()
    if (model[1].startswith('3')):
        return 1
    elif (model[1].startswith('5')):
        return 2
    elif (model[1].startswith('7')):
        return 3


def createYearColumn(X):
    return X
    # if (X <= 2007):
    #     return 1
    # elif (X > 2007 and X <= 2009):
    #     return 2
    # elif (X > 2009 and X <= 2011):
    #     return 3
    # elif (X > 2011 and X <= 2013):
    #     return 4
    # elif (X > 2013 and X <= 2015):
    #     return 5
    # elif (X > 2015 and X <= 2016):
    #     return 6
    # elif (X > 2016 and X <= 2018):
    #     return 7
    # elif (X > 2018):
    #     return 8

    # if (X <= 2009):
    #     return 1
    # elif (X > 2009 and X <= 2013):
    #     return 2
    # elif (X > 2013 and X <= 2016):
    #     return 3
    # elif (X > 2016):
    #     return 4

    # if (X <= 2009):
    #     return 1
    # elif (X > 2009 and X <= 2011):
    #     return 2
    # elif (X > 2011 and X <= 2013):
    #     return 3
    # elif (X > 2013 and X <= 2015):
    #     return 4
    # elif (X > 2015 and X <= 2017):
    #     return 5
    # elif (X > 2017):
    #     return 6


def createHorsePowerColumn(X):
    horsePowerString = X.split('(')
    horsePower = int(horsePowerString[1].replace(')', '').replace(' ', '').replace('CP', ''))
    # return horsePower

    # if (horsePower <= 163):
    #     return 1
    # elif (horsePower > 163 and horsePower <= 184):
    #     return 2
    # elif (horsePower > 184 and horsePower <= 197):
    #     return 3
    # elif (horsePower > 197 and horsePower <= 245):
    #     return 4
    # elif (horsePower > 245 and horsePower <= 258):
    #     return 5
    # elif (horsePower > 258 and horsePower <= 286):
    #     return 6
    # elif (horsePower > 286 and horsePower <= 326):
    #     return 7
    # elif (horsePower > 326):
    #     return 8

    if (horsePower < 184):
        return 1
    elif (horsePower >= 184 and horsePower < 245):
        return 2
    elif (horsePower >= 245 and horsePower < 286):
        return 3
    elif (horsePower >= 286):
        return 4

    # if (horsePower < 180):
    #     return 1
    # elif (horsePower >= 180 and horsePower < 200):
    #     return 2
    # elif (horsePower >= 200 and horsePower < 250):
    #     return 3
    # elif (horsePower >= 250 and horsePower < 300):
    #     return 4
    # elif (horsePower >= 300 and horsePower < 350):
    #     return 5
    # else:
    #     return 6


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
    milage = int(X.replace('.', '').replace('km', '').replace(' ', ''))
    return milage
    # if (milage <= 19000):
    #     return 1
    # elif (milage > 19000 and milage <= 48000):
    #     return 2
    # elif (milage > 48000 and milage <= 82000):
    #     return 3
    # elif (milage > 82000 and milage <= 110000):
    #     return 4
    # elif (milage > 110000 and milage <= 138000):
    #     return 5
    # elif (milage > 138000 and milage <= 170000):
    #     return 6
    # elif (milage > 170000 and milage <= 208000):
    #     return 7
    # elif (milage > 208000):
    #     return 8

    # if (milage <= 30000):
    #     return 1
    # elif (milage > 30000 and milage <= 100000):
    #     return 2
    # elif (milage > 100000 and milage <= 180000):
    #     return 3
    # elif (milage > 180000):
    #     return 4

    # if (milage <= 10000):
    #     return 1
    # elif (milage > 10000 and milage <= 30000):
    #     return 2
    # elif (milage > 30000 and milage <= 50000):
    #     return 3
    # elif (milage > 50000 and milage <= 70000):
    #     return 4
    # elif (milage > 70000 and milage <= 100000):
    #     return 5
    # elif (milage > 100000 and milage <= 125000):
    #     return 6
    # elif (milage > 125000 and milage <= 150000):
    #     return 7
    # elif (milage > 150000 and milage <= 200000):
    #     return 8
    # elif (milage > 200000):
    #     return 9


def createPriceColumn(Y):
    price = int(Y.split()[0].replace('.', ''))
    return price
    # if (price <= 8000):
    #     return 1
    # elif (price > 8000 and price <= 12000):
    #     return 2
    # elif (price > 12000 and price <= 16000):
    #     return 3
    # elif (price > 16000 and price <= 20000):
    #     return 4
    # elif (price > 20000 and price <= 26000):
    #     return 5
    # elif (price > 26000 and price <= 35000):
    #     return 6
    # elif (price > 35000 and price <= 56000):
    #     return 7
    # elif (price > 56000):
    #     return 8

    # if (price <= 12000):
    #     return 1
    # elif (price > 12000 and price <= 21000):
    #     return 2
    # elif (price > 21000 and price <= 36000):
    #     return 3
    # elif (price > 36000):
    #     return 4


def responsPriceInterval(X):
    if (X == 1):
        return '< 10000'
    elif (X == 2):
        return '10000-20000'
    elif (X == 3):
        return '20000-35000'
    elif (X == 4):
        return '>35000'


sf['ModelCategory'] = sf['Model'].apply(createModelColum)
sf['YearCategory'] = sf['Year'].apply(createYearColumn)
sf['HorsePowerCategory'] = sf['HorsePower'].apply(createHorsePowerColumn)
sf['FuelCategory'] = sf['Fuel'].apply(createFuelColumn)
sf['Km'] = sf['Mileage'].apply(createMilageColumn)
sf['TransmissionCategory'] = sf['Transmission'].apply(createTransmisionColumn)
sf['PriceCategory'] = sf['Price'].apply(createPriceColumn)

sf = sf.drop(['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage', 'Transmission', 'Price'], axis=1)

sf = sf.rename(index=str, columns={'ModelCategory': 'Model', "YearCategory": "Year",
                                   "HorsePowerCategory": "HorsePower", "FuelCategory": "Fuel", 'Km': 'Mileage',
                                   'TransmissionCategory': 'Transmission', 'PriceCategory': 'Price'})

# print(sf.head())

################# Training Classifier ################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
import numpy as np
import category_encoders as ce

import time

X = sf[['Model', 'Year', 'HorsePower', 'Fuel', 'Mileage']]
Y = sf['Price']

#X = ce.BinaryEncoder(cols=['HorsePower', 'Fuel']).fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=int(time.time()))
used_features = ['Model', 'Year', 'HorsePower', 'Fuel', 'Transmission', 'Mileage']

# gnb = GradientBoostingClassifier()

regressor = GradientBoostingRegressor()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)

print("------------------------Actual vs Predicted values--------------------------")
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred.astype(int)})
df1 = df.head(25)
print(df1)

print('R2 score:', metrics.r2_score(Y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

# gnb.fit(X_train, Y_train)
# y_pred = gnb.predict(X_test)

# print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
#     .format(
#     X_test.shape[0],
#     (Y_test != y_pred).sum(),
#     100 * (1 - (Y_test != y_pred).sum() / X_test.shape[0])
# ))

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
        parser.add_argument('model')
        parser.add_argument('year')
        parser.add_argument('horsePower')
        parser.add_argument('fuel')
        parser.add_argument('mileage')
        # parser.add_argument('transmission')

        args = parser.parse_args()
        print(args)

        # result_proba = gnb.predict_proba([[
        #     createModelColum(args['model']),
        #     createYearColumn(int(args['year'])),
        #     createHorsePowerColumn(args['horsePower']),
        #     createFuelColumn(args['fuel']),
        #     createMilageColumn(args['mileage']),
        #     createTransmisionColumn(args['transmission'])
        # ]]
        # )

        # print("------------------------Classes probabilities------------------------")
        # print(result_proba)

        result = regressor.predict(
            [[
                createModelColum(args['model']),
                createYearColumn(int(args['year'])),
                createHorsePowerColumn(args['horsePower']),
                createFuelColumn(args['fuel']),
                createMilageColumn(args['mileage']),
                # createTransmisionColumn(args['transmission'])
            ]]
        )

        print("-----------------------RESULT-------------------------")
        print(result)

        responseBody = {
            "score": f"{result}"
        }

        return responseBody, 200


api.add_resource(PricePrediction, "/price")

app.run(debug=True, port=5000)
