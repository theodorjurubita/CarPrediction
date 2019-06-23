from pymongo import MongoClient;

mongoUrl = 'mongodb://localhost:27017/Cars'
client = MongoClient(mongoUrl)
cars = client.Cars
info = cars.Info

import pandas as pd

def createModelColumn(X):
    model = X.split()
    if (model[1].startswith('3')):
        return 1
    elif (model[1].startswith('5')):
        return 2
    elif (model[1].startswith('7')):
        return 3

def createHorsePowerColumn(X):
    horsePowerString = X.split('(')
    horsePower = int(horsePowerString[1].replace(')', '').replace(' ', '').replace('CP', ''))

    return horsePower

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

def createPriceColumn(Y):
    price = int(Y.split()[0].replace('.', ''))
    return price

sf = pd.read_csv('data/toateSeriileCurate.csv')

for i in sf.iterrows():
    car={
        "Model": i[1].Model,
        "Year": i[1].Year,
        "HorsePower": createHorsePowerColumn(i[1].HorsePower),
        "Fuel": i[1].Fuel,
        "Mileage": i[1].Mileage,
        "Transmission": i[1].Transmission,
        "Price": createPriceColumn(i[1].Price)
    }
    info.insert_one(car)
    #print(car)