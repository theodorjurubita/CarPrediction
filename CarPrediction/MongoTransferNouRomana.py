from pymongo import MongoClient;

mongoUrl = 'mongodb://localhost:27017/Cars'
client = MongoClient(mongoUrl)
cars = client.Cars
info = cars.Info

import pandas as pd

def createHorsePowerColumn(X):
    horsePowerString = X.split('(')
    horsePower = int(horsePowerString[1].replace(')', '').replace(' ', '').replace('CP', ''))

    return horsePower

sf = pd.read_csv('data/Seriile37Iunie.csv')

for i in sf.iterrows():
    car={
        "Model": i[1].Model,
        "Year": i[1].Year,
        "HorsePower": createHorsePowerColumn(i[1].HorsePower),
        "Fuel": i[1].Fuel,
        "Mileage": i[1].Mileage,
        "Transmission": i[1].Transmission,
        "Price": i[1].Price
    }
    #print(car)
    info.insert_one(car)