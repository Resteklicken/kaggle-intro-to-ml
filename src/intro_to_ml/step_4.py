import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def main():
    inpath = "/home/rene/coding/intro-to-ml/data/iowa.csv"
    home_data = pd.read_csv(inpath)
    
    # Designate selling price as the prediction target of the model 
    y = home_data.SalePrice
    # Configure the "features" of the model
    feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
    # Extract the feature columns from the home data dataframe
    X = home_data[feature_names]

    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # Define model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit model
    iowa_model.fit(train_X, train_y)

    # print the top few validation predictions
    print(iowa_model.predict(val_X.head()))
    # print the top few actual prices from validation data
    print(y.head())

    # get predicted prices on validation data
    val_predictions = iowa_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))
    