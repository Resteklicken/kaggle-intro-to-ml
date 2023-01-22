import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def main():
    inpath = "/home/rene/coding/intro-to-ml/data/iowa.csv"
    home_data = pd.read_csv(inpath)
    
    # Designate selling price as the prediction target of the model 
    y = home_data.SalePrice
    # Configure the "features" of the model
    feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
    # Extract the feature columns from the home data dataframe
    X = home_data[feature_names]

    # Specify the model
    # For model reproducibility, set a numeric value for random_state when specifying the model
    iowa_model = DecisionTreeRegressor(random_state=8)
    # Fit the model
    iowa_model.fit(X, y)

    # Predict the prediction target (sale price) with the model
    predictions = iowa_model.predict(X)
    print(predictions)

    # Check against actuals
    print("predictions:")
    print(iowa_model.predict(X.head()))
    print("actuals:")
    print(home_data.SalePrice.head())
    