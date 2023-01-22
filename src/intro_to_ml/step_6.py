import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

    # Define model with Decision Tree and optimal maximum leaf nodes of 100
    decision_tree_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=100)
    # Fit model
    decision_tree_model.fit(train_X, train_y)
    # Make predictions and validate the model using Mean Absolute Error
    decision_tree_predictions = decision_tree_model.predict(val_X)
    decision_tree_mae = mean_absolute_error(decision_tree_predictions, val_y)
    print("Decision Tree MAE: {:,.0f}".format(decision_tree_mae))
    
    # Fit another model using Random Forests
    # Supplying max_leaf_nodes=100 actually INCREASES mean absolute error in this model
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    forest_predictions = forest_model.predict(val_X)
    forest_mae = mean_absolute_error(val_y, forest_predictions)
    print("Random Forest MAE: {:,.0f}".format(forest_mae))

    """
    Decision Tree MAE: 27,283
    Random Forest MAE: 21,857

    --> Random forests yields accuracy improvements right out of the box
    """