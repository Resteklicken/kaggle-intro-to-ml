import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Calculate MAE for different maximum leaf nodes 
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

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

    # Make predictions and validate the model using Mean Absolute Error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE: {:,.0f}".format(val_mae))
    
    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    mae_values = []
    for candidate in candidate_max_leaf_nodes:
        mae_values.append((candidate, get_mae(candidate, train_X, val_X, train_y, val_y)))
    
    # Get the min of all MAEs, return the corresponding number of leaf nodes
    best_tree_size = min(mae_values, key = lambda t: t[1])[0]
    print("Best tree size: {}".format(str(best_tree_size)))

    # Train and fit the final model with the whole dataset
    final_model = DecisionTreeRegressor(random_state=8, max_leaf_nodes=best_tree_size)
    final_model.fit(X, y)