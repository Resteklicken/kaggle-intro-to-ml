import pandas as pd

def main():

    inpath = "/home/rene/coding/intro-to-ml/data/iowa.csv"
    home_data = pd.read_csv(inpath)

    #print(home_data.describe())

    # What is the average lot size (rounded to nearest integer)?
    avg_lot_size = home_data.LotArea.mean().round(0)
    print(avg_lot_size)

    # As of today, how old is the newest home (current year - the date in which it was built)
    newest_home_age = pd.Timestamp.today().year - home_data.YearBuilt.max()
    print(newest_home_age)