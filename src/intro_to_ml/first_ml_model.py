import pandas as pd

def main():
    inpath = "/home/rene/coding/intro-to-ml/data/iowa.csv"
    home_data = pd.read_csv(inpath)

    y = home_data.Price