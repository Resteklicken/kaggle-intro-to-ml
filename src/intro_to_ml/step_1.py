import pandas as pd

from . import __version__

def main():
        
    inpath = "/home/rene/coding/intro-to-ml/data/melb_data.csv"
    mel_data = pd.read_csv(inpath)

    print(mel_data.describe())
