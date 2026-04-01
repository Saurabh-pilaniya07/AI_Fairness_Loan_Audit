import pandas as pd

def load_data():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
    df = pd.read_csv(url)
    return df