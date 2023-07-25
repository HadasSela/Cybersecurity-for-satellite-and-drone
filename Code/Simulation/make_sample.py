import pandas as pd

CSV_PATH="C:\\Users\\hadas\\Desktop\\project\\simulation\\CICIDS2017\\data25_CICIDS2017.csv"
NEW_CSV_PATH="C:\\Users\\hadas\\Desktop\\project\\simulation\\CICIDS2017\\NSL_KDD_sample.csv"
if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)
    x = df.sample(frac=0.1)
    x.to_csv(NEW_CSV_PATH, index=False)