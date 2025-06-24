import pandas as pd
from sklearn.linear_model import LinearRegression

def load_and_prepare_dataset(path):
    df = pd.read_csv(path)
    df['velocity'] = df['velocity'].astype(float)
    df['v2'] = df['velocity'] ** 2
    return df

def compute_quartiles(df):
    q1 = df['velocity'].quantile(0.25)
    q2 = df['velocity'].quantile(0.50)
    q3 = df['velocity'].quantile(0.75)
    return q1, q2, q3

def linear_regression_a(df):
    model = LinearRegression().fit(df[['velocity']], df['a'])
    return model.coef_[0], model.intercept_

def linear_regression_b(df):
    model = LinearRegression().fit(df[['v2']], df['b'])
    return model.coef_[0], model.intercept_

def predict_a(v, a_k, a_b):
    return a_k * v + a_b

def predict_b(v, b_k, b_b):
    return b_k * v**2 + b_b