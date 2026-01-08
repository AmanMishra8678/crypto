import numpy as np

def calculate_volatility(df):
    volatility = np.std(df["returns"]) * np.sqrt(365)
    return volatility
