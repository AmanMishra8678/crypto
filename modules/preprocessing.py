def preprocess_data(df):
    df = df.dropna()
    df["returns"] = df["price"].pct_change()
    df["ma_7"] = df["price"].rolling(7).mean()
    df["ma_30"] = df["price"].rolling(30).mean()
    return df
