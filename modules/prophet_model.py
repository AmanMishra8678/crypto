from prophet import Prophet
import pandas as pd

def prophet_forecast(df, days=30):
    data = df.reset_index()
    data.columns = ["ds", "y"]

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]]
