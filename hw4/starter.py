import pickle
import pandas as pd
import numpy as np
import sys
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

def get_ride_id(x):
    year = x["tpep_pickup_datetime"].year
    month = x["tpep_pickup_datetime"].month
    index_val = str(x.name)
    ride_id = f'{year:04d}/{month:02d}_' + index_val
    return ride_id

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = df.apply(lambda x: get_ride_id(x), axis = 1)
    return df

def ride_duration_prediction(df, dv, model, categorical, output_file):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    df["predictions"] = y_pred
    print(f"the mean predicted duration is {np.mean(y_pred)}")
    df_result = df[["ride_id", "predictions"]]
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet',
                   categorical)
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    # output_file = f"yellow_tripdata_{year:04d}-{month:02d}_result.parquet"
    output_file = f's3://mlops-zoomcamp-2023/taxi_type=yellow/year={year:04d}/month={month:02d}/result.parquet'
    ride_duration_prediction(
        df,
        dv,
        model,
        categorical,
        output_file
    )
    return

if __name__ == '__main__':
    run()