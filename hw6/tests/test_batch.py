import pandas as pd
from datetime import datetime
import batch


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    categorical = ['PULocationID', 'DOLocationID']
    
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),     
    ]
    columns = ['PULocationID', 'DOLocationID',
               'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)
    actual_features = batch.prepare_data(df, categorical)

    data = [
        (-1, -1, dt(1, 2), dt(1, 10), 8),
        (1, -1, dt(1, 2), dt(1, 10), 8),
        (1, 2, dt(2, 2), dt(2, 3), 1)
    ]
    columns = ['PULocationID', 'DOLocationID', 
               'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    df = pd.DataFrame(data, columns=columns)
    df[categorical] = df[categorical].astype('str')
    df["duration"] = df["duration"].astype('float')
    expected_features = df

    pd.testing.assert_frame_equal(actual_features, expected_features)