from create_dataset import create_dataset
from train_model import train_model
from prefect import flow

@flow
def data_and_train_flow():
    create_dataset(download=True, clean=True, y="Methyl tert-butyl ether (MTBE)")
    train_model(tracking_server_host="ec2-54-147-5-224.compute-1.amazonaws.com")
    return None