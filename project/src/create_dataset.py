import json
import os
import logging
import requests
import pandas as pd
from prefect import flow

logging.basicConfig(level=logging.INFO)


class DatasetCreator():
    """Class to download and clean dataset."""

    def __init__(self, y):
        """Initialize the dataset creator."""
        self.y = y

    def download_data(self):
        """Download data from the source."""
        if not os.path.exists("../data/"):
            os.makedirs("../data/")
        with open("data_sources.json", "r", encoding="utf-8") as file:
            urls = json.load(file)
        url_field_results = urls["url_field_results"]
        url_lab_results = urls["url_lab_results"]
        for item in [("field_results.csv", url_field_results),
                        ("lab_results.csv", url_lab_results)]:
            try:
                response = requests.get(item[1], allow_redirects=True, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logging.error("The request timed out. Try increasing the timeout value.")
            except requests.exceptions.RequestException as error:
                logging.error("An error occurred: %s", error)
            open(f'../data/{item[0]}', 'wb').write(response.content)
        logging.info("Data download complete")

    def clean_data(self):
        """Load and process the raw results."""
        field_results = pd.read_csv("../data/field_results.csv", low_memory=False)
        field_results = field_results[field_results["station_type"] == "Surface Water"]
        lab_results = pd.read_csv("../data/lab_results.csv", low_memory=False)
        lab_results = lab_results[lab_results["station_type"] == "Surface Water"]
        field_results.rename(columns = {"fdr_result": "result",
                                        "uns_name": "units",
                                        "mth_name": "method_name",
                                        "fdr_reporting_limit": "reporting_limit"},
                                        inplace=True)
        field_results.drop(columns = ['fdr_footnote', 'anl_data_type',
                                      'fdr_text_result', 'fdr_date_result'], inplace=True)
        field_parameters_to_keep = ['DissolvedOxygen',
                                    'SpecificConductance',
                                    'Turbidity',
                                    'WaterTemperature',
                                    'pH'
                                    ]
        field_results = field_results[field_results["parameter"].isin(field_parameters_to_keep)]
        y = self.y
        results = pd.concat([field_results, lab_results], ignore_index=True)
        values_to_keep = ["station_id",
                          "sample_date",
                          "parameter",
                          "result"
                         ]
        results = results[values_to_keep]
        results['sample_date'] = pd.to_datetime(results['sample_date'], format = "mixed")
        # keep only the date part and remove the time information
        results['sample_date'] = results['sample_date'].dt.date
        df = results.pivot_table(index=['station_id', 'sample_date'],
                                 columns='parameter', values='result', aggfunc='first')
        df.reset_index(inplace=True)
        mask = df[y].notnull()
        df = df[mask]
        df = df.dropna(axis=1, how='all')
        values_to_keep = ['station_id',
                          'sample_date',
                          'DissolvedOxygen',
                          'SpecificConductance',
                          "Total Alkalinity",
                          "Total Dissolved Solids",
                          "Total Organic Carbon",
                          'Turbidity',
                          'WaterTemperature',
                          'pH',
                          y
                         ]
        df = df[values_to_keep]
        df.dropna(inplace=True)
        df.replace("< R.L.", 0, inplace=True)
        column_list = ["DissolvedOxygen",
                       "SpecificConductance",
                       "Total Alkalinity",
                       "Total Dissolved Solids",
                       "Total Organic Carbon",
                       "Turbidity",
                       "WaterTemperature",
                       "pH",
                       y
                      ]
        df[column_list] = df[column_list].astype(float)
        df.sort_values(by=['sample_date', "station_id"], inplace=True)
        df.to_parquet("../data/df.parquet")
        logging.info("Data processing complete")

    def train_val_test_split(self):
        """Split data along time axis into training, validation, and test."""
        df = pd.read_parquet("../data/df.parquet")
        train_ratio = 0.80
        val_ratio = 0.10

        num_samples = len(df)
        train_size = int(train_ratio * num_samples)
        val_size = int(val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        # perform the training-validation-test split
        train_df = df.head(train_size)
        val_df = df.iloc[train_size: train_size + val_size]
        test_df = df.tail(test_size)

        train_df.to_parquet("../data/train_df.parquet")
        val_df.to_parquet("../data/val_df.parquet")
        test_df.to_parquet("../data/test_df.parquet")

@flow
def create_dataset(download, clean, y):
    """Main function for dataset creation."""
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    dataset_creator = DatasetCreator(y)
    if download:
        dataset_creator.download_data()
    if clean:
        dataset_creator.clean_data()


if __name__ == "__main__":
    create_dataset(download = True, clean = True, y = "Methyl tert-butyl ether (MTBE)")
