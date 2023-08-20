# pylint: disable=missing-module-docstring
import logging
import json
import os
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)


def download_data():
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
        # open(f'../data/{item[0]}', 'wb').write(response.content)
        file_path = f'../data/{item[0]}'
        with open(file_path, 'wb') as file:
            file.write(response.content)
    logging.info("Data download complete.")


def clean_data(y):
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
    df.replace("< R.L.", 0, inplace=True)
    numerical_list = ["DissolvedOxygen",
                      "SpecificConductance",
                      "Total Alkalinity",
                      "Total Dissolved Solids",
                      "Total Organic Carbon",
                      "Turbidity",
                      "WaterTemperature",
                      "pH",
                      y
                      ]
    df[numerical_list] = df[numerical_list].astype(float)
    df.to_parquet("../data/df.parquet")
    logging.info("Data cleaning complete.")
