import logging
import pandas as pd

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
    df.sort_values(by=['sample_date'], inplace=True)
    df.to_parquet("../data/df.parquet")
    logging.info("Data processing complete")