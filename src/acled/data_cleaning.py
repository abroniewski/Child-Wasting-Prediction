import pandas as pd
from pathlib import Path

# Input file paths (change these paths)
acled_in = "https://raw.githubusercontent.com/abroniewski/Child-Wasting-Prediction/main/data/raw/1900-01-01-2022-09-27-Eastern_Africa-Somalia.csv"
prevalence_in = "https://raw.githubusercontent.com/abroniewski/Child-Wasting-Prediction/main/data/ZHL/prevalence_estimates.csv"

# Output file paths (change these paths)
acled_out = "acled.csv"
prevalence_out = "prevalence.csv"

def clean_data():
    """
    This function performs basic data cleaning in the data.
    """
    global acled_df, prevalence_df
    acled_df = pd.read_csv(acled_in, parse_dates=['event_date'])

    #Drop columns not needed
    acled_df = acled_df.drop(labels=['data_id','iso','event_id_cnty','event_id_no_cnty','time_precision', 'region', 'country', 'admin3','geo_precision','iso3'],axis=1)

    # Drop first column that is just row numbers
    prevalence_df = pd.read_csv(prevalence_in, parse_dates=['date']).iloc[:, 1:]


def write_data():
    """
    This function saves the clean dataframes to the disk
    """
    acled_filepath = Path(acled_out)
    acled_filepath.parent.mkdir(parents=True, exist_ok=True)
    acled_df.to_csv(acled_filepath, index=False)

    prevalence_filepath = Path(prevalence_out)
    prevalence_filepath.parent.mkdir(parents=True, exist_ok=True)
    prevalence_df.to_csv(prevalence_filepath, index=False)


if __name__ == "__main__":
    clean_data()
    write_data()

