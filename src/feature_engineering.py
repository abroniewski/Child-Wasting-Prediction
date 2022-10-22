import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.offsets import MonthEnd, DateOffset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score

# Input file paths (change these paths)
acled_in = "https://raw.githubusercontent.com/abroniewski/Child-Wasting-Prediction/main/data/raw/1900-01-01-2022-09-27-Eastern_Africa-Somalia.csv"
prevalence_in = "https://raw.githubusercontent.com/abroniewski/Child-Wasting-Prediction/main/data/ZHL/prevalence_estimates.csv"

def filter_data():
    """
    This function selects only the necessary columns and filters rows by date.
    """
    global acled_df, prevalence_df
    acled_df = pd.read_csv(acled_in, parse_dates=['event_date'])

    #Drop columns not needed
    acled_df = acled_df.drop(labels=['data_id','iso','event_id_cnty','event_id_no_cnty','time_precision', 'region', 'country', 'admin3','geo_precision','iso3'],axis=1)

    # Filter the data to comply with the dates in the prevalence dataset.
    # Considering the last 18 months prior to the first date in the prevalence data
    acled_df = acled_df.query('event_date > "2016-01-01" and event_date < "2021-07-01"')

    # Select only necessary columns
    prevalence_df = pd.read_csv(prevalence_in, parse_dates=['date'])[['date', 'district', 'GAM Prevalence']]


def create_target_variable():
    """"
    This function creates the target variable "next_prevalence", which represents the prevalence observed 6 months
    after the reference date.
    """
    # Fixing next prevalence issue in old model.
    # We only consider as "next prevalence" the ones observed exactly 6 months after the reference date (date column)
    # The original model considered the next prevalence as being the "next observed prevalence" even when the latter
    # would be measured more than 6 months after the reference date.
    prevalence_df.sort_values('date', inplace=True)
    prevalence_df['next_date'] = prevalence_df.groupby('district')['date'].shift(-1)
    prevalence_df['next_prevalence'] = prevalence_df.groupby('district')['GAM Prevalence'].shift(-1)
    prevalence_df['next_prevalence'] = np.where(
        prevalence_df['next_date'].dt.to_period('M') - prevalence_df['date'].dt.to_period('M') != MonthEnd(6),
        pd.NA,
        prevalence_df['next_prevalence']
    )

    # don't need this column anymore
    prevalence_df.drop(columns=['next_date'], inplace=True)


def feature_engineering():
    """
    This function creates a lot of features from the conflicts in the data and then summarizes this features in a
    time window of 6 months that can be used to create larger time windows.
    """
    global features_df
    # Drop columns we don't find useful as features
    features_df = acled_df.drop(
        columns=['year', 'latitude', 'longitude', 'source', 'notes', 'timestamp', 'inter1', 'inter2', 'location',
                 'admin1'])

    # Add the number of actors involved in the conflict
    features_df['count_actors'] = features_df[['actor1', 'assoc_actor_1', 'actor2', 'assoc_actor_2']].count(axis=1)

    # Add dummies variables per categorical variable
    features_df = pd.get_dummies(features_df, columns=['event_type', 'sub_event_type', 'actor1', 'actor2', 'assoc_actor_1',
                                                    'assoc_actor_2', 'interaction', 'source_scale', 'count_actors'])

    # Add the count of conflicts to be summarized later
    features_df['number_conflicts'] = 1

    # Summarizing features for each semester
    first_day = pd.to_datetime('2017-01-01')
    features_df = features_df.groupby(
        [pd.Grouper(key="event_date", freq='6MS', origin=first_day), 'admin2']).sum().reset_index()


def summarize_features_in_time_windows():
    """
    This function summarizes features in different time windows (6 months, 12 months and 18 months) by considering the
    events that happenned inside each window.
    """
    global features_6m_df, features_1y_df, features_1y6m_df

    # Using data from previous 6 months
    features_6m_df = features_df.copy()
    # Add codes for each district
    features_6m_df['district_encoded'] = features_6m_df['admin2'].astype('category').cat.codes
    # Create column to merge the dataframes and align the dates
    features_6m_df['date_to_join'] = features_6m_df['event_date'] + DateOffset(months=6)

    # Using data from previous 12 months (1 year)
    features_1y_df = features_df.groupby('admin2').rolling(window='365D', on='event_date').sum().reset_index().drop(columns=['level_1'])
    # Add codes for each distric
    features_1y_df['district_encoded'] = features_1y_df['admin2'].astype('category').cat.codes
    # Create column to merge the dataframes and align the dates
    features_1y_df['date_to_join'] = features_1y_df['event_date'] + DateOffset(months=6)

    # Using data from previous 18 months
    features_1y6m_df = features_df.groupby('admin2').rolling(window='547D', on='event_date').sum().reset_index().drop(columns=['level_1'])
    # Add codes for each distric
    features_1y6m_df['district_encoded'] = features_1y6m_df['admin2'].astype('category').cat.codes
    # Create column to merge the dataframes and align the dates
    features_1y6m_df['date_to_join'] = features_1y6m_df['event_date'] + DateOffset(months=6)


def create_input_to_rf():
    """
    This function formats the dataframes to serve as input to Random Forest by merging the conflict data with the
    prevalence data and dropping unnecessary columns.
    """
    global final_6m_df, final_1y_df, final_1y6m_df
    final_6m_df = prevalence_df.merge(features_6m_df, left_on=['district', 'date'], right_on=['admin2', 'date_to_join'])
    final_6m_df.drop(columns=['date_to_join', 'event_date', 'admin2'], inplace=True)
    # Remove the rows where the target variable is null
    final_6m_df = final_6m_df[final_6m_df['next_prevalence'].notnull()]

    final_1y_df = prevalence_df.merge(features_1y_df, left_on=['district', 'date'], right_on=['admin2', 'date_to_join'])
    final_1y_df.drop(columns=['date_to_join', 'event_date', 'admin2'], inplace=True)
    # Remove the rows where the target variable is null
    final_1y_df = final_1y_df[final_1y_df['next_prevalence'].notnull()]

    final_1y6m_df = prevalence_df.merge(features_1y6m_df, left_on=['district', 'date'], right_on=['admin2', 'date_to_join'])
    final_1y6m_df.drop(columns=['date_to_join', 'event_date', 'admin2'], inplace=True)
    # Remove the rows where the target variable is null
    final_1y6m_df = final_1y6m_df[final_1y6m_df['next_prevalence'].notnull()]

    # The final datasets should have the same number of rows
    assert len(final_1y_df) == len(final_6m_df)
    assert len(final_1y6m_df) == len(final_6m_df)


def train_and_evaluate(final_df):
    """
    This function trains a random forest regressor to a given dataframe and prints the evaluation metrics for the
    model. The train-test split is merely to understand the performance of the model. It is not for the purposes of
    model selection.
    """
    # Train and test split. Selecting around 25% of data to test according to dates.
    X_train = final_df.query("date < '2020-07-01'")[final_df.columns.drop(['next_prevalence', 'date', 'district'])]
    y_train = final_df.query("date < '2020-07-01'")['next_prevalence']
    X_test = final_df.query("date >= '2020-07-01'")[final_df.columns.drop(['next_prevalence', 'date', 'district'])]
    y_test = final_df.query("date >= '2020-07-01'")['next_prevalence']

    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train,y_train)

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        mae = mean_absolute_error(test_labels, predictions)
        increase = np.where(test_labels > test_features['GAM Prevalence'], True, False)
        predicted_increase = np.where(predictions > test_features['GAM Prevalence'], True, False)
        acc = accuracy_score(increase, predicted_increase)
        print('MAE:', mae)
        print('Accuracy:', acc)

    evaluate(rf, X_test, y_test)

    return rf


def feature_selection():
    """
    This function selects the best 10 features for each model and saves these features in csv files.
    """
    # Training one model for each time window to select best features
    model_6m = train_and_evaluate(final_6m_df)
    model_1y = train_and_evaluate(final_1y_df)
    model_1y6m = train_and_evaluate(final_1y6m_df)

    # Feature importance according to each model
    importance_6m = list(zip(model_6m.feature_names_in_, model_6m.feature_importances_))
    importance_6m.sort(key=lambda x: x[1])
    importance_1y = list(zip(model_1y.feature_names_in_, model_1y.feature_importances_))
    importance_1y.sort(key=lambda x: x[1])
    importance_1y6m = list(zip(model_1y6m.feature_names_in_, model_1y6m.feature_importances_))
    importance_1y6m.sort(key=lambda x: x[1])

    # Saving results to csv
    final_6m_df[['date', 'next_prevalence', 'district'] + [x[0] for x in importance_6m[-10:]][::-1]].to_csv('./features_6m.csv', index=False)
    final_1y_df[['date', 'next_prevalence', 'district'] + [x[0] for x in importance_1y[-10:]][::-1]].to_csv('./features_1y.csv', index=False)
    final_1y6m_df[['date', 'next_prevalence', 'district'] + [x[0] for x in importance_1y6m[-10:]][::-1]].to_csv('./features_1y6m.csv', index=False)


if __name__ == "__main__":
    filter_data()
    create_target_variable()
    feature_engineering()
    summarize_features_in_time_windows()
    create_input_to_rf()
    feature_selection()

