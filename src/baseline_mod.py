# %%
'''=============================================================
===================== SECTION IMPORTS ==========================
================================================================'''
from time import time

# General imports
import pandas as pd
import numpy as np
import sklearn.base
from numpy import logspace, linspace
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso, ElasticNetCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_validate, RandomizedSearchCV
from tqdm import tqdm
from tabulate import tabulate

# %%
'''=============================================================
==================== SECTION USER VARIABLES ====================
================================================================'''
# Defines the path to the dataset folder
your_datapath = 'data/ZHL/'
#district_name = "Afgooye" #Adan Yabaal, Afgooye, Afmadow

# %%
'''=============================================================
==================== VARIABLE DEFINITION =======================
================================================================'''
# Define search space for number of trees in random forest and depth of trees
num_trees_min = 64
num_trees_max = 128

depth_min = 2
depth_max = 7

# %%
'''=============================================================
==================== SECTION FUNCTIONS =========================
================================================================'''


# Function that creates a pandas dataframe for a single district with columns for the baseline model with semi-yearly entries
def make_district_df_semiyearly(datapath, district_name):
    """
    Function that creates a pandas dataframe for a single district with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder
    district_name : string
        Name of the district

    Returns
    -------
    df : pandas dataframe
    """

    # Read all relevant datasets
    prevalence_df = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    covid_df = pd.read_csv(datapath + 'covid.csv', parse_dates=['date'])
    ipc_df = pd.read_csv(datapath + 'ipc2.csv', parse_dates=['date'])
    risk_df = pd.read_csv(datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    production_df = pd.read_csv(datapath + 'production.csv', parse_dates=['date'])

    # Select data for specific district
    prevalence_df = prevalence_df[prevalence_df['district'] == district_name]
    ipc_df = ipc_df[ipc_df['district'] == district_name]
    risk_df = risk_df[risk_df['district'] == district_name]
    production_df = production_df[production_df['district'] == district_name]

    risk_df = risk_df.groupby(pd.Grouper(key='date', freq='6M')).mean()
    risk_df = risk_df.reset_index()
    risk_df['date'] = risk_df['date'].apply(lambda x: x.replace(day=1))

    covid_df = covid_df.groupby(pd.Grouper(key='date', freq='6M')).sum()
    covid_df = covid_df.reset_index()
    covid_df['date'] = covid_df['date'].apply(lambda x: x.replace(day=1))

    production_df['cropdiv'] = production_df.count(axis=1)

    # Sort dataframes on date
    prevalence_df.sort_values('date', inplace=True)
    covid_df.sort_values('date', inplace=True)
    ipc_df.sort_values('date', inplace=True)
    risk_df.sort_values('date', inplace=True)
    production_df.sort_values('date', inplace=True)

    # Merge dataframes, only joining on current or previous dates as to prevent data leakage
    df = pd.merge_asof(left=prevalence_df, right=ipc_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=production_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=risk_df, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=covid_df, direction='backward', on='date')

    # Calculate prevalence 6lag
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(1)
    df['next_prevalence'] = df['GAM Prevalence'].shift(-1)

    # Select needed columns
    df = df[['date', 'district', 'GAM Prevalence', 'next_prevalence', 'prevalence_6lag', 'new_cases', 'ndvi_score',
             'phase3plus_perc', 'cropdiv', 'total population']]
    df.columns = ['date', 'district', 'prevalence', 'next_prevalence', 'prevalence_6lag', 'covid', 'ndvi', 'ipc',
                  'cropdiv', 'population']

    # Add month column
    df['month'] = df['date'].dt.month

    # Add target variable: increase for next month prevalence (boolean)
    increase = [False if x[1] < x[0] else True for x in list(zip(df['prevalence'], df['prevalence'][1:]))]
    increase.append(False)
    df['increase'] = increase
    df.iloc[-1, df.columns.get_loc('increase')] = np.nan  # No info on next month

    # Add target variable: increase for next month prevalence (boolean)
    increase_numeric = [x[1] - x[0] for x in list(zip(df['prevalence'], df['prevalence'][1:]))]
    increase_numeric.append(0)
    df['increase_numeric'] = increase_numeric
    df.iloc[-1, df.columns.get_loc('increase_numeric')] = np.nan  # No info on next month

    df.loc[(df.date < pd.to_datetime('2020-03-01')), 'covid'] = 0

    return (df)

#%%
# Function that combines the semiyearly dataset (from the function make_district_df_semiyearly) of all districts
def make_combined_df_semiyearly(datapath):
    """
    Function that creates a pandas dataframe for all districts with columns for the baseline model with semiyearly entries

    Parameters
    ----------
    datapath : string
        Path to the datafolder

    Returns
    -------
    df : pandas dataframe
    """

    prevdf = pd.read_csv(datapath + 'prevalence_estimates.csv', parse_dates=['date'])
    districts = prevdf['district'].unique()

    df_list = []
    for district in districts:
        district_df = make_district_df_semiyearly(datapath, district)
        district_df['district'] = district
        df_list.append(district_df)

    df = pd.concat(df_list, ignore_index=True)
    df['district_encoded'] = df['district'].astype('category').cat.codes

    return df

#%%
# Function that returns every possible subset (except the empty set) of the input list l
def subsets(l):
    subset_list = []
    for i in range(len(l) + 1):
        for j in range(i):
            subset_list.append(l[j: i])
    return subset_list


#%%
'''=============================================================
==================== SECTION DATAFRAME CREATION ================
================================================================'''
# Create the dataframe for all districts
df = make_combined_df_semiyearly(your_datapath)

# Drop every row with missing values
df.dropna(inplace=True)

# Sort dataframe on date and reset the index
df.sort_values('date', inplace=True)
df.reset_index(inplace=True, drop=True)

# Drop districts with less than 7 observations: 'Burco', 'Saakow', 'Rab Dhuure', 'Baydhaba', 'Afmadow'
df.drop(df[df['district'].isin(['Burco', 'Saakow', 'Rab Dhuure', 'Baydhaba', 'Afmadow'])].index, inplace=True)


#%%
'''=============================================================
============ SECTION RANDOM FOREST CROSS VALIDATION ============
================================================================'''

# WARNING: this process can take some time, since there are a lot of hyperparameters to investigate. The search space can be manually reduced to speed up the process.

# Create empty list to store model scores
parameter_scores = []

# Define target and explanatory variables
X = df.drop(columns=['increase', 'increase_numeric', 'date', 'district', 'prevalence',
                     'next_prevalence'])  # Note that these columns are dropped, the remaining columns are used as explanatory variables
y = df['next_prevalence']
X_original = X.copy()   # create copy to use in original model
y_original = y.copy()


# ###############################################################
# ###############################################################
# Here we start to build out own model on top of the data changes that were already
# done. To see the original model, scroll down to the location after all calls are made.
# ###############################################################
# ###############################################################
################################
def output_original_datasets_to_csv(joined_dataset: pd.DataFrame, modified_test_set: np.ndarray):
    '''
    This function simply saves the full joined dataframe and the 'X' dataset into a CSV file located at data/processed
    '''
    joined_dataset.to_csv("data/processed/original_df_before_drop.csv", index=False)
    modified_test_set.to_csv("data/processed/original_X_for_model_run.csv", index=False)
################################
# ###############################################################
# Generating Data
# ###############################################################
def generate_train_test_data_with_transformations(X: pd.DataFrame, y: np.ndarray) -> tuple([np.ndarray]):
    '''
    This function generates the train/test split of the dataset. The data will have a scalar transformation completed
    based on the X dataset used for training to prevent data leakage. We stratify the data (i.e. ensure the same
    number of observations exist for each district) with stratify=X[['district_encoded']]. is
    represent. In this case, we will use 2 observations to test, and 5 to train.

    :param X: Data to train model and
    make predictions
    :param y: Resulting variable to be predicted and tested against

    :return: X_train: X_test:
    Y_train: y_test: X: y:
    '''
    # FIXME: encode the test_size denominator to take a count of the number of observations for each district.
    #  Is there a check somewhere that all districts have the same number of observations, or is it just known?
    # NOTE: This approach completely disregards temporal trends in the data, as it does not differentiate between what
    #  year a row of data is from.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(2 / 7), random_state=42, stratify=X[['district_encoded']])

    # We do a scaling transformation based on the X_train dataset. This ensures we are not introducing data leakage.

    # This first set of X,y, test, train datasets is used in the non-cross-validated models.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # This set of X, y datasets is used for the cross-validated model building.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = y.values.ravel()

    return X_train, X_test, y_train, y_test, X, y


    ################################################################
    # Saving Model Results
    ################################################################
def create_dataframe_for_results():
    '''
    Creates a dataframe that is used to append results from each predictive model each training run.

    :return: An empty dataframe that stores RMSE, R-square, and MAE results
    '''
    # NOTE: This is an example of scores we can use. See full list of scores by calling
    #   sklearn.metrics.get_scorer_names().
    results = pd.DataFrame(index=[],
                 columns=['RMSE', 'R-square', 'Mean Absolute Error (MAE)'])
    return results


def append_simple_test_results_to_output_table(model_name: str, y_pred: np.ndarray):
    '''
    Appends the testing results from simple models generated with build_and_test_simple_model() to the results dataframe from create_dataframe_for_results().
    :param model_name: The name of the model that will appear in results table.
    :param y_pred:
    '''
    results_df.loc[model_name,:] = [np.sqrt(mean_squared_error(y_test, y_pred)),
                                    abs(r2_score(y_test, y_pred)),
                                    mean_absolute_error(y_test, y_pred)]    # use GLOBAL y_test


def append_cross_validated_test_results_to_output_table(model_name: str, cross_validated_results: pd.DataFrame):
    '''
    Appends the testing results from cross-validated models generated with build_and_test_cross_validated_model() to the results dataframe from create_dataframe_for_results(). The fucntion modifies values before inserting into results. Cross-validation will return a result for each model produced, so we take a mean() of all scores. The scoring API in scikit always works to maximize the score (i.e. the same API is used for all model scoring). For the API to work, scores that need to be minimized are made negative, and remain reported this way. So we take an abs() of those values.
    :param model_name: The name of the model that will appear in results table.
    :param cross_validated_results: Dataframe holding results from all model predictions
    '''

    # NOTE: (low) I'm not sure exactly why a test_score and train_score exist... Need to figure it out.
    RMSE = abs(cross_validated_results['test_neg_root_mean_squared_error'].mean())
    R_square = abs(cross_validated_results['test_r2'].mean())
    MAE = abs(cross_validated_results['test_neg_mean_absolute_error'].mean())

    results_df.loc[model_name, :] = [RMSE, R_square, MAE]


# ###############################################################
# Building and Testing Models
# ###############################################################
def build_and_test_simple_model(model_title: str, model_object: sklearn.base.BaseEstimator):
    '''
    Generates models using default parameters and tests against the split dataset. This function uses the train/test split data, with no cross validation.
    :param model_title: The name of the model that will appear in results table.
    :param model_object: Scikit-learn model that will be tested
    '''

    model = model_object    # Create a model based on the model_object passed in
    model_fit = model.fit(X_train,y_train)  # Train the model with GLOBAL variables X_train, y_train
    y_pred = model_fit.predict(X_test)    # Predict the response for test dataset (GLOBAL X_test)
    append_simple_test_results_to_output_table(model_title, y_pred)  # Append results to output table


def build_and_test_cross_validated_model(model_title: str, model_object: sklearn.base.BaseEstimator, number_of_splits: int):
    '''
    Generates models with cross validation. Cross validation should be done with stratified data for each distinct district. This results in a very small dataset being used, as there are currently only 7 observations or each district.
    :param model_title: The name of the model that will appear in results table.
    :param model_object: Scikit-learn model that will be tested
    :param number_of_splits: Amount of cross-validations to complete.
    '''
    # FIXME: (high) This KFold needs to be implemented to fold for each distinct 'district_encoded', similar to the way the
    #  non-CV dataset is stratified.
    #  https://www.geeksforgeeks.org/stratified-k-fold-cross-validation/
    cv = KFold(n_splits=number_of_splits, random_state=42, shuffle=True)
    model = model_object    # Create a model based on the model_object passed in

    # TODO: (low) The current method of scaling is introducing data leakage across the splits in the CV. Look at using
    #  make_pipeline(preprocessing.StandardScaler(), model) to fix this.
    #  data leakage issue.

    cross_val_results = pd.DataFrame(cross_validate(model, X, y, cv=cv,
                                       scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                       return_train_score=True))
    append_cross_validated_test_results_to_output_table(model_title, cross_val_results)


def build_and_test_parameter_tuned_model(model_title: str, model_object: sklearn.base.BaseEstimator, number_of_splits: int, parameter_grid:dict):
    '''
    Builds models and complete parameter tuning using RandomizedSearchCV to identify the best parameters. RandomizedSearchCV has been shown in studies to provide as good performance as testing all combinations directly, but saves on computation time.
    :param model_title: The name of the model that will appear in results table.
    :param model_object: Scikit-learn model that will be tested
    :param number_of_splits: Amount of cross-validations to complete.
    :param parameter_grid: Parameters that will be searched though to find the optimum
    '''
    # FIXME: Update KFold for each distinct 'district_encoded'
    # TODO: Update scaling for cross-validated approach.

    cv = KFold(n_splits=number_of_splits, random_state=42, shuffle=True)
    model = model_object
    model_rand_cv = RandomizedSearchCV(model, parameter_grid, cv=cv)  # Create random search space object
    model_rand_cv.fit(X_train, y_train)
    cross_val_results = pd.DataFrame(cross_validate(model_rand_cv.best_estimator_, X, y, cv=cv,
                                       scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
                                       return_train_score=True))    # best_estimator_ returns scores of best model
    append_cross_validated_test_results_to_output_table(model_title, cross_val_results)





def generate_parameter_search_space():
    '''
    Create the dictionary of parameters that GridSearchCV will search through. Each dictionary is created individually.
    :param: None
    :return: Individual dictionaries for each set of parameters required for a model.
    '''
    # NOTE: Does it make sense to use a class or module instead of a function so that I can group all the parameters
    #  together and just call the one I actually need when passing it into the function?
    # NOTE: for any parameters that are continuous, we should use a continuous object. Use uniform(loc=0, scale=1),
    #   where loc=mean, scale=standard deviation.

    decision_tree_params = {"max_depth": [1, 2, 3, 4, 5, 6, 7, None],
                 "min_samples_leaf": range(1, 7),
                 "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]}
    linear_regression_params = {"alpha":logspace(-3,2,base=10), # numbers between 10**-3 and 10**2 (exponential)
                       "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]}
    linear_regression_elasticnet_params = {"alphas":logspace(-3,2,base=10),
                                           "l1_ratio":linspace(0,1,20), # 20 numbers between 0 and 1
                                           "selection": ["random"]}  # random coefficient instead of cycling everything
    neural_net_params = {'hidden_layer_sizes':(range(10,100,10), range(10,100,10)),
                         'learning_rate_init':linspace(0.0001, 0.1, 20),
                         'activation':['logistic', 'relu', 'tanh'],
                         'alpha':logspace(0.0001, 100, 10),
                         'learning_rate':['constant', 'adaptive']}

    return decision_tree_params, linear_regression_params, linear_regression_elasticnet_params, neural_net_params


def build_and_test_all_simple_models():
    '''
    Wrapper function for all simple models to be called together.
    '''
    build_and_test_simple_model('Decision Tree Regressor (Default)', DecisionTreeRegressor())
    build_and_test_simple_model('Linear Regression (Default)', LinearRegression())
    build_and_test_simple_model('Linear Regression: Ridge (Default)', Ridge())
    build_and_test_simple_model('Linear Regression: Lasso (Default)', Lasso())
    build_and_test_simple_model('ElasticNet (Default)', ElasticNet())
    build_and_test_simple_model('Random Forest (Default)', RandomForestRegressor(n_estimators=20, random_state=42, max_depth=4))
    build_and_test_simple_model('Neural Network (Default)', MLPRegressor(hidden_layer_sizes=(100,), alpha=0.0001, activation='relu', max_iter=200, solver='adam', random_state=42))


def build_and_test_all_cross_validated_models():
    '''
    Wrapper function for all cross-validated models to be called together.
    '''
    build_and_test_cross_validated_model('Decision Tree Regressor (CV)', DecisionTreeRegressor(), number_of_splits=2)
    build_and_test_cross_validated_model('Linear Regression (CV)', LinearRegression(), number_of_splits=2)
    build_and_test_cross_validated_model('Random Forest (CV)', RandomForestRegressor(n_estimators=20, random_state=42, max_depth=4), number_of_splits=2)
    build_and_test_cross_validated_model('Neural Network (CV)', MLPRegressor(hidden_layer_sizes=(100,), alpha=0.0001, activation='relu', max_iter=200, solver='adam', random_state=42), number_of_splits=2)


def build_and_test_all_parameter_tuned_models():
    '''
    Wrapper function for all parameter tuned models to be called together.
    '''
    # FIXME: Add output of parameters used in best tuned model
    build_and_test_parameter_tuned_model('Decision Tree Regressor (Tuned)', DecisionTreeRegressor(), number_of_splits=2, parameter_grid=dt_params)
    build_and_test_parameter_tuned_model('Linear Regression: Ridge (Tuned)', Ridge(), number_of_splits=2, parameter_grid=lr_ridge_params)
    # TODO: (low) fix elasticNet below. I think we can use ElasticNet instead of ElasticNetCV here.
    # build_and_test_parameter_tuned_model('Linear Regression: Elastic (Tuned)', ElasticNetCV(), number_of_splits=2, parameter_grid=lr_elasticnet_params)
    # TODO: How many neural nets are not converging (%)?
    #build_and_test_parameter_tuned_model('Neural Network (Tuned)', MLPRegressor(max_iter=500), number_of_splits=2, parameter_grid=neural_params)


def print_all_results():
    '''
    Prints all results to terminal. This produces 4 tables, formatted using tabulate library. The tables are each sorted in terms of the metric being stored.
    '''
    print(f"Results Ordered By Model Name")
    print(tabulate(results_df.sort_index(ascending=True), headers='keys', tablefmt='psql'))
    print(f"\n Results Ordered By RMSE")
    print(tabulate(results_df.sort_values(
        by='RMSE', ascending=True), headers='keys', tablefmt='psql'))
    print(f"\n Results Ordered By R-square")
    print(tabulate(results_df.sort_values(
        by='R-square', ascending=True), headers='keys', tablefmt='psql'))
    print(f"\n Results Ordered By Mean Absolute Error (MAE)")
    print(tabulate(results_df.sort_values(
        by='Mean Absolute Error (MAE)', ascending=True), headers='keys', tablefmt='psql'))
    results_df.to_csv('data/processed/new_results_model_building.csv')

    print(f"\nTuned Parameter Model Time Taken: {(tuned_time_end - tuned_time_start)}")
    print(f"Total Time Taken: {(end_time - start_time)}")
    print(f"Original model data inputs saved to /data/processed/original_")
    print(f"New model metrics saved to /data/processed/new_results_model_building.csv")

# ###############################################################
# Final Calls to Generate Models and Results
# ###############################################################
start_time = time()

output_original_datasets_to_csv(df, X_original) # for assessment of original model inputs

X_train, X_test, y_train, y_test, X, y = generate_train_test_data_with_transformations(X, y)
results_df = create_dataframe_for_results()
build_and_test_all_simple_models()
build_and_test_all_cross_validated_models()
dt_params, lr_ridge_params, lr_elasticnet_params, neural_params = generate_parameter_search_space()

tuned_time_start = time()
build_and_test_all_parameter_tuned_models()
tuned_time_end = time()


# FIXME: Visuals: Create visuals of predictions, datasets, and data flow that can be used in poster
# FIXME: Stratify: For CV model building, stratify on encoded_district
# TODO: (low) Sprint3? -> Data Leakage: Transforming training data for CV
# TODO: (low) Sprint3? -> Consider creating build_and_test_time_series_models() as an improvement over the current approach.

end_time = time()

print_all_results()





# ###############################################################
# ###############################################################
# The code below this point is the original modeling created outside our group.
# ###############################################################
# ###############################################################


################################################################
# How the original model building works
################################################################
# The first 2 nested for loops will try all the different sizes of random forest
# Within the innermost nest for loop (for features in subsets(X.columns)), the RF is
# trying every single combination of features that are available in the dataframe.

# The cross validation is done by calling X[:99]. In this case, the data is ordered by date, and they know already
# that there are 33 unique districts, so they use a multiple of 33 whenever doing cross-validation. So here,
# they use rows 1-99 to train (first 3 observations) and the next 33 rows (100-132, the 4th observation) to test.
# They do this again for a second CV using the first 4 observations to train, and the 5th to test.
# This approach removes the temporality of the data. It essentially creates 2 models: 1 based on the first 3
# observations (CV-1), and one on the first 4 observations (CV-2), and then combines those two models as a final.
# Since the data is sorted by date before being used, this has some naive temporality as it is using the oldest data
# for training and the most recent data for testing. However, there is no differentiation between the relevance of how
# far back data goes (e.g. it is not a time series!).
# ###############################################################
def build_model(trees_min=num_trees_min, trees_max=num_trees_max, depth_start_search=depth_min,
                depth_end_search=depth_max):  # Adam: wrapping the original model creation into a function
    for num_trees in tqdm(range(trees_min, trees_max), desc=" outer", position=0):

        for depth in tqdm(range(depth_start_search, depth_end_search), desc=" inner loop", position=1, leave=False):

            # Investigate every subset of explanatory variables
            for features in subsets(X_original.columns):
                # First CV split. The 99 refers to the first 3 observations for the 33 districts in the data.
                Xtrain = X_original[:99][features].copy().values
                ytrain = y_original[:99]
                Xtest = X_original[99:132][features].copy().values
                ytest = y_original[99:132]

                # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

                # Fit to the training data
                clf.fit(Xtrain, ytrain)

                # Make a prediction on the test data
                predictions = clf.predict(Xtest)

                # Calculate mean absolute error
                MAE1 = mean_absolute_error(ytest, predictions)

                # Second CV split. The 132 refers to the first 4 observations for the 33 districts in the data.
                Xtrain = X_original[:132][features].copy().values
                ytrain = y_original[:132]
                Xtest = X_original[132:165][features].copy().values
                ytest = y_original[132:165]

                # Create a RandomForestRegressor with the selected hyperparameters and random state 0.
                clf = RandomForestRegressor(n_estimators=num_trees, max_depth=depth, random_state=0)

                # Fit to the training data
                clf.fit(Xtrain, ytrain)

                # Make a prediction on the test data
                predictions = clf.predict(Xtest)

                # Calculate mean absolute error
                MAE2 = mean_absolute_error(ytest, predictions)

                # Calculate the mean MAE over the two folds
                mean_MAE = (MAE1 + MAE2) / 2

                # Store the mean MAE together with the used hyperparameters in list
                parameter_scores.append((mean_MAE, num_trees, depth, features))


# # Here we can play with the original model and just change parameters being used. If no parameters are provided then
# # the model will call the original ones set at beginning of code.
# build_model(trees_min=1, trees_max=2, depth_start_search=2, depth_end_search=3)

# # Sort the models based on score and retrieve the hyperparameters of the best model
# parameter_scores.sort(key=lambda x: x[0])
# best_model_score = parameter_scores[0][0]
# best_model_trees = parameter_scores[0][1]
# best_model_depth = parameter_scores[0][2]
# best_model_columns = list(parameter_scores[0][3])


# '''------------SECTION FINAL EVALUATION--------------'''
# X = df[best_model_columns].values
# y = df['next_prevalence'].values
#
# #If there is only one explanatory variable, the values need to be reshaped for the model
# if len(best_model_columns) == 1:
# 	X = X.reshape(-1, 1)
#
# #Peform evaluation on full data
# Xtrain = X[:165]
# ytrain = y[:165]
# Xtest = X[165:]
# ytest = y[165:]
#
# clf = RandomForestRegressor(n_estimators=best_model_trees, max_depth=best_model_depth, random_state=0)
# clf.fit(Xtrain, ytrain)
# predictions = clf.predict(Xtest)
#
# #Calculate MAE
# MAE = mean_absolute_error(ytest, predictions)
#
# #Generate boolean values for increase or decrease in prevalence. 0 if next prevalence is smaller than current prevalence, 1 otherwise.
# increase           = [0 if x<y else 1 for x in df.iloc[165:]['next_prevalence'] for y in df.iloc[165:]['prevalence']]
# predicted_increase = [0 if x<y else 1 for x in predictions                      for y in df.iloc[165:]['prevalence']]
#
# #Calculate accuracy of predicted boolean increase/decrease
# acc = accuracy_score(increase, predicted_increase)
#
# #Print model parameters
# print('no. of trees: ' + str(best_model_trees) + '\nmax_depth: ' + str(best_model_depth) + '\ncolumns: ' + str(best_model_columns))
#
# #Print model scores
# print(MAE, acc)

##### OUTPUT #######
# Total runtime is 13min 53s, with 64 outer loops running for ~13s each
# Final model has:
#   no. of trees: 74
#   max_depth: 6
#   columns: ['district_encoded']
#   MAE: 0.05629900026844118
#   Accuracy: 0.849862258953168
# %%
