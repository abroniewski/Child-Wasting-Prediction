
'''=============================================================
===================== SECTION IMPORTS ==========================
================================================================'''

# General imports
import sys
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None # do not show set copy warnings


'''------------SECTION USER VARIABLES--------------'''

# Defines the path to the ZHL dataset folder
your_datapath = 'data/ZHL/'

# new features data path
acled_datapath = 'data/acled/'

# results save path
result_savepath = 'results/'



'''=============================================================
===================== SECTION FUNCTIONS ==========================
================================================================'''

# Function that creates a pandas dataframe for a single district with columns for the baseline model with semi-yearly entries
def make_district_df_semiyearly(datapath, acled_datapath, district_name):

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
    prevalence_df = pd.read_csv(datapath + 'prevalence_v3.csv', parse_dates=['date'])
    covid_df = pd.read_csv(datapath + 'covid.csv', parse_dates=['date'])
    ipc_df = pd.read_csv(datapath + 'ipc2.csv', parse_dates=['date'])
    risk_df = pd.read_csv(datapath + 'FSNAU_riskfactors.csv', parse_dates=['date'])
    production_df = pd.read_csv(datapath + 'production.csv', parse_dates=['date'])

    # Select data for specific district
    prevalence_df = prevalence_df[prevalence_df['district'] == district_name]
    ipc_df = ipc_df[ipc_df['district'] == district_name]
    risk_df = risk_df[risk_df['district'] == district_name]
    production_df = production_df[production_df['district'] == district_name]    
    
    # risk factor df, as we proposed, use MS(month start) rather then M(month end)
    # the important thing here is we do not have the data before 2017-01-01, so ideally 2017-01-01 nvidi score should be null for all columns
    risk_df = risk_df.groupby(pd.Grouper(key='date', freq='6MS')).mean()
    risk_df = risk_df.shift(periods = 6, freq='M')

    # covid df, as we proposed, use MS(month start) rather then M(month end)
    covid_df = covid_df.groupby(pd.Grouper(key='date', freq='6MS')).sum()
    risk_df = risk_df.shift(periods = 6, freq='M')

    # crop production df
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

    # Calculate prevalence 6lag, we have already adjusted next_prevalence in new prevelence data
    df['prevalence_6lag'] = df['GAM Prevalence'].shift(1)
    # df['next_prevalence'] = df['GAM Prevalence'].shift(-1)

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
    
    # filling covid patients as 0 before covid started
    df.loc[np.isnan(df["covid"]), 'covid'] = 0

    return (df)



# Function to combine conflict features to the base dataframe
def combine_conflict_features(df,district_name):

    """
    Function to combine conflict features to the base dataframe

    Parameters
    ----------
    df : dataframe
        dataframe with baseline features
    district_name : string
        Name of the district

    Returns
    -------
    df : pandas dataframe
    """

    # read conflict datasets/features 6m, 1y and 1.5y
    conflict_6Mdf = pd.read_csv(acled_datapath + 'features_6m.csv', parse_dates=['date'])
    conflict_1ydf = pd.read_csv(acled_datapath + 'features_1y.csv', parse_dates=['date'])
    conflict_1y6mdf = pd.read_csv(acled_datapath + 'features_1y6m.csv', parse_dates=['date'])
    
    # extract district from 6m, 1y and 1.5y, rename it, and drop repeated information which is already present in base dataframe 

    # 6m conflict data
    conflict_df_district_6m = conflict_6Mdf[conflict_6Mdf['district'] == district_name].copy()
    conflict_df_district_6m.drop(columns=['next_prevalence','district','GAM Prevalence','district_encoded'],inplace=True)
    renamed_columns = ['date'] + [col+'_6m' for col in conflict_df_district_6m.columns[1:]]
    conflict_df_district_6m.columns = renamed_columns
    conflict_df_district_6m.sort_values('date', inplace=True)

    # 1y conflict data
    conflict_df_district_1y = conflict_1ydf[conflict_1ydf['district'] == district_name].copy()
    conflict_df_district_1y.drop(columns=['next_prevalence','district','GAM Prevalence','district_encoded'],inplace=True)
    renamed_columns = ['date'] + [col+'_1y' for col in conflict_df_district_1y.columns[1:]]
    conflict_df_district_1y.columns = renamed_columns
    conflict_df_district_1y.sort_values('date', inplace=True)

    #1y6m conflict data
    conflict_df_district_1y6mdf = conflict_1y6mdf[conflict_1y6mdf['district'] == district_name].copy()
    conflict_df_district_1y6mdf.drop(columns=['next_prevalence','district','GAM Prevalence','district_encoded'],inplace=True)
    renamed_columns = ['date'] + [col+'_1y6m' for col in conflict_df_district_1y6mdf.columns[1:]]
    conflict_df_district_1y6mdf.columns = renamed_columns
    conflict_df_district_1y6mdf.sort_values('date', inplace=True)

    # merge dataframe with the base dataframe
    df = pd.merge_asof(left=df, right=conflict_df_district_6m, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=conflict_df_district_1y, direction='backward', on='date')
    df = pd.merge_asof(left=df, right=conflict_df_district_1y6mdf, direction='backward', on='date')

    return df


# Function that combines the semiyearly dataset (from the function make_district_df_semiyearly) of all districts
def make_combined_df_semiyearly(datapath,acled_datapath,implementation):
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
        district_df = make_district_df_semiyearly(datapath, acled_datapath, district)
        if implementation=='model_3':
            district_df = combine_conflict_features(district_df,district)
        district_df['district'] = district
        df_list.append(district_df)

    df = pd.concat(df_list, ignore_index=True)
    df['district_encoded'] = df['district'].astype('category').cat.codes

    return df



# function to pre-process the raw dataframe 
def data_preprocessing(df):

    """
    Function to combine conflict features to the base dataframe

    Parameters
    ----------
    df : dataframe
        dataframe with added/baseline features (dependes on new vs old implementation)

    Returns
    -------
    X : pandas dataframe with input variables
    y : pandas dataframe with target variable
    district_count : total count of districts each with 7 observations
    """


    # - here we have 677 data points
    # - since the cropdiv has only 350 data points, so if we use directly use dropna (dropping values with NaN), the data points 
    # will be below 350 of course because we have NaN values for those crop diversity so we will drop the crop diversity and use the dropna for the rest
    # print("Total no of district before droping are - ",len(df['district'].value_counts().keys()))
    
    # Drop every row with missing values
    df.drop(columns=['cropdiv'],inplace=True) # drop cropdiv column as 50% of the data is missing and then apply dropna
    df.dropna(inplace=True)  # drop the rows with NaN values 

    # Sort dataframe on date and reset the index
    df.sort_values('date', inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    # checking district counts and printing district values with less then threshold_count (minimum number of observations required to feed per district)
    # threshold_count = 7
    # district_counts = df['district'].value_counts()
    # district_counts[district_counts.values < threshold_count]

    # Drop districts with less than 7 observations for conflict data, similar approach as baseline model 
    # these should be dropped from both model_2 and model_3 implementations to keep the data and comparision same
    # to get these districts uncomment the above code
    # - So after dropping we are left with 385 data points almost 1.5 times from the baseline model representing **55** districts
    # - So basically we have 55 districts each with 7 observations
    district_with_less_data = ['Baki', 'Bandarbeyla', 'Burco', 'Xudun', 'Ceel Barde', 'Jariiban', 'Cadaado', 'Cabudwaaq', 'Saakow', 'Laasqoray', 'Lughaye', 'Rab Dhuure', 'Baydhaba']
    df.drop(df[df['district'].isin(district_with_less_data)].index, inplace=True)
    district_count = len(df['district'].value_counts().keys())
    # print("Total no of district after droping are - ",len(df['district'].value_counts().keys()))

    # Define target and explanatory variables
    # dropping unnecessary columns
    X = df.drop(columns = ['increase', 'increase_numeric', 'date', 'district', 'prevalence', 'next_prevalence']) #Note that these columns are dropped, the remaining columns are used as explanatory variables
    y = df['next_prevalence'].values

    return X,y, district_count



# function for splitting the data in train and test for modelling
def train_test_split(X,y, district_count):

    """
    Function to split train and test data

    Parameters
    ----------
    X : dataframe
        dataframe with added/baseline features (dependes on new vs old implementation)
    y : dataframe
        dataframe with target variable
    district_count : integer
        total count of districts each with 7 observations

    Returns
    -------
    Splitted datasets 
    
    """

    # - so we are keeping 5 data points for each districts for training similar to baseline model, and the remaining 2 data points for each district for predictions
    # - Number of traing data rows are = Number of districts * 5 observations = 55*5= 275
    # - Number of testing data rows are = Number of districts * 2 observations = 55* 2 = 110
    # - Total observations = 275 + 110 = 385
    # divide data in train and test as discussed above
    total_observations_per_distrct = 7
    training_observations = 5
    test_observations = 2

    # data to train and test
    train_data_count = district_count*training_observations
    test_data_count = district_count*test_observations
    print(f"number of observations for training are - {train_data_count} and for testing are - {test_data_count} \n")

    # divide data
    Xtrain = X[:train_data_count]
    ytrain = y[:train_data_count]
    Xtest = X[train_data_count:]
    ytest = y[train_data_count:]

    return Xtrain, ytrain, Xtest, ytest



# function for training the model
def model_training(Xtrain, ytrain):
    """
    Function to train the model

    Parameters
    ----------
    Xtrain : dataframe
        dataframe with training data
    ytrain : dataframe
        dataframe with training target variable

    Returns
    -------
    trained randomforest model 
    
    """

    #Create a RandomForestRegressor with the random state 0.
    reg = RandomForestRegressor(random_state=0)

    #Fit to the training data
    reg.fit(Xtrain, ytrain)

    # predicting on training data
    predictions = reg.predict(Xtrain)

    #Calculate MAE for training data
    train_MAE = mean_absolute_error(ytrain, predictions)

    #Training data MAE and accuracy
    print(f"MAE(Mean Absolute Error) score for {implementation} model on training data is - {train_MAE}\n")

    # calculate feature importance and save plot
    feat_importances = pd.Series(reg.feature_importances_, index= Xtrain.columns)
    feat_importances_df = pd.DataFrame({'features':feat_importances.index, 'entropy_score':feat_importances.values}).sort_values(by=['entropy_score'],ascending=False)
    filename = result_savepath+implementation+'_featureimportance.csv' # filname according to paths
    feat_importances_df.to_csv(filename,index=False)     # save important features
    
    # # to save the feature importance plot uncomment below code
    # plt.figure(num=None, figsize=(10,8), dpi=80, facecolor='w', edgecolor='k')
    # feat_importances.sort_values().plot(kind='barh')
    # filename = result_savepath+implementation+'_featureimportance.png' # filname according to paths
    # plt.savefig(filename) # save plot

    return reg



# function to test the model
def model_evaluation(reg,Xtest,ytest):
    """
    Function to test the model

    Parameters
    ----------
    Xtest : dataframe
        dataframe with test data
    ytrain : dataframe
        dataframe with test target variable

    Returns
    -------
    predictions on test data using randomforest model 
    
    """

    # get predictions on data
    predictions = reg.predict(Xtest)
    
    #Calculate MAE
    test_MAE = mean_absolute_error(ytest, predictions)

    #Print model scores
    print(f"MAE(Mean Absolute Error) score for {implementation} model on test data is - {test_MAE} \n")

    return predictions




'''=============================================================
===================== SECTION MAIN FUNCTION ==========================
================================================================'''
# ## =================================================================================================
# ## MAIN FUNCTION
# select between "model_2" and "model_3" implementation to reproduce results, model_2 is baseline, model_3 is new model with added conflict features
def main(implementation):

    # Create the dataframe for all districts
    df = make_combined_df_semiyearly(your_datapath,acled_datapath,implementation)

    # data preprocessing 
    X,y,district_count = data_preprocessing(df)
    print(f"\nTotal no of district after preproecssing are - {district_count} \n")

    # prepare data for train and test
    Xtrain, ytrain, Xtest, ytest = train_test_split(X,y, district_count)
    train_data_count = len(Xtrain)


    '''------------------------------------------------ MODEL TRAINING AND EVALUATION -------------------------------------------------'''

    '''1. ------------------------- MODEL TRAINING------------------'''
    # ### ===========================================================================================
    # ### RANDOM FOREST MODEL

    # The chages made in the baseline model
        # 1. removing the loop and training model on the default parameters of random forest ( since randomeforest itself selects the best parameters(features) we are not performing feature selection. Although 
        # this approach can be extended by selecting best hyperparameters for random forest model using paramter tunning )
        # 2. Removing classification accuracy as a model ealuation method

    # training the randomforest model 
    reg = model_training(Xtrain, ytrain)


    '''2. ------------------------- MODEL EVALUATION ------------------'''
    # ### MODEL EVALUATION
        # ## Updated Evaluation method
        # ### Baseline Evaluation Method ( MAE and Classification Accuracy )
            # - Since it is a regression problem, classifing them as increased or decreased can give wrong estimation regarding the model. Here we are trying to predict the prevelence of wasted children, if the next prevelence increased by 0.0001 and the model preidcted that the prevelence is decreased by 0.0001 then even though the model is vary accurate and the difference between the original and prediction is 0.0002, 
            # it will be classified as wrong prediction, which is not correct measure at all.
            # - So we will measure our model performance using Mean Absoulte Error (MAE) only.

    # evaluating the random forest model
    predictions = model_evaluation(reg,Xtest,ytest)

    # #### SAVE predictions with TEST data    
    Xtest['district'] = df.iloc[train_data_count:]['district']
    Xtest['date'] = df.iloc[train_data_count:]['date']
    Xtest['next_prevalence'] = df.iloc[train_data_count:]['next_prevalence']
    Xtest['predictions'] = predictions
    filename = result_savepath +implementation+'_testresults.csv' # filename
    Xtest.to_csv(filename,index=False)



if __name__=='__main__':
    # extract passed paramter along, default is model_3
    try:
        implementation = sys.argv[1]
    except:
        implementation = ''

    if implementation!='model_2':
        implementation = 'model_3'

    # calling main function with passed implementation method
    main(implementation)

    # # ===============END===================