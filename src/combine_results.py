'''=============================================================
===================== SECTION IMPORTS ==========================
================================================================'''
import pandas as pd
pd.options.mode.chained_assignment = None # do not show set copy warnings


'''------------SECTION USER VARIABLES--------------'''

# Defines the path to result folders
result_savepath = 'results/'

# Defines the path of saved model results
model_1_result_path = 'results/model_1_testresults.csv'
model_2_result_path = 'results/model_2_testresults.csv'
model_3_result_path = 'results/model_3_testresults.csv'



'''------------SECTION MAIN FUNCTION--------------'''

def main():

    # load model_1 (Provieded baseline model) results and rename column
    model_1_results_df = pd.read_csv(model_1_result_path)
    model_1_results_df = model_1_results_df[['district','date','next_prevalence','predictions']]
    model_1_results_df.columns = ['district','date','next_prevalence','predictions_model_1']

    # load model_2 (our baseline model) results and rename column
    model_2_results_df = pd.read_csv(model_2_result_path)
    model_2_results_df = model_2_results_df[['district','date','next_prevalence','predictions']]
    model_2_results_df.columns = ['district','date','next_prevalence','predictions_model_2']

    # load model_3 (our new model which includes conflict features) results and rename column
    model_3_results_df = pd.read_csv(model_3_result_path)
    model_3_results_df = model_3_results_df[['district','date','next_prevalence','predictions']]
    model_3_results_df.columns = ['district','date','next_prevalence','predictions_model_3']

    # combine model 2 and 3 (with 55 districts) and save results
    combined_df_2_3 = pd.merge(model_2_results_df, model_3_results_df, on=['district','date','next_prevalence'], how='inner')
    filename = result_savepath + 'combined_model_2_3'+'_testresults.csv' # filename
    combined_df_2_3.to_csv(filename,index=False)

    # combine model 1(with 33 districts), 2 and 3 (with 55 districts) and save results
    combined_df_1_2_3 = pd.merge(combined_df_2_3, model_1_results_df, on=['district','date','next_prevalence'], how='left')
    combined_df_1_2_3.columns = ['district','date','next_prevalence','predictions_model_2','predictions_model_3','predictions_model_1']
    combined_df_1_2_3 = combined_df_1_2_3[['district','date','next_prevalence','predictions_model_1','predictions_model_2','predictions_model_3']]
    filename = result_savepath + 'combined_model_1_2_3'+'_testresults.csv'# filename
    combined_df_1_2_3.to_csv(filename,index=False)


if __name__=='__main__':
    main()