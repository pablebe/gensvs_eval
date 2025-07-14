#Use this script to calculate the correlation between DMOS and the metrics in a csv.
#Append your metric to this *.csv file @ METRIC_DMOS_PATH to include it in correlation analysis.

import pandas as pd
import scipy.stats as stats

METRIC_DMOS_PATH = './03_evaluation_data/gensvs_eval_data.csv'
generative_models = ['sgmsvs', 'melroformer_bigvgan']
discriminative_models = ['melroformer_small', 'melroformer_large', 'htdemucs']

# load csv file
metric_dmos_df = pd.read_csv(METRIC_DMOS_PATH)
metrics_to_correlate = metric_dmos_df.columns[31:].to_list()  #all columns after the first 30 are metrics

# parse csv file into dmos and metrics
dmos_df = metric_dmos_df[['filepath','model_name','DMOS']].copy()
metrics = metric_dmos_df[['filepath','model_name']+metrics_to_correlate]

# separate into generative and discriminative models
dmos_gen = dmos_df[dmos_df['model_name'].isin(generative_models)].copy()
dmos_disc = dmos_df[dmos_df['model_name'].isin(discriminative_models)].copy()
metrics_gen = metrics[metrics['model_name'].isin(generative_models)].copy()
metrics_disc = metrics[metrics['model_name'].isin(discriminative_models)].copy()
                                
#calculate correlation for discriminative models
corr_df = pd.DataFrame(columns=['metric', 'spearman_correlation (discriminative)', 'spearman_correlation (generative)', 'p_val_spearman (discriminative)', 'p_val_spearman (generative)'])
for metric in metrics_to_correlate:
    #check for correct filepath between dmos and metrics
    if not dmos_disc['filepath'].equals(metrics_disc['filepath']):
        raise ValueError("Filepaths in DMOS and metrics do not match for discriminative models.")
    
    # calculate correlation for discriminative models
    corr_spear_disc, p_val_spear_disc = stats.spearmanr(dmos_disc['DMOS'].to_numpy().astype('float64'), metrics_disc[metric].to_numpy().astype('float64'))
    # calculate correlation for generative models
    corr_spear_gen, p_val_spear_gen = stats.spearmanr(dmos_gen['DMOS'].to_numpy().astype('float64'), metrics_gen[metric].to_numpy().astype('float64'))
    # append to dataframe
    corr_df = corr_df._append({
        'metric': metric,
        'spearman_correlation (discriminative)': corr_spear_disc,
        'spearman_correlation (generative)': corr_spear_gen,
        'p_val_spearman (discriminative)': p_val_spear_disc,
        'p_val_spearman (generative)': p_val_spear_gen
    }, ignore_index=True)

print(corr_df)
