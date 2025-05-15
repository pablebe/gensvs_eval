import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
#import pyaml
import matplotlib
import scipy.stats
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import scipy.stats as stats
import scikit_posthocs as sp
from statannotations.Annotator import Annotator

matplotlib.rcParams['axes.labelsize'] = 14    # Axis labels (x and y)
matplotlib.rcParams['xtick.labelsize'] = 14    # X tick labels
matplotlib.rcParams['ytick.labelsize'] = 11    # Y tick labels
matplotlib.rcParams['legend.fontsize'] = 11  # Set legend font size

SAVE_FIGURES = True

RATINGS_PATH = './04_evaluation_data/dcr_test_ratings.csv'
METRICS_PATH = './01_evaluation_and_correlation/evaluation_metrics'
METRICS_FILE_ORDER = os.path.join(METRICS_PATH,'file_id_order.csv')
OUT_PATH = './04_evaluation_data'
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUT_PATH, 'figures'), exist_ok=True)

ALL_METRICS_OUTPATH = './04_evaluation_data'
P_VAL_THRESHOLD = 0.05
P_VAL_THRESHOLD_2 = 0.01
P_VAL_THRESHOLD_3 = 0.001
gen_models = ['melroformer_bigvgan', 'sgmsvs']
disc_models = ['melroformer_large', 'melroformer_small', 'htdemucs']

def add_gen_disc_column(row):
    if row['model_name'] == 'htdemucs' or row['model_name'] == 'melroformer_large' or row['model_name'] == 'melroformer_small':
        val = 'discriminative'
    else:
        val = 'generative'
    return val
def pvalue_to_stars(p):
    if p <= P_VAL_THRESHOLD_3:
        return '***'
    elif p <= P_VAL_THRESHOLD_2:
        return '**'
    elif p <= P_VAL_THRESHOLD:
        return '*'
    else:
        return 'ns'
    
def draw_significance_bar(ax, pair, y, p_val, h=0.035, color="black", fontsize=12):
    """
    Draws a significance bar with stars between x1 and x2 at height y.
    h = height of the vertical tick lines.
    """
    stars = pvalue_to_stars(p_val)
    if stars == 'ns':
        return
    else:
        ax.plot([pair[0], pair[0], pair[1], pair[1]], [y, y + h, y + h, y], lw=1.5, c=color)
        ax.text(pair[0]+(np.abs(pair[1] - pair[0])/2)*np.sign(pair[1] - pair[0]), y-h , stars, ha='center', va='bottom', color=color, fontsize=fontsize)


metric_files = glob.glob(os.path.join(METRICS_PATH, '*.csv'))
if len([file for file in metric_files if 'file_id_order.csv' in file])>0:
    #exclude file order from evaluation!
    file_order_csv = [file for file in metric_files if 'file_id_order.csv' in file][0]
    metric_files.remove(file_order_csv)
metric_names = [file.split(os.path.sep)[-1].split('.csv')[0] for file in metric_files]
ratings_df = pd.read_csv(RATINGS_PATH)

metrics_df = pd.DataFrame(index=ratings_df.index, columns=['trial_uid','file_id','model_name']+metric_names)
metric_ct = 0
for metric_file in metric_files:
    metric_name = metric_file.split(os.path.sep)[-1].split('.csv')[0]
    if 'Unnamed: 0' in pd.read_csv(metric_file).columns:     
        single_metric_df = pd.read_csv(metric_file).drop(columns=['Unnamed: 0'])
    else:
        single_metric_df = pd.read_csv(metric_file)
    file_id_order_df = pd.read_csv(METRICS_FILE_ORDER).drop(columns=['Unnamed: 0'])
    #add fileid as additional row to metric df if file_id column does not exist
    if 'file_id' not in single_metric_df.columns:
        single_metric_df = pd.concat([file_id_order_df, single_metric_df], axis=1)
    single_metric_df = single_metric_df.set_index('file_id')
    metric_ct += 1
    #replace melroform entries with melroformer ==> wrong id used when calculating metrics
    single_metric_df.columns = single_metric_df.columns.str.replace('melroform_', 'melroformer_')
    single_metric_df.columns = single_metric_df.columns.str.replace('sgmsvs_scratch', 'sgmsvs')
    for row in single_metric_df.iterrows():
        #get file_id from metric_df
        file_id_metric = row[0]
        #get file_id and model_name from metric_df
        model_names_metric = row[1].iloc[0:]
        #get ratings with same fileid
        rating_fileid = ratings_df[ratings_df['file_id'] == file_id_metric]
        for model_metric_row_idx in rating_fileid.index:
            metrics_df.loc[model_metric_row_idx, metric_name] = single_metric_df.loc[rating_fileid.loc[model_metric_row_idx]['file_id'],rating_fileid.loc[model_metric_row_idx]['model_name']]
            metrics_df.loc[model_metric_row_idx, 'file_id'] = rating_fileid.loc[model_metric_row_idx]['file_id']
            metrics_df.loc[model_metric_row_idx, 'model_name'] = rating_fileid.loc[model_metric_row_idx]['model_name']
            metrics_df.loc[model_metric_row_idx, 'trial_uid'] = rating_fileid.loc[model_metric_row_idx]['trial_uid']
            
#save metrics to dataframe
metric_df_2_save = metrics_df.copy()
metric_df_2_save.to_csv(os.path.join(OUT_PATH, 'all_metrics_df.csv'), index=False)

## Shapiro-Wilk to check for normality in test groups for generative, discriminative and individual models
model_order = ['htdemucs', 'melroformer_small', 'melroformer_large', 'sgmsvs', 'melroformer_bigvgan']
dmos_per_test_group = ratings_df[['model_name','DMOS', 'group_id']]#.set_index('group_id')
dmos_per_test_group['model_name'] = pd.Categorical(dmos_per_test_group['model_name'], categories=model_order, ordered=True)
dmos_per_test_group = dmos_per_test_group.sort_values('model_name')
dmos_per_test_group['model_group'] = dmos_per_test_group['model_name'].astype(str) + '_' + dmos_per_test_group.index.astype(str)
dmos_per_test_group['model_type'] = dmos_per_test_group.apply(add_gen_disc_column, axis=1)
dmos_per_test_group.set_index('group_id', inplace=True)

model_name = pd.Series(dtype='string', name='model_name')
model_type = pd.Series(dtype='string', name='model_type')
group_id = pd.Series(dtype='string', name='group_id')
statistic = pd.Series(dtype='float64', name='statistic')
p_value = pd.Series(dtype='float64', name='p_value')
normality = pd.Series(dtype='bool', name='normality')
shapiro_df_model = pd.DataFrame({
    'model_name': model_name,
    'model_type': model_type,
    'group_id': group_id,
    'statistic': statistic,
    'p_value': p_value,
    'normality': normality
})

shapiro_df_gen_disc = pd.DataFrame({
    'group_id': group_id,
    'model_type': model_type,
    'statistic': statistic,
    'p_value': p_value,
    'normality': normality
})

for ct, group_id in enumerate(dmos_per_test_group.index.unique()):
#    ct=0
    
    dmos_per_group_model_name_idx = dmos_per_test_group.loc[group_id]#
    dmos_per_group_model_name_idx.loc[:,'model_name'] = pd.Categorical(dmos_per_group_model_name_idx['model_name'], categories=model_order, ordered=True)
    dmos_per_group_model_name_idx = dmos_per_group_model_name_idx.sort_values('model_name')
    dmos_per_group_model_name_idx.set_index('model_name', inplace=True)
    dmos_data_group = pd.DataFrame()
    for model_name in np.unique(dmos_per_test_group['model_name'].to_numpy()):
        dmos_data_group = pd.concat([dmos_data_group, pd.DataFrame(dmos_per_group_model_name_idx.loc[model_name]['DMOS'].to_numpy(),columns=[model_name])], axis=1)
    
    color_ct = 0
    gen_model_stats = []
    disc_model_stats = []
    for model_name in np.unique(dmos_per_test_group['model_name'].to_numpy()):
        stat_data = dmos_data_group[model_name].to_numpy()[~np.isnan(dmos_data_group[model_name].to_numpy())]
        if model_name in gen_models:
            model_type = 'generative'
            gen_model_stats.append(stat_data)
        if model_name in disc_models:
            model_type = 'discriminative'
            disc_model_stats.append(stat_data)
        stat, p_val = stats.shapiro(stat_data)
        new_row = pd.DataFrame([[model_name, model_type, group_id, stat, p_val, p_val>P_VAL_THRESHOLD]], columns=shapiro_df_model.columns).astype(shapiro_df_model.dtypes.to_dict())
        shapiro_df_model = pd.concat([shapiro_df_model, new_row], ignore_index=True)
    stat_gen, p_val_gen = stats.shapiro(np.concatenate(gen_model_stats))
    stat_disc, p_val_disc = stats.shapiro(np.concatenate(disc_model_stats))
    shapiro_df_gen_disc = pd.concat([shapiro_df_gen_disc, pd.DataFrame([[group_id, 'generative', stat_gen, p_val_gen, p_val_gen>P_VAL_THRESHOLD]], columns=shapiro_df_gen_disc.columns)], ignore_index=True)
    shapiro_df_gen_disc = pd.concat([shapiro_df_gen_disc, pd.DataFrame([[group_id, 'discriminative', stat_disc, p_val_disc, p_val_disc>P_VAL_THRESHOLD]], columns=shapiro_df_gen_disc.columns)], ignore_index=True) 

print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------------------Shapiro-Wilk Test Results per Model------------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print(shapiro_df_model)
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('-------------------------------------------------------Shapiro-Wilk Test Results per Model Group---------------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print(shapiro_df_gen_disc)
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')


## Kruskal-Wallis test to check if generative, discriminative and individual models can be pooled over test groups
significant_difference = pd.Series(dtype='bool', name='data_pooling_allowed')
test_type = pd.Series(dtype='string', name='test_type')

pooling_test_gen_disc_df = pd.DataFrame({
    'model_type': model_type,
    'test_type': test_type,
    'statistic': statistic,
    'p_value': p_value,
    'data_pooling_allowed': significant_difference,
})

pooling_test_models = pd.DataFrame({
    'model_name': model_name,
    'test_type': test_type,
    'statistic': statistic,
    'p_value': p_value,
    'data_pooling_allowed': significant_difference,
})

for model_type_group in ['discriminative', 'generative']:
    #get data for model type
    model_type_data = dmos_per_test_group[dmos_per_test_group['model_type'] == model_type_group]
    #get data for each group
    model_type_data_group = {group_id: [] for group_id in model_type_data.index.unique().to_numpy()}
    for group_id in model_type_data.index.unique().to_numpy():
        model_type_data_group[group_id] = model_type_data.loc[group_id]['DMOS'].to_numpy()
    #perform one way anova
    f_stat_anova, p_val_anova = stats.f_oneway(*model_type_data_group.values())
    new_row = pd.DataFrame([[model_type_group,'ANOVA', f_stat_anova, p_val_anova, p_val>P_VAL_THRESHOLD]], columns=pooling_test_gen_disc_df.columns).astype(pooling_test_gen_disc_df.dtypes.to_dict())
    pooling_test_gen_disc_df = pd.concat([pooling_test_gen_disc_df, new_row], ignore_index=True)
    #perform kruskal-wallis test
    h_stat, p_val_kruskal = stats.kruskal(*model_type_data_group.values())
    new_row = pd.DataFrame([[model_type_group,'Kruskal-Wallis', h_stat, p_val_kruskal, p_val_kruskal>P_VAL_THRESHOLD]], columns=pooling_test_gen_disc_df.columns).astype(pooling_test_gen_disc_df.dtypes.to_dict())
    pooling_test_gen_disc_df = pd.concat([pooling_test_gen_disc_df, new_row], ignore_index=True)

for model_name in dmos_per_test_group['model_name'].unique():
    model_data = dmos_per_test_group[dmos_per_test_group['model_name'] == model_name]
    model_data_group = {group_id: [] for group_id in model_data.index.unique().to_numpy()}
    for group_id in model_data.index.unique().to_numpy():
        model_data_group[group_id] = model_data.loc[group_id]['DMOS'].to_numpy()
    #perform one way anova
    f_stat_anova, p_val_anova = stats.f_oneway(*model_data_group.values())
    new_row = pd.DataFrame([[model_name,'ANOVA', f_stat_anova, p_val_anova, p_val>P_VAL_THRESHOLD]], columns=pooling_test_models.columns).astype(pooling_test_models.dtypes.to_dict())
    pooling_test_models = pd.concat([pooling_test_models, new_row], ignore_index=True)
    #perform kruskal-wallis test
    h_stat, p_val_kruskal = stats.kruskal(*model_data_group.values())
    new_row = pd.DataFrame([[model_name,'Kruskal-Wallis', h_stat, p_val_kruskal, p_val_kruskal>P_VAL_THRESHOLD]], columns=pooling_test_models.columns).astype(pooling_test_models.dtypes.to_dict())
    pooling_test_models = pd.concat([pooling_test_models, new_row], ignore_index=True)
    
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('----------------------------------ANOVA/Kruskal-Wallis test for data pooling of generative and discriminative models-------------------------------------')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print(pooling_test_gen_disc_df)
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('----------------------------------------------ANOVA/Kruskal-Wallis test for data pooling of all models---------------------------------------------------')
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print(pooling_test_models)
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')


## Correlation analysis: Calculate Pearson and Spearman correlation between DMOS and metrics

discriminative_models = ['htdemucs', 'melroformer_small', 'melroformer_large']
generative_models = ['melroformer_bigvgan', 'sgmsvs']
dmos_df_discriminative = ratings_df[ratings_df['model_name'].isin(discriminative_models)]
metrics_df_discriminative = metrics_df[metrics_df['model_name'].isin(discriminative_models)]
dmos_df_generative = ratings_df[ratings_df['model_name'].isin(generative_models)]
metrics_df_generative = metrics_df[metrics_df['model_name'].isin(generative_models)]


corr_df_discriminative = pd.DataFrame(columns=['metric', 'pearson_correlation', 'p_val_pearson', 'linear_relation_by_pearson', 'spearman_correlation', 'p_val_spearman', 'monotonic_relation_by_spearman'])
corr_df_generative = pd.DataFrame(columns=['metric', 'pearson_correlation', 'p_val_pearson', 'linear_relation_by_pearson', 'spearman_correlation', 'p_val_spearman', 'monotonic_relation_by_spearman'])
for metric in metric_names:
    #calculate correlation for discriminative models
    corr_pearson_disc, p_val_pear_disc = scipy.stats.pearsonr(dmos_df_discriminative['DMOS'].to_numpy().astype('float64'), metrics_df_discriminative[metric].to_numpy().astype('float64'))
    corr_spear_disc, p_val_spear_disc = scipy.stats.spearmanr(dmos_df_discriminative['DMOS'].to_numpy().astype('float64'), metrics_df_discriminative[metric].to_numpy().astype('float64'))
    #calculate correlation for generative models
    corr_pearson_gen, p_val_pear_gen = scipy.stats.pearsonr(dmos_df_generative['DMOS'].to_numpy().astype('float64'), metrics_df_generative[metric].to_numpy().astype('float64'))
    corr_spear_gen, p_val_spear_gen = scipy.stats.spearmanr(dmos_df_generative['DMOS'].to_numpy().astype('float64'), metrics_df_generative[metric].to_numpy().astype('float64'))

    if 'multi_res' in metric or 'fad' in metric or 'mse' in metric or 'kad' in metric:
        corr_pearson_disc *=-1
        corr_pearson_gen *=-1
        corr_spear_disc *=-1
        corr_spear_gen *=-1
    new_row = pd.DataFrame([[metric, corr_pearson_disc, p_val_pear_disc, p_val_pear_disc<P_VAL_THRESHOLD, corr_spear_disc, p_val_spear_disc, p_val_spear_disc<P_VAL_THRESHOLD]], columns=corr_df_discriminative.columns).astype(corr_df_discriminative.dtypes.to_dict())
    corr_df_discriminative = pd.concat([corr_df_discriminative, new_row], ignore_index=True)
    new_row = pd.DataFrame([[metric, corr_pearson_gen, p_val_pear_gen, p_val_pear_gen<P_VAL_THRESHOLD, corr_spear_gen, p_val_spear_gen, p_val_spear_gen<P_VAL_THRESHOLD]], columns=corr_df_generative.columns).astype(corr_df_generative.dtypes.to_dict())
    corr_df_generative = pd.concat([corr_df_generative, new_row], ignore_index=True)
merged_srcc = pd.DataFrame({
    'metric': corr_df_discriminative['metric'],
    'corr_disc': corr_df_discriminative['spearman_correlation'],
    'p_val_disc': corr_df_discriminative['p_val_spearman'],
    'corr_gen': corr_df_generative['spearman_correlation'],
    'p_val_gen': corr_df_generative['p_val_spearman'],
})

merged_pcc = pd.DataFrame({
    'metric': corr_df_discriminative['metric'],
    'corr_disc': corr_df_discriminative['pearson_correlation'],
    'p_val_disc': corr_df_discriminative['p_val_pearson'],
    'corr_gen': corr_df_generative['pearson_correlation'],
    'p_val_gen': corr_df_generative['p_val_pearson'],
})


## Listening test evaluation: Dunn's test for post-hoc analysis
model_DMOS = ratings_df[['model_name','DMOS']]
model_DMOS.loc[:,'model_name'] = pd.Categorical(model_DMOS['model_name'], categories=model_order, ordered=True).astype(str)
model_DMOS_sorted = model_DMOS.sort_values('model_name')
#create dataframe with model_names as columns and mean ratings as values
model_DMOS_group = pd.DataFrame()
model_DMOS_model_name_idx = model_DMOS_sorted.set_index('model_name').copy()
#sort according to model_name such that it is the same as in the plot
model_DMOS_model_name_idx.sort_index(inplace=True)
for model_name in np.unique(model_DMOS_sorted['model_name'].to_numpy()):
    model_DMOS_group = pd.concat([model_DMOS_group, pd.DataFrame(model_DMOS_model_name_idx.loc[model_name].to_numpy(),columns=[model_name])], axis=1)
    
# kruskal vallis test for all models
kruskal_stat, kruskal_p_val = stats.kruskal(model_DMOS.set_index('model_name').loc['htdemucs']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_large']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_small']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_bigvgan']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['sgmsvs']['DMOS'].to_numpy())
print(f"Kruskal-Wallis H-test: H={kruskal_stat:.4f}, p={kruskal_p_val:.4f}, significant difference: {kruskal_p_val < P_VAL_THRESHOLD}")
#create dataframe of metrics with same order as ratings_df
if kruskal_p_val < P_VAL_THRESHOLD:
    # perform post-hoc test with Dunn's test and Bonferroni correction
    data = np.concatenate((model_DMOS.set_index('model_name').loc['htdemucs']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_large']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_small']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['melroformer_bigvgan']['DMOS'].to_numpy(), model_DMOS.set_index('model_name').loc['sgmsvs']['DMOS'].to_numpy()))
    groups = ['htdemucs']*len(model_DMOS.set_index('model_name').loc['htdemucs']['DMOS'].to_numpy()) + ['melroformer_large']*len(model_DMOS.set_index('model_name').loc['melroformer_large']['DMOS'].to_numpy()) + ['melroformer_small']*len(model_DMOS.set_index('model_name').loc['melroformer_small']['DMOS'].to_numpy()) + ['melroformer_bigvgan']*len(model_DMOS.set_index('model_name').loc['melroformer_bigvgan']['DMOS'].to_numpy()) + ['sgmsvs']*len(model_DMOS.set_index('model_name').loc['sgmsvs']['DMOS'].to_numpy())
    data_df = pd.DataFrame({
                            'dmos': data,
                            'group': groups
                        })

    dunn_results = sp.posthoc_dunn(data_df,val_col='dmos', group_col='group', p_adjust='bonferroni')
    sig_diff_models = dunn_results < P_VAL_THRESHOLD
    sig_diff_models_2 =dunn_results < P_VAL_THRESHOLD_2
    sig_diff_models_3 = dunn_results < P_VAL_THRESHOLD_3
 #   sig_diff_models_4 = dunn_results < P_VAL_THRESHOLD_4

    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('----------------------------------------------------------Post-hoc Dunn\'s test Results------------------------------------------------------------------')
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(dunn_results)
    print('Statistical significance of model pairs:')
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('p-val threshold: ' + str(P_VAL_THRESHOLD))
    print(sig_diff_models)
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('p-val threshold: ' + str(P_VAL_THRESHOLD_2))
    print(sig_diff_models_2)
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    print('p-val threshold: ' + str(P_VAL_THRESHOLD_3))
    print(sig_diff_models_3)
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
#    print('p-val threshold: ' + str(P_VAL_THRESHOLD_4))
#    print(sig_diff_models_4)
print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')    



## Result plots:

#violin plot of DMOS ratings
# Define significant pairs (p < 0.05)
significant_pairs = [(('htdemucs', 'melroformer_small'), dunn_results['htdemucs']['melroformer_small']),
                     (('htdemucs', 'melroformer_large'), dunn_results['htdemucs']['melroformer_large']),
                     (('htdemucs', 'melroformer_bigvgan'), dunn_results['htdemucs']['melroformer_bigvgan']),
                     (('htdemucs', 'sgmsvs'), dunn_results['htdemucs']['sgmsvs']),
                     (('melroformer_small', 'melroformer_large'), dunn_results['melroformer_small']['melroformer_large']),
                     (('melroformer_small', 'melroformer_bigvgan'), dunn_results['melroformer_small']['melroformer_bigvgan']),
                     (('melroformer_small', 'sgmsvs'), dunn_results['melroformer_small']['sgmsvs']),
                     (('melroformer_large', 'melroformer_bigvgan'), dunn_results['melroformer_large']['melroformer_bigvgan']),
                     (('melroformer_large', 'sgmsvs'), dunn_results['melroformer_large']['sgmsvs']),
                     (('melroformer_bigvgan', 'sgmsvs'), dunn_results['melroformer_bigvgan']['sgmsvs'])]

# Filter for pairs with significant p-values
significant_pairs_1 = [pair[0] for pair in significant_pairs if pair[1] < P_VAL_THRESHOLD]
pval_1 = [pair[1] for pair in significant_pairs if pair[1] < P_VAL_THRESHOLD]
print("Significant Pairs (p < "+str(P_VAL_THRESHOLD)+"): ", significant_pairs)
significant_pairs_2 = [pair[0] for pair in significant_pairs if pair[1] < P_VAL_THRESHOLD_2]
print("Significant Pairs (p <  "+str(P_VAL_THRESHOLD_2)+"): ", significant_pairs_2)
significant_pairs_3 = [pair[0] for pair in significant_pairs if pair[1] < P_VAL_THRESHOLD_3]
print("Significant Pairs (p <  "+str(P_VAL_THRESHOLD_3)+"): ", significant_pairs_3)


model_names_xticks = ['HTDemucs', 'Mel-RoFo. (S)', 'Mel-RoFo. (L)', 'SGMSVS', 'Mel-RoFo. (S) + BigVGAN']
alpha=0.85
custom_color_palette = sns.color_palette("pastel")[:len(model_DMOS['model_name'].unique())]#sns.color_palette("pastel")[:len(violin_data['model_name'].unique())]
custom_color_pallette = [color+(alpha,) for color in custom_color_palette]
box_kwargs = dict(color='black', notch=True, bootstrap=5000, showfliers=False, showmeans=True, widths=0.25, meanprops=dict(marker='x', markersize=8, markeredgecolor='black'), boxprops=dict(color='black'), medianprops=dict(color='black'))

plt.figure(figsize=(8,5.5))
fig=sns.violinplot(data=model_DMOS, x='model_name', y='DMOS', hue=model_DMOS['model_name'], order=model_order, inner=None, palette=[custom_color_palette[1],custom_color_palette[0],custom_color_palette[0],custom_color_palette[1],custom_color_palette[0]],saturation=0.7,cut=0,legend=False)
# Annotate with statannotations
offset = 0.15
tick_dict = {}
for tick in fig.get_xticklabels():
    tick_dict[tick._text] = tick._x
offset_ct = 1
for pair,p_val in zip(significant_pairs_1, pval_1):
    if pair == ('htdemucs', 'melroformer_small'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 4.15, p_val, fontsize=10)
    elif pair == ('melroformer_small', 'melroformer_large'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 4.95, p_val, fontsize=10)
    elif pair == ('melroformer_small', 'melroformer_bigvgan'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 5.14, p_val, fontsize=10)
    elif pair == ('htdemucs', 'melroformer_large'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 5.32, p_val, fontsize=10)
    elif pair == ('htdemucs', 'sgmsvs'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 5.51, p_val, fontsize=10)
    elif pair == ('htdemucs', 'melroformer_bigvgan'):
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 5.7, p_val, fontsize=10)
    else:
        pair_num = (tick_dict[pair[0]], tick_dict[pair[1]])
        draw_significance_bar(fig, pair_num, 5+offset*offset_ct, p_val, fontsize=10)
        offset_ct += 1
box_order = fig.get_xticklabels()
#convert list of Text to list of strings
box_order = [x._text for x in box_order]
box_fig=model_DMOS_group.boxplot(column=box_order, positions=fig.get_xticks(), **box_kwargs, ax=fig)
plt.xlabel('')
fig.set_xticklabels(model_names_xticks, rotation=30, ha='right')
plt.ylabel('DMOS')
plt.yticks(np.arange(1, 5.5, 0.5))
plt.tight_layout()
for i in range(len(box_fig.lines)):
    box_fig.lines[i].set_label(s=None)
box_fig.lines[6].set_label('median')
box_fig.lines[12].set_label('mean')
legend_handles = [
    box_fig.lines[6],
    box_fig.lines[12],
    Patch(facecolor=custom_color_palette[0], edgecolor=[0.5,0.5,0.5], label='discriminative models'),
    Patch(facecolor=custom_color_palette[1], edgecolor=[0.5,0.5,0.5], label='generative models'),
]
plt.legend(
    handles=legend_handles, loc='lower left', bbox_to_anchor=(0, -0.55),  # your existing handles and title
    fontsize=12,            # label font size
    title_fontsize=12,      # title font size
    frameon=True,           # make sure box is visible
    borderpad=0.3,          # padding inside the box
    labelspacing=0.2,       # vertical space between entries
    handlelength=1.2,       # length of the legend handles
    handletextpad=0.3,      # space between handle and label
    borderaxespad=0.1,      # space between box and axes
)

if SAVE_FIGURES:
    plt.savefig(os.path.join(os.path.join(OUT_PATH,'figures'),'violin_plot.pdf'), format='pdf', bbox_inches='tight')

#correlation scatter plot
metric_short_label_dict = {
                     'sar': 'SAR',
                     'sisdr':'SI-SDR',
                     'sdr': 'SDR', 
                     'sir_peass': 'SIR',
                     'sir': 'SIR',
                     'isr': 'ISR',
                     'ops': 'OPS',
                     'aps': 'APS',
                     'ips': 'IPS',
                     'tps': 'TPS',
                     'fad_individal_per_song_clap_audio_df': '$\mathregular{CL^*_{a}}$',
                     'fad_individal_per_song_clap_music_df': '$\mathregular{CL^*_{m}}$',
                     'fad_individal_per_song_mert_df': 'M-L12$^*$',
                     'fad_individal_per_song_music2latent_df': 'M2L$^*$',
                     'fad_song2song_clap_audio_df': '$\mathregular{C_{a}}$',
                     'fad_song2song_clap_music_df': '$\mathregular{C_{m}}$',
                     'fad_song2song_mert_df': 'M-L12',
                     'fad_song2song_music2latent_df': 'M2L',  
                     'fad_clap_audio_per_song_df': '$\mathregular{CL^*_{a}}$',
                     'fad_clap_music_per_song_df': '$\mathregular{CL^*_{m}}$',
                     'fad_mert_per_song_df': 'M-L12$^*$',
                     'fad_music2latent_per_song_df': 'M2L$^*$',
                     'prob_score_clap_audio_df': '$\mathregular{CL^*_{a}}$',
                     'prob_score_clap_music_df': '$\mathregular{CL^*_{m}}$',
                     'prob_score_mert_df': 'M-L12$^*$',
                     'prob_score_music2latent_df': 'M2L$^*$',
                     'kad_clap_audio_per_song_df': '$\mathregular{CL^*_{a}}$',
                     'kad_clap_music_per_song_df': '$\mathregular{CL^*_{m}}$',
                     'kad_mert_per_song_df': 'M-L12$^*$',
                     'kad_music2latent_per_song_df': 'M2L$^*$',
                     'emb_mse_clap_audio_df': '$\mathregular{CL^*_{a}}$',
                     'emb_mse_clap_music_df': '$\mathregular{CL^*_{m}}$',
                     'emb_mse_mert_df': 'M-L12$^*$',
                     'emb_mse_music2latent_df': 'M2L$^*$',
                     'visqol': 'ViSQOL',
                     'multi_res_loss': 'm.-res. loss',
                     'pam': 'PAM',
                     'meta_audiobox_aes_PQ': 'PQ',
                     'meta_audiobox_aes_CU': 'CU',
                     'meta_audiobox_aes_CE': 'CE',
                     'meta_audiobox_aes_PQ_ref_norm': 'PQ_{ref}',
                     'meta_audiobox_aes_CU_ref_norm': 'CU_{ref}',
                     'meta_audiobox_aes_CE_ref_norm': 'CE_{ref}',
                     'singmos': 'SINGMOS',
                     'singmos_ref_norm': 'SINGMOS_{ref}',
                     'xls_r_sqa_full': 'XLS-R-SQA (Full)',
                     'xls_r_sqa': 'XLS-R-SQA',
                     'xls_r_sqa_ref_norm': 'XLS-R-SQA_{ref}',             
}

bss_metrics = ['sisdr', 'sdr', 'sar', 'sir', 'sir_peass', 'isr']
peass_metrics = ['ops', 'aps', 'ips', 'tps']
fad_s2s_metrics = ['fad_song2song_clap_audio_df', 'fad_song2song_clap_music_df', 'fad_song2song_mert_df', 'fad_song2song_music2latent_df','fad_individal_per_song_clap_audio_df', 'fad_individal_per_song_clap_music_df', 'fad_individal_per_song_mert_df', 'fad_individal_per_song_music2latent_df']
fad_s2r_metrics = ['fad_clap_audio_per_song_df', 'fad_clap_music_per_song_df', 'fad_mert_per_song_df', 'fad_music2latent_per_song_df']
mse_metrics = ['emb_mse_clap_audio_df', 'emb_mse_clap_music_df', 'emb_mse_mert_df', 'emb_mse_music2latent_df']
singmos_metrics = ['singmos', 'singmos_ref_norm']
xls_metrics = ['xls_r_sqa_full', 'xls_r_sqa', 'xls_r_sqa_ref_norm']
pam_metrics = ['pam']
#ref_metrics = ['multi_res_loss', 'visqol', 'prob_score_clap_audio_df', 'prob_score_clap_music_df', 'prob_score_mert_df', 'prob_score_music2latent_df', 'kad_clap_audio_per_song_df', 'kad_clap_music_per_song_df', 'kad_mert_per_song_df', 'kad_music2latent_per_song_df']
ab_metrics = ['meta_audiobox_aes_PQ', 'meta_audiobox_aes_CU', 'meta_audiobox_aes_CE', 'meta_audiobox_aes_PQ_ref_norm', 'meta_audiobox_aes_CU_ref_norm', 'meta_audiobox_aes_CE_ref_norm']
marker_types = ['o','s','^','v','D','^','p','*']
ref_marker_types = ['o','s','^','v','D','*']
ref_less_marker_types = ['p','H','<','x']
ref_metrics = bss_metrics + peass_metrics + fad_s2s_metrics + fad_s2r_metrics + mse_metrics + ['visqol', 'multi_res_loss'] 
refless_metrics = singmos_metrics + xls_metrics + pam_metrics + ab_metrics
white_space = 0.001
fig, ax = plt.subplots(figsize=(16,4), ncols=1)
ax.grid()
#ax[1].grid()
metric_type_ct = 0
bss_ct = 0
peass_ct = 0
ab_ct = 0
singmos_ct = 0
pam_ct = 0
xls_ct = 0
fad_s2r_ct = 0
fad_s2s_ct = 0
mse_ct = 0
mres_ct = 0 
visqol_ct = 0
label_ct = 0
#white_space = 0.001
ref_markers = []
refless_markers = []
for metric_id, metric in enumerate(merged_srcc['metric']):
    
    if metric == 'ips':
        #skip ips
        continue
    if metric in bss_metrics:
        marker_type = ref_marker_types[0]
        color = sns.color_palette('pastel')[0]
        label = 'BSS-Eval'
        bss_ct += 1
        label_ct = bss_ct
    elif metric in peass_metrics:
        marker_type = ref_marker_types[1]
        color = sns.color_palette('pastel')[0]
        label = 'PEASS'
        peass_ct += 1
        label_ct = peass_ct
    elif metric in pam_metrics:
        marker_type = ref_less_marker_types[0]
        color = sns.color_palette('pastel')[1]
        label = 'PAM'
        pam_ct += 1
        label_ct = pam_ct
    elif metric in ab_metrics:
        marker_type = ref_less_marker_types[1]
        color = sns.color_palette('pastel')[1]
        label = 'Audiobox-AES'
        ab_ct += 1
        label_ct = ab_ct
    elif metric in singmos_metrics:
        marker_type = ref_less_marker_types[2]
        color = sns.color_palette('pastel')[1]
        label = 'SINGMOS'
        singmos_ct += 1
        label_ct = singmos_ct

    elif metric in xls_metrics:
        marker_type = ref_less_marker_types[3]
        color = sns.color_palette('pastel')[1]
        label = 'XLS-R-SQA'
        xls_ct += 1
        label_ct = xls_ct
    elif metric in fad_s2s_metrics:
        marker_type = ref_marker_types[2]
        color = sns.color_palette('pastel')[0]
        label = '$\mathregular{FAD_{song2song}}$'
        fad_s2s_ct += 1
        label_ct = fad_s2s_ct
    elif metric in mse_metrics:
        marker_type = ref_marker_types[3]
        color = sns.color_palette('pastel')[0]
        label = 'MSE'
        mse_ct += 1
        label_ct = mse_ct
    elif metric == 'visqol':
        marker_type = ref_marker_types[4]
        color =sns.color_palette('pastel')[0]
        label = 'VISQOL'
        visqol_ct += 1
        label_ct = visqol_ct
    elif metric == 'multi_res_loss':
        marker_type = ref_marker_types[5]
        color =sns.color_palette('pastel')[0]
        label = 'm.-res. loss'
        mres_ct += 1
        label_ct = mres_ct
    else:
        assert False, 'metric not found in any category!'

    sc=ax.scatter(merged_srcc[merged_srcc['metric']==metric]['corr_disc'].iloc[0], merged_srcc[merged_srcc['metric']==metric]['corr_gen'].iloc[0], color=color, edgecolor=[0.5,0.5,0.5], alpha=1, marker=marker_type, label=label if label_ct==1 else "",s=60)
    if metric == 'sisdr' or metric == 'sdr'  or 'emb_mse_clap' in metric or metric == 'emb_mse_music2latent_df' or 'audiobox' in metric or metric == 'fad_individal_per_song_mert_df':
        ax.text(merged_srcc[merged_srcc['metric']==metric]['corr_disc'].iloc[0]+white_space, merged_srcc[merged_srcc['metric']==metric]['corr_gen'].iloc[0], metric_short_label_dict[metric], fontsize=13, ha='left', va='top')  # Adjust alignment as needed
    elif  metric == 'fad_individal_per_song_clap_audio_df' :
        ax.text(merged_srcc[merged_srcc['metric']==metric]['corr_disc'].iloc[0]-white_space, merged_srcc[merged_srcc['metric']==metric]['corr_gen'].iloc[0], metric_short_label_dict[metric], fontsize=13, ha='right', va='top')  # Adjust alignment as needed
    elif metric == 'sar' or metric == 'isr'  or metric == 'fad_individal_per_song_clap_music_df':
        ax.text(merged_srcc[merged_srcc['metric']==metric]['corr_disc'].iloc[0]-white_space, merged_srcc[merged_srcc['metric']==metric]['corr_gen'].iloc[0], metric_short_label_dict[metric], fontsize=13, ha='right', va='bottom')  # Adjust alignment as needed
    else:
        ax.text(merged_srcc[merged_srcc['metric']==metric]['corr_disc'].iloc[0]+white_space/2, merged_srcc[merged_srcc['metric']==metric]['corr_gen'].iloc[0], metric_short_label_dict[metric], fontsize=13, ha='left', va='bottom')  # Adjust alignment as needed
    if metric in ref_metrics:
        if label_ct == 1:
            ref_markers.append(plt.Line2D([0], [0], linestyle='none', marker=marker_type, color=color, label=label, markeredgecolor=[0.5,0.5,0.5], markersize=np.sqrt(sc.get_sizes()[0])))
    if metric in refless_metrics:
        if label_ct == 1:
            if metric in xls_metrics:
                refless_markers.append(plt.Line2D([0], [0], linestyle='none', marker=marker_type, color=color, label=label, markeredgecolor=None, markersize=np.sqrt(sc.get_sizes()[0])))
            else: 
                refless_markers.append(plt.Line2D([0], [0], linestyle='none', marker=marker_type, color=color, label=label, markeredgecolor=[0.5,0.5,0.5], markersize=np.sqrt(sc.get_sizes()[0])))

rel_width = 0.2
rel_height = 0.01
first_legend = ax.legend(handles=ref_markers,loc='lower left', 
                         alignment='left', title="Intrusive Metrics\u00A0\u00A0\u00A0\u00A0\u00A0\u00A0\u2009\u2009", title_fontproperties={'weight': 'bold', 'size': 13},
                         bbox_to_anchor =(1, 0.4, rel_width, rel_height),fontsize=13, borderpad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)
ax.add_artist(first_legend)
second_legend = ax.legend(handles=refless_markers,loc='upper left', 
                          alignment='left', title="Non-intrusive Metrics", title_fontproperties={'weight': 'bold', 'size': 13},
                          bbox_to_anchor =(1, 0.4, rel_width, rel_height), fontsize=13, borderpad=0.3, labelspacing=0.2, handlelength=1.5, handletextpad=0.3)#                          borderaxespad=0, frameon=True)
ax.set_yticks(np.arange(-0.1,0.9,0.1))
ax.set_xlabel('$\mathregular{SRCC_{disc}}$')
ax.set_ylabel('$\mathregular{SRCC_{gen}}$')
ax.tick_params(axis='y',labelsize=14)
plt.tight_layout()
if SAVE_FIGURES:
        plt.savefig(os.path.join(os.path.join(OUT_PATH,'figures'),'gen_disc_srcc_tradeoff.pdf'), format='pdf', bbox_inches='tight')
        plt.savefig(os.path.join(os.path.join(OUT_PATH,'figures'),'gen_disc_srcc_tradeoff.png'), format='png',dpi=300, bbox_inches='tight')
        






