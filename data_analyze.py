import pandas as pd
from pandas import DataFrame
from scipy.stats import pearsonr, spearmanr
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")

chemical_composition = ['C', 'Si', 'Mn', 'Mg', 'Cu', 'Ni', 'Mo', 'S', 'P', 'Cr']
heat_treatment = ['aust_temp', 'aust_czas', 'ausf_temp', 'ausf_czas']
physical_properties = ['Rm', 'Rp02', 'A5', 'HB', 'KV', 'K']
additional_features = ['zakres_grubosci', 'sklad_produkcyjny', 'obrobka_produkcyjna']
filled_properties = ['Rm_filled', 'Rp02_filled', 'A5_filled', 'HB_filled', 'K_filled']

all_cols = chemical_composition + heat_treatment + physical_properties + additional_features + filled_properties
# df = pd.read_csv("data_with_indexes.csv")
df = pd.read_excel("../../magisterka/zebrane-dane.xlsx", sheet_name="Dane", header=1, usecols=all_cols)

print(df.describe())


def check_nans(row, analyzed, is_not_nan, is_nan):
    for column in analyzed:
        value = row[column]
        if column in is_nan and not pd.isnull(value):
            return False
        if column in is_not_nan and pd.isnull(value):
            return False
    return True


# target_columns_without_KV = ['Rm', 'Rp02', 'A5', 'HB', 'K']
# target_columns_without_K = ['Rm', 'Rp02', 'A5', 'HB', 'KV']
#
#
# def get_subsets(given_set):
#     n = len(given_set)
#     subsets = []
#     for i in range(1 << n):
#         subset = []
#         for j in range(n):
#             if (i & (1 << j)) > 0:
#                 subset.append(given_set[j])
#         if 0 < len(subset) < n:
#             subsets.append(subset)
#     return subsets
#
#
# def get_subsets_nan_not_nan(target_columns):
#     subsets = get_subsets(target_columns)
#     result = []
#     for subset in subsets:
#         result.append((subset, list(set(target_columns) - set(subset))))
#     return result
#
#
# subsets_without_KV = get_subsets_nan_not_nan(target_columns_without_KV)
# subsets_without_K = get_subsets_nan_not_nan(target_columns_without_K)
#
# for subset in subsets_without_KV:
#     not_nan, nan = subset[0], subset[1]
#     column_name = "K_without_{}".format("_".join(nan))
#     df[column_name] = df.apply(check_nans, axis=1, args=[target_columns_without_KV,
#                                                          not_nan, nan])
#     count = df[df[column_name]].shape[0]
#     if count > 0:
#         print(column_name + ": " + str(count))
#
# print('==========')
# for subset in subsets_without_K:
#     not_nan, nan = subset[0], subset[1]
#     column_name = "KV_without_{}".format("_".join(nan))
#     df[column_name] = df.apply(check_nans, axis=1, args=[target_columns_without_K,
#                                                          not_nan, nan])
#     count = df[df[column_name]].shape[0]
#     if count > 0:
#         print(column_name + ": " + str(count))
#
# print('==========')
# df.to_csv("with_conditions.csv")
# print("Records with all values without K value: " + str(df[~(df[target_columns_without_K].isnull().any(axis=1))].shape[0]))
# print("Records with all values without KV value: " + str(df[~(df[target_columns_without_KV].isnull().any(axis=1))].shape[0]))
# print('==========\n')

def calculate_correlations(df: DataFrame, left_cols, right_cols, function, details):
    result = {}
    for col1 in left_cols:
        result[col1] = {}
        for col2 in right_cols:
            data = df[[col1, col2]].dropna()
            data1 = data[col1].to_numpy()
            data2 = data[col2].to_numpy()
            # print("Calculating " + col1 + " " + col2)
            try:
                corr, _ = function(data1, data2)
            except:
                continue
            if abs(corr) >= 0.6:
                title = details + ": " + col1 + " to " + col2 + " corr: " + "{:.2f}".format(corr) + ", cnt: " + str(data.shape[0])
                print(title)
                data.plot.scatter(x=col1, y=col2, title=title)
                plt.savefig("./wykresy/"+details+"_"+col1 + "_to_" + col2 + ".png")
                # plt.show()
            result[col1][col2] = corr
    return result

def calculate_correlations2(df: DataFrame, cols, function, details):
    result = {}
    for i in range(len(cols)):
        col1 = cols[i]
        result[col1] = {}
        for j in range(i+1, len(cols)):
            col2 = cols[j]
            data = df[[col1, col2]].dropna()
            if data.empty:
                continue
            data1 = data[col1].to_numpy()
            data2 = data[col2].to_numpy()
            # print("Calculating " + col1 + " " + col2)
            try:
                corr, _ = function(data1, data2)
            except:
                continue

            if abs(corr) >= 0.6:
                title = details + ": " + col1 + " to " + col2 + " corr: " + "{:.2f}".format(corr) + ", cnt: " + str(data.shape[0])
                print(title)
                data.plot.scatter(x=col1, y=col2, title=title)
                plt.savefig("./wykresy/"+details+"_"+col1 + "_to_" + col2 + ".png")
                # plt.show()

            result[col1][col2] = corr
    return result


left_cols = ['C', 'Si', 'Mn', 'Mg', 'Cu', 'Ni', 'Mo', 'S', 'P', 'Cr', 'aust_temp', 'aust_czas',
             'ausf_temp', 'ausf_czas']
right_cols = ['Rm', 'Rp02', 'A5', 'HB', 'K']

right_cols2 = filled_properties

# print("Pearson correlation")
# calculate_correlations(df, left_cols, right_cols, pearsonr)
print("\n")
# print("Spearman correlation")
# calculate_correlations(df, left_cols, right_cols, spearmanr)
# print("\n")
zakresy_grubosci = [1, 2, 3]
sklad_produkcyjny = [0, 1]
obrobka_produkcyjna = [0, 1]

for zakres in zakresy_grubosci:
    for sklad in sklad_produkcyjny:
        for obrobka in obrobka_produkcyjna:
            tmp = df.query("zakres_grubosci=={} & sklad_produkcyjny=={} & obrobka_produkcyjna=={}".format(zakres, sklad, obrobka))
            calculate_correlations(tmp, left_cols, right_cols, pearsonr, "zakres={},sklad={},obrobka={}".format(zakres, sklad, obrobka))
            calculate_correlations2(tmp, right_cols, pearsonr, "zakres={},sklad={},obrobka={}".format(zakres, sklad, obrobka))


# print("Pearson correlation")
# calculate_correlations2(df, right_cols, pearsonr)
# print("\n")

# print("Spearman correlation")
# calculate_correlations2(df, right_cols, spearmanr)
# print("\n")

# df_without_2 = df_clean.loc[df['indeks_artykułu'] != 2]
# records_in_article = df_without_2.groupby('indeks_artykułu')
#
# records_in_article.size().plot.bar()
# print(np.average(records_in_article.size()))
# plt.show()

