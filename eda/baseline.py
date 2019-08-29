import pandas as pd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path = '../data/original/'

train_sales = pd.read_csv(path + 'train_sales_data.csv')
train_search = pd.read_csv(path + 'train_search_data.csv')
train_user = pd.read_csv(path + 'train_user_reply_data.csv')

evaluation_public = pd.read_csv(path + 'evaluation_public.csv')
submit_example = pd.read_csv(path + 'submit_example.csv')

# 2018年1月，提取方式历史月份销量比例，考虑时间衰减，月份越近占比越高，即考虑同比与环比
m1_12 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 12), 'salesVolume'].values
m1_11 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 11), 'salesVolume'].values
m1_10 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 10), 'salesVolume'].values
m1_09 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 9), 'salesVolume'].values
m1_08 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 8), 'salesVolume'].values
m1_07 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
        train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 7), 'salesVolume'].values

m1_12_volum = train_sales.loc[
                  (train_sales.regYear == 2017) & (train_sales.regMonth == 12), 'salesVolume'].values * m1_12
m1_11_volum = train_sales.loc[
                  (train_sales.regYear == 2017) & (train_sales.regMonth == 11), 'salesVolume'].values * m1_11
m1_10_volum = train_sales.loc[
                  (train_sales.regYear == 2017) & (train_sales.regMonth == 10), 'salesVolume'].values * m1_10
m1_09_volum = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 9), 'salesVolume'].values * m1_09
m1_08_volum = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 8), 'salesVolume'].values * m1_08
m1_07_volum = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 7), 'salesVolume'].values * m1_07

evaluation_public.loc[
    evaluation_public.regMonth == 1, 'forecastVolum'] = m1_12_volum / 2 + m1_11_volum / 4 + m1_10_volum / 8 + \
                                                        m1_09_volum / 16 + m1_08_volum / 32 + m1_07_volum / 32


# 2018年2、3、4月，提取方式历史月份销量比例，考虑时间衰减，月份越近占比越高
m16_1_2 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values
m16_1_3 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values
m16_1_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
m16_1_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values
m16_1_6 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 6), 'salesVolume'].values

m16_2_3 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values
m16_2_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
m16_2_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values
m16_2_6 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 6), 'salesVolume'].values
m16_2_7 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 7), 'salesVolume'].values

m16_3_4 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 4), 'salesVolume'].values
m16_3_5 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 5), 'salesVolume'].values
m16_3_6 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 6), 'salesVolume'].values
m16_3_7 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 7), 'salesVolume'].values
m16_3_8 = train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2016) & (train_sales.regMonth == 8), 'salesVolume'].values

m17_1_2 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values
m17_1_3 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values
m17_1_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
m17_1_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values
m17_1_6 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 1), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 6), 'salesVolume'].values

m17_2_3 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values
m17_2_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
m17_2_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values
m17_2_6 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 6), 'salesVolume'].values
m17_2_7 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 2), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 7), 'salesVolume'].values

m17_3_4 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 4), 'salesVolume'].values
m17_3_5 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 5), 'salesVolume'].values
m17_3_6 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 6), 'salesVolume'].values
m17_3_7 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 7), 'salesVolume'].values
m17_3_8 = train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 3), 'salesVolume'].values / \
          train_sales.loc[(train_sales.regYear == 2017) & (train_sales.regMonth == 8), 'salesVolume'].values

m16_1 = m16_1_2 / 2 + m16_1_3 / 4 + m16_1_4 / 8 + m16_1_5 / 16 + m16_1_6 / 16
m16_2 = m16_2_3 / 2 + m16_2_4 / 4 + m16_2_5 / 8 + m16_2_6 / 16 + m16_2_7 / 16
m16_3 = m16_3_4 / 2 + m16_3_5 / 4 + m16_3_6 / 8 + m16_3_7 / 16 + m16_3_8 / 16

m17_1 = m17_1_2 / 2 + m17_1_3 / 4 + m17_1_4 / 8 + m17_1_5 / 16 + m16_1_6 / 16
m17_2 = m17_2_3 / 2 + m17_2_4 / 4 + m17_2_5 / 8 + m17_2_6 / 16 + m16_2_7 / 16
m17_3 = m17_3_4 / 2 + m17_3_5 / 4 + m17_3_6 / 8 + m17_3_7 / 16 + m16_3_8 / 16

m1 = m16_1 * 0.4 + m17_1 * 0.6
m2 = m16_2 * 0.4 + m17_2 * 0.6
m3 = m16_3 * 0.4 + m17_3 * 0.6

evaluation_public.loc[evaluation_public.regMonth == 2, 'forecastVolum'] = evaluation_public.loc[(evaluation_public.regYear == 2018) & (evaluation_public.regMonth == 1), 'forecastVolum'].values / m1
evaluation_public.loc[evaluation_public.regMonth == 3, 'forecastVolum'] = evaluation_public.loc[(evaluation_public.regYear == 2018) & (evaluation_public.regMonth == 2), 'forecastVolum'].values / m2
evaluation_public.loc[evaluation_public.regMonth == 4, 'forecastVolum'] = evaluation_public.loc[(evaluation_public.regYear == 2018) & (evaluation_public.regMonth == 3), 'forecastVolum'].values / m3


evaluation_public[['id', 'forecastVolum']].round().astype(int).to_csv('../data/submit/rule_submit.csv', index=False)
