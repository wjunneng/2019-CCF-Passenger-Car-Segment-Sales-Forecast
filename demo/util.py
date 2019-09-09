from config import DefaultConfig
from pseudoLabeler import PseudoLabeler


def get_train_sales_data(**params):
    """
    返回train_sales_data
    :param params:
    :return:
    """
    import pandas as pd

    df = pd.read_csv(DefaultConfig.train_sales_data_path, encoding='utf-8')

    return df


def get_train_search_data(**params):
    """
    返回 train_search_data
    :param params:
    :return:
    """
    import pandas as pd

    df = pd.read_csv(DefaultConfig.train_search_data_path, encoding='utf-8')

    return df


def get_train_user_reply_data(**params):
    """
    返回 train_user_reply_data
    :param params:
    :return:
    """
    import pandas as pd

    df = pd.read_csv(DefaultConfig.train_user_reply_data_path, encoding='utf_8')

    return df


def get_evaluation_public(**params):
    """
    返回evaluation_public
    :param params:
    :return:
    """
    import pandas as pd

    df = pd.read_csv(DefaultConfig.evaluation_public_path, encoding='utf-8')

    return df


def semi_suprivised_learning(X_train, y_train, X_test, target_col, **params):
    """
    半监督学习
    :param X_train:
    :param y_train:
    :param X_test:
    :param params:
    :return:
    """
    model = None
    if DefaultConfig.select_model is 'xgb':
        from lightgbm import LGBMClassifier

        model = PseudoLabeler(
            model=LGBMClassifier(n_jobs=10),
            unlabled_data=X_test,
            features=X_test.columns,
            target=target_col,
            sample_rate=0.3
        )

    if DefaultConfig.select_model is 'lgb':
        from lightgbm import LGBMClassifier

        model = PseudoLabeler(
            model=LGBMClassifier(nthread=10),
            unlabled_data=X_test,
            features=X_test.columns,
            target=target_col,
            sample_rate=0.4
        )

    if DefaultConfig.select_model is 'cat':
        from catboost import CatBoostClassifier

        model = PseudoLabeler(
            model=CatBoostClassifier(thread_count=10),
            unlabled_data=X_test,
            features=X_test.columns,
            target=target_col,
            sample_rate=0.4
        )

    model.fit(X_train, y_train)
    X_test[target_col] = model.predict(X_test)

    return X_test


def feature_transform(df, **params):
    """
    特征分布转换
    :param df:
    :param params:
    :return:
    """
    from sklearn import preprocessing
    import pandas as pd

    pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)

    df[DefaultConfig.outlier_columns] = pd.DataFrame(columns=DefaultConfig.outlier_columns,
                                                     data=pt.fit_transform(df[DefaultConfig.outlier_columns]))

    return df


def reduce_mem_usage(df, verbose=True):
    """
    减少内存消耗
    :param df:
    :param verbose:
    :return:
    """
    import numpy as np

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


def deal_id(df, **params):
    """
    处理id
    :param df:
    :param params:
    :return:
    """
    df['id'] = df['id'].fillna(0).astype(int)

    return df


def deal_forecastVolum(df, **params):
    """
    处理forecastVolum
    :param df:
    :param params:
    :return:
    """
    del df['forecastVolum']

    return df


def deal_bodyType(df, train_sales_data, **params):
    """
    处理bodyType
    :param df:
    :param params:
    :return:
    """
    from sklearn.preprocessing import LabelEncoder

    df['bodyType'] = df['model'].map(train_sales_data.drop_duplicates('model').set_index('model')['bodyType'])

    # 编码
    # for column in ['bodyType', 'model']:
    #     df[column] = df[column].map(dict(zip(df[column].unique(), range(df[column].nunique()))))

    df['model'] = LabelEncoder().fit_transform(df['model'])
    df['bodyType'] = LabelEncoder().fit_transform(df['bodyType'])

    return df


def deal_adcode(df, **params):
    """
    处理adcode
    """
    df['adcode'] = df['adcode'].apply(lambda x: int(str(x)[:2]))

    return df


def add_feature(df, **params):
    """
    添加新的feature
    :param df:
    :param params:
    :return:
    """
    from sklearn import preprocessing
    import datetime

    # 一、
    df['month'] = (df['regYear'] - 2016) * 12 + df['regMonth']
    df['date'] = df['regYear'].astype(str) + '-' + df['regMonth'].astype(str)
    df['date'] = df['date'].apply(
        lambda x: datetime.datetime(year=int(x.split('-')[0]), month=int(x.split('-')[1]), day=1))

    # 二、效果不好
    df['dayofweek'] = df['date'].dt.dayofweek
    del df['date']

    # 三、效果不好 毒特征
    df['bodyType_model'] = df['bodyType'] * 100 + df['model']

    # 四、
    for column_i in ['model']:
        # 数值列
        for column_j in ['popularity', 'carCommentVolum', 'newsReplyVolum']:
            stats = df.groupby(column_i)[column_j].agg(['mean', 'max', 'min', 'std', 'sum'])
            stats.columns = ['mean_' + column_j, 'max_' + column_j, 'min_' + column_j, 'std_' + column_j,
                             'sum_' + column_j]
            df = df.merge(stats, left_on=column_i, right_index=True, how='left')

    df['adcode_model'] = df['adcode'] *100 + df['model']

    # ################################################################### yeo-johnson 变换
    # columns = ['mean_popularity', 'max_popularity', 'min_popularity', 'std_popularity', 'sum_popularity',
    #            'mean_carCommentVolum', 'max_carCommentVolum', 'min_carCommentVolum', 'std_carCommentVolum',
    #            'sum_carCommentVolum',
    #            'mean_newsReplyVolum', 'max_newsReplyVolum', 'min_newsReplyVolum', 'std_newsReplyVolum',
    #            'sum_newsReplyVolum']
    #
    # print('进行yeo-johnson变换的特征列：')
    # pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
    # df[columns] = pt.fit_transform(df[columns])
    #
    # print(list(df.columns))
    # shift_feat = []
    # for i in [11]:
    #     i = i + 1
    #     shift_feat.append('shift_model_adcode_month_salesVolume_{0}'.format(i))
    #     shift_feat.append('model_adcode_month_{0}'.format(i))
    #
    #     df['model_adcode_month_{0}'.format(i)] = df['model_adcode_month'] + i
    #     df_last = df[~df.salesVolume.isnull()].set_index('model_adcode_month_{0}'.format(i))
    #     df['shift_model_adcode_month_salesVolume_{0}'.format(i)] = df['model_adcode_month'].map(
    #         df_last['salesVolume'])

    # current columns: ['adcode', 'bodyType', 'id', 'model', 'province', 'regMonth', 'regYear',
    #        'salesVolume', 'popularity', 'carCommentVolum', 'newsReplyVolum',
    #        'month', 'model_adcode', 'model_adcode_month', 'model_adcode_month_12',
    #        'shift_model_adcode_month_salesVolume_12']

    # numerical_feature = ['regYear', 'regMonth', 'month', 'adcode',
    #                      'mean_popularity', 'max_popularity', 'min_popularity', 'std_popularity', 'sum_popularity',
    #                      'mean_carCommentVolum', 'max_carCommentVolum', 'min_carCommentVolum', 'std_carCommentVolum',
    #                      'sum_carCommentVolum',
    #                      'mean_newsReplyVolum', 'max_newsReplyVolum', 'min_newsReplyVolum', 'std_newsReplyVolum',
    #                      'sum_newsReplyVolum']
    numerical_feature = ['regYear', 'regMonth', 'month', 'adcode', 'adcode_model', 'model']
    category_feature = []

    features = numerical_feature + category_feature

    return df, numerical_feature, category_feature, features


def split_data(df, features, **params):
    """
    划分数据集
    :param df:
    :param params:
    :return:
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    train_idx = (df['month'] <= 24)
    test_idx = (df['month'] > 24)

    X_train = df[train_idx][features]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, df[train_idx]['salesVolume'], test_size=0.1,
                                                          random_state=23)
    y_train = np.log1p(y_train)
    y_valid = np.log1p(y_valid)

    X_test = df[test_idx]
    X_test_id = X_test['id']
    X_test = X_test[features]

    # train_idx = (df['month'] <= 20)
    # valid_idx = (df['month'].between(21, 24))
    # test_idx = (df['month'] > 24)
    #
    # X_train = df[train_idx][features]
    # y_train = np.log1p(df[train_idx]['salesVolume'])
    #
    # X_valid = df[valid_idx][features]
    # y_valid = np.log1p(df[valid_idx]['salesVolume'])
    #
    # X_test = df[test_idx]
    # X_test_id = X_test['id']
    # X_test = X_test[features]

    return X_train, y_train, X_valid, y_valid, X_test, X_test_id


def preprocess(save=True, **param):
    """
    合并
    :param param:
    :return:
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if DefaultConfig.X_train_cache_path and DefaultConfig.no_replace:
        X_train = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.X_train_cache_path, key='X_train', mode='r'))
        X_valid = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.X_valid_cache_path, key='X_valid', mode='r'))
        y_train = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.y_train_cache_path, key='y_train', mode='r'))
        y_valid = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.y_valid_cache_path, key='y_valid', mode='r'))
        X_test_id = reduce_mem_usage(
            pd.read_hdf(path_or_buf=DefaultConfig.X_test_id_cache_path, key='X_test_id', mode='r'))
        X_test = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.X_test_cache_path, key='X_test', mode='r'))

    else:
        # ['province', 'adcode', 'model', 'bodyType', 'regYear', 'regMonth', 'salesVolume'] (31680, 7)
        train_sales_data = get_train_sales_data()
        # ['province', 'adcode', 'model', 'regYear', 'regMonth', 'popularity'] (31680, 6)
        train_search_data = get_train_search_data()
        # ['model', 'regYear', 'regMonth', 'carCommentVolum', 'newsReplyVolum'] (1440, 5)
        train_user_reply_data = get_train_user_reply_data()
        # ['id', 'province', 'adcode', 'model', 'regYear', 'regMonth', 'forecastVolum'] (5280, 7)
        evaluation_public = get_evaluation_public()

        # 合并train_sales_data与evaluation_public
        data = pd.concat([train_sales_data, evaluation_public], axis=0, ignore_index=True)
        # 合并data与train_search_data
        data = data.merge(train_search_data, how='left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
        # 合并data与train_user_reply_data
        data = data.merge(train_user_reply_data, how='left', on=['model', 'regYear', 'regMonth'])

        # 处理id
        data = deal_id(data)
        # 处理forecastVolum
        data = deal_forecastVolum(data)
        # 处理bodyType和model
        data = deal_bodyType(data, train_sales_data)
        # 处理adcode
        data = deal_adcode(data)
        # 添加新的特征
        data, numerical_feature, category_feature, features = add_feature(data)
        # split
        X_train, y_train, X_valid, y_valid, X_test, X_test_id = split_data(data, features)

        if save:
            X_train.to_hdf(DefaultConfig.X_train_cache_path, key='X_train', mode='w')
            X_valid.to_hdf(DefaultConfig.X_valid_cache_path, key='X_valid', mode='w')
            pd.DataFrame(y_train, index=None).to_hdf(DefaultConfig.y_train_cache_path, key='y_train', mode='w')
            pd.DataFrame(y_valid, index=None).to_hdf(DefaultConfig.y_valid_cache_path, key='y_valid', mode='w')
            pd.DataFrame(X_test_id).to_hdf(DefaultConfig.X_test_id_cache_path, key='X_test_id', mode='w')
            X_test.to_hdf(DefaultConfig.X_test_cache_path, key='X_test', mode='w')

    X_train.reset_index(drop=True, inplace=True)
    X_valid.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_valid.reset_index(drop=True, inplace=True)
    X_test_id.reset_index(drop=True, inplace=True)

    return X_train, X_valid, y_train, y_valid, X_test_id, X_test[list(X_train.columns)]


def rmspe(y, yhat):
    """
    rmspe评价指标
    :param y:
    :param yhat:
    :return:
    """
    import numpy as np

    return np.sqrt(np.mean((yhat / y - 1) ** 2))


def rmspe_xg(yhat, y):
    """
    xgboost -> rmspe
    :param yhat:
    :param y:
    :return:
    """
    import numpy as np

    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat)


def rmspe_lgb(yhat, y):
    """
    lightgbm -> rmspe
    :param yhat:
    :param y:
    :return:
    """
    import numpy as np

    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y, yhat), False


def rmspe_gscv(y, yhat):
    """
    sklearn interface
    :param y:
    :param yhat:
    :return:
    """
    import numpy as np

    y = np.expm1(y)
    yhat = np.expm1(yhat)
    return rmspe(y, yhat)


def xgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test):
    """
    xgb 模型
    :param X_train:
    :param X_valid:
    :param y_train:
    :param y_valid:
    :param X_test_id:
    :return:
    """
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train.values)
    dvalid = xgb.DMatrix(X_valid, label=y_valid.values)

    # ########################################## Tuning Paramters ##########################################
    xgb_best_params = {}
    params = {'booster': 'gbtree',
              'objective': 'reg:squarederror',
              'max_depth': 6,
              'learning_rate': 1,
              'gamma': 0,
              'min_child_weight': 1,
              'subsample': 1,
              'colsample_bytree': 1,
              'reg_alpha': 0,
              'reg_lambda ': 1,
              'random_state': 23,
              'gpu_id': 0,
              'max_bin': 16,
              'tree_method': 'gpu_exact'
              }

    # ########################################### n_estimators  ############################################
    min_merror = np.inf
    for n_estimators in range(10, 1000, 10):
        params['n_estimators'] = n_estimators
        cv_results = xgb.cv(params, dtrain, nfold=3, num_boost_round=1000, early_stopping_rounds=30, feval=rmspe_xg,
                            seed=23)
        mean_error = min(cv_results['test-rmspe-mean'])

        if mean_error < min_merror:
            min_merror = mean_error
            xgb_best_params["n_estimators"] = n_estimators

    params["n_estimators"] = xgb_best_params["n_estimators"]

    # ########################################### max_depth & min_child_weight #############################
    min_merror = np.inf
    for max_depth in range(6, 11, 1):
        for min_child_weight in range(4, 10, 1):
            params['max_depth'] = max_depth
            params['min_child_weight'] = min_child_weight
            cv_results = xgb.cv(params, dtrain, nfold=3, num_boost_round=1000, early_stopping_rounds=50, feval=rmspe_xg,
                                seed=23)
            mean_error = np.argmin(cv_results['test-rmspe-mean'])

            if mean_error < min_merror:
                min_merror = mean_error
                xgb_best_params["max_depth"] = max_depth
                xgb_best_params["min_child_weight"] = min_child_weight

    params['max_depth'] = xgb_best_params['max_depth']
    params["min_child_weight"] = xgb_best_params["min_child_weight"]

    # ########################################### gamma #####################################################
    for gamma in [i / 10.0 for i in range(0, 1)]:
        params['gamma'] = gamma
        cv_results = xgb.cv(params, dtrain, nfold=3, early_stopping_rounds=50, feval=rmspe_xg, seed=23)
        mean_error = min(cv_results['test-rmspe-mean'])

        if mean_error < min_merror:
            min_merror = mean_error
            xgb_best_params["gamma"] = gamma

    params["gamma"] = xgb_best_params["gamma"]

    # ############################################# subsample & colsample_bytree ############################
    min_merror = np.inf
    for subsample in [i / 10.0 for i in range(6, 10)]:
        for colsample_bytree in [i / 10.0 for i in range(6, 10)]:
            params['subsample'] = subsample
            params['colsample_bytree'] = colsample_bytree
            cv_results = xgb.cv(params, dtrain, nfold=3, early_stopping_rounds=50, feval=rmspe_xg, seed=23)
            mean_error = min(cv_results['test-rmspe-mean'])

            if mean_error < min_merror:
                min_merror = mean_error
                xgb_best_params["subsample"] = subsample
                xgb_best_params["colsample_bytree"] = colsample_bytree

    params["subsample"] = xgb_best_params["subsample"]
    params["colsample_bytree"] = xgb_best_params["colsample_bytree"]

    # ############################################# reg_alpha ################################################
    min_merror = np.inf
    for reg_alpha in [0.8, 0.9, 1, 1.1, 1.2]:
        params['reg_alpha'] = reg_alpha
        cv_results = xgb.cv(params, dtrain, nfold=3, early_stopping_rounds=50, feval=rmspe_xg, seed=23)
        mean_error = min(cv_results['test-rmspe-mean'])

        if mean_error < min_merror:
            min_merror = mean_error
            xgb_best_params["reg_alpha"] = reg_alpha

    params["reg_alpha"] = xgb_best_params["reg_alpha"]

    # ############################################# reg_lambda ################################################
    min_merror = np.inf
    for reg_lambda in [0.8, 0.9, 1, 1.1, 1.2]:
        params['reg_lambda'] = reg_lambda
        cv_results = xgb.cv(params, dtrain, nfold=3, early_stopping_rounds=50, feval=rmspe_xg, seed=23)
        mean_error = min(cv_results['test-rmspe-mean'])

        if mean_error < min_merror:
            min_merror = mean_error
            xgb_best_params["reg_lambda"] = reg_lambda

    params["reg_lambda"] = xgb_best_params["reg_lambda"]

    # ############################################# learning_rate ################################################
    min_merror = np.inf
    for learning_rate in [0.001, 0.005, 0.01, 0.03, 0.05]:
        params['learning_rate'] = learning_rate
        cv_results = xgb.cv(params, dtrain, nfold=3, early_stopping_rounds=50, feval=rmspe_xg, seed=23)
        mean_error = min(cv_results['test-rmspe-mean'])

        if mean_error < min_merror:
            min_merror = mean_error
            xgb_best_params["learning_rate"] = learning_rate

    params["learning_rate"] = xgb_best_params["learning_rate"]

    print(params)
    bst_params = {
        "eta": 0.3,
        "alpha": 1,
        "silent": 1,
        "seed": 42,
        "objective": params['objective'],
        "booster": params['booster'],
        "max_depth": params['max_depth'],
        'min_child_weight': params['min_child_weight'],
        "subsample": params['subsample'],
        "colsample_bytree": params['colsample_bytree'],
        "reg_alpha": params['reg_alpha'],
        "gpu_id": params['gpu_id'],
        "max_bin": params['max_bin'],
        "tree_method": params['tree_method'],
        "n_estimators": params['n_estimators']
    }

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    xgb_model = xgb.train(bst_params, dtrain, num_boost_round=1000, evals=watchlist, early_stopping_rounds=100,
                          feval=rmspe_xg, verbose_eval=True)
    print("Validating")
    yhat = xgb_model.predict(xgb.DMatrix(X_valid))
    error = rmspe(np.expm1(y_valid.values), np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))

    xgb_test_prod = xgb_model.predict(xgb.DMatrix(X_test))
    xgb_test_prod = np.expm1(xgb_test_prod)
    sub_df = pd.DataFrame(data=list(X_test_id), columns=['id'])
    sub_df["forecastVolum"] = [int(i) for i in xgb_test_prod]
    sub_df.to_csv(DefaultConfig.project_path + "/data/submit/" + DefaultConfig.select_model + "_submission.csv",
                  index=False,
                  encoding='utf-8')


def lgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test):
    """
    lgb model
    :param X_train:
    :param X_valid:
    :param y_train:
    :param y_valid:
    :param X_test_id:
    :param X_test:
    :return:
    """
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from matplotlib import pyplot as plt

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    if DefaultConfig.single_model:
        lgbm_params = {'task': 'train',
                       'boosting_type': 'gbdt',
                       'objective': 'regression',
                       'learning_rate': 0.05,
                       'num_leaves': 100,
                       'max_bin': 255,
                       'min_data_in_leaf ': 20,
                       'min_data_in_leaf': 20,
                       'feature_fraction': 0.6,
                       'bagging_fraction': 0.3,
                       'bagging_freq': 0,
                       'lambda_l1': 0.5,
                       'lambda_l2': 0.3,
                       'min_split_gain': 0,
                       'num_iterations': 2019}

        lgb_model = lgb.train(lgbm_params, dtrain, num_boost_round=30000, valid_sets=[dvalid, dtrain],
                              valid_names=['eval', 'train'],
                              early_stopping_rounds=50, feval=rmspe_lgb, verbose_eval=True)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(X_train.columns)
        fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='split',
                                                                        iteration=lgb_model.best_iteration)

        if fold_importance_df is not None:
            plt.figure(figsize=(8, 8))

            print(list(
                fold_importance_df.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).index))

            # 5折数据取平均值
            fold_importance_df.groupby(['feature'])['importance'].agg('mean').sort_values(ascending=False).head(
                40).plot.barh()
            plt.show()

        print("Validating")
        yhat = lgb_model.predict(X_valid)
        error = rmspe(np.expm1(y_valid.values), np.expm1(yhat))
        print('RMSPE: {:.6f}'.format(error))
        lgb_test_prod = lgb_model.predict(X_test)
        lgb_test_prod = np.expm1(lgb_test_prod)
        sub_df = pd.DataFrame(data=list(X_test_id), columns=['id'])
        sub_df["forecastVolum"] = [int(i) for i in lgb_test_prod]
        sub_df.to_csv(DefaultConfig.project_path + "/data/submit/" + DefaultConfig.select_model + "_submission.csv",
                      index=False, encoding='utf-8')
    else:
        # ## Tuning Paramters
        lgbm_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            "learning_rate": 1,
            "num_leaves": 31,
            "max_bin": 255,
            "min_data_in_leaf ": 20
        }
        lgb_best_params = {}

        # ## num_leaves
        min_merror = np.inf
        for num_leaves in range(30, 200, 10):
            lgbm_params["num_leaves"] = num_leaves

            cv_results = lgb.cv(params=lgbm_params,
                                train_set=dtrain,
                                num_boost_round=2000,
                                stratified=False,
                                nfold=5,
                                feval=rmspe_lgb,
                                verbose_eval=50,
                                seed=23,
                                early_stopping_rounds=20)
            mean_error = min(cv_results['rmspe-mean'])

            if mean_error < min_merror:
                min_merror = mean_error
                lgb_best_params["num_leaves"] = num_leaves

        lgbm_params["num_leaves"] = lgb_best_params["num_leaves"]

        # ## max_bin,  min_data_in_leaf
        min_merror = np.inf
        for max_bin in range(255, 260, 5):
            for min_data_in_leaf in range(20, 100, 10):
                lgbm_params["max_bin"] = max_bin
                lgbm_params["min_data_in_leaf"] = min_data_in_leaf

                cv_results = lgb.cv(params=lgbm_params,
                                    train_set=dtrain,
                                    num_boost_round=3000,
                                    stratified=False,
                                    nfold=5,
                                    feval=rmspe_lgb,
                                    verbose_eval=50,
                                    seed=23,
                                    early_stopping_rounds=30)
                mean_error = min(cv_results['rmspe-mean'])

                if mean_error < min_merror:
                    min_merror = mean_error
                    lgb_best_params["max_bin"] = max_bin
                    lgb_best_params["min_data_in_leaf"] = min_data_in_leaf

        lgbm_params["max_bin"] = lgb_best_params["max_bin"]
        lgbm_params["min_data_in_leaf"] = lgb_best_params["min_data_in_leaf"]

        # ## feature_fraction, bagging_fraction and bagging_freq
        min_merror = np.inf
        for feature_fraction in [i / 10.0 for i in range(3, 11)]:
            for bagging_fraction in [i / 10.0 for i in range(3, 11)]:
                for bagging_freq in range(0, 10, 2):
                    lgbm_params["feature_fraction"] = feature_fraction
                    lgbm_params["bagging_fraction"] = bagging_fraction
                    lgbm_params["bagging_freq"] = bagging_freq

                    cv_results = lgb.cv(params=lgbm_params,
                                        train_set=dtrain,
                                        num_boost_round=3000,
                                        stratified=False,
                                        nfold=5,
                                        feval=rmspe_lgb,
                                        verbose_eval=100,
                                        seed=23,
                                        early_stopping_rounds=30)
                    mean_error = min(cv_results['rmspe-mean'])

                    if mean_error < min_merror:
                        min_merror = mean_error
                        lgb_best_params["feature_fraction"] = feature_fraction
                        lgb_best_params["bagging_fraction"] = bagging_fraction
                        lgb_best_params["bagging_freq"] = bagging_freq

        lgbm_params["feature_fraction"] = lgb_best_params["feature_fraction"]
        lgbm_params["bagging_fraction"] = lgb_best_params["bagging_fraction"]
        lgbm_params["bagging_freq"] = lgb_best_params["bagging_freq"]

        # ## lambda_l1, lambda_l2 and min_gain_to_split
        min_merror = np.inf
        for lambda_l1 in [i / 10.0 for i in range(0, 11)]:
            for lambda_l2 in [i / 10.0 for i in range(0, 11)]:
                for min_split_gain in [i / 10.0 for i in range(0, 11)]:
                    lgbm_params["lambda_l1"] = lambda_l1
                    lgbm_params["lambda_l2"] = lambda_l2
                    lgbm_params["min_split_gain"] = min_split_gain

                    cv_results = lgb.cv(params=lgbm_params,
                                        train_set=dtrain,
                                        num_boost_round=3000,
                                        stratified=False,
                                        nfold=5,
                                        feval=rmspe_lgb,
                                        verbose_eval=100,
                                        seed=23,
                                        early_stopping_rounds=30)
                    mean_error = min(cv_results['rmspe-mean'])

                    if mean_error < min_merror:
                        min_merror = mean_error
                        lgb_best_params["lambda_l1"] = lambda_l1
                        lgb_best_params["lambda_l2"] = lambda_l2
                        lgb_best_params["min_split_gain"] = min_split_gain

        lgbm_params["lambda_l1"] = lgb_best_params["lambda_l1"]
        lgbm_params["lambda_l2"] = lgb_best_params["lambda_l2"]
        lgbm_params["min_split_gain"] = lgb_best_params["min_split_gain"]

        lgbm_params["num_leaves"] = 100
        lgbm_params["bagging_fraction"] = 0.3
        lgbm_params["bagging_freq"] = 0
        lgbm_params["feature_fraction"] = 0.6
        lgbm_params["max_bin"] = 255
        lgbm_params["min_data_in_leaf"] = 20
        lgbm_params["lambda_l1"] = 0.5
        lgbm_params["lambda_l2"] = 0.3
        lgbm_params["min_split_gain"] = 0

        # ## learning_rate
        lgbm_params["learning_rate"] = 0.05

        # ## num_iterations
        lgbm_params["num_iterations"] = 2019

        print(lgbm_params)
        lgb_model = lgb.train(lgbm_params, dtrain, num_boost_round=30000, valid_sets=[dvalid, dtrain],
                              valid_names=['eval', 'train'],
                              early_stopping_rounds=2019, feval=rmspe_lgb, verbose_eval=True)
        print("Validating")
        yhat = lgb_model.predict(X_valid)
        error = rmspe(np.expm1(y_valid.values), np.expm1(yhat))
        print('RMSPE: {:.6f}'.format(error))
        lgb_test_prod = lgb_model.predict(X_test)
        lgb_test_prod = np.expm1(lgb_test_prod)
        sub_df = pd.DataFrame(data=list(X_test_id), columns=['id'])
        sub_df["forecastVolum"] = [int(i) for i in lgb_test_prod]
        sub_df.to_csv(DefaultConfig.project_path + "/data/submit/" + DefaultConfig.select_model + "_submission.csv",
                      index=False, encoding='utf-8')


def cbt_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test):
    """
    cbt_model
    :param X_train:
    :param y_train:
    :param X_test:
    :param columns:
    :param params:
    :return:
    """
    import numpy as np
    import pandas as pd
    from catboost import CatBoostRegressor

    cbt_params = {
        'learning_rate': 0.01,
        'depth': 8,
        'l2_leaf_reg': 5.0,
        'loss_function': 'RMSE',
        'iterations': 800,
        'random_seed': 2019,
        'logging_level': 'Silent',
        'thread_count': 10}

    X_train = X_train.fillna(0)
    X_valid = X_valid.fillna(0)

    y_train = y_train.fillna(0)
    y_valid = y_valid.fillna(0)
    cbt_model = CatBoostRegressor(**cbt_params).fit(X=X_train, y=y_train, eval_set=(X_valid, y_valid),
                                                    early_stopping_rounds=2000, verbose_eval=True,
                                                    cat_features=['model', 'bodyType', 'dayofweek'])
    print("Validating")
    yhat = cbt_model.predict(X_valid)
    error = rmspe(np.expm1(y_valid.values), np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))
    cbt_test_prod = cbt_model.predict(X_test)
    cbt_test_prod = np.expm1(cbt_test_prod)
    sub_df = pd.DataFrame(data=list(X_test_id), columns=['id'])
    sub_df["forecastVolum"] = [int(i) for i in cbt_test_prod]
    sub_df.to_csv(DefaultConfig.project_path + "/data/submit/" + DefaultConfig.select_model + "_submission.csv",
                  index=False, encoding='utf-8')


def merge(**params):
    """
    合并
    :param params:
    :return:
    """
    import pandas as pd
    import numpy as np

    lgb = pd.read_csv(filepath_or_buffer=DefaultConfig.lgb_submission_path)
    rule = pd.read_csv(filepath_or_buffer=DefaultConfig.rule_submission_path)

    submission = pd.read_csv(filepath_or_buffer=DefaultConfig.submission_path)

    tmp = np.inf
    result = None
    for coefficient in np.arange(0.55, 0.6, 0.01):
        rule['forecastVolum'] = coefficient * rule['forecastVolum'] + (1 - coefficient) * lgb['forecastVolum']
        rule['forecastVolum'] = rule['forecastVolum'].astype(int)

        current = abs(rule['forecastVolum'] - submission['forecastVolum']).sum()
        if tmp > current:
            tmp = current
            result = rule.copy()

            print('coefficient: ', coefficient)
            print('差距： ', current)
    result.to_csv(path_or_buf=DefaultConfig.rule_lgb_submission_path, encoding='utf-8', index=None)

    # rule['forecastVolum'] = 0.7 * rule['forecastVolum'] + 0.3 * xgb['forecastVolum']
    # rule['forecastVolum'] = rule['forecastVolum'].astype(int)

    # rule.to_csv(path_or_buf=DefaultConfig.rule_xgb_submission_path, encoding='utf-8', index=None)

# if __name__ == '__main__':
#     preprocessing()
#     import lightgbm as lgb
#     from sklearn.metrics import mean_squared_error
#
#     lgb_model = lgb.LGBMRegressor(
#         num_leaves=32, reg_alpha=1, reg_lambda=0.1, objective='mse',
#         max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=np.random.randint(1000),
#         n_estimators=5000, subsample=0.8, colsample_bytree=0.8, n_jobs=10
#     )
#
#     lgb_model.fit(X_train, y_train, eval_set=[
#         (X_valid, y_valid),
#     ], categorical_feature=category_feature, early_stopping_rounds=100, verbose=100)
#
#     data['pred_label'] = lgb_model.predict(data[features]) * data['model_weight']
#
#
#     def score(data, pred='pred_label', label='salesVolume', group='model'):
#         data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
#         data_agg = data.groupby('model').agg({
#             pred: list,
#             label: [list, 'mean'],
#
#         }).reset_index()
#
#         data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
#         nrmse_score = []
#         for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
#             nrmse_score.append(
#                 mean_squared_error(raw[0], raw[1]) ** 0.5 / raw[2]
#             )
#         print(1 - np.mean(nrmse_score))
#         return 1 - np.mean(nrmse_score)
#
#
#     score(data.loc[list(X_valid.index)])
#     lgb_model.n_estimators = 666
#
#     lgb_model.fit(pd.concat([X_train, X_valid], ignore_index=True),
#                   pd.concat([y_train, y_valid], ignore_index=True), categorical_feature=category_feature)
#     data['forecastVolum'] = lgb_model.predict(data[features]) * data['model_weight']
#     sub = pd.DataFrame(data=list(X_test_id), columns=['id'])
#     print(data.loc[list(X_test.index)])
#     sub['forecastVolum'] = data.loc[list(X_test.index)]['forecastVolum'].apply(
#         lambda x: 0 if x < 0 else x).round().astype(int).reset_index(drop=True)
#     sub.to_csv(DefaultConfig.project_path + '/data/submit/lgb_submission.csv', index=False)
