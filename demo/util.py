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


def preprocessing(save=True, **param):
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
        X_test_id = reduce_mem_usage(pd.read_hdf(path_or_buf=DefaultConfig.X_test_id_cache_path, key='X_test_id', mode='r'))
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

        # 合并了sales + search
        train_sales_search = pd.merge(train_sales_data, train_search_data,
                                      on=['province', 'adcode', 'model', 'regYear', 'regMonth'])

        # 删除特征
        del evaluation_public['forecastVolum']
        del evaluation_public['province']
        del train_sales_search['province']
        del train_sales_search['bodyType']

        # 先保存id
        X_test_id = evaluation_public['id']
        # 后删除
        del evaluation_public['id']

        # 合并训练集和测试集
        X = pd.concat([train_sales_search, evaluation_public], axis=0, ignore_index=True)

        # 截取前两个字符
        X['adcode'] = X['adcode'].apply(lambda x: int(str(x)[:2]))
        # 编码
        X['model'] = LabelEncoder().fit_transform(X['model'])

        # 测试集
        X_test = X.iloc[train_sales_search.shape[0]:, :]
        # 训练集
        X = X.iloc[:train_sales_search.shape[0], :]

        del X['popularity']
        # 学习popularity
        # columns = list(X.columns)
        # columns.remove('popularity')
        # columns.remove('salesVolume')
        # X_test = semi_suprivised_learning(X[columns], X['popularity'], X_test[columns], 'popularity')
        # X_test.reset_index(drop=True, inplace=True)
        # for column in list(X_test.columns):
        #     X_test[column] = X_test[column].astype(int)
        #     X_test[column] = X_test[column].astype(int)
        # print(X_test['popularity'].values)

        if 'salesVolumne' in list(X_test.columns):
            # 删除测试集的salesVolume列
            del X_test['salesVolume']

        columns = list(X.columns)
        columns.remove('salesVolume')

        X, y = X[columns], np.log1p(X['salesVolume'])
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=23)
        y_train = pd.DataFrame(y_train, index=None)
        y_valid = pd.DataFrame(y_valid, index=None)

        if save:
            X_train.to_hdf(DefaultConfig.X_train_cache_path, key='X_train', mode='w')
            X_valid.to_hdf(DefaultConfig.X_valid_cache_path, key='X_valid', mode='w')
            y_train.to_hdf(DefaultConfig.y_train_cache_path, key='y_train', mode='w')
            y_valid.to_hdf(DefaultConfig.y_valid_cache_path, key='y_valid', mode='w')
            pd.DataFrame(X_test_id).to_hdf(DefaultConfig.X_test_id_cache_path, key='X_test_id', mode='w')
            X_test.to_hdf(DefaultConfig.X_test_cache_path, key='X_test', mode='w')

    X_train.reset_index(drop=True, inplace=True)
    X_valid.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

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
    for n_estimators in range(10, 100, 10):
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
        for min_child_weight in range(1, 6, 1):
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

    bst_params = {"objective": params['objective'],
                  "booster": params['booster'],
                  "eta": 0.3,
                  "max_depth": params['max_depth'],
                  'min_child_weight': params['min_child_weight'],
                  "subsample": params['subsample'],
                  "colsample_bytree": params['colsample_bytree'],
                  "reg_alpha": params['reg_alpha'],
                  "silent": 1,
                  "seed": 2019,
                  "gpu_id": params['gpu_id'],
                  "max_bin": params['max_bin'],
                  "tree_method": params['tree_method']
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

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

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

    lgb_model = lgb.train(lgbm_params, dtrain, num_boost_round=30000, valid_sets=[dvalid, dtrain],
                          valid_names=['eval', 'train'],
                          early_stopping_rounds=50, feval=rmspe_lgb, verbose_eval=True)
    print("Validating")
    yhat = lgb_model.predict(X_valid)
    error = rmspe(np.expm1(y_valid.values), np.expm1(yhat))
    print('RMSPE: {:.6f}'.format(error))
    lgb_test_prod = lgb_model.predict(X_test)
    lgb_test_prod = np.expm1(lgb_test_prod)
    sub_df = sub_df = pd.DataFrame(data=list(X_test_id), columns=['id'])
    sub_df["forecastVolum"] = [int(i) for i in lgb_test_prod]
    sub_df.to_csv(DefaultConfig.project_path + "/data/submit/" + DefaultConfig.select_model + "_submission.csv",
                  index=False,
                  encoding='utf-8')

    # test_prod = 0.1 * xgb_test_prod + 0.9 * lgb_test_prod
    # sub_df = pd.DataFrame({"id": X_test_id.values})
    # sub_df["forecastVolum"] = [int(i) for i in test_prod]
    # sub_df.to_csv("xgb_lgb_submission.csv", index=False, encoding='utf-8')
