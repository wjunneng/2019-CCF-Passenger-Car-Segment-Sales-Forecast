# -*- coding: utf-8 -*-
"""
    配置文件
"""
import os


class DefaultConfig(object):
    """
    参数配置
    """

    def __init__(self):
        pass

    # 项目路径
    project_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])

    # train_sales_data
    train_sales_data_path = project_path + '/data/original/train_sales_data.csv'
    # train_search_data
    train_search_data_path = project_path + '/data/original/train_search_data.csv'
    # train_user_reply_data
    train_user_reply_data_path = project_path + '/data/original/train_user_reply_data.csv'

    # evaluation_public
    evaluation_public_path = project_path + '/data/original/evaluation_public.csv'
    # submit_example.csv
    submit_example_path = project_path + '/data/original/submit_example.csv'

    # no_replace
    no_replace = False

    # 选中的模型
    select_model = 'lgb'

    # cache
    X_train_cache_path = project_path + '/data/cache/X_train.h5'
    X_valid_cache_path = project_path + '/data/cache/X_valid.h5'
    y_train_cache_path = project_path + '/data/cache/y_train.h5'
    y_valid_cache_path = project_path + '/data/cache/y_valid.h5'
    X_test_id_cache_path = project_path + '/data/cache/X_test_id.h5'
    X_test_cache_path = project_path + '/data/cache/X_test.h5'

    # lgb_path
    lgb_submission_path = project_path + '/data/submit/lgb_submission.csv'
    # xgb_path
    xgb_submission_path = project_path + '/data/submit/xgb_submission.csv'
    # rule_submit
    rule_submission_path = project_path + '/data/submit/rule_submission.csv'
    # submit
    rule_lgb_submission_path = project_path + '/data/submit/rule_lgb_submission.csv'
    rule_xgb_submission_path = project_path + '/data/submit/rule_xgb_submission.csv'
