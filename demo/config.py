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
    train_sales_data_path = '/data/original/Train/train_sales_data.csv'
    # train_search_data
    train_search_data_path = '/data/original/Train/train_search_data.csv'
    # train_user_reply_data
    train_user_reply_data_path = '/data/original/Train/train_user_reply_data.csv'

    # evaluation_public
    evaluation_public_path = '/data/original/evaluation_public.csv'
    # submit_example.csv
    submit_example_path = '/data/original/submit_example.csv'

    # no_replace
    no_replace = False
