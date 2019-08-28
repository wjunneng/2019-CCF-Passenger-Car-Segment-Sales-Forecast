from demo.config import DefaultConfig


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

