from config import *
import warnings
from util_1 import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def main(**params):
    """
    主函数
    :param params:
    :return:
    """
    X_train, X_valid, y_train, y_valid, X_test_id, X_test = preprocessing()

    if DefaultConfig.select_model is 'xgb':
        xgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)
    elif DefaultConfig.select_model is 'lgb':
        lgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)
    elif DefaultConfig.select_model is 'cbt':
        cbt_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)

    merge()


if __name__ == '__main__':
    main()
