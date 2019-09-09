from config import *
import warnings
from util import *

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def main(**params):
    """
    主函数
    :param params:
    :return:
    """
    X_train, X_valid, y_train, y_valid, X_test_id, X_test = preprocess()

    if DefaultConfig.select_model is 'xgb':
        xgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)
    elif DefaultConfig.select_model is 'lgb':
        lgb_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)
    elif DefaultConfig.select_model is 'cbt':
        cbt_model(X_train, X_valid, y_train, y_valid, X_test_id, X_test)

    merge()


if __name__ == '__main__':
    main()
