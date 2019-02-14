from thex_data.data_init import *
from thex_data.data_prep import get_train_test, get_data
from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR


from model_performance.performance import *
from models.base_model.cmd_interpreter import get_model_data
from models.tree_model.hmc_tree import run_tree


def main():
    X_train, X_test, y_train, y_test = get_model_data()
    run_tree(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
