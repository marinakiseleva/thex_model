from thex_data.data_init import *
from thex_data.data_prep import get_train_test, get_data
from thex_data.data_consts import code_cat, TARGET_LABEL, ROOT_DIR


from model_performance.performance import *
from model_performance.init_classifier import *

from hmc_tree import run_tree


def main():
    col_list, incl_redshift, test_on_train = collect_args()
    run_tree(col_list, incl_redshift, test_on_train)


if __name__ == '__main__':
    main()
