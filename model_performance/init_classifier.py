import argparse
from thex_data.data_init import *


def collect_args():
    parser = argparse.ArgumentParser(description='Classify transients')
    parser.add_argument('-cols', '--cols', nargs='+',
                        help='<Required> Pass column names', required=False)
    parser.add_argument('-col_names', '--col_names', nargs='+',
                        help='Pass in string by which columns will be selected. For example: AllWISE will use all AlLWISE columns.', required=False)
    parser.add_argument('-incl_redshift', '--incl_redshift', nargs='+',
                        help='<Required> Set flag', required=False)
    args = parser.parse_args()

    col_list = []
    if args.cols is None:
        if args.col_names is not None:
            col_list = collect_cols(args.col_names)
        else:
            # Use all columns in data set with number values
            cols = list(collect_data())
            drop_cols = ['event', 'ra', 'dec', 'ra_deg', 'dec_deg', 'radec_err', 'redshift', 'claimedtype', 'host', 'host_ra', 'host_dec', 'ebv', 'host_ra_deg',
                         'host_dec_deg', 'host_dist', 'host_search_radius', 'is_confirmed_host', 'by_primary_cand', 'by_transient', "Err", "_e_", 'AllWISE_IsVar',
                         'HyperLEDA_objtype',
                         'HyperLEDA_type',
                         'HyperLEDA_bar',
                         'HyperLEDA_ring',
                         'HyperLEDA_multiple',
                         'HyperLEDA_compactness',
                         'HyperLEDA_agnclass'
                         ]
            col_list = []
            for c in cols:
                if not any(drop in c for drop in drop_cols):
                    col_list.append(c)

    else:
        col_list = args.cols

    incl_redshift = args.incl_redshift if args.incl_redshift is not None else False

    print("\n\nUsing data columns:\n\n" + str(col_list))
    return col_list, args.incl_redshift
