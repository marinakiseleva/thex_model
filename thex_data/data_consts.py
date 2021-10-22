"""
data_consts
Constant data-based information needed in THEx data program. Includes file locations of data, mapping of transient types, as well as program-specific column names and references.
"""

import os

# ROOT_DIR = /full/path/up/to/thex_model
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."

# Default data path - FITS file of transient/galaxy data
DATA_PATH = ROOT_DIR + "/../../data/catalogs/v8/THEx-v8.0-release.mags-xcalib.min-xcal.fits"

# Runtime specs
CPU_COUNT = 12
TREE_ROOT = "_ROOT"
LOSS_FUNCTION = "brier_score_loss"
DEFAULT_KERNEL = "exponential"

# Plotting Specifications
FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 600
TICK_S = 16
LAB_S = 14
TITLE_S = 16

P_BAR_COLOR = "#3385ff"
C_BAR_COLOR = "#ff9933"
INTVL_COLOR = "black"
BSLN_COLOR = "#ff1a1a"
BAR_EDGE_COLOR = "black"

# ORIG_TARGET_LABEL: Database defined column name - converted to TARGET_LABEL in project
ORIG_TARGET_LABEL = 'revised_type'
TARGET_LABEL = 'transient_type'
UNKNOWN_LABEL = 'Unknown'
PRED_LABEL = 'predicted_class'
UNDEF_CLASS = 'Unspecified '


"""
ORDERED_CLASSES
Transient classes in order they are to be visualized. If class is Unspecified, it will take the place of its unspecified name
Update 2/4/2021.
"""
ORDERED_CLASSES = ['Ia',
                   # 'Ia Pec',
                   'Ia-91bg',
                   'Ia-91T',
                   'Ia-02cx',
                   'Ia-CSM',
                   'Ia-09dc',
                   'Ia-02es',
                   'Ia-99aa',
                   'Ia-00cx',
                   'Ia-HV',
                   'Ca-rich',

                   'CC',
                   'SE',
                   'Ib',
                   'Ibn',
                   'IIb',
                   # 'Ib Pec',
                   # 'Ib-Ca',
                   'Ic',
                   'Ic BL',
                   # 'Ic Pec',
                   # 'Ic-SL',
                   'Ibc',
                   'Ib/c',
                   # 'Ib/c Pec',
                   'II (cust.)',
                   'II',
                   'II P',
                   'II L',
                   'IIn',
                   # 'IIn Pec',
                   # 'IIn-09ip',
                   # 'II Pec',
                   # 'II P Pec',
                   # 'II P-97D',
                   # 'IIb Pec',
                   # 'IIb-n',

                   'SLSN',
                   'SLSN-R',
                   'SLSN-I',
                   'SLSN-II',

                   # 'nIa',

                   'TDE',
                   'UVOptTDE',
                   'XrayTDE',
                   # 'LikelyXrayTDE',
                   # 'PossibleXrayTDE',

                   'GRB',
                   'SGRB',
                   'LGRB',

                   'FRB',
                   'Kilonova',
                   'GW'
                   ]


"""
CLASS_HIERARCHY
{Parent : [Child1, child2, ...] }
Hierarchy of transient types, used in hierarchical multilabel classifiers
"""

CLASS_HIERARCHY = {
    "Ia":                    ["Ia-00cx", "Ia-02cx", "Ia-09dc", "Ia-91T",
                              "Ia-91bg", "Ia-99aa", "Ia-HV", "Ia CSM"],

    "CC":                    ["SE", "II", "SLSN"],

    "SE":                    ["Ib", "Ic", "Ib/c"],


    "Ib":                    ["Ibn",   "IIb"],
    # "Ib Pec":                ["Ib-Ca"],
    # "Ic":                    ["Ic Pec"],
    # "Ic Pec":                ["Ic BL", "Ic-SL"],
    # "IIb":                   ["IIb Pec"],
    # "IIb Pec":               ["IIb-n"],

    "II":                    ["II L", "II P", "IIn"],

    # "II P":                  ["II P Pec"],
    # "II P Pec":              ["II P-97D"],
    # "IIn":                   ["IIn Pec"],
    # "IIn Pec":               ["IIn-09ip"],

    "SLSN":                  ["SLSN-R", "SLSN-I", "SLSN-II"],

    "GRB":                   ["LGRB", "SGRB"],

    "TDE":                   ["XrayTDE", "UVOptTDE"],
    "XrayTDE":               ["LikelyXrayTDE", "PossibleXrayTDE"],
    "_ROOT":                 ["Ia", "CC",  "TDE", "GRB",
                              "FRB", "Kilonova", "GW"],

    # "MiscRadio":             ["Radio", "maser"],
    # "Other":                 ["NT", "MiscRadio"],
    # "Impostor":              ["Star", "SolarSystem", "AGN", "Galaxy",
    # "Galactic", "blue"],
    # "Star":                  ["Variable"],
    # "Variable":              ["CV", "LBV", "LRV", "LRV", "LPV", "XRB", "YSO",
    #                           "Nova"],
    # "SolarSystem":           ["Comet", "mover"],

    # "_SN":                   ["I", "II", "SLSN"],

    # "_PEC_SN":               ["I Pec", "Ia Pec", "Ib Pec", "Ic Pec",
    #                           "Ib/c Pec", "II Pec", "II P Pec", "IIb Pec",
    #                           "IIn Pec", "Pec"],

    # "_W_RADIO":              ["FRB", "MiscRadio"],
    # "_W_UVOPT":              ["I", "II", "SLSN", "UVOptTDE", "Kilonova"],
    # "_W_HIENERGY":           ["GRB", "XrayTDE"],
    # "_W_GW":                 ["GW"],


    # "_W_UVOPT", "_W_HIENERGY", "_PEC_SN",
    # "Variable", "Candidate"],

    # "_UNCLEAR_LABELS":       ["_UNCLEAR", "CN", "LCH", "LSQ", "c"],
    # "_IGNORED_LABELS":       ["removed"],
    # "_CONFUSION":            ["_CONFUSION_Ia_CC", "_CONFUSION_CC",
    #                           "_CONFUSION_SN_Other"],
    # "__CONFLICT_CASES":      [["Ia", "CC"], ["I", "II"], ["_SN", "Impostor"],
    #                           ["II L", "II P"], ["IIn", "IIb"], ["Ib", "Ic"]]
}


"""
EXCLUDE_COLS
Columns with non-numeric values that are not used in analysis
"""
EXCLUDE_COLS = [
    "_id",  # object
    "name",  # object
    "ra",  # object
    "dec",  # object
    "ra_deg",  # float64
    "radec_err",  # float32
    "dec_deg",  # float64
    "host_dist",  # float32
    "claimedtype",  # object
    "N_candidates",  # int32
    "host_dec",  # float64
    "host_id",  # object
    "host_ra",  # float64
    "host_radec_src",  # object
    "identified_by",  # object
    "is_identified",  # bool
    "valid_crossid",  # bool
    "host_confusion",  # bool
    "host_confusion_cats",  # object
]

"""
ORDERED_MAGS
Magnitudes adjacent to one another in spectrum that can be subtracted to get colors.
"""
ORDERED_MAGS = {
    'NUV_mag': 'FUV_mag',
    'u_mag': 'NUV_mag',
    'g_mag': 'u_mag',
    'r_mag': 'g_mag',
    'i_mag': 'r_mag',
    'z_mag': 'i_mag',
    'y_mag': 'z_mag',
    'J_mag': 'y_mag',
    'H_mag': 'J_mag',
    'K_mag': 'H_mag',
    'W1_mag': 'K_mag',
    'W2_mag': 'W1_mag',
    'W3_mag': 'W2_mag',
    'W4_mag': 'W3_mag',
    # 'GALEXAIS_NUV': 'GALEXAIS_FUV',
    # 'PS1_gmag':  'GALEXAIS_NUV',
    # 'PS1_rmag': 'PS1_gmag',
    # 'PS1_imag': 'PS1_rmag',
    # 'PS1_zmag': 'PS1_imag',
    # 'PS1_ymag': 'PS1_zmag',
    # 'AllWISE_W1mag': 'PS1_ymag',
    # 'AllWISE_W2mag': 'AllWISE_W1mag',
    # 'AllWISE_W3mag': 'AllWISE_W2mag',
    # 'AllWISE_W4mag': 'AllWISE_W3mag'
}
