"""
data_consts
Constant data-based information needed in THEx data program. Includes file locations of data, mapping of transient types and codes, as well as program-specific column names and references.
"""

import os

CPU_COUNT = 6

# Plotting Specifications
FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 600

#***************************************************************
# LOCAL_DATA_PATH to relative path of THEx FITS file, relative to root of
# package: thex_model


# Testing
# LOCAL_DATA_PATH = '/../../data/test_data.fits'

# June 3 Assembled - More data?
# LOCAL_DATA_PATH = '/../../data/assembled.fits'

# GALEX/WISE/PANSTARRS Versions 1, 2, 3
# LOCAL_DATA_PATH = '/../../data/THEx-training-set.v0_0_1.fits'
# LOCAL_DATA_PATH = '/../../data/THEx-training-set-v0_0_2.fits'
# LOCAL_DATA_PATH = '/../../data/THEx-training-set-v0_0_3.fits'
LOCAL_DATA_PATH = '/../../data/assembled-magcols.fits'

# All data Version 4
# LOCAL_DATA_PATH = '/../../data/THEx-catalog.v0_0_4.fits'


#***************************************************************


# ROOT_DIR = /full/path/up/to/thex_model
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."

# FITS file of transient/galaxy data
DATA_PATH = ROOT_DIR + LOCAL_DATA_PATH

# ORIG_TARGET_LABEL: Database defined column name; converted to TARGET_LABEL in project
ORIG_TARGET_LABEL = 'claimedtype'
# TARGET_LABEL: Program-designated label of target
TARGET_LABEL = 'transient_type'
UNKNOWN_LABEL = 'Unknown'
PRED_LABEL = 'predicted_class'
UNDEF_CLASS = 'Unspecified '
TREE_ROOT = "_ROOT"


LOSS_FUNCTION = "brier_score_loss"


"""
Transient classes in order they are to be visualized. If class is Unspecified, it will take the place of its unspecified name
"""
ORDERED_CLASSES = ['Ia',
                   'Ia Pec',
                   'Ia-91bg',
                   'Ia-91T',
                   'Ia-02cx',
                   'Ia CSM',
                   'Ia-00cx',
                   'Ia-09dc',
                   'Ia-99aa',
                   'Ia-HV',

                   'CC',
                   'SE',
                   'Ib',
                   'Ibn',
                   'IIb',
                   'Ib Pec',
                   'Ib-Ca',
                   'Ic',
                   'Ic BL',
                   'Ic Pec',
                   'Ic-SL',
                   'Ib/c',
                   'Ib/c Pec',

                   'II',
                   'II P',
                   'II L',
                   'IIn',
                   'IIn Pec',
                   'IIn-09ip',
                   'II Pec',
                   'II P Pec',
                   'II P-97D',
                   'IIb Pec',
                   'IIb-n',

                   'SLSN',
                   'SLSN-R',
                   'SLSN-I',
                   'SLSN-II',

                   'nIa',

                   'TDE',
                   'XrayTDE',
                   'UVOptTDE',
                   'LikelyXrayTDE',
                   'PossibleXrayTDE',

                   'GRB',
                   'SGRB',
                   'LGRB',

                   'FRB',
                   'Kilonova',
                   'GW'
                   ]


"""
class_to_subclass
{Parent : [Child1, child2, ...] }
Hierarchy of transient types, used in hierarchical multilabel classifiers
"""

# Version 4
class_to_subclass = {
    "Ia":                    ["Ia Pec"],
    "Ia Pec":                ["Ia-00cx", "Ia-02cx", "Ia-09dc", "Ia-91T",
                              "Ia-91bg", "Ia-99aa", "Ia-HV", "Ia CSM"],
    "CC":                    ["nIa", "SE", "II", "SLSN"],

    "SE":                    ["Ib", "Ic", "Ib/c"],

    "Ib/c":                  ["Ib/c Pec"],

    "Ib":                    ["Ibn", "Ib Pec", "IIb"],
    "Ib Pec":                ["Ib-Ca"],

    "Ic":                    ["Ic Pec"],
    "Ic Pec":                ["Ic BL", "Ic-SL"],

    "IIb":                   ["IIb Pec"],
    "IIb Pec":               ["IIb-n"],

    "II":                    ["II Pec", "II L", "II P", "IIn"],

    "II P":                  ["II P Pec"],
    "II P Pec":              ["II P-97D"],

    "IIn":                   ["IIn Pec"],
    "IIn Pec":               ["IIn-09ip"],

    "SLSN":                  ["SLSN-R", "SLSN-I", "SLSN-II"],

    "GRB":                   ["LGRB", "SGRB"],

    "TDE":                   ["XrayTDE", "UVOptTDE"],
    "XrayTDE":               ["LikelyXrayTDE", "PossibleXrayTDE"],
    "_ROOT":                 ["Ia", "CC",  "TDE", "GRB",
                              "FRB", "GW", "Kilonova"],

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
drop_cols
Columns with non-numeric values that are not used in analysis
"""
drop_cols = [
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
Magnitudes adjacent to one another that can be subtracted to get colors.
"""

adjacent_mags = {
    'NUV_mag': 'FUV_mag',
    'g_mag': 'NUV_mag',
    'r_mag': 'g_mag',
    'i_mag': 'r_mag',
    'z_mag': 'i_mag',
    'y_mag': 'z_mag',
    'W1_mag': 'y_mag',
    'W2_mag': 'W1_mag',
    'W3_mag': 'W2_mag',
    'W4_mag': 'W3_mag',
    'GALEXAIS_NUV': 'GALEXAIS_FUV',
    'PS1_gmag':  'GALEXAIS_NUV',
    'PS1_rmag': 'PS1_gmag',
    'PS1_imag': 'PS1_rmag',
    'PS1_zmag': 'PS1_imag',
    'PS1_ymag': 'PS1_zmag',
    'AllWISE_W1mag': 'PS1_ymag',
    'AllWISE_W2mag': 'AllWISE_W1mag',
    'AllWISE_W3mag': 'AllWISE_W2mag',
    'AllWISE_W4mag': 'AllWISE_W3mag'
}
