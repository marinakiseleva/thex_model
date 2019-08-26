"""
data_consts
Constant data-based information needed in THEx data program. Includes file locations of data, mapping of transient types and codes, as well as program-specific column names and references.
"""

import os

CPU_COUNT = 16

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
LOCAL_DATA_PATH = '/../../data/THEx-training-set-v0_0_3.fits'


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
TREE_ROOT = "_ROOT"  # 'TTypes'  #

"""
class_to_subclass
{Parent : [Child1, child2, ...] }
Hierarchy of transient types, used in hierarchical multilabel classifiers
"""
# Version 3
class_to_subclass_old = {
    "TTypes": ["Ia", "CC", "TDE", "GRB", "FRB", "Kilonova",  "GW"],
    "Ia": ["Ia-91bg", "Ia-91T", "Ia-02cx",  "Ia-CSM"],
    "CC": ["SE", "II", "SLSN"],
    "SE": ["Ib", "Ic", "Ib/c"],
    "Ib": ["Ibn", "IIb", "Ib-Ca?"],
    "Ic": ["Ic-BL"],
    "II": ["II P", "II L", "IIn"],
    "SLSN": ["SLSN-I", "SLSN-II", "SLSN-R"],
    "GRB": ["LGRB", "SGRB"],
    "TDE": ["UVOptTDE", "XrayTDE"]
}
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

    "MiscRadio":             ["Radio", "maser"],
    "Other":                 ["NT", "MiscRadio"],
    "Impostor":              ["Star", "SolarSystem", "AGN", "Galaxy",
                              "Galactic", "blue"],
    "Star":                  ["Variable"],
    "Variable":              ["CV", "LBV", "LRV", "LRV", "LPV", "XRB", "YSO",
                              "Nova"],
    "SolarSystem":           ["Comet", "mover"],

    # "_SN":                   ["I", "II", "SLSN"],

    "_PEC_SN":               ["I Pec", "Ia Pec", "Ib Pec", "Ic Pec",
                              "Ib/c Pec", "II Pec", "II P Pec", "IIb Pec",
                              "IIn Pec", "Pec"],

    # "_W_RADIO":              ["FRB", "MiscRadio"],
    "_W_UVOPT":              ["I", "II", "SLSN", "UVOptTDE", "Kilonova"],
    "_W_HIENERGY":           ["GRB", "XrayTDE"],
    # "_W_GW":                 ["GW"],

    "_ROOT":                 ["Ia", "CC", "GRB", "TDE",
                              "FRB", "GW", "Kilonova"],

    # "_UNCLEAR_LABELS":       ["_UNCLEAR", "CN", "LCH", "LSQ", "c"],
    # "_IGNORED_LABELS":       ["removed"],
    # "_CONFUSION":            ["_CONFUSION_Ia_CC", "_CONFUSION_CC",
    #                           "_CONFUSION_SN_Other"],
    # "__CONFLICT_CASES":      [["Ia", "CC"], ["I", "II"], ["_SN", "Impostor"],
    #                           ["II L", "II P"], ["IIn", "IIb"], ["Ib", "Ic"]]
}

"""
cat_code
{transient type name : transient type class code}
Transient Type categories, mapped to codes (integer values).
"""
cat_code = {
    'AGN': 1,
    'CC': 2,
    'Candidate': 3,
    'False': 4,
    'GRB': 5,
    'GW': 6,
    'Galaxy': 7,
    'HII Region': 8,
    'He + SMBH': 9,
    'I': 10,
    'I Pec': 11,
    'I-faint': 12,
    'I-rapid': 13,
    'II': 14,
    'II L': 15,
    'II P': 16,
    'II P Pec': 17,
    'II P-97D': 18,
    'II Pec': 19,
    'IIb': 20,
    'IIb Pec': 21,
    'IIc': 22,
    'IIn': 23,
    'IIn L': 24,
    'IIn P': 25,
    'IIn Pec': 26,
    'Ia': 27,
    'Ia CSM': 28,
    'Ia Pec': 29,
    'Ia-00cx': 30,
    'Ia-02cx': 31,
    'Ia-09dc': 32,
    'Ia-91T': 33,
    'Ia-91bg': 34,
    'Ia-99aa': 35,
    'Ia-HV': 36,
    'Ia/b': 37,
    'Ia/c': 38,
    'Ib': 39,
    'Ib Pec': 40,
    'Ib-Ca': 41,
    'Ib/c': 42,
    'Ib/c Pec': 43,
    'Ibn': 44,
    'Ic': 45,
    'Ic BL': 46,
    'Ic Pec': 47,
    'Ic-lum': 48,
    'Impostor': 49,
    'Kilonova': 50,
    'KilonovaCand': 51,
    'LGRB': 52,
    'Lensing': 53,
    'Low-mass TDE': 54,
    'MS + SMBH': 55,
    'Minor Planet': 56,
    'Nova': 57,
    'Other': 58,
    'PISN': 59,
    'Planet + WD': 60,
    'SGRB': 61,
    'SLSN': 62,
    'SLSN-I': 63,
    'SLSN-II': 64,
    'SLSN-R': 65,
    'Star': 66,
    'TDE': 67,
    'Variable': 68,
    'WD + IMBH': 69,
    'nIa': 70,
    'Ic-BL': 71,  # New class
    'UVOptTDE': 72,  # New class
    '_W_UVOPT': 73,
    '_W_HIENERGY': 74,
    '_PEC_SN': 75,
    "UVOptTDE": 76,
    "XrayTDE": 77,
    "PossibleXrayTDE": 78,
    "LikelyXrayTDE": 79,
    'Unknown': 100,
    'TTypes': 200
}

"""
code_cat
{transient type class code : transient type name }
Transient Type codes mapped to categories.
"""
code_cat = {v: k for k, v in cat_code.items()}


"""
groupings
{data values : transient type }
The groupings below map specific transient types in the data set to claimed type groups. For example: nIa, Ia, Ia*, and Ia-HV all map to Ia.
"""
groupings = {
    'Candidate': 'Candidate',
    'PSN': 'Candidate',
    'Cand': 'Candidate',
    'Stale': 'Candidate',
    'CandFaint': 'Candidate',
    'CC': 'CC',
    'CCSN': 'CC',
    'Core-Collapse': 'CC',
    'I': 'I',
    'I-rapid': 'I-rapid',
    'Ia': 'Ia',
    'Ia-norm': 'Ia',
    'Ia- norm': 'Ia',
    'Lensed SN Ia': 'Ia',
    'Computed-Ia': 'Ia',
    'Ia(N)': 'Ia',
    'Ia*': 'Ia',
    'I Pec': 'I Pec',
    'I pec': 'I Pec',
    'I-pec': 'I Pec',
    'I-Pec': 'I Pec',
    'IPec': 'I Pec',
    'Ipec': 'I Pec',
    'Ia Pec': 'Ia Pec',
    'Ia pec': 'Ia Pec',
    'Ia-pec': 'Ia Pec',
    'Iapec': 'Ia Pec',
    'IaPec': 'Ia Pec',
    'Ia-Pec': 'Ia Pec',
    'Iap': 'Ia Pec',
    'Ia P': 'Ia Pec',
    'IaP': 'Ia Pec',
    'Ia-p': 'Ia Pec',
    'IaPec(N)': 'Ia Pec',
    'Ia? P': 'Ia Pec',
    'Ia-pec.': 'Ia Pec',
    'Ia CSM': 'Ia CSM',
    'Ia-CSM': 'Ia CSM',
    'Ia-csm': 'Ia CSM',
    'Ia+CSM': 'Ia CSM',
    'Ia-CSM/IIn': 'IIn',
    'Ia-02ic-like': 'Ia CSM',
    'Ia-91bg': 'Ia-91bg',
    'Ia-pec (1991bg)': 'Ia-91bg',
    'Ia-91bg-like': 'Ia-91bg',
    'Ia-91bg like': 'Ia-91bg',
    'Ia-92bg': 'Ia-91bg',
    'Ia-91T': 'Ia-91T',
    'Ia-pec 1991T': 'Ia-91T',
    'Ia-91T-like': 'Ia-91T',
    'Ia-91T like': 'Ia-91T',
    'Ia 91T-like': 'Ia-91T',
    'Ia-T': 'Ia-91T',
    'IaT': 'Ia-91T',
    'Ia-91Tlike': 'Ia-91T',
    'Ia-02cx': 'Ia-02cx',
    'Ia-02cx-like': 'Ia-02cx',
    'Iax': 'Ia-02cx',
    'Iax[02cx-like]': 'Ia-02cx',
    'Ib-pec/Iax': 'Ib Pec',
    'Ia-00cx': 'Ia-00cx',
    'Ia-99aa': 'Ia-99aa',
    'Ia-09dc': 'Ia-09dc',
    'Ia-HV': 'Ia-HV',
    'I-faint': 'I-faint',
    'Ib': 'Ib',
    'Ib-norm': 'Ib',
    'Ib/IIb': 'IIb',
    'Ib-IIb': 'IIb',
    'IIb/Ib': 'IIb',
    'II/Ib': 'II',
    'Ibn': 'Ibn',
    'Ib-n': 'Ibn',
    'IIn/Ibn': 'IIn',
    'Ib-n/IIb-n': 'IIb',
    'Ib Pec': 'Ib Pec',
    'Ib pec': 'Ib Pec',
    'Ib-pec': 'Ib Pec',
    'Ibpec': 'Ib Pec',
    'IbPec': 'Ib Pec',
    'Ic': 'Ic',
    '1c': 'Ic',
    'Ic/Ic-BL': 'Ic BL',
    'IIn/Ic': 'IIn',
    'II/Ic': 'II',
    'Ic Pec': 'Ic Pec',
    'Ic pec': 'Ic Pec',
    'Ic-pec': 'Ic Pec',
    'Icpec': 'Ic Pec',
    'IcPec': 'Ic Pec',
    'Ia/c': 'Ia/c',
    'Ic/Ia': 'Ia/c',
    'Iac': 'Ia/c',
    'Ia/Ic': 'Ia/c',
    'Ia/b': 'Ia/b',
    'Ib/c': 'Ib/c',
    'Ibc': 'Ib/c',
    'Ibc-norm': 'Ib/c',
    'Ib/c-norm': 'Ib/c',
    'Ib/Ic': 'Ib/c',
    'IIb/Ib/Ic (Ca rich)': 'Ib-Ca',
    'Ib/Ic (Ca rich)': 'Ib-Ca',
    'II/Ib/c': 'II',
    'Ib/c-BL': 'Ic BL',
    'Ic/b': 'Ib/c',
    'Ib/c Pec': 'Ib/c Pec',
    'Ib/c-pec': 'Ib/c Pec',
    'Ibc pec': 'Ib/c Pec',
    'Ib/c pec': 'Ib/c Pec',
    'Ib/cPec': 'Ib/c Pec',
    'Ib/cpec': 'Ib/c Pec',
    'Ibc-pec': 'Ib/c Pec',
    'Ib/cPe': 'Ib/c Pec',
    'II': 'II',
    'II/IIB': 'IIb',
    'II/LBV': 'Variable Star',
    'II/IIb': 'IIb',
    'II P': 'II P',
    'IIP': 'II P',
    'IIp': 'II P',
    'II p': 'II P',
    'II-P': 'II P',
    'IIP/IIL': 'II L',
    'Computed-IIP': 'II P',
    'II-p': 'II P',
    'II-P/L': 'II L',
    'II Pec': 'II Pec',
    'II pec': 'II Pec',
    'IIpec': 'II Pec',
    'IIPec': 'II Pec',
    'II-pec': 'II Pec',
    'II-Pec': 'II Pec',
    'II P Pec': 'II P Pec',
    'IIP-pec': 'II P Pec',
    'IIPpec': 'II P Pec',
    'IIPp': 'II P Pec',
    'II P pec': 'II P Pec',
    'II-Pp.': 'II P Pec',
    'II L': 'II L',
    'IIL': 'II L',
    'II-L': 'II L',
    'IIL/IIn': 'IIn',
    'IIn': 'IIn',
    'II n': 'IIn',
    'IIN': 'IIn',
    'LBV to IIn': 'Variable Star',
    'IIn/LBV': 'Variable Star',
    'II-09ip': 'IIn',
    'IIn-09ip': 'IIn',
    'IIn P': 'IIn P',
    'IInP': 'IIn P',
    'IIn Pec': 'IIn Pec',
    'IIn pec': 'IIn Pec',
    'IIn-pec': 'IIn Pec',
    'IInPec': 'IIn Pec',
    'IInpec': 'IIn Pec',
    'IIn-pec/LBV': 'Variable Star',
    'IIn L': 'IIn L',
    'IInL': 'IIn L',
    'IIb': 'IIb',
    'IIB': 'IIb',
    'II-b': 'IIb',
    'Computed-IIb': 'IIb',
    'IIb Pec': 'IIb Pec',
    'IIb-pec': 'IIb Pec',
    'IIb: pec': 'IIb Pec',
    'IIb: Pec': 'IIb Pec',
    'IIb pec': 'IIb Pec',
    'IIb:pec': 'IIb Pec',
    'IIb:Pec': 'IIb Pec',
    'IIc': 'IIc',
    'nIa': 'nIa',
    'non-Ia': 'nIa',
    'not-Ia': 'nIa',
    'non Ia': 'nIa',
    'not Ia': 'nIa',
    'SLSN': 'SLSN',
    'SL': 'SLSN',
    'SLSN-R': 'SLSN-R',
    'SLSN I': 'SLSN-I',
    'SL-I': 'SLSN-I',
    'SLI': 'SLSN-I',
    'SLSN-Ic': 'SLSN-I',
    'SLSN Ic': 'SLSN-I',
    'SL-Ic': 'SLSN-I',
    'SLIc': 'SLSN-I',
    'SLSN-I-R': 'SLSN-R',
    'SLSN II': 'SLSN-II',
    'SL-II': 'SLSN-II',
    'SLII': 'SLSN-II',
    'SLSN-IIn': 'SLSN-II',
    'SLSN-I': 'SLSN-I',
    'SLSN-II': 'SLSN-II',
    'Ib-Ca': 'Ib-Ca',
    'Ib - Ca-rich': 'Ib-Ca',
    'Ib (Ca rich)': 'Ib-Ca',
    'Ca-rich': 'Ib-Ca',
    'Ib-Ca-rich': 'Ib-Ca',
    'II P-97D': 'II P-97D',
    'IIP-pec 1997D': 'II P-97D',
    'Ic BL': 'Ic BL',
    'Ic-broad': 'Ic BL',
    'Ic-BL': 'Ic BL',
    'Ic-bl': 'Ic BL',
    'BL-Ic': 'Ic BL',
    'c-BL': 'Ic BL',
    'Ic/Ic-bl': 'Ic BL',
    'Ic-hyp': 'Ic BL',
    'BL': 'Ic BL',
    'Ic-lum': 'Ic-lum',
    'PISN': 'PISN',
    'Computed-PISN': 'PISN',
    # 'Unconf': '',
    # 'unconf': '',
    # 'Uncomfirmed': '',
    # 'uncomfirmed': '',
    # 'Unk': '',
    # 'unknown': '',
    # 'Unknown': '',
    # 'SN': '',
    # '?': '',
    # '-': '',
    # 'transient': '',
    # 'Transient': '',
    # 'Not': '',
    # '---': '',
    # 'removed': '',
    'Nova': 'Nova',
    'Dwarf Nova': 'Nova',
    'DN': 'Nova',
    'star': 'Star',
    '*': 'Star',
    'Stellar': 'Star',
    'stellar': 'Star',
    'M-star': 'Star',
    'Star': 'Star',
    'Steller': 'Star',
    'YSO': 'Variable Star',
    'C-star': 'Star',
    'M-Dwarf': 'Star',
    'Dwarf': 'Star',
    'symbiotic star': 'Star',
    'varstar': 'Variable Star',
    'Varstar': 'Variable Star',
    'VS': 'Variable Star',
    'GV': 'Variable Star',
    'Mira': 'Variable Star',
    'Variable Star': 'Variable Star',
    'Variable': 'Variable Star',
    'LBV': 'Variable Star',
    'LPV': 'Variable Star',
    'LRV': 'Variable Star',
    'HII Region': 'HII Region',
    'HII': 'HII Region',
    'galaxy': 'Galaxy',
    'gal': 'Galaxy',
    'Gal': 'Galaxy',
    'Galaxy': 'Galaxy',
    'AGN / QSO': 'AGN',
    'AGN': 'AGN',
    'QSO': 'AGN',
    'BL Lac': 'AGN',
    'Blazar': 'AGN',
    'Blazer': 'AGN',
    'Quasar': 'AGN',
    'imposter': 'Impostor',
    'Imposter': 'Impostor',
    'impostor': 'Impostor',
    'Impostor': 'Impostor',
    'Asteroid': 'Minor Planet',
    'MP': 'Minor Planet',
    'Minor Planet': 'Minor Planet',
    'Asteroids': 'Minor Planet',
    'Mover': 'Minor Planet',
    'mover': 'Minor Planet',
    'Microlens': 'Lensing',
    'microlensing': 'Lensing',
    'TDE': 'TDE',
    'TDE?': 'TDE',
    'MS + SMBH': 'MS + SMBH',
    'WD + IMBH': 'WD + IMBH',
    'Planet + WD': 'Planet + WD',
    'Low-mass TDE': 'Low-mass TDE',
    'He + SMBH': 'He + SMBH',
    'LGRB': 'LGRB',
    'SGRB': 'SGRB',
    'GRB': 'GRB',
    'Kilonova': 'Kilonova',
    'KN': 'Kilonova',
    'KilonovaCand': 'KilonovaCand',
    'GW': 'GW',
    'False': 'False',
    'Fake': 'False',
    'False Positive': 'False',
    'Erroneous Entry': 'False',
    'BadRef': 'False',
    'Other': 'Other',
    'maser': 'Other',
    'Radio': 'Other',
    'Flare': 'Other',
    'Foreground': 'Other',
    'Galactic': 'Other',
    'blue': 'Other',
    'dwarf': 'Other',
    'Pec': 'Other',
    'XRB': 'Other',
    'Jovan': 'Other',
    'c': 'Other',
    'LCH': 'Other',
    'Ii': 'Other',
    'Ia-SC': 'Other',
    'CN': 'Other',
    'CV': 'Other',
    'LSQ': 'Other'
}


"""
grouping_lists
{transient type : possible data values}
inverse of groupings
"""
grouping_lists = {
    "Candidate":     ["Candidate", "PSN", "Cand", "Stale", "CandFaint"],
    "CC":            ["CC", "CCSN", "Core-Collapse"],
    "I":             ["I"],
    "I-rapid":       ["I-rapid"],
    "Ia":            ["Ia", "Ia-norm", "Ia- norm", "Lensed SN Ia", "Computed-Ia", "Ia(N)", "Ia*"],
    "I Pec":         ["I Pec", "I pec", "I-pec", "I Pec", "I-Pec", "IPec", "Ipec"],
    "Ia Pec":        ["Ia Pec", "Ia pec", "Ia-pec", "Iapec", "IaPec", "Ia-Pec", "Iap",
                      "Ia P", "IaP", "Ia-p", "IaPec(N)", "Ia? P", "Ia-pec."],
    "Ia CSM":        ["Ia CSM", "Ia-CSM", "Ia-csm", "Ia-csm", "Ia+CSM", "Ia-CSM/IIn", "Ia-02ic-like"],
    "Ia-91bg":       ["Ia-91bg", "Ia-pec (1991bg)", "Ia-91bg-like", "Ia-91bg like", "Ia-92bg"],
    "Ia-91T":        ["Ia-91T", "Ia-pec 1991T", "Ia-91T-like", "Ia-91T like",
                      "Ia 91T-like", "Ia-T", "IaT", "Ia-91Tlike"],
    "Ia-02cx":       ["Ia-02cx", "Ia-02cx-like", "Iax", "Iax[02cx-like]", "Ib-pec/Iax"],
    "Ia-00cx":       ["Ia-00cx"],
    "Ia-99aa":       ["Ia-99aa"],
    "Ia-09dc":       ["Ia-09dc"],
    "Ia-HV":         ["Ia-HV"],
    "I-faint":       ["I-faint"],
    "Ib":            ["Ib", "Ib-norm", "Ib/IIb", "Ib-IIb", "IIb/Ib", "II/Ib"],
    "Ibn":           ["Ibn", "Ib-n", "IIn/Ibn", "Ib-n/IIb-n"],
    "Ib Pec":        ["Ib Pec", "Ib pec", "Ib-pec", "Ibpec", "IbPec", "Ib-pec/Iax"],
    "Ic":            ["Ic", "1c", "Ic/Ic-BL", "IIn/Ic", "II/Ic"],
    "Ic Pec":        ["Ic Pec", "Ic pec", "Ic-pec", "Icpec", "IcPec"],
    "Ia/c":          ["Ia/c", "Ic/Ia", "Iac", "Ia/Ic"],
    "Ia/b":          ["Ia/b"],
    "Ib/c":          ["Ib/c", "Ibc", "Ibc-norm", "Ib/c-norm", "Ib/Ic", "IIb/Ib/Ic (Ca rich)",
                      "Ib/Ic (Ca rich)", "II/Ib/c", "Ib/c-BL", "Ic/b"],
    "Ib/c Pec":      ["Ib/c Pec", "Ib/c-pec", "Ibc pec", "Ib/c pec", "Ib/cPec", "Ib/cpec",
                      "Ibc-pec", "Ib/cPe", "IIb/Ib/Ic (Ca rich)"],
    "II":            ["II", "II/IIB", "II/Ic", "II/Ib/c", "II/LBV", "II/IIb", "II/Ib"],
    "II P":          ["II P", "IIP", "IIp", "II p", "II-P", "IIP/IIL", "Computed-IIP", "II-p", "II-P/L"],
    "II Pec":        ["II Pec", "II pec", "IIpec", "II Pec", "IIPec", "II-pec", "II-Pec"],
    "II P Pec":      ["II P Pec", "IIP-pec", "IIPpec", "IIPp", "II P pec", "II-Pp."],
    "II L":          ["II L", "IIL", "II-L", "IIL/IIn", "IIP/IIL", "II-P/L"],
    "IIn":           ["IIn", "II n", "IIN", "LBV to IIn", "IIL/IIn", "IIn/Ic", "IIn/Ibn",
                      "Ia-CSM/IIn", "IIn/LBV", "II-09ip", "IIn-09ip"],
    "IIn P":         ["IIn P", "IInP"],
    "IIn Pec":       ["IIn Pec", "IIn pec", "IIn-pec", "IInPec", "IInpec", "IIn-pec/LBV"],
    "IIn L":         ["IIn L", "IInL"],
    "IIb":           ["IIb", "IIB", "II-b", "Computed-IIb", "II/IIB", "Ib/IIb",
                      "IIb/Ib/Ic (Ca rich)", "II/IIb", "Ib-IIb", "IIb/Ib", "Ib-n/IIb-n"],
    "IIb Pec":       ["IIb Pec", "IIb-pec", "IIb: pec", "IIb: Pec", "IIb pec", "IIb:pec", "IIb:Pec"],
    "IIc":           ["IIc"],
    "nIa":           ["nIa", "non-Ia", "not-Ia", "non Ia", "not Ia"],
    "SLSN":          ["SLSN", "SL", "SLSN-R", "SLSN I", "SL-I", "SLI", "SLSN-Ic", "SLSN Ic",
                      "SL-Ic", "SLIc", "SLSN-I-R", "SLSN II", "SL-II", "SLII", "SLSN-IIn",
                      "SLSN-R", "SLSN-I-R"],
    "SLSN-I":        ["SLSN-I", "SLSN I", "SL-I", "SLI", "SLSN-Ic", "SLSN Ic",
                      "SL-Ic", "SLIc", "SLSN-I-R"],
    "SLSN-II":       ["SLSN-II", "SLSN II", "SL-II", "SLII", "SLSN-IIn"],
    "SLSN-R":        ["SLSN-R", "SLSN-I-R"],
    "Ib-Ca":         ["Ib-Ca", "Ib - Ca-rich", "Ib (Ca rich)", "IIb/Ib/Ic (Ca rich)",
                      "Ib/Ic (Ca rich)", "Ca-rich", "Ib-Ca-rich"],
    "II P-97D":      ["II P-97D", "IIP-pec 1997D"],
    "Ic BL":         ["Ic BL", "Ic-broad", "Ic-BL", "Ic-bl", "BL-Ic", "c-BL", "Ic/Ic-bl",
                      "Ic/Ic-BL", "Ib/c-BL", "Ic-hyp", "BL"],
    "Ic-lum":        ["Ic-lum"],
    "PISN":          ["PISN", "Computed-PISN"],

    "":              ["Unconf", "unconf", "Uncomfirmed", "uncomfirmed", "Unk",
                      "unknown", "Unknown", "SN", "?", "-", "transient",
                      "Transient", "Not", "---", "removed"],
    "Nova":          ["Nova", "Dwarf Nova", "DN"],
    "Star":          ["star", "*", "Stellar", "stellar", "M-star", "Star",
                      "Stellar", "Steller", "YSO", "C-star", "M-Dwarf", "Dwarf", "symbiotic star"],
    "Variable Star": ["varstar", "Varstar", "VS", "GV", "Mira", "Variable Star",
                      "Variable", "Varstar", "VS", "YSO", "LBV", "LPV", "LBV to IIn",
                      "IIn/LBV", "II/LBV", "IIn-pec/LBV", "LRV"],
    "HII Region":    ["HII Region", "HII"],
    "Galaxy":        ["galaxy", "gal", "Gal", "Galaxy"],
    "AGN":           ["AGN / QSO", "AGN", "QSO", "BL Lac", "Blazar", "Blazer", "Quasar"],
    "Impostor":      ["imposter", "Imposter", "impostor", "Imposter", "Impostor"],
    "Minor Planet":  ["Asteroid", "MP", "Minor Planet", "Asteroids", "Mover", "mover"],
    "Lensing":       ["Microlens", "microlensing"],
    "TDE":           ["TDE", "TDE?", "MS + SMBH", "WD + IMBH", "Planet + WD", "Low-mass TDE", "He + SMBH"],
    "LGRB":          ["LGRB"],
    "SGRB":          ["SGRB"],
    "GRB":           ["GRB"],
    "Kilonova":      ["Kilonova", "KN", "KilonovaCand"],
    "KilonovaCand":  ["KilonovaCand"],
    "GW":            ["GW"],
    "False":         ["False", "Fake", "False Positive", "Erroneous Entry", "BadRef"],
    "Other":         ["Other", "maser", "Radio", "Flare", "Foreground", "Galactic", "blue", "dwarf",
                      "Pec", "XRB", "Jovan", "c", "LCH", "Ii", "Ia-SC", "CN", "CV", "LSQ"],
    "MS + SMBH":     ["MS + SMBH"],
    "WD + IMBH":     ["WD + IMBH"],
    "Planet + WD":   ["Planet + WD"],
    "Low-mass TDE":  ["Low-mass TDE"],
    "He + SMBH":     ["He + SMBH"]
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
    "redshift",  # float32
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
mag_cols
Column names corresponding to magntiude, which can be subtracted from one another to produce color
"""
mag_cols = ['GALEXAIS_FUV',
            'GALEXAIS_NUV',
            'GALEXAIS_FUV.b',
            'GALEXAIS_e_FUV.b',
            'GALEXAIS_NUV.b',
            'GALEXAIS_e_NUV.b',
            'GALEXAIS_FUV.a',
            'GALEXAIS_e_FUV.a',
            'GALEXAIS_NUV.a',
            'GALEXAIS_e_NUV.a',
            'GALEXAIS_FUV.4',
            'GALEXAIS_e_FUV.4',
            'GALEXAIS_NUV.4',
            'GALEXAIS_e_NUV.4',
            'GALEXAIS_FUV.6',
            'GALEXAIS_e_FUV.6',
            'GALEXAIS_NUV.6',
            'GALEXAIS_e_NUV.6',
            'AllWISE_W1mag',
            'AllWISE_e_W1mag',
            'AllWISE_W2mag',
            'AllWISE_e_W2mag',
            'AllWISE_W3mag',
            'AllWISE_e_W3mag',
            'AllWISE_W4mag',
            'AllWISE_e_W4mag',
            'PS1_gmag',
            'PS1_gmagStd',
            'PS1_b_gmag',
            'PS1_B_gmag',
            'PS1_gKmag',
            'PS1_rmag',
            'PS1_rmagStd',
            'PS1_b_rmag',
            'PS1_B_rmag',
            'PS1_rKmag',
            'PS1_imag',
            'PS1_imagStd',
            'PS1_b_imag',
            'PS1_B_imag',
            'PS1_iKmag',
            'PS1_zmag',
            'PS1_zmagStd',
            'PS1_b_zmag',
            'PS1_B_zmag',
            'PS1_zKmag',
            'PS1_ymag',
            'PS1_ymagStd',
            'PS1_b_ymag',
            'PS1_B_ymag',
            'PS1_yKmag',
            'NED_GALEX_FUV',
            'NED_GALEX_NUV',
            'NED_2MASS_J',
            'NED_2MASS_H',
            'NED_2MASS_Ks',
            'NED_SDSS_u',
            'NED_SDSS_g',
            'NED_SDSS_r',
            'NED_SDSS_i',
            'NED_SDSS_z',
            'AllWISE_Jmag',
            'AllWISE_Hmag',
            'AllWISE_Kmag'
            ]
