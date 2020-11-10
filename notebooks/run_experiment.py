from models.binary_model.binary_model import BinaryModel
from models.ind_model.ind_model import OvAModel
from models.multi_model.multi_model import MultiModel


mags = ["g_mag",  "r_mag", "i_mag", "z_mag", "y_mag",
        "W1_mag", "W2_mag",
        "J_mag", "K_mag", "H_mag"]

v7mags = []
v4mags = []
for i in mags:
    v4mags.append(i + "_v4")
    v7mags.append(i + "_v7")


# mags = ["g_mag",  "r_mag", "i_mag", "z_mag", "y_mag", "u_mag",
#         "W1_mag", "W2_mag", "W3_mag", "W4_mag",
#         "J_mag", "K_mag", "H_mag",
#         "NUV_mag", "FUV_mag"]


from thex_data.data_init import *
from thex_data.data_consts import ROOT_DIR, DATA_PATH

# v4
df_v4ab = collect_data(ROOT_DIR + "/../../data/catalogs/v4ab/assembled-magcols.fits")

df_v7a = collect_data(
    ROOT_DIR + "/../../data/catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits")
merged_df = df_v4ab.merge(right=df_v7a, how='inner', on=[
                          'name'], suffixes=['_v4', '_v7'])

v4_labels_list = merged_df['claimedtype_v4'].tolist()
v7_labels_list = merged_df['claimedtype_v7'].tolist()

keep_indices = []
for index, v4_label in enumerate(v4_labels_list):
    v7_label = v7_labels_list[index]
    if v7_label == v4_label:
        keep_indices.append(index)
merged_df = merged_df.loc[keep_indices]
merged_df['claimedtype'] = merged_df['claimedtype_v4']


print("V4 Model - Inner.")
v4_model = MultiModel(
    folds=40,
    min_class_size=40,
    transform_features=True,
    cols=v4mags,
    data=merged_df,
)

v4_model.run_model()
