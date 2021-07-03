import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('typing-extensions')
install('imageio')

# # Try and get keras plot to work
# !pip install -q pydot
# !pip install -q pydotplus
# !apt-get install -q graphviz
# !pip install -q tensorflow
# !pip install -q tensorflow_addons

print("\n... OTHER IMPORTS STARTING ...\n")
print("\n\tVERSION INFORMATION")

# Machine Learning and Data Science Imports
import tensorflow_addons as tfa; print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}");
import tensorflow as tf; print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}");
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import scipy; print(f"\t\t– SCIPY VERSION: {scipy.__version__}");

# Built In Imports
from collections import Counter
from datetime import datetime
import multiprocessing
from glob import glob
import warnings
import requests
import imageio
import IPython
import urllib
import zipfile
import pickle
import random
import shutil
import string
import math
import tqdm
import time
import gzip
import io
import os
import gc
import re

# Visualization Imports
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import plotly.express as px
import seaborn as sns
from PIL import Image
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
import plotly
import PIL
import cv2
import ast

# PRESETS
LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x:LBL_NAMES[x] for x in np.arange(19)}
INT_2_STR_LOWER = {k:v.lower().replace(" ", "_") for k,v in INT_2_STR.items()}
STR_2_INT_LOWER = {v:k for k,v in INT_2_STR_LOWER.items()}
STR_2_INT = {v:k for k,v in INT_2_STR.items()}
FIG_FONT = dict(family="Helvetica, Arial", size=14, color="#7f7f7f")
LABEL_COLORS = [px.colors.label_rgb(px.colors.convert_to_RGB_255(x)) for x in sns.color_palette("Spectral", len(LBL_NAMES))]
LABEL_COL_MAP = {str(i):x for i,x in enumerate(LABEL_COLORS)}

print("\n\n... IMPORTS COMPLETE ...\n")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    print(gpus)
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
    
 # Define the path to the root data directory
ROOT_DIR = "/usr/xtmp/hs285/"

# Define the path to the competition data directory
COMP_DIR = os.path.join("/usr/xtmp/jwl50/hpa-kaggle/")

# Define path to the filtered TP IDs for each class
#PKL_DIR = os.path.join(ROOT_DIR, "hpa-rule-based-single-cell-filtering")

# Define the paths to the training tiles for the cell-wise classification dataset
# RED_TILE_DIR = os.path.join(ROOT_DIR, "red_channel_tiles/red_tiles")
# GREEN_TILE_DIR = os.path.join(ROOT_DIR, "green_channel_tiles/green_tiles")
# BLUE_TILE_DIR = os.path.join(ROOT_DIR, "blue_channel_tiles/blue_tiles")
# YELLOW_TILE_DIR = os.path.join(ROOT_DIR, "yellow_channel_tiles/yellow_tiles")
RED_TILE_DIR = os.path.join(ROOT_DIR, "combined_data_channel_tiles/red_tiles")
GREEN_TILE_DIR = os.path.join(ROOT_DIR, "combined_data_channel_tiles/green_tiles")
BLUE_TILE_DIR = os.path.join(ROOT_DIR, "combined_data_channel_tiles/blue_tiles")
YELLOW_TILE_DIR = os.path.join(ROOT_DIR, "combined_data_channel_tiles/yellow_tiles")

# Define the paths to the training and testing tfrecord and 
# image folders respectively for the competition data
TRAIN_IMG_DIR = os.path.join(COMP_DIR, "train")
TRAIN_TFREC_DIR = os.path.join(COMP_DIR, "train_tfrecords")
TEST_IMG_DIR = os.path.join(COMP_DIR, "test")
TEST_TFREC_DIR = os.path.join(COMP_DIR, "test_tfrecords")

# Capture all the relevant full image paths for the competition dataset
TRAIN_IMG_PATHS = sorted([os.path.join(TRAIN_IMG_DIR, f_name) for f_name in os.listdir(TRAIN_IMG_DIR)])
TEST_IMG_PATHS = sorted([os.path.join(TEST_IMG_DIR, f_name) for f_name in os.listdir(TEST_IMG_DIR)])
print(f"\n... Recall that 4 training images compose one example (R,G,B,Y) ...")
print(f"... \t– i.e. The first 4 training files are:")
for path in [x.rsplit('/',1)[1] for x in TRAIN_IMG_PATHS[:4]]: print(f"... \t\t– {path}")
print(f"\n... The number of training images is {len(TRAIN_IMG_PATHS)} i.e. {len(TRAIN_IMG_PATHS)//4} 4-channel images ...")
print(f"... The number of testing images is {len(TEST_IMG_PATHS)} i.e. {len(TEST_IMG_PATHS)//4} 4-channel images ...")

# Capture all the relevant full tfrec paths
TRAIN_TFREC_PATHS = sorted([os.path.join(TRAIN_TFREC_DIR, f_name) for f_name in os.listdir(TRAIN_TFREC_DIR)])
TEST_TFREC_PATHS = sorted([os.path.join(TEST_TFREC_DIR, f_name) for f_name in os.listdir(TEST_TFREC_DIR)])
print(f"\n... The number of training tfrecord files is {len(TRAIN_TFREC_PATHS)} ...")
print(f"... The number of testing tfrecord files is {len(TEST_TFREC_PATHS)} ...\n")

# Random Useful Info
ORIGINAL_DIST_MAP = {0: 37472, 1: 4845, 2: 12672, 3: 12882, 4: 17527, 5: 15337, 6: 10198, 7: 18825, 8: 11194, 9: 5322, 10: 7789, 11: 10, 12: 13952, 13: 21168, 14: 27494, 15: 2275, 16: 22738, 17: 5619, 18: 952}

# Define paths to the relevant csv files
TRAIN_CSV = os.path.join(ROOT_DIR, "updated_train.csv")

print("\n... Loading massive train dataframe ...\n")
# Create the relevant dataframe objects
train_df = pd.read_csv(TRAIN_CSV)
# train_df.mask_rles = train_df.mask_rles.apply(lambda x: ast.literal_eval(x))
# train_df.mask_bboxes = train_df.mask_bboxes.apply(lambda x: ast.literal_eval(x))
    
print("\n\nTRAIN DATAFRAME\n\n")
display(train_df.head(3))

def load_image_scaled(img_id, img_dir, img_size=512, load_style="tf"):
    """ Load An Image Using ID and Directory Path - Composes 4 Individual Images """
    def __load_with_tf(path, img_size=512):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        return tf.image.resize(img, (img_size, img_size))[..., 0]
    
    def __load_with_pil(path, img_size=512):
        img = Image.open(path)
        img = img.resize((img_size, img_size))
        return np.asarray(img)
    
    def __load_with_cv2(path, img_size=512):
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (img_size, img_size))
        return img
        
    if load_style is "tf":
        load_fn = __load_with_tf
    elif load_style is "PIL":
        load_fn = __load_with_pil
    else:
        load_fn = __load_with_cv2
    
    return np.stack(
        [np.asarray(load_fn(os.path.join(img_dir, img_id+f"_{c}.png"), img_size)/255.) for c in ["red", "yellow", "blue"]], axis=2
    )


def decode_img(img, img_size=(224,224)):
    """TBD"""
    
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)

    # resize the image to the desired size
    return tf.cast(tf.image.resize(img, img_size), tf.uint8)


def get_color_path_maps(color_dirs, tp_id_map):
    c_p_maps = [{k:[] for k in INT_2_STR.keys()} for _ in range(len(color_dirs))]
    color_d_paths = [
        [d_path for d_path in os.listdir(color_dir) if d_path.endswith("_256")] \
        for color_dir in color_dirs
    ]
    for c in tqdm(color_d_paths[0], total=len(color_d_paths[0])):
        
        # Get class stuff
        cls = c.split("_", 1)[1].rsplit("_",1)[0]
        cls_idx = STR_2_INT_LOWER[cls]
        
        # Get the relevant color directories
        c_dirs = [
            os.path.join(color_dir, c.replace("red", clr)) \
            for clr, color_dir in zip(["red", "green", "blue", "yellow"], color_dirs)
        ]

        # Update map
        for f_name in tqdm(os.listdir(c_dirs[0]), total=len(os.listdir(c_dirs[0]))):
            # get the relevant full paths
            full_paths = [os.path.join(c_dir, f_name.replace("red", clr)) for clr, c_dir in zip(["red", "green", "blue", "yellow"], c_dirs)]
            if tp_id_map==None:
                for c_p_map, full_path in zip(c_p_maps, full_paths):
                    c_p_map[cls_idx].append(full_path)
            elif (f_name.endswith(".png") and ("negative" in full_paths[0] or f_name.rsplit("_", 1)[0] in tp_id_map[cls_idx])):
                for c_p_map, full_path in zip(c_p_maps, full_paths):
                    c_p_map[cls_idx].append(full_path)
            else:
                for c_p_map, full_path in zip(c_p_maps, full_paths):
                    c_p_map[STR_2_INT["Negative"]].append(full_path)
    return [{k:sorted(v) for k,v in c_p_map.items()} for c_p_map in c_p_maps]


def get_tp_id_map(pkl_dir):
    """ TBD """
    # Capture all relevant paths
    pkl_paths = [
        os.path.join(pkl_dir, f_name) \
        for f_name in os.listdir(pkl_dir) \
        if f_name.endswith(".pkl")
    ]
    
    # REMOVE AFTER UPDATING CLASSBASED NOTEBOOK
    pkl_paths.append("/kaggle/input/tmp-intermediate-filaments-pkl-file/intermediate_filaments_tp_list.pkl")
    
    # Initialize
    tp_id_map = {}
    for path in pkl_paths:
        class_id = STR_2_INT_LOWER[path.rsplit("/", 1)[1].replace("_tp_list.pkl", "")]
        with open(path, "rb") as f:
            tp_id_map[class_id] = pickle.load(f)
    return tp_id_map

    
def plot_rgb(arr, figsize=(12,12)):
    """ Plot 3 Channel Microscopy Image """
    plt.figure(figsize=figsize)
    plt.title(f"RGB Composite Image", fontweight="bold")
    plt.imshow(arr)
    plt.axis(False)
    plt.show()    

    
def convert_rgby_to_rgb(arr):
    """ Convert a 4 channel (RGBY) image to a 3 channel RGB image.
    
    Advice From Competition Host/User: lnhtrang

    For annotation (by experts) and for the model, I guess we agree that individual 
    channels with full range px values are better. 
    In annotation, we toggled the channels. 
    For visualization purpose only, you can try blending the channels. 
    For example, 
        - red = red + yellow
        - green = green + yellow/2
        - blue=blue.
        
    Args:
        arr (numpy array): The RGBY, 4 channel numpy array for a given image
    
    Returns:
        RGB Image
    """
    
    rgb_arr = np.zeros_like(arr[..., :-1])
    rgb_arr[..., 0] = arr[..., 0]
    rgb_arr[..., 1] = arr[..., 1]+arr[..., 3]/2
    rgb_arr[..., 2] = arr[..., 2]
    
    return rgb_arr
    
    
def plot_ex(arr, figsize=(20,6), title=None, plot_merged=True, rgb_only=False):
    """ Plot 4 Channels Side by Side """
    if plot_merged and not rgb_only:
        n_images=5 
    elif plot_merged and rgb_only:
        n_images=4
    elif not plot_merged and rgb_only:
        n_images=4
    else:
        n_images=3
    plt.figure(figsize=figsize)
    if type(title) == str:
        plt.suptitle(title, fontsize=20, fontweight="bold")

    for i, c in enumerate(["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus", "Yellow – Endoplasmic Reticulum"]):
        if not rgb_only:
            ch_arr = np.zeros_like(arr[..., :-1])        
        else:
            ch_arr = np.zeros_like(arr)
        if c in ["Red Channel – Microtubles", "Green Channel – Protein of Interest", "Blue - Nucleus"]:
            ch_arr[..., i] = arr[..., i]
        else:
            if rgb_only:
                continue
            ch_arr[..., 0] = arr[..., i]
            ch_arr[..., 1] = arr[..., i]
        plt.subplot(1,n_images,i+1)
        plt.title(f"{c.title()}", fontweight="bold")
        plt.imshow(ch_arr)
        plt.axis(False)
        
    if plot_merged:
        plt.subplot(1,n_images,n_images)
        
        if rgb_only:
            plt.title(f"Merged RGB", fontweight="bold")
            plt.imshow(arr)
        else:
            plt.title(f"Merged RGBY into RGB", fontweight="bold")
            plt.imshow(convert_rgby_to_rgb(arr))
        plt.axis(False)
        
    plt.tight_layout(rect=[0, 0.2, 1, 0.97])
    plt.show()
    
    
def flatten_list_of_lists(l_o_l):
    return [item for sublist in l_o_l for item in sublist]


def create_input_list(crp, cgp, cbp, cyp, shuffle=True, val_split=0.025):
    lbl_arr = flatten_list_of_lists([[k,]*len(v) for k, v in sorted(crp.items())])
    cr_arr = flatten_list_of_lists([v for k,v in sorted(crp.items())])
    cg_arr = flatten_list_of_lists([v for k,v in sorted(cgp.items())])
    cb_arr = flatten_list_of_lists([v for k,v in sorted(cbp.items())])
    cy_arr = flatten_list_of_lists([v for k,v in sorted(cyp.items())])
    
    val_samps = random.sample(range(len(lbl_arr)), int(len(lbl_arr)*val_split))
    train_samps = list(set(range(len(lbl_arr))) - set(val_samps))
    
    if val_split is not None:
        val_lbl_arr = []
        [val_lbl_arr.append(lbl_arr[i]) for i in val_samps]
        train_lbl_arr = []
        [train_lbl_arr.append(lbl_arr[i]) for i in train_samps]
        
        val_cr_arr = []
        [val_cr_arr.append(cr_arr[i]) for i in val_samps]
        train_cr_arr = []
        [train_cr_arr.append(cr_arr[i]) for i in train_samps]
        
        val_cg_arr = []
        [val_cg_arr.append(cg_arr[i]) for i in val_samps]
        train_cg_arr = []
        [train_cg_arr.append(cg_arr[i]) for i in train_samps]
        
        val_cb_arr = []
        [val_cb_arr.append(cb_arr[i]) for i in val_samps]
        train_cb_arr = []
        [train_cb_arr.append(cb_arr[i]) for i in train_samps]

        val_cy_arr = []
        [val_cy_arr.append(cy_arr[i]) for i in val_samps]
        train_cy_arr = []
        [train_cy_arr.append(cy_arr[i]) for i in train_samps]
        
    if shuffle:
        to_shuffle = list(zip(train_cr_arr, train_cg_arr, train_cb_arr, train_cy_arr, train_lbl_arr))
        random.shuffle(to_shuffle)
        train_cr_arr, train_cg_arr, train_cb_arr, train_cy_arr, train_lbl_arr = zip(*to_shuffle)
        
        if val_split is not None:
            val_to_shuffle = list(zip(val_cr_arr, val_cg_arr, val_cb_arr, val_cy_arr, val_lbl_arr))
            random.shuffle(val_to_shuffle)
            val_cr_arr, val_cg_arr, val_cb_arr, val_cy_arr, val_lbl_arr = zip(*val_to_shuffle)
    
    if val_split is None:
        return list(cr_arr), list(cg_arr), list(cb_arr), list(cy_arr), list(lbl_arr)
    else:
        return (list(train_cr_arr), list(train_cg_arr), list(train_cb_arr), list(train_cy_arr), list(train_lbl_arr)), \
               (list(val_cr_arr), list(val_cg_arr), list(val_cb_arr), list(val_cy_arr), list(val_lbl_arr))


def get_class_wts(single_ch_paths, n_classes=19, exclude_mitotic=True, multiplier=10, return_counts=False):
    """ TBD """
    # Get class counts
    class_counts = {c_idx:len(single_ch_paths[c_idx]) for c_idx in range(n_classes)}

    # Exclude mitotic spindle
    if exclude_mitotic:
        real_min_count = list(sorted(class_counts.values(), reverse=True))[-2]
    else:
        real_min_count = list(sorted(class_counts.values(), reverse=True))[-1]

    # Calculate weights
    class_wts = {k:min(1, multiplier*(real_min_count/v)) for k,v in class_counts.items()}

    if exclude_mitotic:
        # Manually adjust mitotic spindle to a more appropriate value
        class_wts[min(class_counts, key=class_counts.get)] = 1.0

    if return_counts:
        return class_wts, class_counts
    else:
        return class_wts
      
TILE_DIRS = [RED_TILE_DIR, GREEN_TILE_DIR, BLUE_TILE_DIR, YELLOW_TILE_DIR]
# TP_ID_MAP = get_tp_id_map(PKL_DIR)

# Define the paths to the training files for the tile dataset as a map from class index to list of paths
RED_FILE_MAP, GREEN_FILE_MAP, BLUE_FILE_MAP, YELLOW_FILE_MAP = \
    get_color_path_maps(TILE_DIRS, None)

VAL_FRAC = 0.075

# red_inputs, green_inputs, blue_inputs, yellow_inputs, labels
train_inputs, val_inputs = create_input_list(
    RED_FILE_MAP, 
    GREEN_FILE_MAP, 
    BLUE_FILE_MAP, 
    YELLOW_FILE_MAP, 
    shuffle=True,
    val_split=VAL_FRAC,
)

# i = 0
# for label in train_inputs[4]:
#     protein = cv2.imread(train_inputs[1][i],0)
    
#     #Nucleoplasm
#     if label == 0:
#         threshold = 15
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-10)
#         intensity = np.mean(thresh)
# #         if intensity < 1.8:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Nuclear membrane
#     if label == 1:
#         threshold = 0.2
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-40)
#         intensity = np.mean(thresh)
# #         if intensity < 0.2:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Nucleoli
#     if label == 2:
#         threshold = 17
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 2.1:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Nucleoli fibrilar center
#     if label == 3:
#         threshold = 10
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 1.9:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Nuclear speckles
#     if label == 4:
#         threshold = 1
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 0.1:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Nuclear bodies
#     if label == 5:
#         threshold = 0.5
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 0.08:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Endoplasmic reticulum
#     if label == 6:
#         threshold = 0.8
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-15)
#         intensity = np.mean(thresh)
# #         if intensity < 0.7:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
    
#     #Golgi apparatus
#     if label == 7:
#         threshold = 1
#         dst = cv2.medianBlur(protein,13)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-15)
#         intensity = np.mean(thresh)
# #         if intensity < 0.2:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Intermediate filaments
#     if label == 8:
#         threshold = 26
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-5)
#         intensity = np.mean(thresh)
# #         if intensity < 2.5:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Actin
    
#     #Microtubules
    
#     #Mitotic spindle
            
    
#     #Centrosome
#     if label == 12:
#         threshold = 10
#         dst = cv2.medianBlur(protein, 5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 1.2:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Plasma membrane
#     if label == 13:
#         threshold = 1
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-20)
#         intensity = np.mean(thresh)
# #         if intensity < 0.6:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Mitochondria
#     if label == 14:
#         threshold = 10
#         dst = cv2.GaussianBlur(protein,(5,5), sigmaX = 1 )
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20) 
#         intensity = np.mean(thresh)
# #         if intensity < 2.2:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Aggresome
#     if label == 15:
#         threshold = 0.1
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-70)
#         intensity = np.mean(thresh)
# #         if intensity < 0.1:
# #             train_inputs[4][i] = 18
# #             i = i + 1
        
#         if intensity <= 0.05:
#             train_inputs[4][i] = 18
#             i = i +1
#         elif 0.05 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Cytosol
#     if label == 16:
#         threshold = 1
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,501,-15)
#         intensity = np.mean(thresh)
# #         if intensity < 0.8:
# #             train_inputs[4][i] = 18
        
#         if intensity <= 0.1:
#             train_inputs[4][i] = 18
#             i = i + 1
#         elif 0.1 < intensity < threshold:
#             del train_inputs[0][i]
#             del train_inputs[1][i]
#             del train_inputs[2][i]
#             del train_inputs[3][i]
#             del train_inputs[4][i]
#         else:
#             i = i + 1
            
#     #Vesicles and punctate cytosolic patterns
    
#     #Negative
# #     i = i + 1
#     print('Current Cell Number {}'.format(i), end = '\r')

### POTENTIAL LOSS FN ###
# def macro_double_soft_f1(y, y_hat):
#     """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
#     Use probability values instead of binary predictions.
#     This version uses the computation of soft-F1 for both positive and negative class for each label.
    
#     Args:
#         y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
#     Returns:
#         cost (scalar Tensor): value of the cost function for the batch
#     """
#     y = tf.cast(y, tf.float32)
#     y_hat = tf.cast(y_hat, tf.float32)
#     tp = tf.reduce_sum(y_hat * y, axis=0)
#     fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
#     fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
#     tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
#     soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
#     soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
#     cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
#     cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
#     cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
#     macro_cost = tf.reduce_mean(cost) # average on all labels
#     return macro_cost

### POTENTIAL LOSS FN ###
# def macro_soft_f1(y, y_hat):
#     """Compute the macro soft F1-score as a cost.
#     Average (1 - soft-F1) across all labels.
#     Use probability values instead of binary predictions.
    
#     Args:
#         y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
#     Returns:
#         cost (scalar Tensor): value of the cost function for the batch
#     """
    
#     y = tf.cast(y, tf.float32)
#     y_hat = tf.cast(y_hat, tf.float32)
#     tp = tf.reduce_sum(y_hat * y, axis=0)
#     fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
#     fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
#     soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
#     cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
#     macro_cost = tf.reduce_mean(cost) # average on all labels
    
#     return macro_cost

### POTENTIAL METRIC ###
# def macro_f1(y, y_hat, thresh=0.5):
#     """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
#     Args:
#         y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
#         y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
#         thresh: probability value above which we predict positive
        
#     Returns:
#         macro_f1 (scalar Tensor): value of macro F1 for the batch
#     """
#     y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
#     tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
#     fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
#     fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
#     f1 = 2*tp / (2*tp + fn + fp + 1e-16)
#     macro_f1 = tf.reduce_mean(f1)
#     return macro_f1


# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.
N_EPOCHS=10
LR_START = 0.0005
LR_MAX = 0.0011
LR_MIN = 0.0005
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 2
LR_EXP_DECAY = 0.75

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

# VIEW SCHEDULE
rng = [i for i in range(N_EPOCHS)]
y = [lrfn(x) for x in rng]

plt.figure(figsize=(10,4))
plt.plot(rng, y)
plt.title("CUSTOM LR SCHEDULE", fontweight="bold")
plt.show()

print(f"Learning rate schedule: {y[0]:.3g} to {max(y):.3g} to {y[-1]:.3g}")

#PARAMS
MODEL_CKPT_DIR = os.path.join(ROOT_DIR, "models/public_ebnet_b2_wdensehead")
DROP_YELLOW = True
NO_NEG_CLASS = False

if NO_NEG_CLASS:
    class_wts = {k:v for k,v in class_wts.items() if k!=18}
    class_cnts = {k:v for k,v in class_cnts.items() if k!=18}
    n_classes = 18
else:
    n_classes=19
    
BATCH_SIZE=32
OPTIMIZER = tf.keras.optimizers.Adam(lr=LR_START)
LOSS_FN = "binary_crossentropy"
SHUFF_BUFF = 500


# AUTO-CALCULATED
N_EX = len(RED_FILE_MAP[0])
N_VAL = int(VAL_FRAC*N_EX)
N_TRAIN = N_EX-N_VAL

if not os.path.isdir(MODEL_CKPT_DIR):
    os.makedirs(MODEL_CKPT_DIR, exist_ok=True)
    
print(f"{N_TRAIN:<7} TRAINING EXAMPLES")
print(f"{N_VAL:<7} VALIDATION EXAMPLES")

# TRAIN DATASET
train_path_ds = tf.data.Dataset.zip(
    tuple([tf.data.Dataset.from_tensor_slices(input_ds) for input_ds in train_inputs])
)

# VALIDATION DATASET
val_path_ds = tf.data.Dataset.zip(
    tuple([tf.data.Dataset.from_tensor_slices(input_ds) for input_ds in val_inputs])
)

print(f"\n ... THERE ARE {N_EX} CELL TILES IN OUR FULL DATASET ... ")
print(f" ... THERE ARE {N_TRAIN} CELL TILES IN OUR TRAIN DATASET ... ")
print(f" ... THERE ARE {N_VAL} CELL TILES IN OUR VALIDATION DATASET ... \n")

print(train_path_ds)

for a,b,c,d,e in train_path_ds.take(1): 
    print(f"\tRed Path      --> {a}\n\t" \
          f"Green Path    --> {b}\n\t" \
          f"Blue Path     --> {c}\n\t" \
          f"Yellow Path   --> {d}\n\t" \
          f"Example Label --> {e} ({INT_2_STR[e.numpy()]})\n")
    
   def preprocess_path_ds(rp, gp, bp, yp, lbl, img_size=(224,224), combine=True, drop_yellow=True, no_neg=True):
    """ TBD """
    
    # Adjust class output expectation
    if no_neg:
        if lbl==18:
            lbl_arr = tf.zeros((18,), dtype=tf.uint8)
        else:
            lbl_arr = tf.one_hot(lbl, 18, dtype=tf.uint8)
    else:
        lbl_arr = tf.one_hot(lbl, 19, dtype=tf.uint8)
    
    ri = decode_img(tf.io.read_file(rp), img_size)
    gi = decode_img(tf.io.read_file(gp), img_size)
    bi = decode_img(tf.io.read_file(bp), img_size)

    if combine and drop_yellow:
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0]], axis=-1), lbl_arr
    elif combine:
        yi = decode_img(tf.io.read_file(yp), img_size)
        return tf.stack([ri[..., 0], gi[..., 0], bi[..., 0], yi[..., 0]], axis=-1), lbl_arr
    elif drop_yellow:
        return ri, gi, bi, lbl_arr
    else:
        yi = decode_img(tf.io.read_file(yp), img_size)
        return ri, gi, bi, yi, lbl_arr
    

def augment(img_batch, lbl_batch):
    # SEEDING & KERNEL INIT
    K = tf.random.uniform((1,), minval=0, maxval=4, dtype=tf.dtypes.int32)[0]
    
    img_batch = tf.image.random_flip_left_right(img_batch)
    img_batch = tf.image.random_flip_up_down(img_batch)
    img_batch = tf.image.rot90(img_batch, K)
    
    img_batch = tf.image.random_saturation(img_batch, 0.85, 1.15)
    img_batch = tf.image.random_brightness(img_batch, 0.1)
    img_batch = tf.image.random_contrast(img_batch, 0.85, 1.15)

    return img_batch, lbl_batch
  train_ds = train_path_ds.map(
    lambda r,g,b,y,l: preprocess_path_ds(r,g,b,y,l, drop_yellow=DROP_YELLOW, no_neg=NO_NEG_CLASS), 
    num_parallel_calls=tf.data.AUTOTUNE
)
  
  TRAIN_CACHE_DIR = os.path.join(ROOT_DIR,"train_cache")
VAL_CACHE_DIR = os.path.join(ROOT_DIR,"val_cache")

if not os.path.isdir(TRAIN_CACHE_DIR):
    os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
if not os.path.isdir(VAL_CACHE_DIR):
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)

train_ds = train_path_ds.map(
    lambda r,g,b,y,l: preprocess_path_ds(r,g,b,y,l, drop_yellow=DROP_YELLOW, no_neg=NO_NEG_CLASS), 
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_path_ds.map(
    lambda r,g,b,y,l: preprocess_path_ds(r,g,b,y,l, drop_yellow=DROP_YELLOW, no_neg=NO_NEG_CLASS), 
    num_parallel_calls=tf.data.AUTOTUNE
)

# VISUALIZE EXAMPLES
print("\n\t\t... TRAIN EXAMPLES ...\n")
for x,y in train_ds.take(3):
    if y.numpy().sum()==0:
        title_str = INT_2_STR[18]
    else:
        title_str = INT_2_STR[np.argmax(y.numpy())]
    plot_ex(x.numpy(), title=f"{title_str}", rgb_only=DROP_YELLOW)

print("\n\t\t... VAL EXAMPLES ...\n")
for x,y in val_ds.take(3):
    if y.numpy().sum()==0:
        title_str = INT_2_STR[18]
    else:
        title_str = INT_2_STR[np.argmax(y.numpy())]
    plot_ex(x.numpy(), title=f"{title_str}", rgb_only=DROP_YELLOW)
    
train_ds = train_ds.cache(TRAIN_CACHE_DIR) \
                   .shuffle(SHUFF_BUFF) \
                   .batch(BATCH_SIZE) \
                   .map(augment, num_parallel_calls=tf.data.AUTOTUNE) \
                   .prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.cache(VAL_CACHE_DIR) \
               .batch(BATCH_SIZE) \
               .prefetch(tf.data.AUTOTUNE)

def get_backbone(efficientnet_name="efficientnet_b0", input_shape=(224,224,3), include_top=False, weights="imagenet", pooling="avg"):
    if "b0" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB0(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b1" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB1(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b2" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB2(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b3" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB3(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b4" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB4(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b5" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB5(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b6" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB6(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    elif "b7" in efficientnet_name:
        eb = tf.keras.applications.EfficientNetB7(
            include_top=include_top, weights=weights, pooling=pooling, input_shape=input_shape
            )
    else:
        raise ValueError("Invalid EfficientNet Name!!!")
    return eb


def add_head_to_bb(bb, n_classes=19, dropout=0.05, head_layer_nodes=(512,)):
    x = tf.keras.layers.BatchNormalization()(bb.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    for n_nodes in head_layer_nodes:
        x = tf.keras.layers.Dense(n_nodes, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout/2)(x)
    
    output = tf.keras.layers.Dense(n_classes, activation="sigmoid")(x)
    return tf.keras.Model(inputs=bb.inputs, outputs=output)

tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    eb = get_backbone("b7")
    eb = add_head_to_bb(eb, n_classes, dropout=0.5)
    eb.compile(optimizer=OPTIMIZER, loss=LOSS_FN, metrics=["acc", tf.keras.metrics.AUC(name="auc", multi_label=True)])

#tf.keras.utils.plot_model(eb, show_shapes=True, show_dtype=True, dpi=55)

#eb = tf.keras.models.load_model("../input/hpa-cellwise-classification-training/ebnet_b2_wdensehead/ckpt-0009-0.0801.ckpt")

history = eb.fit(
    train_ds, 
    validation_data=val_ds, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_CKPT_DIR, "ckpt-{epoch:04d}-{val_loss:.4f}.ckpt"), verbose=1),
        tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
    ], 
    #class_weight=class_wts, 
    epochs=N_EPOCHS
)
#eb.save("./model_b2_224_w_densehead_classifier")
