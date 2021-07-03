# In[1]:


from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


path = Path('/usr/xtmp/hs285')
df = pd.read_csv(path/'hpa_public_formatted.csv')
cell_dir = '/usr/xtmp/agk21/hpa-public-mask1/hpa_cell_mask'
green_image_dir = '/usr/xtmp/agk21/hpa_public_png_green'
image_dir = '/usr/xtmp/agk21/hpa_public_png1'


# In[3]:


# ROOT = '../input/hpa-single-cell-image-classification/'
# train_or_test = 'train'


# In[3]:


for i in range(len(df)):
    df.iloc[i,:].Image = df.iloc[i,:].Image.replace('https://images.proteinatlas.org/','').replace('/','-')


# In[5]:


labels = [str(i) for i in range(19)]
for x in labels: df[x] = df['Label'].apply(lambda r: int(x in r.split('|')))


# In[7]:


df["Label"] = df["Label"].str.split("|")

# class labels
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

# binarizing each label/class
for label in tqdm(class_labels):
    df[label] = df['Label'].map(lambda result: 1 if label in result else 0)

# rename column
df.columns = ['ID', 'Label', 'Cellline','Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center',
                    'Nuclear speckles', 'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus', 'Intermediate filaments',
                    'Actin filaments', 'Microtubules', 'Mitotic spindle', 'Centrosome', 'Plasma membrane', 'Mitochondria',
                    'Aggresome', 'Cytosol', 'Vesicles and punctate cytosolic patterns', 'Negative']
df['lengths'] = df['Label'].str.len()


# In[8]:


df = df[(df['lengths'] == 1)].reset_index(drop=True)


# In[7]:


# dfs_0 = df[df['Label'] == '0'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_1 = df[df['1'] == 1].sample(n=4, random_state=42).reset_index(drop=True)
# dfs_1u = df[df['Label'] == '1'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_2 = df[df['Label'] == '2'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_3 = df[df['Label'] == '3'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_4 = df[df['Label'] == '4'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_5 = df[df['Label'] == '5'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_6 = df[df['6'] == 1].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_7 = df[df['Label'] == '7'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_8 = df[df['Label'] == '8'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_9 = df[df['9'] == 1].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_9u = df[df['Label'] == '9'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_10 = df[df['10'] == 1].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_10u = df[df['Label'] == '10'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_11 = df[df['11'] == 1].reset_index(drop=True)
# dfs_12 = df[df['Label'] == '12'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_13 = df[df['Label'] == '13'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_14 = df[df['Label'] == '14'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_15 = df[df['15'] == 1].reset_index(drop=True)
# dfs_16 = df[df['Label'] == '16'].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_17 = df[df['17'] == 1].sample(n=1, random_state=42).reset_index(drop=True)
# dfs_18 = df[df['18'] == 1].reset_index(drop=True)
# dfs_ = [dfs_0, dfs_1, dfs_1u, dfs_2, dfs_3, dfs_4, dfs_5, dfs_6, dfs_7, dfs_8, dfs_9, dfs_9u, dfs_10, dfs_10u,
#         dfs_11, dfs_12, dfs_13, dfs_14, dfs_15, dfs_16, dfs_17, dfs_18]


# In[8]:


# dfs = pd.concat(dfs_, ignore_index=True)
# dfs.drop_duplicates(inplace=True, ignore_index=True)
# len(dfs)


# In[9]:


# unique_counts = {}
# for lbl in labels:
#     unique_counts[lbl] = len(dfs[dfs.Label == lbl])

# full_counts = {}
# for lbl in labels:
#     count = 0
#     for row_label in dfs['Label']:
#         if lbl in row_label.split('|'): count += 1
#     full_counts[lbl] = count
    
# counts = list(zip(full_counts.keys(), full_counts.values(), unique_counts.values()))
# counts = np.array(sorted(counts, key=lambda x:-x[1]))
# counts = pd.DataFrame(counts, columns=['label', 'full_count', 'unique_count'])
# counts.set_index('label').T


# In[9]:


def get_cropped_cell(img, msk):
    bmask = msk.astype(int)[...,None]
    masked_img = img * bmask
    true_points = np.argwhere(bmask)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = masked_img[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
    return cropped_arr


# In[10]:


def get_stats(cropped_cell):
    x = (cropped_cell/255.0).reshape(-1,3).mean(0)
    x2 = ((cropped_cell/255.0)**2).reshape(-1,3).mean(0)
    return x, x2


# In[11]:


def read_img(image_id, color, image_size=None):
    if color == "green":
        filename = f'{green_image_dir}/{image_id}_{color}.png'
    else:
        filename = f'{image_dir}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img


# In[11]:


# image_id = df.iloc[8].Image
# #cell_mask = np.load(f'{cell_dir}/{image_id}.npz')['arr_0']
# green = read_img(image_id, "green", None)
# dst = cv2.medianBlur(protein,5)
# thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,151,-10)


# In[60]:


images = set()
for filename in os.listdir(cell_dir):
    image_id = filename.replace('.npz','')
    images.add(image_id)
images = np.array(list(images))


# In[62]:


x_tot,x2_tot = [],[]
lbls = []
num_files = len(df)
all_cells = []



for image in tqdm(images):
    idx = df.loc[df['ID'] == image].index[0]
    image_id = df.iloc[idx].ID
    image_label = df.iloc[idx].Label
    cell_mask = np.load(f'{cell_dir}/{image_id}.npz')['arr_0']
    red = read_img(image_id, "red", None)
    protein = read_img(image_id, "green", None)
    blue = read_img(image_id, "blue", None)
    #yellow = read_img(image_id, "yellow", train_or_test, image_size)
    stacked_image = np.transpose(np.array([blue, protein, red]), (1,2,0))
#     #Nucleoplasm
#     if image_label == ['0']:
#         threshold = 15
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,151,-10)

#     #Nuclear membrane
#     if image_label == ['1']:
#         threshold = 0.2
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,75,-40)

#     #Nucleoli
#     if image_label == ['2']:
#         threshold = 17
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20)

#     #Nucleoli fibrilar center
#     if image_label == ['3']:
#         threshold = 10
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20)

#     #Nuclear speckles
#     if image_label == ['4']:
#         threshold = 1
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-20)

#     #Nuclear bodies
#     if image_label == ['5']:
#         threshold = 0.5
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-20)

#     #Endoplasmic reticulum
#     if image_label == ['6']:
#         threshold = 0.8
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,61,-15)

#     #Golgi apparatus
#     if image_label == ['7']:
#         threshold = 1
#         dst = cv2.medianBlur(protein,13)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-15)

#     #Intermediate filaments
#     if image_label == ['8']:
#         threshold = 26
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,551,-5)

#     #Actin
#     if image_label == ['9']:
#         threshold = 0
#         thresh = protein

#     #Microtubules
#     if image_label == ['10']:
#         threshold = 0
#         thresh = protein

#     #Mitotic spindle
#     if image_label == ['11']:
#         threshold = 0
#         thresh = protein

#     #Centrosome
#     if image_label == ['12']:
#         threshold = 10
#         dst = cv2.medianBlur(protein, 5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,105,-20)

#     #Plasma membrane
#     if image_label == ['13']:
#         threshold = 1
#         dst = cv2.medianBlur(protein,7)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,-20)

#     #Mitochondria
#     if image_label == ['14']:
#         threshold = 10
#         dst = cv2.GaussianBlur(protein,(5,5), sigmaX = 1 )
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,501,-20) 

#     #Aggresome
#     if image_label == ['15']:
#         threshold = 0.2
#         dst = cv2.medianBlur(protein,5)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,105,-70)

#     #Cytosol
#     if image_label == ['16']:
#         threshold = 1
#         dst = cv2.medianBlur(protein,11)
#         thresh = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,61,-15)

#     #Vesicles and punctate cytosolic patterns
#     if image_label == ['17']:
#         threshold = 0
#         thresh = protein

#     #Negative
#     if image_label == ['18']:
#         threshold = 0
#         thresh = protein
    for cell in range(1, np.max(cell_mask) + 1):
        cell_label = image_label
        bmask = (cell_mask == cell)
#         intensity = np.mean(thresh[bmask])
#         if (intensity <= 0.1) | (intensity >= threshold):
#             if intensity <= 0.1:
#                 cell_label = ['18']
        cropped_cell = get_cropped_cell(stacked_image, bmask)
#             fname = f'{image_id}_{cell}.png'
#             im = cv2.imencode('.png', cropped_cell)[1]
#             img_out.writestr(fname, im)

        x, x2 = get_stats(cropped_cell)
        all_cells.append({
            'image_id': image_id,
            'r_mean': x[0],
            'g_mean': x[1],
            'b_mean': x[2],
            'cell_id': cell,
            'cell_label': cell_label,
            'size1': cropped_cell.shape[0],
            'size2': cropped_cell.shape[1],
        })

cell_df = pd.DataFrame(all_cells)
cell_df.to_csv("/usr/xtmp/hs285/public_cropped_cells1_nothresh.csv", index=False)
print('mean:',img_avr, ', std:', img_std)
