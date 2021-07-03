import pandas as pd
import numpy as np
import os
#from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt



# In[23]:


data = pd.read_csv("/usr/xtmp/hs285/cleaned_public.csv")


# In[24]:




# In[26]:
cell_dir = '/usr/xtmp/agk21/hpa-public-mask/hpa_cell_mask'
green_image_dir = '/usr/xtmp/agk21/hpa_public_png_green'
image_dir = '/usr/xtmp/agk21/hpa_public_png'

images = set()
for filename in os.listdir(cell_dir):
    image_id = filename.replace('.npz','')
    images.add(image_id)
images = np.array(list(images))

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
def get_cropped_cell(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


# In[11]:


color = 'yellow'
os.makedirs('/usr/xtmp/hs285/public_data_channel_tiles/no_thresh/yellow_tiles/')
classes = ['nucleoplasm', 'nuclear_membrane', 'nucleoli', 'nucleoli_fibrillar_center',
                    'nuclear_speckles', 'nuclear_bodies', 'endoplasmic_reticulum', 'golgi_apparatus', 'intermediate_filaments',
                    'actin_filaments', 'microtubules', 'mitotic_spindle', 'centrosome', 'plasma_membrane', 'mitochondria',
                    'aggresome', 'cytosol', 'vesicles', 'negative']
for i in range(19):
    os.makedirs('/usr/xtmp/hs285/public_data_channel_tiles/no_thresh/yellow_tiles/yellow_{}'.format(classes[i]) + '_256')


# In[ ]:





i = 0

for image in images:
    idx = data.loc[data['ID'] == image].index[0]
    label = data.iloc[idx].Label
    cell_mask = np.load(f'{cell_dir}/{image}.npz')['arr_0']
#    red = read_img(image, "red", None)
#    green = read_img(image, "green", None)
#    blue = read_img(image, "blue", None)
    yellow = read_img(image, "yellow", None)
#    stacked_image = np.transpose(np.array([blue, protein, red]), (1,2,0))
    path = "/usr/xtmp/hs285/public_data_channel_tiles/no_thresh/yellow_tiles/yellow_{}".format(classes[int(label.replace('[\'','').replace('\']',''))]) + '_256'
    for cell in range(1, np.max(cell_mask) + 1):
        bmask = (cell_mask == cell)
        cropped_cell = get_cropped_cell(yellow, mask = bmask)
        cv2.imwrite(os.path.join(path, f"{image}_{cell}_{color}.png"), cropped_cell)
        print(i, end = '\r')
        i = i + 1
    
