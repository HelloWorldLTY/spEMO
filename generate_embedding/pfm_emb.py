import scanpy as sc
import squidpy as sq

adata = sc.read_h5ad("../dplfc_data/adata_visium.h5ad")

adata

adata.obsm['X_pca'] = adata.obsm['PCA'].values[:,0:20]

adata.obs['cluster'] = adata.obs['spatialLIBD']
adata.obs['cluster'] = [str(i) for i in adata.obs['cluster']]


for idx in adata.obs['sample_id'].unique():
    print(idx)
    import pandas as pd
    import numpy as np
    adata_s = adata[adata.obs['sample_id'] == str(idx)]
    image_id = pd.read_csv(f"../dplfc_data/{idx}/tissue_positions_list.txt", header=None)

    image_id.index = image_id[0]

    image_id = image_id.loc[adata_s.obs_names]

    adata_s.obsm['spatial'] = np.stack((image_id[4].values, image_id[5].values), axis=1)
    
    img = sq.im.ImageContainer(f"../dplfc_data/{idx}_full_image.tif")
    
    image_list = []

    for i in adata_s.obsm['spatial']:
        crop_center = img.crop_center(i[1], i[0], radius=112) # the shape is (y,x)
        img_array = np.array(crop_center.data.image)
        image_list.append(img_array)

    import pickle
    with open(f'../dplfc_data/visium_{idx}_112.pkl', 'wb') as f:
        pickle.dump(image_list, f)

# use GPFM as an example
from models import get_model, get_custom_transformer
from PIL import Image
import numpy as np
import PIL.Image
import torch
PIL.Image.MAX_IMAGE_PIXELS = 7793202000

model = get_model('GPFM', 0, 1)
transformer = get_custom_transformer('GPFM')
model = model.cuda()
print(transformer)
print(model)

for item in ['151507',
 '151508',
 '151509',
 '151510',
 '151669',
 '151670',
 '151671',
 '151672',
 '151673',
 '151674',
 '151675',
 '151676']:
    import pickle
    print(item)
    with open(f"../dplfc_data/visium_{item}_112.pkl", "rb") as f:
        data_all = pickle.load(f)
    feature_all = torch.zeros((len(data_all), 1024))
    count = 0
    for i in data_all:
        myarray = i[:,:,0,:]
        image = Image.fromarray(np.uint8(myarray)).convert('RGB')
        image = transformer(image)[None]
        image = image.cuda()
        with torch.inference_mode():
            feature_emb = model(image) # Extracted features (torch.Tensor) with shape [1,1024]
            feature_all[count,:] = feature_emb.cpu()
            count += 1
    torch.save(feature_all, f"./visium_{item}_allspot_gpfm_112.pkl")

