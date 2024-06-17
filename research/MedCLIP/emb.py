import torch
import torchvision.transforms as transforms
from PIL import Image
import umap
import numpy as np
import matplotlib.pyplot as plt
from examples.medclip import constants
from examples.medclip import MedCLIPModel, MedCLIPVisionModelViT
from examples.medclip import MedCLIPProcessor
import os
import pandas as pd

MAX_EMBEDDINGS = 800

def define_class(row):
    for i, col in enumerate(df.columns):
        if row[col] == 1:
            return i

df = pd.read_csv('local_data/mimic-cxr-train-meta.csv')
class_list = ['Atelectasis','Cardiomegaly','Consolidation','Edema', 'Pleural Effusion']
df_class = df[(df[class_list] == 1).sum(axis=1) == 1]

new_columns = [*class_list, *["imgpath"]]

df = df_class[new_columns]
df['class'] = df.apply(define_class, axis=1)
df = df.drop(columns=class_list)
df = df.sample(frac=1, random_state=42)

# Função para extrair embeddings de uma imagem usando seu modelo
def extract_embedding(image_path, model, transform):

    processor = MedCLIPProcessor()
    image = Image.open(image_path)
    image = transform(image)
    input = processor(
        text=[""],#"There is no focal consolidation, pleural effusion or pneumothorax"], 
        images=image, 
        return_tensors="pt", 
        padding=True
        )
    embedding = model(**input)
    return embedding['img_embeds'].cpu().squeeze().detach().numpy()

image_path = df['imgpath'][0:MAX_EMBEDDINGS]

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint='checkpoints/vision_text_pretrain/78000')
model.cuda()
model.eval()
torch.manual_seed(42)

transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.25),
                transforms.ColorJitter(0.2,0.2),
                #transforms.GaussianBlur(3, (0.1, 1)),
                transforms.RandomAffine(degrees=10, scale=(0.8,1.1), translate=(0.0625,0.0625)),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[constants.IMG_MEAN],std=[constants.IMG_STD]),
                transforms.ToPILImage()
                ],
            )

embeddings = [extract_embedding(path, model, transform) for path in image_path]
embeddings_array = np.array(embeddings)
print(embeddings_array)

reducer = umap.UMAP(n_components=2, metric='euclidean',random_state=42)
embeddings_2d = reducer.fit_transform(embeddings_array)

import matplotlib.colors as mcolors

color_dict = {
    0: 'Atelectasis',
    1: 'Cardiomegaly',
    2: 'Consolidation',
    3: 'Edema',
    4: 'Pleural Effusion'
}

c = df['class'][0:MAX_EMBEDDINGS]
cores = [mcolors.to_rgba(f'C{i}') for i in c]

# Plotar os embeddings em 2D
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color=cores, edgecolor='k', s=50) #c=labels
plt.title('Visualização dos Embeddings Visuais no Espaço 2D')
plt.grid(True)

#mfcolor = [mcolors.to_rgba(f'C{i}') for i in range(5)]
# Criar uma lista de handles para a legenda
#handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mfcolor, markersize=10, label=f'{i}') for i in color_dict]
#plt.legend(handles=handles, title="Classes")

plt.show()