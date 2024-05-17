import os
os.chdir('../')

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from medclip import PromptClassifier
import torch

# init models
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT, checkpoint='MedCLIP/checkpoints/vision_text_pretrain/2000')
clf = PromptClassifier(model, ensemble=True)
clf.cuda()

# prepare input image
from PIL import Image
image_list = ['MedCLIP/evaluation_data/atelectasis.jpeg', 'MedCLIP/evaluation_data/cardiomegally.png',
              'MedCLIP/evaluation_data/consolidation.jpg', 'MedCLIP/evaluation_data/edema.jpeg']
for image_path in image_list:
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # prepare input prompt texts
    from medclip.prompts import generate_chexpert_class_prompts, process_class_prompts

    cls_prompts = process_class_prompts(generate_chexpert_class_prompts(n=10))
    inputs['prompt_inputs'] = cls_prompts

    output = clf(**inputs)
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print(f'{image_path}: {output["class_names"][torch.argmax(output["logits"]).item()]}')
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print(output)