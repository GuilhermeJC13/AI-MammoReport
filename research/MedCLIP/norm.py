from torchvision import transforms
import examples.medclip.constants as constants
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def trans(image):
    image_array = np.array(image)

    # Normalizar a imagem
    normalized_image_array = (image_array - constants.IMG_MEAN) / constants.IMG_STD

    # Para evitar valores muito grandes ou muito pequenos, podemos opcionalmente converter o array para um intervalo padrão
    # Por exemplo, convertemos de volta para o intervalo [0, 255] para visualização (opcional):
    #normalized_image_array = ((normalized_image_array - normalized_image_array.min()) / 
    #                        (normalized_image_array.max() - normalized_image_array.min())) * 255
    normalized_image_array = normalized_image_array / 255

    # Converter o array NumPy normalizado de volta para imagem PIL (e converter para uint8)
    normalized_image = Image.fromarray(normalized_image_array.astype(np.uint8))
    return normalized_image

# Carregar a imagem
image_path = 'mimic-cxr/mimic-cxr-images/p10/p10000032/s53911762/68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.jpg'
image = Image.open(image_path)

# Transformações para normalizar
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[constants.IMG_MEAN], std=[constants.IMG_STD])
])

# Aplicar a transformação
transformed_image = transform(image)
trans_image = trans(image)

# Converter tensor para numpy array
to_pil = transforms.ToPILImage()
image_pil = to_pil(transformed_image)

image.save("x_original.png")
trans_image.save("x_medclip.png", format='PNG')

# Plotar a imagem original e a imagem transformada
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#axs[0].imshow(image, cmap='gray')
#axs[0].set_title('Original')
#axs[0].axis('off')

#axs[1].imshow(trans_image, cmap='gray')
#axs[1].set_title('Transformada')
#axs[1].axis('off')

#plt.savefig("img_medclip.png")

#plt.tight_layout()
#plt.show()

# Calcular e plotar o histograma da imagem original e da transformada
#fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Histograma da imagem original
#axs[0].hist(np.array(image).flatten(), bins=256, color='black', alpha=0.7)
#axs[0].set_title('Histograma da Imagem Original')

# Histograma da imagem transformada
#axs[1].hist(transformed_image.numpy().flatten(), bins=256, color='black', alpha=0.7)
#axs[1].set_title('Histograma da Imagem Transformada')

#plt.tight_layout()
#plt.show()