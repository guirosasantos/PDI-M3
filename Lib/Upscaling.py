import numpy as np
from cv2 import dnn_superres

def ApplyUpscaling(images, scale=4):
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel("C:/Users/guiro/Downloads/Python Processamento de Imagens/Trabalhos/trabalho-m3/Lib/EDSR_x4.pb")
    sr.setModel("edsr", scale)
    
    count = 1
    upscaled_images = []
    for img in images:
        print("Upscaling image " + str(count))

        # Converter a imagem para uint8 e ajustar valores para [0, 255]
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img

        # Aplicar o upscaling
        upscaled_img = sr.upsample(img_uint8)

        # Converter de volta para float32 e normalizar para [0, 1] para uso no modelo
        upscaled_img = upscaled_img.astype(np.float32) / 255.0

        upscaled_images.append(upscaled_img)
        count += 1
    return np.array(upscaled_images)