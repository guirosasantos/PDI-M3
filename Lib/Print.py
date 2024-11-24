import numpy as np
import matplotlib.pyplot as plt

def PrintResults(
    original_img,
    sharpened_img,
    upscaled_img,
    pred_orig_model_sharpened,
    pred_upscaled_model,
    img_number):

    # Definir os rótulos das classes CIFAR-10
    class_labels = ['avião', 'automóvel', 'pássaro', 'gato', 'veado', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

    # Obter a classe predita para cada previsão
    pred_orig_model_sharpened_class = np.argmax(pred_orig_model_sharpened)

    # Obter o rótulo da classe predita
    pred_orig_model_sharpened_label = class_labels[pred_orig_model_sharpened_class]

    # Obter a classe predita para cada previsão do modelo upscalado
    pred_upscaled_model_class = np.argmax(pred_upscaled_model)

    # Obter o rótulo da classe predita pelo modelo upscalado
    pred_upscaled_model_label = class_labels[pred_upscaled_model_class]

    # Criar uma figura com 1 linha e 3 colunas
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Exibir a imagem original
    axs[0].imshow(original_img)
    axs[0].axis('off')
    axs[0].set_title('Imagem Original')

    # Exibir a imagem com sharpening
    axs[1].imshow(sharpened_img)
    axs[1].axis('off')
    axs[1].set_title('Imagem com Sharpening')

    # Exibir a imagem upscalada
    axs[2].imshow(upscaled_img[0])
    axs[2].axis('off')
    axs[2].set_title('Imagem Upscalada')

    # Ajustar layout
    plt.tight_layout()
    plt.show()

    # Imprimir as previsões
    print("\nPrevisões na Imagem com Sharpening para a " + str(img_number) + "ª imagem:")
    print(f"Modelo Original: {pred_orig_model_sharpened_label}")
    print(f"Modelo Upscalado: {pred_upscaled_model_label}")