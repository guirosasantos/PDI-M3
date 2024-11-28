import numpy as np
import matplotlib.pyplot as plt

def PrintResults(
    sharpened_img,
    upscaled_img,
    pred_orig_model_sharpened,
    pred_upscaled_model,
    estimated_img,
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

    # Estimated
    estimated_class = np.argmax(estimated_img)

    # Estimated label
    estimated_label = class_labels[estimated_class]

    # Criar uma figura com 1 linha e 2 colunas
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Exibir a imagem com sharpening
    axs[0].imshow(sharpened_img)
    axs[0].axis('off')
    axs[0].set_title('Imagem com Sharpening')

    # Exibir a imagem upscalada
    axs[1].imshow(upscaled_img[0])
    axs[1].axis('off')
    axs[1].set_title('Imagem Upscalada')

    # Ajustar layout
    plt.tight_layout()
    plt.show()

    # Imprimir as previsões
    print("\nPrevisões na Imagem com Sharpening para a " + str(img_number) + "ª imagem:")
    print(f"Modelo Original: {pred_orig_model_sharpened_label}")
    print(f"Modelo Upscalado: {pred_upscaled_model_label}")
    print(f"Valor real: {estimated_label}\n")

def ShowTrainingResults(history, epochs_range):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treinamento e Validação')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], label='Perda de Treinamento')
    plt.plot(epochs_range, history.history['val_loss'], label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treinamento e Validação')
    
    plt.show()