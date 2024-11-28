import cv2
import numpy as np

def ApplySharpening(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    sharpened_img = cv2.filter2D(img, -1, kernel)
    sharpened_img = np.clip(sharpened_img, 0, 1)
    sharpened_img = sharpened_img.astype(np.float32)
    return sharpened_img

def ApplySharpeningToImages(images):
    sharpened_images = []
    counter = 1
    for image in images:
        print("Applying Sharpening to image " + str(counter))
        counter += 1
        sharpened_images.append(ApplySharpening(image))
    return np.array(sharpened_images)
