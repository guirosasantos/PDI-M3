import cv2

def ApplyGrayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return gray_img

def ApplyGrayscaleToImages(images):
    count = 1
    gray_images = []
    for img in images:
        gray_img = ApplyGrayscale(img)
        print("Grayscaling image " + str(count))
        count += 1
        gray_images.append(gray_img)
    return gray_images