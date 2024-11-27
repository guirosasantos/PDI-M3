import cv2

def ApplyCanny(img):
    canny_img = cv2.Canny(img, 100, 200)
    canny_img = cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR)
    return canny_img

def ApplyCannyToImages(images):
    count = 1
    canny_images = []
    for img in images:
        canny_img = ApplyCanny(img)
        print("Applying Canny to image " + str(count))
        count += 1
        canny_images.append(canny_img)
    return canny_images

