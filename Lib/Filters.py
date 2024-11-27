from Lib import Sharpening, Grayscale, Canny

def ApplyFilters(image):
    # Apply sharpening
    sharpened_image = Sharpening.ApplySharpening(image)
    # Apply grayscale
    gray_image = Grayscale.ApplyGrayscale(sharpened_image)
    # Apply Canny
    #canny_image = Canny.ApplyCanny(gray_image)
    return gray_image

def ApplyFiltersToImages(images):
    count = 1
    filtered_images = []
    for img in images:
        print("Applying filters to image " + str(count))
        filtered_img = ApplyFilters(img)
        filtered_images.append(filtered_img)
        count += 1
    return filtered_images