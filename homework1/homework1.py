import random

import cv2
import numpy as np

""" === 2 === """
def details_of_image(image):
    if image is None:
        print("[Error] Image not found.")
        return

    img_size = image.size
    img_height = image.shape[0]
    img_width = image.shape[1]
    cv2.imshow('Original', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print("Height: " + str(img_height) + " Width: " + str(img_width) + " Size: " + str(img_size))

""" === 3 === """
def apply_blur_filter(image, parameter_1=5, parameter_2=5):
    if image is None:
        print("[Error] Image not found.")
        return

    img_blur = cv2.blur(src=image,
                        ksize=(parameter_1, parameter_2))

    cv2.imshow('Original', image)
    cv2.imshow('Blurred', img_blur)

    cv2.waitKey()
    cv2.imwrite('blur.jpg', img_blur)
    cv2.destroyAllWindows()

def apply_sharpen_filter(image):
    if image is None:
        print("[Error] Image not found.")
        return

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])

    img_sharpened = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    cv2.imshow('Original', image)
    cv2.imshow('Sharpened', img_sharpened)

    cv2.waitKey()
    cv2.imwrite('sharpened.jpg', img_sharpened)
    cv2.destroyAllWindows()

""" === 4 === """
def apply_custom_filter(image, a):
    if image is None:
        print("[Error] Image not found.")
        return

    kernel = np.array([[0, -1, 0],
                       [-1, a + 8, -1],
                       [0, -1, 0]])

    img_custom_filtered = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    cv2.imshow('Original', image)
    cv2.imshow('Custom Filtered', img_custom_filtered)

    cv2.waitKey()
    cv2.imwrite('custom_filtered.jpg', img_custom_filtered)
    cv2.destroyAllWindows()

""" === 5 === """
def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)   # (center, angle, scale) creaza o matrice noua de transformare care va fi folosita pentru a roti imaginea
    return cv2.warpAffine(image, m, (w, h)) # ia fiecare pixel al imaginii si il pune la pozitia noua

def apply_rotation(image):
    if image is None:
        print("[Error] Image not found.")
        return

    angles = [0, 90, 180, 270, 75, -30, 45, -45]
    for angle in angles:
        img_rotated = rotate(image, angle)
        cv2.imshow('Rotated', img_rotated)
        cv2.waitKey()
        cv2.imwrite('rotated.jpg', img_rotated)
    cv2.destroyAllWindows()

""" ===6=== """

def crop(image, x, y, width, height):
    H, W = image.shape[:2]

    if not (0 <= x < W and 0 <= y < H):
        raise ValueError("Starting position INVALID")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")

    x_end = min(x + width, W)
    y_end = min(y + height, H)

    return image[y:y_end, x:x_end].copy() # extrage pixelii din zona selectata si returneaza o copie a acestora


def apply_crop(image, x, y, width, height):
    if image is None:
        print("[Error] Image not found.")
        return

    cropped_image = crop(image, x, y, width, height)

    cv2.imshow('Original', image)
    cv2.imshow('Cropped', cropped_image)

    cv2.waitKey()
    cv2.imwrite('cropped.jpg', cropped_image)
    cv2.destroyAllWindows()

""" === 7 === """
import cv2
import numpy as np

def make_emoji(size=400, face_color=(0, 215, 255)):
    emoji = np.full((size, size, 3), 255, dtype=np.uint8)
    cx = cy = size // 2
    r = int(size * 0.45)            # raza fetei
    s = size / 400                  # factor de scalare pentru dimensiuni

    def p(v):                       # transformarea dimensiunilor
        return int(v * s)

    # Fata
    cv2.circle(emoji, (cx, cy), r, face_color, -1) # -1 fata umpluta

    # Ochii
    for ex in (cx - p(65), cx + p(65)):
        cv2.circle(emoji, (ex, cy - p(50)), p(25), (0, 0, 0), -1)

    # Smile
    cv2.ellipse(emoji, (cx, cy + p(40)), (p(95), p(70)), 0, 15, 165,
                (0, 0, 0), max(1, p(10)))

    # Rosy cheeks (translucent)
    overlay = emoji.copy()
    return cv2.addWeighted(overlay, 0.5, emoji, 0.5, 0)

def apply_emoji(size=400, face_color=(0, 215, 255)):
    emoji = make_emoji(size, face_color)
    cv2.imshow('Emoji', emoji)
    cv2.waitKey()
    cv2.imwrite('emoji.jpg', emoji)
    cv2.destroyAllWindows()


def main():
    img = cv2.imread('lena.tif')

    """ === 2 === """
   # details_of_image(img)

    """ === 3 === """
   # apply_blur_filter(img)
   # apply_blur_filter(img,10,10)
   # apply_sharpen_filter(img)

    """ === 4 === """
    """
    a = 0 se extrag muchiile imaginii
    a = 0.5 -> 1 detaliile devin mai clare
    a = 2 accentuare excesiva
    """
    #a=2
    #print(f"Applying custom filter with a={a}")
    #apply_custom_filter(img, a)


    """ === 5 === """
    # apply_rotation(img)

    """ === 6 === """
   # x, y, width, height = 100, 100, 200, 200
   # apply_crop(img, x, y, width, height)

    """ === 7 === """
    apply_emoji(size=400, face_color=(0, 215, 255))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()