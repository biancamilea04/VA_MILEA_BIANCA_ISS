import cv2
import numpy as np

# === metoda 1 ===

def get_bgr_colors(img_bgr):
    
    # in openCV imaginile color sunt stocate implicit in format BGR (Blue, Green, Red)
    
    B, G, R = img_bgr[:, :, 0].astype(int), img_bgr[:, :, 1].astype(int), img_bgr[:, :, 2].astype(int)
    return B, G, R

def skin_method1(img):
    B, G, R = get_bgr_colors(img)

    """FORMULA DE DETECTIE A PIELII (METODA 1)"""
    max_rgb = np.maximum(R, np.maximum(G, B))
    min_rgb = np.minimum(R, np.minimum(G, B))
    mask = (R > 95) & (G > 40) & (B > 20) & ((max_rgb - min_rgb) > 15) & (np.abs(R - G) > 15) & (R > G) & (R > B)
    # inmultim masca cu 255 pentru a obtine o imagine binara in care pixelii pielii sunt albi (255) si restul pixelilor sunt negri (0)
    return (mask * 255).astype(np.uint8)

def apply_skin_method1(img):

    img_copy = img.copy()
    img_alb_negru = skin_method1(img_copy)

    cv2.imshow('imagine originala', img)
    cv2.imshow('skin method1', img_alb_negru)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rezultat_masca1.jpg", img_alb_negru)

# === metoda 2 ===

def get_hsv_colors(img_bgr):

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # H in OpenCV este 0-179. Inmultim cu 2.0 pentru a-l aduce la 0-360.
    H = img_hsv[:, :, 0].astype(float) * 2.0
    # S si V in OpenCV sunt 0-255. Impartim la 255.0 pentru a obtine intervalul 0.0 - 1.0.
    S = img_hsv[:, :, 1].astype(float) / 255.0
    V = img_hsv[:, :, 2].astype(float) / 255.0

    return H, S, V


def skin_method2(img):
    H, S, V = get_hsv_colors(img)

    """FORMULA DE DETECTIE A PIELII (METODA 2)"""
    mask = (H >= 0) & (H <= 50) & (S >= 0.23) & (S <= 0.68) & (V >= 0.35) & (V <= 1.0)

    # Inmultim cu 255 pentru a obtine pixelii albi (255) pentru piele
    return (mask * 255).astype(np.uint8)


def apply_skin_method2(img):
    img_copy = img.copy()
    img_alb_negru = skin_method2(img_copy)

    cv2.imshow('imagine originala', img)
    cv2.imshow('skin method2', img_alb_negru)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rezultat_masca2.jpg", img_alb_negru)

#   === metoda 3 ===

def get_ycbcr_colors(img_bgr):

    B, G, R = get_bgr_colors(img_bgr)

    # Convertim in float pentru calcule matematice precise
    B, G, R = B.astype(float), G.astype(float), R.astype(float)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

    return Y, Cb, Cr


def skin_method3(img):
    """
    Returneaza masca binara corespunzatoare pielii folosind metoda 3 (YCbCr).
    """
    Y, Cb, Cr = get_ycbcr_colors(img)

    """FORMULA DE DETECTIE A PIELII (METODA 3)"""
    mask = (Y > 80) & (Cb > 85) & (Cb < 135) & (Cr > 135) & (Cr < 180)

    # Inmultim cu 255 pentru a obtine pixelii albi (255) pentru piele
    return (mask * 255).astype(np.uint8)


def apply_skin_method3(img):
    img_copy = img.copy()
    img_alb_negru = skin_method3(img_copy)

    cv2.imshow('imagine originala', img)
    cv2.imshow('skin method3', img_alb_negru)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rezultat_masca3.jpg", img_alb_negru)

# === method 4 ===

def skin_method4(img):
    B, G, R = get_bgr_colors(img)
    H, S, V = get_hsv_colors(img)
    Y, Cb, Cr = get_ycbcr_colors(img)

    """FORMULA DE DETECTIE A PIELII (METODA 4)"""
    # conditia pe canalele RGB (comuna pentru ambele variante din articol)
    conditie_rgb = (R > 95) & (G > 40) & (B > 20) & (R > G) & (R > B) & (np.abs(R - G) > 15)

    # conditia pe canalele HSV (H in grade, S in [0, 1])
    conditie_hsv = (H >= 0) & (H <= 50) & (S >= 0.23) & (S <= 0.68)

    # conditia pe canalele YCbCr
    conditie_ycbcr = (Y > 80) & (Cb > 85) & (Cr > 135) & \
                     (Cr <= 1.5862 * Cb + 20) & \
                     (Cr >= 0.3448 * Cb + 76.2069) & \
                     (Cr >= -4.5652 * Cb + 234.5652) & \
                     (Cr <= -1.15 * Cb + 301.75) & \
                     (Cr <= -2.2857 * Cb + 432.85)

    # un pixel este piele daca respecta conditia RGB si cel putin una dintre
    # conditiile HSV sau YCbCr: (RGB si HSV) sau (RGB si YCbCr)
    # conditia A > 15 este ignorata, imaginile nu au transparenta
    mask = conditie_rgb & (conditie_hsv | conditie_ycbcr)

    # inmultim masca cu 255 pentru a obtine o imagine binara in care pixelii pielii sunt albi (255) si restul pixelilor sunt negri (0)
    return (mask * 255).astype(np.uint8)


def apply_skin_method4(img):
    img_copy = img.copy()
    img_alb_negru = skin_method4(img_copy)

    cv2.imshow('imagine originala', img)
    cv2.imshow('skin method4', img_alb_negru)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rezultat_masca4.jpg", img_alb_negru)

def main():
    img = cv2.imread('t26.jpg')
    apply_skin_method1(img)
    apply_skin_method2(img)
    apply_skin_method3(img)
    apply_skin_method4(img)

if __name__ == "__main__":
    main()