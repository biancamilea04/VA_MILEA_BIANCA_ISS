import cv2
import numpy as np

from homework2_ex1 import skin_method4

def clean_mask(mask):
    """
    opening elimina petele mici de zgomot, iar closing umple golurile din interiorul fetei

    kernerul este o matrice nxn cu 1 in interiorul elipsei si 0 in exterior
    Dimensiunea elipsei este aleasa astfel incat sa elimine zgomotul, dar sa nu afecteze fata
    """
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # o elipsa de 7x7 pixeli
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) # o elipsa de 15x15 pixeli
    # imaginea de intrare, tipul de operatie morfologica, kernelul
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open) # elimina petele mici de zgomot
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close) # umple golurile din interiorul fetei
    return mask

def get_face_square(img):
    mask = clean_mask(skin_method4(img))

    # componentele conexe ale mastii,
    # componenta 0 este fundalul
    # nr_comp este numarul de componente/regiuni de pixeli
    # o imagine in care fiecare pixel are numarul regiunii din care face parte
    # stats tabel cu care un rand pe regiune [x,y,latimea, inaltimea, aria]
    # centroidul este centrul regiunii
    nr_comp, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if nr_comp < 2:
        return None

    # fata este considerata componenta conexa cu aria cea mai mare
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h = stats[idx, :4]

    # cel mai mic patrat care contine dreptunghiul (x, y, w, h), centrat pe acesta
    # limitam latura la inaltimea si latimea imaginii
    latura = min(max(w, h), img.shape[0], img.shape[1])

    #centrul dreptunghiului
    cx = x + w / 2
    cy = y + h / 2
    x0 = int(round(cx - latura / 2))
    y0 = int(round(cy - latura / 2))

    # ne asiguram ca patratul ramane in interiorul imaginii
    x0 = max(0, min(x0, img.shape[1] - latura))
    y0 = max(0, min(y0, img.shape[0] - latura))

    return x0, y0, latura



def apply_face_detection(img):
    img_copy = img.copy()
    rezultat = get_face_square(img_copy)

    if rezultat is not None:
        x, y, latura = rezultat
        cv2.rectangle(img_copy, (x, y), (x + latura, y + latura), (0, 255, 0), 3)

    cv2.imshow('imagine originala', img)
    cv2.imshow('fata detectata', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("rezultat_fata.png", img_copy)


if __name__ == "__main__":
    apply_face_detection(cv2.imread("2.jpg"))