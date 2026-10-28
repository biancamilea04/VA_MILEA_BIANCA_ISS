import os
import cv2
import numpy as np

from homework2_ex1 import skin_method1, skin_method2, skin_method3, skin_method4

def get_confusion_matrix(mask_pred, mask_gt):

    # convertim mastile in valori booleane (True = piele); folosim pragul 127 deoarece
    # imaginile ground truth pot avea valori intermediare (compresie, redimensionare)
    pred = mask_pred > 127
    gt = mask_gt > 127

    TP = np.sum(gt & pred)      # piele in ground truth si piele prezisa
    FN = np.sum(gt & ~pred)     # piele in ground truth, dar non-piele prezisa
    FP = np.sum(~gt & pred)     # non-piele in ground truth, dar piele prezisa
    TN = np.sum(~gt & ~pred)    # non-piele in ground truth si non-piele prezisa

    return np.array([[TP, FN], [FP, TN]], dtype=np.int64)


def get_accuracy(conf_matrix):

    TP, FN = conf_matrix[0]
    FP, TN = conf_matrix[1]
    return (TP + TN) / (TP + FN + FP + TN)


def find_ground_truth(image_name, folder_gt):
    nume = os.path.splitext(image_name)[0]
    for fisier in os.listdir(folder_gt):
        if os.path.splitext(fisier)[0] == nume:
            return os.path.join(folder_gt, fisier)
    return None


def evaluate_method(skin_method, folder_imagini, folder_gt):
    conf_matrix_total = np.zeros((2, 2), dtype=np.int64)

    for image_name in sorted(os.listdir(folder_imagini)):
        img = cv2.imread(os.path.join(folder_imagini, image_name))
        if img is None:
            continue    # fisierul nu este o imagine

        gt_path = find_ground_truth(image_name, folder_gt)
        if gt_path is None:
            continue    # nu exista ground truth pentru aceasta imagine
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # daca dimensiunile difera, redimensionam ground truth-ul cu interpolare
        # nearest neighbor pentru a pastra masca strict binara
        if gt.shape != img.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_pred = skin_method(img)
        conf_matrix_total += get_confusion_matrix(mask_pred, gt)

    return conf_matrix_total


def display_results(conf_matrix, nume_metoda):
    TP, FN = conf_matrix[0]
    FP, TN = conf_matrix[1]
    print("\n===== " + nume_metoda + " =====")
    print("%-16s %-18s %-18s" % ("", "Predictie piele", "Predictie non-piele"))
    print("%-16s %-18d %-18d" % ("GT piele", TP, FN))
    print("%-16s %-18d %-18d" % ("GT non-piele", FP, TN))
    print("Acuratete = %.4f" % get_accuracy(conf_matrix))


def exercise_b(folder_imagini, folder_gt):
    metode = {
        "Metoda 1 (RGB)": skin_method1,
        "Metoda 2 (HSV)": skin_method2,
        "Metoda 3 (YCbCr)": skin_method3,
        "Metoda 4 z": skin_method4,
    }
    for nume, metoda in metode.items():
        conf_matrix = evaluate_method(metoda, folder_imagini, folder_gt)
        display_results(conf_matrix, nume)


if __name__ == "__main__":
    print("######## Face_Photo ########")
    exercise_b("Face_Dataset/Pratheepan_Dataset/FacePhoto", "Face_Dataset/Ground_Truth/GroundT_FacePhoto")

    print("\n######## Family_Photo ########")
    exercise_b("Face_Dataset/Pratheepan_Dataset/FamilyPhoto", "Face_Dataset/Ground_Truth/GroundT_FamilyPhoto")

# CONCLUZII
# Pe setul Pratheepan, metodele bazate pe praguri de culoare ating 78% - 84% acuratete.
# Nicio metoda nu este cea mai buna pe ambele seturi: YCbCr (metoda 3) are cea mai buna
# acuratete pe Face_Photo (0.80), iar HSV (metoda 2) pe Family_Photo (0.84).
# Metoda 1 (RGB) are recall mare, dar si multe fals pozitive. Metoda 2 (HSV) este cea mai
# restrictiva: are putine fals pozitive, dar pierde mai mult din pielea reala.
# Metoda 4 (Kolkur et al.) este cea mai echilibrata: este a doua ca acuratete pe ambele
# seturi (0.797 si 0.827), deoarece combina conditiile RGB, HSV si YCbCr si reduce
# numarul de fals pozitive fata de RGB si YCbCr.
# Acuratetea este influentata de dezechilibrul dintre clase (pielea ocupa doar ~12% - 26%
# din pixeli), iar precizia ramane scazuta (0.36 - 0.58). Erorile apar din cauza fundalurilor
# cu culori apropiate de piele, a parului si umbrelor si a ground truth-ului desenat manual.
