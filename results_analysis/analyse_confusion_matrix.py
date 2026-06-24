import numpy as np

path_confusion_matrix = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/test_dinov3/unfreeze1/focal_05/20260602-115513/inference_outputs/confusion_matrix.npy"
cm = np.load(path_confusion_matrix)

for i in range(cm.shape[0]):
    row = cm[i].astype(float)
    total = row.sum()

    row[i] = 0  # ignorer la diagonale

    if total > 0:
        percentages = row / total * 100
    else:
        percentages = row

    top2 = np.argsort(percentages)[-2:][::-1]

    print(f"Classe {i}:")
    for j in top2:
        print(f"  -> Classe {j}: {cm[i,j]} échantillons ({percentages[j]:.2f}%)")
    print()