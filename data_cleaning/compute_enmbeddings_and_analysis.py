import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModel

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONFIG
# ============================================================

DATA_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/Datasets/PERLA_cropped_Sarah_20042026_vuLPL"

model_name = "dinov3"

if model_name == "dinov2":
    MODEL_ID = "facebook/dinov2-base"
elif model_name == "dinov3":
    MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
else:
    raise ValueError("Choisir entre 'dinov2' ou 'dinov3'")

EMB_FIELD = model_name + "_embedding"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. Charger modèle
# ============================================================

print("Chargement modèle...")
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()


# ============================================================
# 3. Charger dataset (structure dossier = classes)
# ============================================================

filepaths = []
labels = []
class_to_idx = {}

for idx, class_name in enumerate(sorted(os.listdir(DATA_DIR))):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    class_to_idx[class_name] = idx

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for fname in os.listdir(class_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            filepaths.append(os.path.join(class_path, fname))
            labels.append(idx)

filepaths = np.array(filepaths)
labels = np.array(labels)

print(f"{len(filepaths)} images chargées")
print(f"{len(class_to_idx)} classes")


# ============================================================
# 4. Embeddings
# ============================================================


@torch.inference_mode()
def compute_embeddings(model, processor, filepaths, device, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(filepaths), batch_size)):
        batch_paths = filepaths[i : i + batch_size]

        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        feats = outputs.last_hidden_state

        emb = feats.mean(dim=1)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings).numpy().astype(np.float32)


print("Calcul embeddings...")
embeddings = compute_embeddings(model, processor, filepaths, device)

np.save(f"{EMB_FIELD}.npy", embeddings)
np.save("labels.npy", labels)


# ============================================================
# 5. Analyse embeddings
# ============================================================

embeddings_norm = normalize(embeddings)
classes = np.unique(labels)


# ----- Centroïdes
centroids = []
intra_mean = []
intra_var = []

for c in classes:
    Xc = embeddings_norm[labels == c]
    centroid = Xc.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    centroids.append(centroid)

    dist = cosine_distances(Xc, centroid.reshape(1, -1)).ravel()
    intra_mean.append(dist.mean())
    intra_var.append(dist.var())

centroids = np.vstack(centroids)


# ----- Distances inter
dist_matrix = cosine_distances(centroids)


# ============================================================
# 6. Classes proches / éloignées
# ============================================================

pairs = [(i, j, dist_matrix[i, j]) for i in range(len(classes)) for j in range(i + 1, len(classes))]

print("\nClasses les plus proches:")
for i, j, d in sorted(pairs, key=lambda x: x[2])[:10]:
    print(idx_to_class[i], "-", idx_to_class[j], "|", round(d, 4))

print("\nClasses les plus éloignées:")
for i, j, d in sorted(pairs, key=lambda x: x[2], reverse=True)[:10]:
    print(idx_to_class[i], "-", idx_to_class[j], "|", round(d, 4))


# ============================================================
# 7. Silhouette
# ============================================================

sil_global = silhouette_score(embeddings_norm, labels, metric="cosine")
sil_samples = silhouette_samples(embeddings_norm, labels, metric="cosine")

sil_per_class = {c: sil_samples[labels == c].mean() for c in classes}

print("\nSilhouette global:", sil_global)


# ============================================================
# 8. Séparabilité
# ============================================================

inter_mean = [np.delete(dist_matrix[i], i).mean() for i in range(len(classes))]

inter_mean = np.array(inter_mean)
intra_mean = np.array(intra_mean)

ratio = inter_mean / (intra_mean + 1e-8)


# ============================================================
# 9. Clustering
# ============================================================

kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init="auto")
clusters = kmeans.fit_predict(embeddings_norm)

print("\nARI:", adjusted_rand_score(labels, clusters))
print("NMI:", normalized_mutual_info_score(labels, clusters))


# ============================================================
# 10. Heatmap
# ============================================================

plt.imshow(cosine_similarity(centroids))
plt.title("Similarité entre classes")
plt.colorbar()
plt.show()


# ============================================================
# 11. Export
# ============================================================

df = pd.DataFrame({"classe": classes, "silhouette": [sil_per_class[c] for c in classes], "intra": intra_mean, "inter": inter_mean, "ratio": ratio})

df.to_csv("analyse_embeddings.csv", index=False)

print("\nAnalyse terminée ✔️")

# ============================================================
# 12. Suggestions automatiques de fusion de classes
# ============================================================


def suggest_class_merges(dist_matrix, classes, idx_to_class, sil_per_class, ratio, top_k=30, max_centroid_distance=0.05, max_mean_silhouette=0.20, max_mean_ratio=None):
    """
    Propose des paires de classes candidates à fusionner.

    Critères :
    - distance entre centroïdes faible
    - silhouette moyenne faible
    - ratio de séparabilité faible optionnel
    """

    suggestions = []

    class_to_pos = {c: pos for pos, c in enumerate(classes)}

    for a in classes:
        for b in classes:
            if a >= b:
                continue

            i = class_to_pos[a]
            j = class_to_pos[b]

            centroid_dist = dist_matrix[i, j]
            mean_sil = (sil_per_class[a] + sil_per_class[b]) / 2
            mean_ratio = (ratio[i] + ratio[j]) / 2

            if centroid_dist > max_centroid_distance:
                continue

            if mean_sil > max_mean_silhouette:
                continue

            if max_mean_ratio is not None and mean_ratio > max_mean_ratio:
                continue

            # Score faible = fusion plus probable
            merge_score = centroid_dist + max(mean_sil, 0) + 0.01 * mean_ratio

            suggestions.append(
                {
                    "class_id_1": int(a),
                    "class_name_1": idx_to_class[int(a)],
                    "class_id_2": int(b),
                    "class_name_2": idx_to_class[int(b)],
                    "centroid_distance": float(centroid_dist),
                    "mean_silhouette": float(mean_sil),
                    "mean_ratio": float(mean_ratio),
                    "merge_score": float(merge_score),
                }
            )

    suggestions = sorted(suggestions, key=lambda x: x["merge_score"])
    return suggestions[:top_k]


merge_suggestions = suggest_class_merges(
    dist_matrix=dist_matrix, classes=classes, idx_to_class=idx_to_class, sil_per_class=sil_per_class, ratio=ratio, top_k=50, max_centroid_distance=0.05, max_mean_silhouette=0.20
)

print("\nSuggestions de classes à fusionner :")
for s in merge_suggestions:
    print(
        f"{s['class_id_1']} ({s['class_name_1']}) <-> {s['class_id_2']} ({s['class_name_2']}) | dist={s['centroid_distance']:.4f} | silhouette={s['mean_silhouette']:.4f} | ratio={s['mean_ratio']:.2f}"
    )

df_merges = pd.DataFrame(merge_suggestions)
df_merges.to_csv("/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Results/PERLA/suggestions_fusion_classes.csv", index=False)
