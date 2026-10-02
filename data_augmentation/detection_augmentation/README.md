# Copy and Paste Data Augmentation for Detection pipeline

Le principe est de fournir une augmentation des données robustes pour le pipeline de détection. En effet, pour pouvoir générer des données avec plus de variabilité pour un pipeline de détection, les transformations simples ne suffisent pas : il convient de founir une méthode qui permette de généraliser à une autre échelle, avec des interactions entre plusieurs bêtes.

## Fonctionnement du pipeline
A partir des images de base pour la détection, nous allons extraire les images des boîtes, pour avoir la bête isolée. Une fois qu'on détient la bête isolée, on va appliquer une méthode pour extraire un masque de la bête et donc une image segmentée. On fait cela pour l'ensemble des bêtes.

Pour les méthodes de segmentation, on va tester plusieurs méthodes.

Ensuite, on va les coller sur un fond blanc, ou avec du bruit style des cailloux etc... cela nous donnera donc une nouvelle banque d'image pour la détection !
et on pourra rotater les masques pour insérer les bêtes, créer de nouvelles bounding boxes, etc...
