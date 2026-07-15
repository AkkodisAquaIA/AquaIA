un fichier 'aqua_ia_conf.ini' permet de régler certains paramètres. Si ce fichier n'est pas correcte ou absent, le programme s'arrête 

Le programme commence par une analyse complète du Dataset afin de vérifier sa conformité et sa qualité. 
Cette analyse repose sur une série de contrôles classés en deux catégories :

🔍 1. Contrôles effectués sur le Dataset

Les vérifications suivantes sont réalisées :

Détection des labels orphelins
Vérification que les images sont dans un format valide
Vérification que chaque image possède un label associé
Validation du contenu des fichiers de labels (voir ci-dessous)

🧾 2. Contrôles sur les fichiers de labels

Chaque fichier ‘label’ est analysé selon les règles suivantes :

Pas de ligne vide
Chaque ligne doit contenir exactement 5 champs
La classe doit être dans les limites définies
Tous les champs doivent être numériques
Les valeurs doivent être positives

Détection de bounding boxes (Bboxes) en double
Détection de classes différentes pour une même coordonnée
Identification d’un IoU suspect (Voir en fin de fichier)

👉 Important :
A la fin de cette analyse, les fichiers qui présentent des défauts sont déplacés dans un répertoire ‘problems’ pour chaque type. Pour les images, leurs fichiers ‘label’ associés est lui aussi déplacé. 



📊 3. Calculs statistiques 

Les calculs se font donc sur un Dataset propre. Le programme calcule plusieurs statistiques sur le Dataset :

📈 Informations générales
Nombre total d’images
Nombre total de labels
Nombre total de Bboxes
Moyenne de Bboxes par image
Nombre de classes détectées
📐 Caractéristiques des Bboxes

Pour la largeur, la hauteur et l’aire :

Minimum
Maximum
Moyenne
Écart-type


🏷️ 4. Analyse des classes

Le programme compare les classes définies dans le fichier YAML avec celles présentes dans les labels :

Classes présentes dans le YAML mais absentes du Dataset
Classes présentes dans le Dataset mais absentes du YAML

Ensuite, il fournit :

Une liste des classes avec :
Nombre d’occurrences
Pourcentage d’apparition
Un graphe de distribution
🎯 Classification des classes

Deux seuils permettent de catégoriser les classes en trois groupes :

🟢 Dominantes
🟡 Moyennes
🔴 Rares

👉 Les classes rares sont affichées explicitement.


🖼️ Évaluation de la qualité des images

Le programme calcule un score de qualité par image afin d’identifier les images les plus problématiques du Dataset.

Cette évaluation repose sur l’analyse des anomalies détectées sur les bounding boxes (Bboxes).

⚖️ 1. Pondération des anomalies
Chaque type d’anomalie est associé à un poids (niveau de gravité) :

Bbox trop petite → poids 1
Surface trop petite → poids 1
Bbox trop grande → poids 2
Surface trop grande → poids 2
Bbox hors limite (warning) → poids 3
Bbox hors limite (erreur) → poids 5

👉 Plus le poids est élevé, plus l’anomalie est considérée comme critique.

📊 2. Agrégation des données par image
Pour chaque image, le programme calcule :

Le nombre total de Bboxes
Le nombre d’anomalies détectées
La somme des niveaux de gravité des anomalies

🧮 3. Calcul du score par image
Un score est calculé pour chaque image à partir de deux facteurs :

    Taux d’erreur   = nombre d’anomalies / nombre total de Bboxes
    Gravité moyenne = somme des poids / nombre d’anomalies

👉 Le score final est défini comme :

    𝑠𝑐𝑜𝑟𝑒 = 𝑡𝑎𝑢𝑥 𝑑′𝑒𝑟𝑟𝑒𝑢𝑟×𝑔𝑟𝑎𝑣𝑖𝑡𝑒ˊ 𝑚𝑜𝑦𝑒𝑛𝑛𝑒

🔎 Interprétation :
Score faible → image globalement propre
Score élevé → image avec beaucoup d’erreurs et/ou des erreurs graves

📉 4. Identification des images problématiques
Les images sont ensuite :

Triées par score décroissant
Les 10 pires images sont affichées avec :
Leur score
Le nombre d’anomalies
Le nombre total de Bboxes
Le pourcentage de Bboxes problématiques

👉 Cela permet de cibler rapidement les images nécessitant une correction.

📈 5. Indicateurs globaux du Dataset
Le programme calcule également des métriques globales :

Nombre d’images problématiques (au moins une anomalie détectée)
Pourcentage d’images impactées
Nombre total de Bboxes problématiques
Score moyen du Dataset

👉 Le score moyen donne une vision globale de la qualité du Dataset.

🎯 Objectif
Cette approche permet de :
    Prioriser les corrections (focus sur les pires images)
    Évaluer rapidement la qualité globale du Dataset
    Suivre l’amélioration du Dataset au fil des itérations


⚖️ Analyse avancée du déséquilibre du Dataset

Le programme réalise une analyse du déséquilibre entre les classes afin d’évaluer la qualité de la distribution des données.

Cette analyse repose sur deux métriques principales, combinées ensuite en un score global.

📊 1. Ratio max / min

Le ratio max/min mesure l’écart entre :

La classe la plus représentée
La classe la moins représentée
🔎 Interprétation :
Ratio faible → distribution équilibrée
Ratio élevé → forte dominance de certaines classes

👉 Exemple :

Ratio = 10 → acceptable
Ratio = 100+ → Dataset très déséquilibré
🧠 2. Entropie normalisée
L’entropie normalisée mesure la diversité globale des classes, en tenant compte de leur distribution.

    𝐻𝑛𝑜𝑟𝑚 = −∑𝑝𝑖log⁡(𝑝𝑖) / log⁡(𝑁)

Avec :

    𝑝𝑖  = proportion de la classe i
    N = nombre total de classes

🔎 Interprétation :
Proche de 1 → distribution homogène (équilibrée)
Proche de 0 → distribution très déséquilibrée

🧮 3. Score global
Un score global est calculé en combinant :

Le ratio (déséquilibre extrême)
L’entropie (répartition globale)

👉 Ce score permet d’obtenir une évaluation synthétique du dataset :

Score faible → Dataset déséquilibré
Score élevé → Dataset bien équilibré

🎨 4. Visualisation des métriques
Chaque métrique est affichée avec :

Une barre de progression
Un code couleur (ex : rouge / orange / vert)
Un statut qualitatif :
Ratio : Très déséquilibré / Déséquilibré / Équilibré
Entropie : Déséquilibré / Moyen / Équilibré
Score : Faible / Moyen / Bon

👉 Les seuils sont configurables via des constantes.
🩺 5. Diagnostic automatique

Le programme génère un diagnostic basé sur les métriques :

Ratio élevé (> 100)
→ Dataset très déséquilibré
Entropie faible (< 0.75)
→ Mauvaise répartition globale
Entropie correcte + ratio élevé
→ Présence de nombreuses classes rares malgré une diversité globale acceptable

💡 6. Recommandations
En fonction des résultats, plusieurs actions sont suggérées :

Augmenter les classes rares (notamment < 1%)
Rééquilibrer la distribution globale
Appliquer de la data augmentation ciblée
Mettre en place un sampling équilibré

🎯 Objectif
Cette analyse permet de :

Détecter rapidement les problèmes de distribution
Comprendre la structure du dataset
Orienter les actions d’amélioration avant entraînement


==========================================================================================================================
------------ IoU suspect ------------
🔍 Détection des IoU suspects

L’IoU (Intersection over Union) est une métrique utilisée pour mesurer le chevauchement entre deux bounding boxes (Bboxes).

    𝐼𝑜𝑈 = (A𝑖𝑟𝑒 𝑑′𝑖𝑛𝑡𝑒𝑟𝑠𝑒𝑐𝑡𝑖𝑜𝑛) / (𝐴𝑖𝑟𝑒 𝑑′𝑢𝑛𝑖𝑜𝑛)

IoU = 0 → aucun chevauchement
IoU = 1 → superposition parfaite
0 < IoU < 1 → chevauchement partiel
⚠️ Cas considérés comme suspects

Le programme identifie comme IoU suspects les situations suivantes :

1. Chevauchement élevé entre deux Bboxes de classes différentes
Exemple : IoU > seuil élevé (ex : 0.7 ou 0.8)
Problème potentiel :
Mauvaise annotation
Confusion entre classes
Duplication d’objets avec des labels différents

👉 Impact : incohérence dans les données d’entraînement

2. Chevauchement quasi total (doublons cachés)
IoU très proche de 1 entre deux Bboxes
Peut indiquer :
Une duplication accidentelle
Une annotation répétée

👉 Souvent lié à :

Bboxes en double
Erreurs de tooling d’annotation
3. Chevauchement anormal dans une même image
Plusieurs Bboxes fortement superposées
Peut révéler :
Une mauvaise séparation des objets
Un problème de granularité dans l’annotation
4. IoU incohérent avec les classes attendues
Certaines classes ne devraient jamais se chevaucher fortement
Exemple : objets exclusifs (selon ton cas métier)

👉 Ce type de règle peut être :

Générique (basée sur seuil)
Ou spécifique (basée sur une matrice de compatibilité entre classes)
⚙️ Paramétrage des seuils

Les seuils d’IoU doivent être définis selon le contexte du dataset :

Seuil bas (~0.3 - 0.5) : détection de chevauchements modérés
Seuil élevé (~0.7 - 0.9) : détection de cas critiques

👉 Recommandation :

Utiliser deux seuils :
Seuil warning
Seuil critique (erreur)
🧮 Utilisation dans le score d’anomalie

Les IoU suspects contribuent au score global d’anomalie :

Plus l’IoU est élevé → plus la pénalité est forte
Si les classes sont différentes → pénalité augmentée
Si répétition dans une image → effet cumulatif
📝 Bonnes pratiques
Ajuster les seuils selon :
Le type d’objets (petits vs grands)
Le niveau de précision attendu
Analyser visuellement les cas détectés pour affiner les règles
Combiner avec :
Détection de doublons
Vérification des classes
