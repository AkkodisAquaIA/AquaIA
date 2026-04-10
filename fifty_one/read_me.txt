
Dans un premier temps, le programme réalise une première analyse du Dataset. 
Il réalise plusieurs contrôles. Les contrôles sont classés en deux types les ‘warnings’ et les ‘erreurs’. Les warning n’interrompent pas la suite des contrôles contrairement aux erreurs qui arrête le déroulement programme. 
Les contrôles sont les suivants :
-	Vérifications que les images soient du types valides
-	Chaque image a un label associé
-	Le contenu du fichier label est conforme
-	Les labels orphelins

Pour les fichiers labels, les contrôles sont les suivants :
-	Chaque lignes doit avoir 5 champs
-	La classe doit être dans les limites
-	Les champs ne doivent être que des nombres
-	Les nombres doivent être positifs
-	Bboxes en double
-	Classes différentes pour une même coordonnée 
-	IOU suspect

Les labels orphelins déclenchent un warning, mais n’interrompent pas le programme. Tous les autres contrôles arrêtent le programme.

S’il n’y a pas d’erreur des analyses statistiques sont réalisées sur le Dataset. 

Dans un premier temps, on récapitule le nombres d’images trouvées, le nombre de labels, le nombres de Bboxes. La moyenne de Bboxes par image et le nombre de classes trouvées.

En suite on calcul les caractéristiques des Bboxes, à savoir le minimum, le maximum et la moyenne pour la largeur, la hauteur et l’air des Bboxes.

On recherche ensuite le nombre et les classes qui sont présentes dans le fichier ‘YAML’ mais que l’on ne retrouve pas dans les labels du Dataset, ainsi que les classes trouvés dans les labels mais pas présentes dans le fichier ‘TAML’.

Ensuite, on liste l’ensemble des classes trouvées avec le nombre d’apparition en nombre et en %, ainsi qu’un bargraphe. Deux seuils sont définis pour permettre de visualiser rapidement la distribution des différentes classes. Ces seuils permettent de définir trois zones dominant, moyen et rare. Ils seront représentés par trois couleurs :
-	Vert : dominant
-	Jaune : moyen
-	Rouge : rare
On affiche ensuite la liste des classes rares

On réalise ensuite des analyses sur les Bboxes concernant :
-	Taille trop petite ou trop grande          : (Poids = 1)
-	Aire . trop petite ou trop grande          : (Poids = 2)
-	Bboxe débordants seuils : ‘warning’ : (Poids = 3) 
-	Bboxe débordants seuils : ‘error’        : (Poids = 5)
Les seuils sont à définir 

Pour les Bboxes présentant des anomales, un score est calculé en fonction du nombre de Bboxes dans l’image et du types d’anomalies Les anomalies ont des niveaux de gravité.


