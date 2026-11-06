# Prototype Streamlit de détection AquaIA

L'interface sélectionne un checkpoint à partir des choix expérimentaux, affiche ses métriques de validation et de test disponibles, puis exécute une détection sur une image TIF, TIFF, PNG ou JPEG. Le résultat contient l'image complète annotée et une galerie de toutes les détections découpées, avec leur score de confiance.

## Lancement

Depuis la racine du projet et avec l'environnement virtuel activé :

```bash
streamlit run deploy/streamlit_detection/app.py --server.port 8501
```

L'application est ensuite accessible à l'adresse `http://localhost:8501`.

La limite d'upload est configurée à 1 Go dans `.streamlit/config.toml`. Pour un fichier local volumineux, le mode « Chemin local sur la machine » évite le transfert par le navigateur.

## Métriques d'une image

Sans annotation, l'application affiche le nombre de détections, les confiances et le temps d'inférence. Un fichier TXT au format YOLO peut être fourni avec l'image pour calculer une précision, un rappel et un F1 à l'échelle de cette image.

Ces mesures sur une image ne remplacent pas la mAP calculée sur un jeu de test complet.
