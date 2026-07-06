# Déploiement ADIAB — VM Ubuntu 24.04

> **VM :** 172.30.24.4 · Ubuntu 24.04 · 4 vCPU · 16 GB RAM · pas de GPU  
> **User :** ubuntu  
> **Clé SSH :** `key-adiab.pem` (dans ~/Downloads sur ta machine)  
> **Stack :** `aquaia-dataset-builder/` — backend FastAPI + frontend Next.js + Nginx (port 80)

---

## Architecture déployée

```
Internet
    │
    ▼
Nginx :80
    ├── /api/*   → backend FastAPI :8000
    └── /*       → frontend Next.js :3000
                       │
                       ▼
                 SQLite (storage/adiab.db)
```

Tous les services tournent dans des conteneurs Docker sur la VM.  
Le code source est cloné sur la VM et les images sont buildées localement (pas de registry externe nécessaire).

---

## 1. Première connexion à la VM

Depuis ta machine (Mac) :

```bash
chmod 400 ~/Downloads/key-adiab.pem
ssh -i ~/Downloads/key-adiab.pem ubuntu@172.30.24.4
```

---

## 2. Bootstrap de la VM (une seule fois)

Copier le script sur la VM depuis ta machine :

```bash
scp -i ~/Downloads/key-adiab.pem \
    deploy/adiab/bootstrap-adiab.sh \
    ubuntu@172.30.24.4:/tmp/
```

Puis sur la VM :

```bash
sudo bash /tmp/bootstrap-adiab.sh
rm /tmp/bootstrap-adiab.sh
```

Le script est idempotent — le relancer sur une VM déjà provisionnée est sans danger.

**Ce que le script installe :**
- Docker Engine + Docker Compose plugin (v2)
- git
- Dossier `/opt/aquaia-dataset-builder/storage/` (owned by ubuntu)
- Ajoute `ubuntu` au groupe `docker`

⚠️ Après l'ajout au groupe docker, il faut se déconnecter et reconnecter pour que le groupe prenne effet :

```bash
exit
ssh -i ~/Downloads/key-adiab.pem ubuntu@172.30.24.4
```

---

## 3. Déploiement initial

Sur la VM :

```bash
# Cloner le dépôt
git clone <URL_DU_REPO> /opt/aquaia-dataset-builder
cd /opt/aquaia-dataset-builder/aquaia-dataset-builder

# Si la base de données existe déjà sur un autre serveur, la copier d'abord :
# (depuis ta machine Mac)
# scp -i ~/Downloads/key-adiab.pem /chemin/vers/adiab.db ubuntu@172.30.24.4:/opt/aquaia-dataset-builder/storage/

# Builder et démarrer
docker compose -f docker-compose.prod.yml up -d --build
```

**Temps estimé du premier build :**
- Frontend (Next.js multi-stage) : ~3-5 min
- Backend (Python 3.11-slim) : ~1-2 min

---

## 4. Vérification

```bash
# Status des conteneurs
docker compose -f docker-compose.prod.yml ps

# Health check backend
curl http://localhost/health

# Logs en temps réel
docker compose -f docker-compose.prod.yml logs -f --tail 100

# Logs d'un service spécifique
docker compose -f docker-compose.prod.yml logs -f backend
docker compose -f docker-compose.prod.yml logs -f frontend
docker compose -f docker-compose.prod.yml logs -f nginx
```

L'application est accessible depuis un navigateur à : **http://172.30.24.4**

---

## 5. Opérations courantes

### Mettre à jour l'application

```bash
cd /opt/aquaia-dataset-builder/aquaia-dataset-builder
git pull origin development
docker compose -f docker-compose.prod.yml up -d --build
```

### Arrêter la stack

```bash
docker compose -f docker-compose.prod.yml down
```

### Redémarrer sans rebuild

```bash
docker compose -f docker-compose.prod.yml restart
```

### Exec dans le backend

```bash
docker compose -f docker-compose.prod.yml exec backend bash
```

### Sauvegarder la base de données

```bash
# Sur la VM
cp /opt/aquaia-dataset-builder/storage/adiab.db \
   /opt/aquaia-dataset-builder/storage/adiab.db.bak-$(date +%Y%m%d)

# Ou récupérer sur ta machine Mac
scp -i ~/Downloads/key-adiab.pem \
    ubuntu@172.30.24.4:/opt/aquaia-dataset-builder/storage/adiab.db \
    ~/Downloads/adiab-backup-$(date +%Y%m%d).db
```

---

## 6. Rollback

Chaque `git pull` + rebuild crée une nouvelle version. Pour revenir en arrière :

```bash
cd /opt/aquaia-dataset-builder/aquaia-dataset-builder
git log --oneline -10         # identifier le commit stable
git checkout <sha>
docker compose -f docker-compose.prod.yml up -d --build
```

---

## 7. Structure des volumes sur la VM

```
/opt/aquaia-dataset-builder/
└── aquaia-dataset-builder/       ← repo cloné
    ├── backend/
    ├── frontend/
    ├── nginx/
    ├── docker-compose.prod.yml
    └── storage/                  ← bind mount Docker (données persistantes)
        └── adiab.db              ← base SQLite
```

Le dossier `storage/` est un bind mount — il n'est jamais supprimé par `docker compose down`.

---

## 8. Notes Azure

- Port 80 doit être ouvert dans le **Network Security Group** Azure (règle entrante TCP:80).
- Pas de GPU → pas de NVIDIA Container Toolkit nécessaire.
- La VM n'a pas besoin d'accéder à Docker Hub (build local depuis les sources).
