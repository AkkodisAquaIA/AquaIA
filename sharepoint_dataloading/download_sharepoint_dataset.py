import requests
import os
from msal import ConfidentialClientApplication
from pathlib import Path, PurePosixPath

# =========================
# Configuration
# =========================

#Avant de lancer le script python => rentrer les ID et secret dans le systeme avec :
#export TENANT_ID="xxx"
#export CLIENT_ID="xxx"
#export CLIENT_SECRET="xxx"

TENANT_ID = os.getenv("TENANT_ID") # ID de l'organisation de l'entreprise
CLIENT_ID = os.getenv("CLIENT_ID") # ID de l'application
CLIENT_SECRET = os.getenv("CLIENT_SECRET") # Secret généré 

# Vérification immédiate
if not TENANT_ID or not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError(
        "Variables d'environnement manquantes : "
        "TENANT_ID, CLIENT_ID, CLIENT_SECRET"
    )

# URL complète du site SharePoint
SITE_URL = "https://tobumo.sharepoint.com/teams/FRProjetDecarbonation-AQUA-IA"

# Chemin du dossier à lire dans la bibliothèque Documents
FOLDER_PATH = "AQUA-IA/Data/datasets/FIN-Benthic_clean_splited"

# Nom attendu de la bibliothèque SharePoint
DOCUMENTS_DRIVE_NAME = "Documents"

# Timeout par défaut pour les appels HTTP
HTTP_TIMEOUT = 30

LOCAL_DOWNLOAD_DIR = "/home/sarah.laroui/Bureau/AQUA-IA/Python_code/Data/sharepoint_dataset_test"   # dossier local de destination

# True = télécharge aussi les sous-dossiers
RECURSIVE_DOWNLOAD = True


# =========================
# Authentification Microsoft
# =========================

authority = f"https://login.microsoftonline.com/{TENANT_ID}"

app = ConfidentialClientApplication(
    client_id=CLIENT_ID,
    client_credential=CLIENT_SECRET,
    authority=authority
)

scope = ["https://graph.microsoft.com/.default"]


# =========================
# Exceptions personnalisées
# =========================

class GraphApiError(Exception):
    """Erreur générique liée à Microsoft Graph."""
    pass


class SharePointResolutionError(Exception):
    """Erreur lors de la résolution du site, drive ou dossier SharePoint."""
    pass


# =========================
# Fonctions utilitaires
# =========================

def get_token() -> str:
    result = app.acquire_token_for_client(scopes=scope)

    access_token = result.get("access_token")
    if not access_token:
        error = result.get("error", "unknown_error")
        error_description = result.get("error_description", "No error description provided.")
        raise GraphApiError(
            f"Échec de récupération du token. "
            f"Erreur: {error} | Détail: {error_description}"
        )

    return access_token


def build_headers() -> dict:
    return {
        "Authorization": f"Bearer {get_token()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def graph_get(url: str, headers: dict) -> dict:
    try:
        response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise GraphApiError(f"Timeout lors de l'appel Graph: {url}") from e
    except requests.exceptions.HTTPError as e:
        response_text = ""
        try:
            response_text = e.response.text
        except Exception:
            response_text = "<réponse non disponible>"

        raise GraphApiError(
            f"Erreur HTTP lors de l'appel Graph: {url}\n"
            f"Statut: {e.response.status_code if e.response else 'inconnu'}\n"
            f"Réponse: {response_text}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise GraphApiError(f"Erreur réseau lors de l'appel Graph: {url}\n{e}") from e

    try:
        return response.json()
    except ValueError as e:
        raise GraphApiError(f"Réponse non JSON reçue depuis: {url}") from e


def build_site_graph_url(site_url: str) -> str:
    prefix = "https://"
    graph_prefix = "https://graph.microsoft.com/v1.0/sites/"

    if not site_url.startswith(prefix):
        raise ValueError(f"SITE_URL invalide : {site_url}")

    without_scheme = site_url[len(prefix):]
    parts = without_scheme.split("/", 1)

    domain = parts[0]
    path = f"/{parts[1]}" if len(parts) > 1 else ""

    if "sharepoint.com" not in domain:
        raise ValueError(f"Le domaine ne semble pas être un domaine SharePoint valide : {domain}")

    return f"{graph_prefix}{domain}:{path}"


def get_site_info(site_url: str, headers: dict) -> dict:
    graph_site_url = build_site_graph_url(site_url)
    site = graph_get(graph_site_url, headers)

    if "id" not in site:
        raise SharePointResolutionError(
            f"Impossible de résoudre l'ID du site à partir de l'URL : {site_url}\n"
            f"Réponse reçue : {site}"
        )

    return site


def get_site_drives(site_id: str, headers: dict) -> list:
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    data = graph_get(url, headers)

    drives = data.get("value")
    if not isinstance(drives, list):
        raise SharePointResolutionError(
            f"Réponse invalide lors de la récupération des drives du site {site_id}: {data}"
        )

    return drives


def find_drive_by_name(drives: list, drive_name: str) -> dict:
    for drive in drives:
        if drive.get("name") == drive_name:
            return drive

    available_names = [drive.get("name", "<sans nom>") for drive in drives]
    raise SharePointResolutionError(
        f"Bibliothèque '{drive_name}' introuvable.\n"
        f"Bibliothèques disponibles : {available_names}"
    )


def list_folder_contents(site_id: str, drive_id: str, folder_path: str, headers: dict) -> list:
    normalized_path = folder_path.strip("/")

    if not normalized_path:
        raise ValueError("FOLDER_PATH ne doit pas être vide.")

    url = (
        f"https://graph.microsoft.com/v1.0/sites/{site_id}"
        f"/drives/{drive_id}/root:/{normalized_path}:/children"
    )

    data = graph_get(url, headers)

    items = data.get("value")
    if not isinstance(items, list):
        raise SharePointResolutionError(
            f"Réponse invalide lors de la lecture du dossier '{folder_path}': {data}"
        )

    return items


# =========================
# Téléchargement
# =========================

def sanitize_relative_path(path_str: str) -> Path:
    """
    Convertit un chemin relatif SharePoint en Path local sûr.
    """
    return Path(path_str.strip("/"))


def download_file_from_url(download_url: str, destination_path: Path) -> None:
    """
    Télécharge un fichier via l'URL de téléchargement directe.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(download_url, stream=True, timeout=HTTP_TIMEOUT) as response:
            response.raise_for_status()
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise GraphApiError(
            f"Échec du téléchargement vers '{destination_path}': {e}"
        ) from e



def download_folder_contents(
    site_id: str,
    drive_id: str,
    remote_folder_path: str,
    local_base_dir: Path,
    headers: dict,
    recursive: bool = True
) -> None:
    """
    Télécharge tout le contenu d'un dossier SharePoint vers un dossier local,
    en conservant l'arborescence à partir du dossier FOLDER_PATH.
    """
    items = list_folder_contents(site_id, drive_id, remote_folder_path, headers)

    print(f"\nContenu du dossier distant '{remote_folder_path}' :")
    if not items:
        print("  (dossier vide)")
        return

    root_remote_path = PurePosixPath(FOLDER_PATH.strip("/"))
    current_remote_path = PurePosixPath(remote_folder_path.strip("/"))

    for item in items:
        item_name = item.get("name", "<sans nom>")
        is_folder = "folder" in item

        if is_folder:
            print(f"  [DOSSIER] {item_name}")

            if recursive:
                sub_remote_path = f"{remote_folder_path.rstrip('/')}/{item_name}"
                download_folder_contents(
                    site_id=site_id,
                    drive_id=drive_id,
                    remote_folder_path=sub_remote_path,
                    local_base_dir=local_base_dir,
                    headers=headers,
                    recursive=True
                )
        else:
            download_url = item.get("@microsoft.graph.downloadUrl")
            if not download_url:
                print(f"  [SKIP] Impossible de récupérer l'URL de téléchargement pour '{item_name}'")
                continue

            relative_path_inside_root = current_remote_path.relative_to(root_remote_path)
            local_file_path = local_base_dir / relative_path_inside_root / item_name

            print(f"  [FICHIER] Téléchargement de '{item_name}' vers '{local_file_path}'")
            download_file_from_url(download_url, local_file_path)


# =========================
# Fonction principale
# =========================

def download_sharepoint_folder() -> None:
    """
    Fonction principale :
    - récupère un token,
    - résout le site SharePoint,
    - récupère les drives du site,
    - trouve la bibliothèque Documents,
    - télécharge le contenu du dossier demandé.
    """
    try:
        headers = build_headers()

        # 1) Résolution du site SharePoint
        site = get_site_info(SITE_URL, headers)
        site_id = site["id"]

        # 2) Récupération des bibliothèques du site
        drives = get_site_drives(site_id, headers)

        # 3) Recherche de la bibliothèque demandée
        documents_drive = find_drive_by_name(drives, DOCUMENTS_DRIVE_NAME)
        drive_id = documents_drive["id"]

        # 4) Préparation du dossier local
        root_folder_name = Path(FOLDER_PATH.strip("/")).name
        local_base_dir = Path(LOCAL_DOWNLOAD_DIR) / root_folder_name
        local_base_dir.mkdir(parents=True, exist_ok=True)

        # 5) Téléchargement
        download_folder_contents(
            site_id=site_id,
            drive_id=drive_id,
            remote_folder_path=FOLDER_PATH,
            local_base_dir=local_base_dir,
            headers=headers,
            recursive=RECURSIVE_DOWNLOAD
        )

        print("\nTéléchargement terminé.")

    except (GraphApiError, SharePointResolutionError, ValueError) as e:
        print(f"[ERREUR] {e}")
    except Exception as e:
        print(f"[ERREUR INATTENDUE] {type(e).__name__}: {e}")


# =========================
# Point d'entrée
# =========================

if __name__ == "__main__":
    download_sharepoint_folder()