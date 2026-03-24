import requests


HTTP_TIMEOUT = 30


class GraphApiError(Exception):
    """Erreur générique liée à Microsoft Graph."""
    pass


class SharePointResolutionError(Exception):
    """Erreur lors de la résolution du site, drive ou dossier SharePoint."""
    pass


def graph_get(url: str, headers: dict) -> dict:
    """
    Envoie une requête GET à Microsoft Graph avec gestion des erreurs.
    """
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
    """
    Convertit une URL SharePoint classique en URL Microsoft Graph.

    Exemple :
        https://tobumo.sharepoint.com/sites/MonSite
    devient :
        https://graph.microsoft.com/v1.0/sites/tobumo.sharepoint.com:/sites/MonSite
    """
    prefix = "https://"
    graph_prefix = "https://graph.microsoft.com/v1.0/sites/"

    if not site_url.startswith(prefix):
        raise ValueError(f"SITE_URL invalide : {site_url}")

    without_scheme = site_url[len(prefix):]
    parts = without_scheme.split("/", 1)

    domain = parts[0]
    path = f"/{parts[1]}" if len(parts) > 1 else ""

    if "sharepoint.com" not in domain:
        raise ValueError(
            f"Le domaine ne semble pas être un domaine SharePoint valide : {domain}"
        )

    return f"{graph_prefix}{domain}:{path}"


def get_site_info(site_url: str, headers: dict) -> dict:
    """
    Résout les informations du site SharePoint via Microsoft Graph.
    """
    graph_site_url = build_site_graph_url(site_url)
    site = graph_get(graph_site_url, headers)

    if "id" not in site:
        raise SharePointResolutionError(
            f"Impossible de résoudre l'ID du site à partir de l'URL : {site_url}\n"
            f"Réponse reçue : {site}"
        )

    return site


def get_site_drives(site_id: str, headers: dict) -> list:
    """
    Récupère les bibliothèques de documents (drives) d'un site SharePoint.
    """
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    data = graph_get(url, headers)

    drives = data.get("value")
    if not isinstance(drives, list):
        raise SharePointResolutionError(
            f"Réponse invalide lors de la récupération des drives du site {site_id}: {data}"
        )

    return drives


def find_drive_by_name(drives: list, drive_name: str) -> dict:
    """
    Recherche une bibliothèque de documents par son nom.
    """
    for drive in drives:
        if drive.get("name") == drive_name:
            return drive

    available_names = [drive.get("name", "<sans nom>") for drive in drives]
    raise SharePointResolutionError(
        f"Bibliothèque '{drive_name}' introuvable.\n"
        f"Bibliothèques disponibles : {available_names}"
    )


def list_folder_contents(site_id: str, drive_id: str, folder_path: str, headers: dict) -> list:
    """
    Liste le contenu d'un dossier SharePoint via Microsoft Graph.
    """
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


def list_subfolders(site_id: str, drive_id: str, folder_path: str, headers: dict) -> list:
    """
    Retourne uniquement les sous-dossiers d'un dossier.
    """
    items = list_folder_contents(site_id, drive_id, folder_path, headers)
    return [item for item in items if "folder" in item]


def list_files(site_id: str, drive_id: str, folder_path: str, headers: dict) -> list:
    """
    Retourne uniquement les fichiers d'un dossier.
    """
    items = list_folder_contents(site_id, drive_id, folder_path, headers)
    return [item for item in items if "file" in item]


def download_file_content(drive_id: str, file_id: str, headers: dict) -> bytes:
    """
    Télécharge le contenu binaire d'un fichier via Microsoft Graph.
    """
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"

    try:
        response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout as e:
        raise GraphApiError(f"Timeout lors du téléchargement du fichier {file_id}") from e
    except requests.exceptions.HTTPError as e:
        response_text = ""
        try:
            response_text = e.response.text
        except Exception:
            response_text = "<réponse non disponible>"

        raise GraphApiError(
            f"Erreur HTTP lors du téléchargement du fichier {file_id}\n"
            f"Statut: {e.response.status_code if e.response else 'inconnu'}\n"
            f"Réponse: {response_text}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise GraphApiError(f"Erreur réseau lors du téléchargement du fichier {file_id}\n{e}") from e

    return response.content


def display_items(folder_path: str, items: list) -> None:
    """
    Affiche proprement les éléments trouvés dans le dossier.
    """
    print(f"\nContenu du dossier '{folder_path}' :")

    if not items:
        print("  (dossier vide)")
        return

    for item in items:
        item_name = item.get("name", "<sans nom>")
        item_type = "Folder" if "folder" in item else "File"
        item_size = item.get("size", 0)

        print(f"  - {item_name} ({item_type}, {item_size} bytes)")