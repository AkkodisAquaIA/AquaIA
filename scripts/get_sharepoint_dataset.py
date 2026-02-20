import requests
import os
from msal import ConfidentialClientApplication


# Variables
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
TENANT_ID = "YOUR_TENANT_ID"
SITE_URL = "https://tobumo.sharepoint.com/sites/FRProjetDecarbonation-AQUA-IA"
FOLDER_PATH = "/sites/FRProjetDecarbonation-AQUA-IA/Documents partages/AQUA-IA/Data/datasets/coco128"
OUTPUT_DIR = "./dataset"

app = ConfidentialClientApplication(client_id=CLIENT_ID, client_credential=CLIENT_SECRET, authority=f"https://login.microsoftonline.com/{TENANT_ID}")


def get_token():
	result = app.acquire_token_for_client(scopes=["https://tobumo.sharepoint.com/.default"])
	if "access_token" not in result:
		raise Exception(f"Failed to acquire token: {result.get('error_description')}")
	return result["access_token"]


def get_headers():
	# Re-acquire token each time — MSAL caches it and refreshes automatically
	return {"Authorization": f"Bearer {get_token()}", "Accept": "application/json;odata=verbose"}


def download_folder(folder_server_relative_url, local_dir):
	os.makedirs(local_dir, exist_ok=True)

	# --- Download files ---
	files_endpoint = f"{SITE_URL}/_api/web/GetFolderByServerRelativeUrl('{folder_server_relative_url}')/Files"

	while files_endpoint:
		response = requests.get(files_endpoint, headers=get_headers())
		response.raise_for_status()
		data = response.json()["d"]

		for file in data["results"]:
			file_name = file["Name"]
			file_url = file["ServerRelativeUrl"]
			local_path = os.path.join(local_dir, file_name)

			print(f"Downloading: {file_url}")
			download_resp = requests.get(f"{SITE_URL}/_api/web/GetFileByServerRelativeUrl('{file_url}')/$value", headers=get_headers())
			download_resp.raise_for_status()

			with open(local_path, "wb") as f:
				f.write(download_resp.content)

		# Follow pagination if present
		files_endpoint = data.get("__next")


download_folder(FOLDER_PATH, OUTPUT_DIR)
print("Done.")
