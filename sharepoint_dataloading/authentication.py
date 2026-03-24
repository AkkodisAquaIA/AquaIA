import os
import threading
import time
from msal import ConfidentialClientApplication


class GraphApiError(Exception):
    pass


TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

if not TENANT_ID or not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("TENANT_ID, CLIENT_ID et CLIENT_SECRET doivent être définis")


class SharePointTokenProvider:
    def __init__(self, client_id: str, tenant_id: str, client_secret: str, refresh_margin_seconds: int = 300):
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.client_secret = client_secret
        self.refresh_margin_seconds = refresh_margin_seconds

        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]

        self.app = ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority,
        )

        self._lock = threading.Lock()
        self._access_token = None
        self._expires_at = 0

    def _acquire_token(self) -> None:
        result = self.app.acquire_token_for_client(scopes=self.scope)
        access_token = result.get("access_token")
        if not access_token:
            raise GraphApiError(
                f"Échec token: {result.get('error')} | {result.get('error_description')}"
            )

        self._access_token = access_token
        expires_in = int(result.get("expires_in", 3600))
        self._expires_at = time.time() + expires_in

    def get_token(self) -> str:
        now = time.time()
        with self._lock:
            if self._access_token is None or now >= (self._expires_at - self.refresh_margin_seconds):
                self._acquire_token()
            return self._access_token

    def build_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


token_provider = SharePointTokenProvider(
    client_id=CLIENT_ID,
    tenant_id=TENANT_ID,
    client_secret=CLIENT_SECRET,
)


def build_headers() -> dict:
    return token_provider.build_headers()