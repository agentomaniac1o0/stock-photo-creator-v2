"""
Shared Nextcloud WebDAV client.
Used by all modules that need to read/write to Nextcloud.
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class NextcloudClient:
    """WebDAV client for Nextcloud file operations."""

    def __init__(self, host: str, user: str, password: str):
        self.host = host.rstrip("/")
        self.user = user
        self.password = password
        self.base_url = f"{self.host}/remote.php/dav/files/{self.user}"

    def _url(self, remote_path: str) -> str:
        parts = remote_path.strip("/").split("/")
        encoded = "/".join(quote(p, safe="") for p in parts)
        return f"{self.base_url}/{encoded}"

    def _auth(self) -> tuple:
        return (self.user, self.password)

    def list_dir(self, remote_path: str) -> list[dict]:
        """List files in a Nextcloud directory."""
        url = self._url(remote_path)
        headers = {"Depth": "1"}
        resp = requests.request("PROPFIND", url, headers=headers,
                                auth=self._auth(), timeout=60, verify=False)
        if resp.status_code not in (200, 207):
            return []
        import xml.etree.ElementTree as ET
        ns = {"d": "DAV:"}
        root = ET.fromstring(resp.content)
        items = []
        for resp_elem in root.findall("d:response", ns):
            href = resp_elem.find("d:href", ns)
            if href is None:
                continue
            href_text = href.text or ""
            name = href_text.rstrip("/").split("/")[-1]
            name = requests.utils.unquote(name)
            if name in ("", remote_path.strip("/").split("/")[-1]):
                continue
            props = {}
            for propstat in resp_elem.findall("d:propstat", ns):
                prop = propstat.find("d:prop", ns)
                if prop is not None:
                    cl = prop.find("{DAV:}getcontentlength", ns)
                    ct = prop.find("{DAV:}getcontenttype", ns)
                    if cl is not None and cl.text:
                        props["size"] = int(cl.text)
                    if ct is not None and ct.text:
                        props["contenttype"] = ct.text
            items.append({"name": name, "href": href_text, "props": props})
        return items

    def download_file(self, remote_path: str, local_path: Path) -> bool:
        """Download a single file from Nextcloud."""
        url = self._url(remote_path)
        try:
            resp = requests.get(url, auth=self._auth(), timeout=300, verify=False, stream=True)
            if resp.status_code != 200:
                return False
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            resp.close()
            return True
        except Exception:
            return False

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """Upload a single file to Nextcloud."""
        url = self._url(remote_path)
        try:
            parent = "/".join(remote_path.strip("/").split("/")[:-1])
            self.mkdir(parent)
            with open(local_path, "rb") as f:
                resp = requests.put(url, data=f, auth=self._auth(), timeout=300, verify=False)
            return resp.status_code in (201, 204)
        except Exception:
            return False

    def mkdir(self, remote_path: str) -> bool:
        """Create directory chain on Nextcloud."""
        parts = remote_path.strip("/").split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            url = self._url(current)
            requests.request("MKCOL", url, auth=self._auth(), timeout=30, verify=False)
        return True

    def delete_file(self, remote_path: str) -> bool:
        """Delete a file on Nextcloud."""
        url = self._url(remote_path)
        resp = requests.delete(url, auth=self._auth(), timeout=30, verify=False)
        return resp.status_code in (200, 204, 404)

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on Nextcloud."""
        url = self._url(remote_path)
        resp = requests.head(url, auth=self._auth(), timeout=30, verify=False)
        return resp.status_code == 200


def init_nextcloud() -> Optional[NextcloudClient]:
    """Initialize Nextcloud client from environment or ~/.env."""
    host = os.getenv("NEXTCLOUD_HOST", "").rstrip("/")
    user = os.getenv("NEXTCLOUD_USER", "")
    password = os.getenv("NEXTCLOUD_APP_PASSWORD", "")
    if not host or not user or not password:
        try:
            env_path = Path.home() / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            k, _, v = line.partition("=")
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k == "NEXTCLOUD_HOST" and not host:
                                host = v.rstrip("/")
                            elif k == "NEXTCLOUD_USER" and not user:
                                user = v
                            elif k == "NEXTCLOUD_APP_PASSWORD" and not password:
                                password = v
        except Exception:
            pass
    if not host or not user or not password:
        return None
    return NextcloudClient(host, user, password)
