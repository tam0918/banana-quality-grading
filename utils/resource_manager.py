from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    path: str
    message: str


class ResourceManager:
    """Auto-provisioning for large/binary resources (weights, fonts, etc.).

    - Checks if a file exists
    - If missing and a URL is provided, downloads it with streaming
    - Creates parent folders automatically

    Designed to be safe for demos: failures don't crash the app unless you choose to.
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self._session = session or requests.Session()

    def check_and_download(self, file_path: str, url: str, description: str, timeout_s: int = 30) -> DownloadResult:
        path = Path(file_path)

        if path.exists() and path.is_file() and path.stat().st_size > 0:
            print(f"[OK] Found {description}: {path}")
            return DownloadResult(ok=True, path=str(path), message="found")

        if not url or not url.strip():
            msg = f"Missing {description} at {path} and no URL provided."
            print(f"[WARN] {msg}")
            return DownloadResult(ok=False, path=str(path), message=msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading {description}...")
        try:
            with self._session.get(url, stream=True, timeout=timeout_s) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or "0")

                tmp_path = path.with_suffix(path.suffix + ".part")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

                if tqdm is not None and total > 0:
                    bar = tqdm(total=total, unit="B", unit_scale=True, desc=description)
                else:
                    bar = None

                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if bar is not None:
                            bar.update(len(chunk))

                if bar is not None:
                    bar.close()

                # Basic sanity check
                if total > 0 and downloaded != total:
                    raise IOError(f"Download incomplete: {downloaded}/{total} bytes")

                os.replace(tmp_path, path)

            print(f"[OK] Downloaded {description} -> {path}")
            return DownloadResult(ok=True, path=str(path), message="downloaded")
        except Exception as e:
            msg = f"Failed to download {description} from {url}: {type(e).__name__}: {e}"
            print(f"[WARN] {msg}")
            return DownloadResult(ok=False, path=str(path), message=msg)
