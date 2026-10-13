from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from shared.schemas import FeatureVector

LOG = logging.getLogger(__name__)


class FogTransport:
    def __init__(self, fog_url: str, spool_path: Path = Path("edge_spool.jsonl")) -> None:
        self.url = fog_url.rstrip("/") + "/api/features"
        self.spool_path = spool_path

    def send(self, features: FeatureVector) -> bool:
        try:
            response = httpx.post(self.url, json=features.model_dump(mode="json"), timeout=3)
            response.raise_for_status()
            return True
        except (httpx.HTTPError, OSError) as error:
            LOG.warning("Fog unreachable; spooling vector locally: %s", error)
            with self.spool_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(features.model_dump(mode="json")) + "\n")
            return False

    def flush(self) -> int:
        if not self.spool_path.exists():
            return 0
        lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        remaining: list[str] = []
        delivered = 0
        for line in lines:
            item = FeatureVector.model_validate_json(line)
            if self.send(item):
                delivered += 1
            else:
                remaining.append(line)
        self.spool_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        return delivered
