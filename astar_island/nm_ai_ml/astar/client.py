"""Astar Island API client."""
import logging
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_URL = "https://api.ainm.no"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubles each retry


class AstarClient:
    def __init__(self, access_token: str | None = None):
        self.access_token = access_token or os.getenv("AINM_ACCESS_TOKEN", "")
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {self.access_token}"},
            timeout=30.0,
        )

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            r = self._client.request(method, url, **kwargs)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning("Rate limited (429), retrying in %.1fs (attempt %d/%d)",
                               wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
        r.raise_for_status()
        return r  # unreachable, but satisfies type checker

    def get_rounds(self) -> list[dict]:
        return self._request("GET", "/astar-island/rounds").json()

    def get_active_round(self) -> dict | None:
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> dict:
        return self._request("GET", f"/astar-island/rounds/{round_id}").json()

    def get_budget(self) -> dict:
        return self._request("GET", "/astar-island/budget").json()

    def simulate(self, round_id: str, seed_index: int,
                 viewport_x: int = 0, viewport_y: int = 0,
                 viewport_w: int = 15, viewport_h: int = 15) -> dict:
        return self._request("POST", "/astar-island/simulate", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        }).json()

    def submit(self, round_id: str, seed_index: int, prediction: list) -> dict:
        return self._request("POST", "/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }).json()

    def get_my_rounds(self) -> dict:
        return self._request("GET", "/astar-island/my-rounds").json()

    def get_my_predictions(self, round_id: str) -> dict:
        return self._request("GET", f"/astar-island/my-predictions/{round_id}").json()

    def get_analysis(self, round_id: str, seed_index: int) -> dict:
        return self._request("GET", f"/astar-island/analysis/{round_id}/{seed_index}").json()

    def get_leaderboard(self) -> dict:
        return self._request("GET", "/astar-island/leaderboard").json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
