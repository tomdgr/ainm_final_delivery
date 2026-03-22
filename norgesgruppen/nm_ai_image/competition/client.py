import logging
import os

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


class CompetitionClient:
    def __init__(self, base_url: str | None = None, access_token: str | None = None):
        self.base_url = (base_url or os.getenv("AINM_API_URL", "https://api.ainm.no")).rstrip("/")
        self.access_token = access_token or os.getenv("AINM_ACCESS_TOKEN", "")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._auth_headers(),
            timeout=30.0,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def request_token(self, map_id: str) -> dict:
        response = self._client.post("/token", json={"map_id": map_id})
        response.raise_for_status()
        data = response.json()
        logger.info("Token received for map %s", map_id)
        return data

    def request_task(self) -> dict:
        response = self._client.post("/task")
        response.raise_for_status()
        data = response.json()
        logger.info("Task received: %s", data.get("task_id", "unknown"))
        return data

    def submit(self, predictions: dict, task_id: str | None = None) -> dict:
        payload = {"predictions": predictions}
        if task_id:
            payload["task_id"] = task_id

        response = self._client.post("/submit", json=payload)
        response.raise_for_status()
        data = response.json()
        logger.info("Submission result: %s", data)
        return data

    def get_leaderboard(self) -> dict:
        response = self._client.get("/leaderboard")
        response.raise_for_status()
        return response.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncCompetitionClient:
    def __init__(self, base_url: str | None = None, access_token: str | None = None):
        self.base_url = (base_url or os.getenv("AINM_API_URL", "https://api.ainm.no")).rstrip("/")
        self.access_token = access_token or os.getenv("AINM_ACCESS_TOKEN", "")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._auth_headers(),
            timeout=30.0,
        )

    def _auth_headers(self) -> dict[str, str]:
        headers = {}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def submit(self, predictions: dict, task_id: str | None = None) -> dict:
        payload = {"predictions": predictions}
        if task_id:
            payload["task_id"] = task_id
        response = await self._client.post("/submit", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_leaderboard(self) -> dict:
        response = await self._client.get("/leaderboard")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
