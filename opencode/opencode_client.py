from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
from opencode_ai import Opencode


def send_message(message: str) -> str:
    client = Opencode()
    config = client.config.get()
    model_ref = getattr(config, "model", None)
    if not model_ref:
        raise RuntimeError("Opencode model not configured")

    provider_id, model_id = model_ref.split("/", 1)
    session_resp = client._client.post(
        "/session",
        json={},
        timeout=httpx.Timeout(60.0),
        headers={"Content-Type": "application/json"},
    )
    session_resp.raise_for_status()
    session_id = session_resp.json()["id"]

    payload: dict[str, Any] = {
        "messageID": f"msg_{uuid4().hex}",
        "model": {
            "providerID": provider_id,
            "modelID": model_id,
        },
        "parts": [
            {
                "type": "text",
                "text": message,
            }
        ],
    }

    response = client._client.post(
        f"/session/{session_id}/message",
        json=payload,
        timeout=httpx.Timeout(180.0),
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    body = response.json()
    return extract_text(body)


def extract_text(data: dict[str, Any]) -> str:
    for part in data.get("parts", []):
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text")
    return ""