from __future__ import annotations

import json
import logging
import os
import threading

_LOGGER = logging.getLogger("klassed.state_store")
_KEY_PREFIX = os.environ.get("KLASSED_STATE_PREFIX", "klassed:v1:")


class SharedStateStore:
    def __init__(self, client=None, backend: str | None = None):
        self._client = client
        self.backend = backend or ("custom" if client is not None else "none")
        self.errors = 0

    @property
    def active(self) -> bool:
        return self._client is not None

    def _key(self, key: str) -> str:
        return f"{_KEY_PREFIX}{key}"

    def get_json(self, key: str):
        if self._client is None:
            return None
        try:
            raw = self._client.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception as exc:
            self.errors += 1
            _LOGGER.warning("state store read failed for %s: %s", key, exc)
            return None

    def set_json(self, key: str, value, ttl_seconds: int | None = None) -> bool:
        if self._client is None:
            return False
        try:
            payload = json.dumps(value, separators=(",", ":"))
            if ttl_seconds:
                self._client.set(self._key(key), payload, ex=int(ttl_seconds))
            else:
                self._client.set(self._key(key), payload)
            return True
        except Exception as exc:
            self.errors += 1
            _LOGGER.warning("state store write failed for %s: %s", key, exc)
            return False

    def delete(self, key: str) -> None:
        if self._client is None:
            return
        try:
            self._client.delete(self._key(key))
        except Exception as exc:
            self.errors += 1
            _LOGGER.warning("state store delete failed for %s: %s", key, exc)

    def health(self) -> dict:
        ok = None
        if self._client is not None:
            try:
                ok = bool(self._client.ping()) if hasattr(self._client, "ping") else True
            except Exception:
                ok = False
        return {"backend": self.backend, "active": self.active, "reachable": ok, "errors": self.errors}


def _build_from_env() -> SharedStateStore:
    url = os.environ.get("KLASSED_REDIS_URL") or os.environ.get("REDIS_URL")
    if not url:
        return SharedStateStore()
    try:
        import redis

        client = redis.Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.5)
        client.ping()
        return SharedStateStore(client, backend="redis")
    except Exception as exc:
        _LOGGER.warning("REDIS_URL is set but Redis is unavailable (%s); using per-process state", exc)
        return SharedStateStore(backend="redis_unavailable")


_STORE: SharedStateStore | None = None
_LOCK = threading.Lock()


def shared_store() -> SharedStateStore:
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = _build_from_env()
        return _STORE


def set_shared_store(store: SharedStateStore | None) -> None:
    global _STORE
    with _LOCK:
        _STORE = store
