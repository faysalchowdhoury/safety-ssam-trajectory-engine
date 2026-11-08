from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ssam_engine.api.app import create_app
from ssam_engine.api.settings import ApiSettings


def _client() -> TestClient:
    return TestClient(create_app(ApiSettings(redis_url=None, max_runs=2)))


def test_upload_query_and_cache(fzp_file: Path) -> None:
    with _client() as c:
        assert c.get("/health").json()["status"] == "ok"
        with open(fzp_file, "rb") as fh:
            r = c.post("/v1/runs", files={"file": ("sample.fzp", fh, "application/octet-stream")})
        assert r.status_code == 201, r.text
        run = r.json()
        assert run["events"] == 2
        assert run["by_type"]["crossing"] == 1 and run["by_type"]["rear_end"] == 1
        rid = run["run_id"]

        q = {"min_x": -10, "min_y": -10, "max_x": 10, "max_y": 10}
        r1 = c.get(f"/v1/runs/{rid}/conflicts", params=q)
        assert r1.status_code == 200 and r1.headers["x-cache"] == "MISS"
        body = r1.json()
        assert body["count"] == 1 and body["conflicts"][0]["conflict_type"] == "crossing"
        r2 = c.get(f"/v1/runs/{rid}/conflicts", params=q)
        assert r2.headers["x-cache"] == "HIT" and r2.json() == body

        rr = c.get(f"/v1/runs/{rid}/conflicts", params={"conflict_type": "rear_end"}).json()
        assert rr["count"] == 1
        assert c.get(f"/v1/runs/{rid}/heatmap", params={"cell_m": 5}).json()["cells"]
        assert c.get(f"/v1/runs/{rid}").json()["run_id"] == rid


def test_errors(fzp_file: Path, tmp_path: Path) -> None:
    with _client() as c:
        assert c.get("/v1/runs/nope/conflicts").status_code == 404
        bad = tmp_path / "bad.fzp"
        bad.write_text("not a vehicle record\n")
        with open(bad, "rb") as fh:
            assert c.post("/v1/runs", files={"file": ("bad.fzp", fh)}).status_code == 422
        with open(fzp_file, "rb") as fh:
            assert c.post("/v1/runs", files={"file": ("x.csv", fh)}).status_code == 415
        with open(fzp_file, "rb") as fh:
            rid = c.post("/v1/runs", files={"file": ("s.fzp", fh)}).json()["run_id"]
        assert c.get(f"/v1/runs/{rid}/conflicts", params={"min_x": 0}).status_code == 422
        assert c.get(f"/v1/runs/{rid}/conflicts", params={"min_x": 5, "min_y": 0, "max_x": 1, "max_y": 1}).status_code == 422


def test_run_eviction(fzp_file: Path) -> None:
    with _client() as c:
        ids = []
        for _ in range(3):
            with open(fzp_file, "rb") as fh:
                ids.append(c.post("/v1/runs", files={"file": ("s.fzp", fh)}).json()["run_id"])
        assert c.get(f"/v1/runs/{ids[0]}").status_code == 404
        assert c.get(f"/v1/runs/{ids[2]}").status_code == 200
