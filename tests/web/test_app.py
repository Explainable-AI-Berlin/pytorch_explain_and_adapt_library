import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnx2torch")
from fastapi.testclient import TestClient  # noqa: E402

from peal.architectures.onnx_predictor import export_to_onnx, inspect_onnx  # noqa: E402
from peal.web.app import create_app  # noqa: E402
from tests.web.test_ingest import make_zip  # noqa: E402
from tests.web.test_onnx_predictor import TinyNet  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAL_RAE_WEIGHTS", "hf://org/peal-rae")
    monkeypatch.setenv("PEAL_WEB_MIN_IMAGES", "1")  # the test zips are tiny
    app = create_app(jobs_root=str(tmp_path / "jobs"), start_worker=False)
    with TestClient(app) as c:
        yield c, app


def _onnx_bytes(tmp_path, n_out=5):
    path = str(tmp_path / "m.onnx")
    export_to_onnx(
        TinyNet(n_out).eval(), path, input_shape=[3, 64, 64], opset_version=11
    )
    return open(path, "rb").read()


def test_full_submission_flow(client, tmp_path):
    c, app = client
    assert c.get("/").status_code == 200
    q = c.get("/api/queue").json()
    assert q["queued"] == 0 and q["estimated_wait_seconds_for_new_job"] == 0

    r = c.post(
        "/api/uploads/model",
        files={"file": ("m.onnx", _onnx_bytes(tmp_path), "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model = r.json()
    assert model["num_outputs"] == 5 and model["input_shape"] == [None, 3, 64, 64]

    zpath = make_zip(tmp_path / "d.zip", {"cat": 3, "dog": 3, "bird": 2})
    r = c.post(
        "/api/uploads/dataset",
        files={"file": ("d.zip", open(zpath, "rb").read(), "application/zip")},
    )
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["classes"] == {"cat": 3, "dog": 3, "bird": 2}

    form = dict(
        model_token=model["token"],
        dataset_token=ds["token"],
        class_a="dog",
        class_b="cat",
        class_a_name="Dog",
        class_b_name="Cat",
        output_index_a=4,
        output_index_b=1,
        normalization="imagenet",
        input_height=64,
        input_width=64,
        accept_terms="true",
    )
    # mandatory-field validation
    bad = dict(form, input_height=32)
    assert c.post("/api/jobs", data=bad).status_code == 400
    bad = dict(form, output_index_a=7)
    assert "outside" in c.post("/api/jobs", data=bad).json()["detail"]
    bad = dict(form, class_b="dog")
    assert c.post("/api/jobs", data=bad).status_code == 400
    bad = dict(form, accept_terms="false")
    assert c.post("/api/jobs", data=bad).status_code == 400
    bad = dict(form, class_a_name=" ")
    assert c.post("/api/jobs", data=bad).status_code == 400

    r = c.post("/api/jobs", data=form)
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "queued" and job["queue_position"] == 1
    job_dir = app.state.store.job_dir(job["id"])
    assert inspect_onnx(os.path.join(job_dir, "model.onnx"))["num_outputs"] == 2
    rows = (
        open(os.path.join(job_dir, "dataset", "data.csv")).read().strip().splitlines()
    )
    assert len(rows) == 7
    assert os.path.isfile(os.path.join(job_dir, "config.yaml"))
    # uploads are consumed
    assert not os.path.isdir(os.path.join(app.state.uploads_root, model["token"]))

    s = c.get(f"/api/jobs/{job['id']}").json()
    assert s["status"] == "queued" and s["estimated_wait_seconds"] == 0
    assert c.get("/api/jobs/deadbeefdead").status_code == 404
    assert c.get(f"/api/jobs/{job['id']}/results").status_code == 404

    # second job waits one median behind the first
    r = c.post(
        "/api/uploads/model",
        files={"file": ("m.onnx", _onnx_bytes(tmp_path), "application/octet-stream")},
    )
    r2 = c.post(
        "/api/uploads/dataset",
        files={"file": ("d.zip", open(zpath, "rb").read(), "application/zip")},
    )
    form2 = dict(form, model_token=r.json()["token"], dataset_token=r2.json()["token"])
    job2 = c.post("/api/jobs", data=form2).json()
    assert job2["queue_position"] == 2 and job2["estimated_wait_seconds"] > 0

    # cancel a queued job
    assert c.post(f"/api/jobs/{job2['id']}/cancel").json()["status"] == "cancelled"


def test_feedback_round_trip_and_files(client, tmp_path):
    c, app = client
    store = app.state.store
    job = store.create({"class_a_name": "a", "class_b_name": "b"})
    store.update(job["id"], status="awaiting_feedback")
    job_dir = store.job_dir(job["id"])
    fdir = os.path.join(job_dir, "feedback")
    cdir = os.path.join(job_dir, "run", "direction_collages", "rank000_dir1_x")
    os.makedirs(fdir)
    os.makedirs(cdir)
    png = os.path.join(cdir, "c0.png")
    open(png, "wb").write(b"png")
    json.dump(
        {
            "round": 0,
            "items": [
                {
                    "index": 0,
                    "collage": png,
                    "source_class": 0,
                    "target_class": 1,
                    "confidence": 0.9,
                },
                {
                    "index": 2,
                    "collage": png,
                    "source_class": 1,
                    "target_class": 0,
                    "confidence": 0.8,
                },
            ],
        },
        open(os.path.join(fdir, "feedback_request_0.json"), "w"),
    )
    s = c.get(f"/api/jobs/{job['id']}").json()
    assert s["feedback_pending"] is True
    fb = c.get(f"/api/jobs/{job['id']}/feedback").json()
    assert fb["round"] == 0 and fb["n_items"] == 2
    assert c.get(fb["items"][0]["collage_url"]).content == b"png"
    # incomplete / invalid verdicts are refused
    assert (
        c.post(
            f"/api/jobs/{job['id']}/feedback",
            json={"round": 0, "verdicts": {"0": "true"}},
        ).status_code
        == 400
    )
    assert (
        c.post(
            f"/api/jobs/{job['id']}/feedback",
            json={"round": 0, "verdicts": {"0": "maybe", "2": "false"}},
        ).status_code
        == 400
    )
    r = c.post(
        f"/api/jobs/{job['id']}/feedback",
        json={"round": 0, "verdicts": {"0": "true", "2": "false"}},
    )
    assert r.status_code == 200
    assert json.load(open(os.path.join(fdir, "feedback_response_0.json")))[
        "verdicts"
    ] == {"0": "true", "2": "false"}
    assert c.get(f"/api/jobs/{job['id']}/feedback").json()["round"] is None
    # file serving: traversal and non-allowed paths
    assert c.get(
        f"/api/jobs/{job['id']}/files/../../{job['id']}/job.json"
    ).status_code in (403, 404)
    assert c.get(f"/api/jobs/{job['id']}/files/job.json").status_code == 403
    assert (
        c.get(
            f"/api/jobs/{job['id']}/files/run/direction_collages/rank000_dir1_x/c0.png"
        ).status_code
        == 200
    )


def test_bad_uploads(client, tmp_path):
    c, _ = client
    r = c.post(
        "/api/uploads/model",
        files={"file": ("m.onnx", b"not onnx", "application/octet-stream")},
    )
    assert r.status_code == 400
    z = make_zip(tmp_path / "one.zip", {"only": 2})
    r = c.post(
        "/api/uploads/dataset",
        files={"file": ("d.zip", open(z, "rb").read(), "application/zip")},
    )
    assert r.status_code == 400
    r = c.post(
        "/api/uploads/dataset", files={"file": ("d.zip", b"zip?", "application/zip")}
    )
    assert r.status_code == 400


def test_group_by_direction(tmp_path):
    from peal.web.app import group_by_direction

    run = tmp_path / "direction_collages"
    folder = run / "rank000_dir7_OFF_crane_SAE_4599_TO_ON_pool_SAE_5433"
    folder.mkdir(parents=True)
    (folder / "0000001_collage.png").write_bytes(b"png")
    items = [
        {"index": 0, "collage": str(run / "rank001_dir2_pair0_conf0.60.png")},
        {"index": 1, "collage": str(run / "rank000_dir7_pair0_conf0.55.png")},
        {"index": 2, "collage": str(run / "rank000_dir7_pair1_conf0.52.png")},
        {"index": 3, "collage": str(run / "unexpected.png")},
    ]
    groups = group_by_direction(items)
    assert [g["rank"] for g in groups] == [0, 1, None]
    assert groups[0]["name"] == "OFF crane #4599 → ON pool #5433"
    assert [it["index"] for it in groups[0]["items"]] == [1, 2]
    assert "detail_collage" not in groups[0]["items"][0]
    assert groups[0]["items"][1]["detail_collage"].endswith("0000001_collage.png")
    assert groups[1]["name"] == "direction 2"


def test_too_small_dataset_is_rejected(client, tmp_path, monkeypatch):
    c, _ = client
    monkeypatch.setenv("PEAL_WEB_MIN_IMAGES", "50")
    zpath = make_zip(tmp_path / "small.zip", {"cat": 5, "dog": 5})
    with open(zpath, "rb") as f:
        r = c.post(
            "/api/uploads/dataset", files={"file": ("small.zip", f, "application/zip")}
        )
    assert r.status_code == 400
    assert "at least 50" in r.json()["detail"]
