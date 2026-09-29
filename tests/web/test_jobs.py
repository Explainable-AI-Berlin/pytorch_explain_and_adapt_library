import os
import time

from peal.web.jobs import JobStore, Worker, latest_activity, pending_feedback_round


def test_store_and_wait_estimate(tmp_path):
    store = JobStore(str(tmp_path))
    assert store.list() == []
    assert store.stats()["queued"] == 0
    a = store.create({"x": 1})
    store.update(a["id"], status="running", started=time.time() - 600)
    b = store.create({})
    store.update(b["id"], status="queued")
    c = store.create({})
    store.update(c["id"], status="queued")
    # no finished jobs yet -> default median
    median = store.median_duration()
    assert store.queue_position(a["id"]) == 1
    assert store.queue_position(c["id"]) == 3
    est_b = store.estimate_wait_seconds(b["id"])
    est_c = store.estimate_wait_seconds(c["id"])
    assert abs(est_b - max(60.0, median - 600)) < 5
    assert abs(est_c - (est_b + median)) < 5
    assert store.stats()["queued"] == 2 and store.stats()["running"] == 1
    # finished jobs feed the median
    for d in (100.0, 300.0, 200.0):
        j = store.create({})
        store.update(j["id"], status="finished", duration=d)
    assert store.median_duration() == 200.0
    assert store.next_queued()["id"] == b["id"]


def test_pending_feedback_round(tmp_path):
    assert pending_feedback_round(str(tmp_path / "none")) is None
    (tmp_path / "feedback_request_0.json").write_text("{}")
    (tmp_path / "feedback_request_1.json").write_text("{}")
    assert pending_feedback_round(str(tmp_path)) == 0
    (tmp_path / "feedback_response_0.json").write_text("{}")
    assert pending_feedback_round(str(tmp_path)) == 1
    (tmp_path / "feedback_response_1.json").write_text("{}")
    assert pending_feedback_round(str(tmp_path)) is None


def test_steps_and_queue_listing(tmp_path):
    store = JobStore(str(tmp_path))
    a = store.create({})
    store.mark_step(a["id"], "queued", status="queued")
    first = store.get(a["id"])["steps"]["queued"]
    store.mark_step(a["id"], "queued", status="queued")
    assert store.get(a["id"])["steps"]["queued"] == first  # first time kept
    store.mark_step(a["id"], "2-6", status="running", started=time.time())
    job = store.get(a["id"])
    assert job["step"] == "2-6" and set(job["steps"]) == {"queued", "2-6"}
    b = store.create({})
    store.mark_step(b["id"], "queued", status="queued")
    listing = store.stats()["jobs"]
    assert [(j["position"], j["status"]) for j in listing] == [
        (1, "running"),
        (2, "queued"),
    ]
    assert all("id" not in j for j in listing)


def test_latest_activity(tmp_path):
    log = tmp_path / "log.txt"
    assert latest_activity(str(log)) is None
    log.write_text(
        "/usr/bin/python run_didae.py --config x.yaml\n"
        "[DiDAE] Step 2-6: Sweeping\n"
        "bounds:  10%|#  | 1/10\rbounds:  20%|## | 2/10\n"
        "Loading config from {'a': 1}\n\n"
    )
    assert latest_activity(str(log)) == "bounds:  20%|## | 2/10"
    log.write_text("x" * 500 + "\n")
    assert len(latest_activity(str(log), max_chars=50)) == 50


def test_worker_runs_a_fake_job(tmp_path):
    """The worker runs the job's command, tracks the stage from the log, and
    marks the job finished with a duration; the command is faked here."""
    store = JobStore(str(tmp_path))
    job = store.create({})
    job_dir = store.job_dir(job["id"])
    os.makedirs(os.path.join(job_dir, "run"))
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys, time\n"
        "print('[DiDAE] Step 1: distil'); sys.stdout.flush(); time.sleep(0.3)\n"
        "print('[DiDAE] Step 9: finetune'); sys.stdout.flush()\n"
    )
    worker = Worker(store, poll_interval=0.05)
    worker.command = lambda d: [worker.python, str(script)]
    checked = []
    worker.check_accuracy = lambda d, log, env=None: checked.append(d)
    store.update(job["id"], status="queued")
    worker.run_job(job["id"])
    j = store.get(job["id"])
    assert j["status"] == "finished"
    assert checked == [job_dir]
    assert list(j["steps"])[:2] == ["check", "starting"]
    assert j["duration"] > 0
    assert os.path.isfile(os.path.join(job_dir, "results.json"))
    assert "[DiDAE] Step 9" in open(os.path.join(job_dir, "log.txt")).read()
    assert Worker._stage_from_log(os.path.join(job_dir, "log.txt")) == "9"


def test_worker_failure(tmp_path):
    store = JobStore(str(tmp_path))
    job = store.create({})
    script = tmp_path / "fail.py"
    script.write_text("import sys; print('boom'); sys.exit(3)\n")
    worker = Worker(store, poll_interval=0.05)
    worker.command = lambda d: [worker.python, str(script)]
    worker.run_job(job["id"])
    j = store.get(job["id"])
    assert j["status"] == "failed" and "code 3" in j["error"]
