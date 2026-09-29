import json
import os
import threading
import time

import pytest

from peal.teachers.web_feedback_teacher import WebFeedbackTeacher
from peal.teachers.teacher_factory import get_teacher


def test_round_trip(tmp_path):
    t = WebFeedbackTeacher(str(tmp_path), poll_interval=0.05, timeout_s=10)
    collages = [str(tmp_path / f"c{i}.png") for i in range(4)]
    # 0: judged, 1: student wrong, 2: not swapped, 3: judged
    args = dict(
        collage_path_list=collages,
        y_target_end_confidence_list=[0.9, 0.9, 0.2, 0.7],
        y_source_list=[0, 0, 1, 1],
        y_list=[0, 1, 1, 1],
        y_target_list=[1, 1, 0, 0],
    )

    def answer():
        req_path = t.request_path(0)
        while not os.path.isfile(req_path):
            time.sleep(0.02)
        req = json.load(open(req_path))
        assert [it["index"] for it in req["items"]] == [0, 3]
        assert req["items"][0]["target_class"] == 1
        # an incomplete response must be ignored, then the full one accepted
        json.dump({"verdicts": {"0": "true"}}, open(t.response_path(0), "w"))
        time.sleep(0.15)
        json.dump(
            {"verdicts": {"0": "true", "3": "false"}}, open(t.response_path(0), "w")
        )

    th = threading.Thread(target=answer)
    th.start()
    fb = t.get_feedback(**args)
    th.join()
    assert fb == ["true", "student incorrect!", "student not swapped!", "false"]
    assert t.round == 1


def test_timeout(tmp_path):
    t = WebFeedbackTeacher(str(tmp_path), poll_interval=0.02, timeout_s=0.1)
    with pytest.raises(TimeoutError):
        t.get_feedback(
            collage_path_list=["a.png"],
            y_target_end_confidence_list=[0.9],
            y_source_list=[0],
            y_list=[0],
            y_target_list=[1],
        )


def test_auto_verdict_and_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAL_WEB_FEEDBACK_AUTO", "false")
    t = get_teacher(
        teacher={"type": "web", "dir": str(tmp_path / "fb")},
        output_size=2,
        adaptor_config=None,
        dataset=None,
    )
    assert isinstance(t, WebFeedbackTeacher)
    fb = t.get_feedback(
        collage_path_list=["a.png", "b.png"],
        y_target_end_confidence_list=[0.9, 0.9],
        y_source_list=[0, 1],
        y_list=[0, 1],
        y_target_list=[1, 0],
    )
    assert fb == ["false", "false"]
    assert not os.path.exists(t.request_path(0))
    t2 = get_teacher(
        teacher="web:" + str(tmp_path / "fb2"),
        output_size=2,
        adaptor_config=None,
        dataset=None,
    )
    assert t2.feedback_dir.endswith("fb2")
