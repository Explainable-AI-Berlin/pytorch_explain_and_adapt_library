import json

import torch

from peal.web.summarize import summarize


def test_summarize(tmp_path):
    run = tmp_path / "run"
    (run / "direction_collages" / "rank000_dir42_fries").mkdir(parents=True)
    (
        run
        / "direction_collages"
        / "rank000_dir42_fries"
        / "rank000_dir42_pair0_conf0.90.png"
    ).write_bytes(b"png")
    torch.save(
        [
            {
                "direction_idx": 42,
                "dimension_name": "fries",
                "success_count": 5,
                "ambient_flip_count": 4,
                "latent_flip_count": 50,
                "total_attempted": 8,
            },
            {
                "direction_idx": 7,
                "dimension_name": "bun",
                "success_count": 2,
                "ambient_flip_count": 2,
                "latent_flip_count": 20,
                "total_attempted": 8,
            },
        ],
        run / "sweep_results.pt",
    )
    (run / "direction_feedback.txt").write_text(
        "direction=42, feedback=false, success_count=5, n_true=1, n_false=3, n_pairs=4\n"
        "direction=7, feedback=true, success_count=2, n_true=2, n_false=0, n_pairs=2\n"
    )
    (run / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "log.txt").write_text("[DiDAE] Post-CFKD test accuracy: 0.9125\n")
    res = summarize(str(tmp_path))
    assert [d["direction_idx"] for d in res["directions"]] == [42, 7]
    assert res["directions"][0]["feedback"] == "false"
    assert res["directions"][0]["collages"] == [
        "run/direction_collages/rank000_dir42_fries/rank000_dir42_pair0_conf0.90.png"
    ]
    assert res["false_directions"] == [42]
    assert res["accuracies"]["post_cfkd_test_accuracy"] == 0.9125
    assert res["outputs"]["model.onnx"] == "run/model.onnx"
    assert res["corrected"] is True
    assert json.load(open(tmp_path / "results.json"))["corrected"] is True
