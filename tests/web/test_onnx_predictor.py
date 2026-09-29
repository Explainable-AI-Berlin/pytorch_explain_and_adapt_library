import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx2torch")

from peal.architectures.onnx_predictor import (  # noqa: E402
    OnnxPredictor,
    export_to_onnx,
    inspect_onnx,
    load_onnx_as_torch,
    load_onnx_predictor,
    normalize_onnx_graph,
    select_onnx_outputs,
)


class TinyNet(torch.nn.Module):
    def __init__(self, n_out=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(
            8, 8, 3, padding=1
        )  # zero-init bias -> Identity alias
        self.fc = torch.nn.Linear(8, n_out)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = x.mean((2, 3))
        return self.fc(x)


@pytest.fixture
def tiny_onnx(tmp_path):
    torch.manual_seed(0)
    net = TinyNet().eval()
    with torch.no_grad():
        net.conv1.bias.zero_()
        net.conv2.bias.zero_()
    path = str(tmp_path / "tiny.onnx")
    export_to_onnx(net, path, input_shape=[3, 32, 32], opset_version=11)
    return path, net


def test_inspect(tiny_onnx):
    path, _ = tiny_onnx
    info = inspect_onnx(path)
    assert info["input_shape"] == [None, 3, 32, 32]
    assert info["num_outputs"] == 4
    assert info["input_name"] == "input"


def test_normalize_resolves_identity_aliases(tiny_onnx):
    path, _ = tiny_onnx
    model = onnx.load(path)
    model, n_const, n_ident = normalize_onnx_graph(model)
    assert not any(
        n.op_type == "Identity"
        and n.input[0] in {i.name for i in model.graph.initializer}
        for n in model.graph.node
    )
    onnx.checker.check_model(model)


def test_convert_train_pickle_export(tiny_onnx, tmp_path):
    path, net = tiny_onnx
    pred = load_onnx_as_torch(path, device="cpu")
    assert isinstance(pred, OnnxPredictor)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(pred(x), net(x), atol=1e-4)
    # trainable
    params = [p for p in pred.parameters() if p.requires_grad]
    assert params
    opt = torch.optim.SGD(params, lr=0.1)
    loss = torch.nn.functional.cross_entropy(pred(x), torch.tensor([0, 1, 2]))
    loss.backward()
    opt.step()
    with torch.no_grad():
        assert not torch.allclose(pred(x), net(x), atol=1e-4)
    # pickles like a .cpl
    torch.save(pred, tmp_path / "m.cpl")
    back = torch.load(tmp_path / "m.cpl", weights_only=False)
    with torch.no_grad():
        assert torch.allclose(back(x), pred(x))
    # exports with the original interface
    out = export_to_onnx(back, str(tmp_path / "ft.onnx"))
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": x.numpy()})[0]
    with torch.no_grad():
        assert np.abs(y - back(x).numpy()).max() < 1e-4
    assert inspect_onnx(out)["input_shape"] == [None, 3, 32, 32]


def test_runtime_fallback_env(tiny_onnx, monkeypatch):
    path, net = tiny_onnx
    monkeypatch.setenv("PEAL_ONNX_CONVERT", "0")
    fn = load_onnx_predictor(path, device="cpu")
    assert not isinstance(fn, torch.nn.Module)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(fn(x), net(x), atol=1e-4)


def test_select_outputs(tiny_onnx, tmp_path):
    path, net = tiny_onnx
    dst = select_onnx_outputs(path, str(tmp_path / "bin.onnx"), [3, 1])
    info = inspect_onnx(dst)
    assert info["num_outputs"] == 2
    x = torch.randn(2, 3, 32, 32)
    sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    y = sess.run(None, {"input": x.numpy()})[0]
    with torch.no_grad():
        ref = net(x)[:, [3, 1]].numpy()
    assert np.abs(y - ref).max() < 1e-5
    # and the selected model converts to torch too
    pred = load_onnx_as_torch(dst, device="cpu")
    with torch.no_grad():
        assert np.abs(pred(x).numpy() - ref).max() < 1e-4
    with pytest.raises(ValueError):
        select_onnx_outputs(path, str(tmp_path / "bad.onnx"), [0, 9])
    with pytest.raises(ValueError):
        select_onnx_outputs(path, str(tmp_path / "bad.onnx"), [1, 1])


def test_select_outputs_is_idempotent(tmp_path):
    from peal.architectures.onnx_predictor import (
        export_to_onnx,
        inspect_onnx,
        select_onnx_outputs,
    )

    src = str(tmp_path / "m.onnx")
    export_to_onnx(TinyNet(), src, input_shape=[3, 8, 8])
    once = select_onnx_outputs(src, str(tmp_path / "once.onnx"), [1, 0])
    # re-selecting from an already selected model used to fail with
    # "output_selected_indices initializer name is not unique"
    twice = select_onnx_outputs(once, str(tmp_path / "twice.onnx"), [1, 0])
    same = select_onnx_outputs(once, str(tmp_path / "same.onnx"), [0, 1])
    assert inspect_onnx(twice)["num_outputs"] == 2
    assert open(same, "rb").read() == open(once, "rb").read()


def test_final_linear_roundtrip(tmp_path):
    import onnxruntime as ort

    from peal.architectures.onnx_predictor import (
        export_to_onnx,
        find_final_linear,
        select_onnx_outputs,
        truncate_to_features,
        write_final_linear,
    )

    src = str(tmp_path / "m.onnx")
    export_to_onnx(TinyNet(), src, input_shape=[3, 8, 8])
    sel = select_onnx_outputs(src, str(tmp_path / "sel.onnx"), [1, 0])
    info = find_final_linear(sel)
    assert info is not None and info["selected"] == [1, 0]
    feat = truncate_to_features(sel, str(tmp_path / "f.onnx"), info["feature_tensor"])
    x = np.random.rand(3, 3, 8, 8).astype(np.float32)
    name = ort.InferenceSession(sel).get_inputs()[0].name
    f = ort.InferenceSession(feat).run(None, {name: x})[0].reshape(3, -1)
    assert f.shape[1] == info["n_features"]
    W = np.random.randn(2, info["n_features"]).astype(np.float32)
    b = np.array([0.5, -0.5], np.float32)
    out = write_final_linear(sel, str(tmp_path / "w.onnx"), info, W, b)
    y = ort.InferenceSession(out).run(None, {name: x})[0]
    assert np.allclose(y, f @ W.T + b, atol=1e-5)


class FixedBatchNet(torch.nn.Module):
    """Model-zoo style graph: batch fixed to 1, also inside a Reshape constant
    (ShuffleNet's channel shuffle)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(1, 2, 2, 8, 8).transpose(1, 2).reshape(1, 4, 8, 8)
        return self.fc(x.mean((2, 3)))


@pytest.mark.parametrize("convert", ["1", "0"])
def test_fixed_batch_graph_runs_on_batches(tmp_path, monkeypatch, convert):
    torch.manual_seed(0)
    net = FixedBatchNet().eval()
    path = str(tmp_path / "fixed.onnx")
    torch.onnx.export(
        net,
        torch.randn(1, 3, 8, 8),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=13,
    )
    assert inspect_onnx(path)["input_shape"][0] == 1
    monkeypatch.setenv("PEAL_ONNX_CONVERT", convert)
    model = load_onnx_predictor(path, device="cpu")
    if convert == "1":
        assert isinstance(model, OnnxPredictor)  # no onnxruntime fallback
    x = torch.randn(5, 3, 8, 8)
    with torch.no_grad():
        want = torch.cat([net(x[i : i + 1]) for i in range(5)])
        got = model(x)
    assert got.shape == (5, 2)
    assert torch.allclose(got, want, atol=1e-5)


def test_truncated_fixed_batch_graph_keeps_its_batch(tmp_path):
    from peal.architectures.onnx_predictor import (
        find_final_linear,
        truncate_to_features,
    )

    torch.manual_seed(0)
    net = FixedBatchNet().eval()
    path = str(tmp_path / "fixed.onnx")
    torch.onnx.export(
        net,
        torch.randn(1, 3, 8, 8),
        path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=13,
    )
    info = find_final_linear(path)
    assert info is not None
    feat = truncate_to_features(
        path, str(tmp_path / "feat.onnx"), info["feature_tensor"]
    )
    assert inspect_onnx(feat)["input_shape"][0] == 1
    model = load_onnx_predictor(feat, device="cpu")
    with torch.no_grad():
        f = model(torch.randn(5, 3, 8, 8))
    assert f.shape[0] == 5
