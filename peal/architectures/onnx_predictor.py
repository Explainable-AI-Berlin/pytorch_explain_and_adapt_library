"""
ONNX classifiers as PEAL predictors.

PEAL accepted ``.onnx`` predictors before, but only as an onnxruntime closure:
fine for the analysis half (DiDAE's sweep, CFKD's counterfactual search only
ever *query* the student), useless for the correction half, whose finetuning
needs an ``nn.Module`` with parameters. This module closes that gap:

* :func:`load_onnx_as_torch` converts the graph with ``onnx2torch`` into a
  trainable :class:`OnnxPredictor` (an ``nn.Module`` that pickles, deep-copies
  and re-exports), after normalising the graph so the converter accepts what
  ``torch.onnx.export`` produces (weights aliased through ``Identity`` nodes,
  ``Constant`` nodes instead of initializers). The converted module is checked
  against onnxruntime on a random batch before it is trusted.
* :func:`load_onnx_runtime` is the old closure, kept as the fallback for graphs
  onnx2torch cannot convert, with a CPU provider fallback instead of the
  hard-coded CUDA provider (the CPU-only wheel is what pip gives you on
  aarch64 hosts such as the DGX Spark).
* :func:`export_to_onnx` writes a finetuned predictor back out, so a user who
  uploaded an ONNX file gets a corrected ONNX file back.
* :func:`inspect_onnx` reads the declared input/output shapes, which the web
  demo uses to validate a user's mandatory fields against the graph.

Set ``PEAL_ONNX_CONVERT=0`` to skip the conversion and always use the runtime
closure (the behaviour every pre-2026-09-25 config saw).
"""

import copy
import os
import warnings

import numpy as np
import torch
from torch import nn
from peal.log import get_logger

_log = get_logger(__name__)


def _ort_providers():
    """Execution providers to request: CUDA then CPU, restricted to those the
    installed onnxruntime wheel offers."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available]
    return providers or available


def _ort_session(path):
    """Open an ``InferenceSession`` for ``path`` with explicit thread counts."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    # Explicit thread counts: onnxruntime otherwise pins threads to cores and
    # logs pthread_setaffinity_np errors inside cgroup-limited containers.
    opts.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=_ort_providers())


def inspect_onnx(path):
    """Declared interface of an ONNX graph.

    Returns a dict with ``input_name``, ``input_shape`` (list; symbolic dims are
    None), ``output_name``, ``output_shape``, ``num_outputs`` (the size of the
    last output axis, i.e. the number of classes for a classifier, or None when
    symbolic), ``opset`` and ``ir_version``. Raises ``ValueError`` when the graph
    is not a single-input single-output model.

    Parameters
    ----------
    path : str
        Path of the ``.onnx`` file. External weight data is not loaded.

    Returns
    -------
    dict
        Keys ``input_name``, ``input_shape``, ``output_name``, ``output_shape``,
        ``num_outputs``, ``opset``, ``ir_version`` and ``producer``.

    Raises
    ------
    ValueError
        If the graph has more than one non-initializer input or no output.
    """
    import onnx

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1 or len(graph.output) < 1:
        raise ValueError(
            f"{path}: expected exactly one graph input and at least one output, "
            f"found {len(inputs)} inputs and {len(graph.output)} outputs"
        )

    def dims(value_info):
        tt = value_info.type.tensor_type
        out = []
        for d in tt.shape.dim:
            out.append(int(d.dim_value) if d.HasField("dim_value") else None)
        return out

    in_shape = dims(inputs[0])
    out_shape = dims(graph.output[0])
    opset = None
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            opset = int(o.version)
    return {
        "input_name": inputs[0].name,
        "input_shape": in_shape,
        "output_name": graph.output[0].name,
        "output_shape": out_shape,
        "num_outputs": out_shape[-1] if len(out_shape) > 0 else None,
        "opset": opset,
        "ir_version": int(model.ir_version),
        "producer": model.producer_name,
    }


def normalize_onnx_graph(model):
    """Rewrite an ONNX model in place so onnx2torch accepts it.

    ``torch.onnx.export`` de-duplicates identical weight tensors (every zero
    conv bias, say) into one initializer plus ``Identity`` nodes, and some
    exporters emit weights as ``Constant`` nodes. onnx2torch looks weights up in
    ``graph.initializer`` only, so both forms fail with a ``KeyError`` on the
    aliased name. Returns ``(model, n_constants_moved, n_identities_resolved)``.

    Parameters
    ----------
    model : onnx.ModelProto
        Loaded model; its graph is modified in place.

    Returns
    -------
    tuple
        ``(model, n_constants_moved, n_identities_resolved)``: the same model
        object, the number of ``Constant`` nodes turned into initializers and
        the number of ``Identity`` aliases replaced by copied initializers.
        Both node kinds are removed from ``graph.node``.
    """
    import onnx

    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    n_const = n_ident = 0
    changed = True
    while changed:
        changed = False
        keep = []
        for node in graph.node:
            if (
                node.op_type == "Constant"
                and len(node.output) == 1
                and node.output[0] not in inits
            ):
                tensor = None
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = onnx.helper.get_attribute_value(attr)
                if tensor is not None:
                    tensor = copy.deepcopy(tensor)
                    tensor.name = node.output[0]
                    graph.initializer.append(tensor)
                    inits[tensor.name] = tensor
                    n_const += 1
                    changed = True
                    continue
            if (
                node.op_type == "Identity"
                and node.input[0] in inits
                and node.output[0] not in inits
            ):
                tensor = copy.deepcopy(inits[node.input[0]])
                tensor.name = node.output[0]
                graph.initializer.append(tensor)
                inits[tensor.name] = tensor
                n_ident += 1
                changed = True
                continue
            keep.append(node)
        del graph.node[:]
        graph.node.extend(keep)
    return model, n_const, n_ident


class OnnxPredictor(nn.Module):
    """A trainable torch module converted from an ONNX classifier.

    ``forward`` takes the same ``[B, C, H, W]`` float tensor the ONNX graph
    declared and returns its first output. ``source`` and ``input_shape`` are
    kept so the corrected model can be exported with the interface the user
    uploaded.

    Parameters
    ----------
    graph_module : torch.fx.GraphModule
        The module ``onnx2torch.convert`` produced.
    source : str, optional
        Absolute path of the ONNX file the module came from.
    input_name : str
        Name of the graph input, kept for re-export.
    input_shape : list of int or None, optional
        Declared input shape (symbolic dims as ``None``); used by
        :func:`export_to_onnx` when no shape is given.
    """

    def __init__(self, graph_module, source=None, input_name="input", input_shape=None):
        """Wrap ``graph_module`` and remember the ONNX interface it came from."""
        super().__init__()
        self.graph_module = graph_module
        self.source = source
        self.input_name = input_name
        self.input_shape = list(input_shape) if input_shape is not None else None

    def forward(self, x):
        """Run the converted graph and return its first output tensor (in
        chunks when the graph declares a fixed batch size)."""
        return _run_in_fixed_batches(
            self._forward_graph, x, _fixed_batch(self.input_shape)
        )

    def _forward_graph(self, x):
        out = self.graph_module(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _fixed_batch(input_shape):
    """The batch size a graph declares as a fixed number (model-zoo exports
    often fix it to 1, also in their Reshape constants), or None if symbolic."""
    if input_shape and isinstance(input_shape[0], int) and input_shape[0] > 0:
        return int(input_shape[0])
    return None


def _run_in_fixed_batches(fn, x, batch):
    """Call ``fn`` on ``x`` in chunks of exactly ``batch`` samples (the last
    chunk padded by repeating its final sample) and concatenate the outputs."""
    if batch is None or x.shape[0] == batch:
        return fn(x)
    outs = []
    for start in range(0, x.shape[0], batch):
        chunk = x[start : start + batch]
        n = chunk.shape[0]
        if n < batch:
            chunk = torch.cat([chunk, chunk[-1:].expand(batch - n, *chunk.shape[1:])])
        outs.append(fn(chunk)[:n])
    return torch.cat(outs)


def load_onnx_runtime(path, device="cuda"):
    """The pre-conversion behaviour: an onnxruntime closure (not trainable).

    Parameters
    ----------
    path : str
        Path of the ``.onnx`` file.
    device : str or torch.device
        Device the output tensor is moved to.

    Returns
    -------
    callable
        ``onnx_model(input_data)`` that feeds the tensor to onnxruntime as
        float32 and returns the first output as a tensor on ``device``. The
        closure carries the attribute ``onnx_path``.
    """
    session = _ort_session(path)
    input_name = session.get_inputs()[0].name
    batch = _fixed_batch(inspect_onnx(path)["input_shape"])
    device = torch.device(device) if not isinstance(device, torch.device) else device

    def run(x):
        session_output = session.run(
            None, {input_name: x.detach().cpu().numpy().astype(np.float32)}
        )
        return torch.from_numpy(session_output[0])

    def onnx_model(input_data):
        return _run_in_fixed_batches(run, input_data, batch).to(device)

    onnx_model.onnx_path = path
    return onnx_model


def load_onnx_as_torch(path, device="cuda", check=True, atol=1e-3):
    """Convert an ONNX classifier into an :class:`OnnxPredictor`.

    Raises ``RuntimeError`` (with the converter's message) when the graph has an
    operator onnx2torch does not support, or when the converted module disagrees
    with onnxruntime by more than ``atol`` on a random batch.

    Parameters
    ----------
    path : str
        Path of the ``.onnx`` file.
    device : str or torch.device
        Device the returned module is moved to.
    check : bool
        Compare the converted module against onnxruntime on a random batch of
        two inputs (symbolic dims set to 1, ``[1, 3, 224, 224]`` when the
        graph declares no shape) and print a one-line summary.
    atol : float
        Maximum absolute difference tolerated by the check.

    Returns
    -------
    OnnxPredictor
        Trainable module in eval mode on ``device``.

    Raises
    ------
    RuntimeError
        If onnx2torch cannot convert the graph or the check fails.
    """
    import onnx
    import onnx2torch

    info = inspect_onnx(path)
    model = onnx.load(path)
    model, n_const, n_ident = normalize_onnx_graph(model)
    try:
        graph_module = onnx2torch.convert(model)
    except Exception as exc:  # KeyError / NotImplementedError from the converter
        raise RuntimeError(
            f"onnx2torch could not convert {path}: {type(exc).__name__}: {exc}"
        ) from exc
    predictor = OnnxPredictor(
        graph_module,
        source=os.path.abspath(path),
        input_name=info["input_name"],
        input_shape=info["input_shape"],
    )
    predictor.eval()

    if check:
        shape = [d if d is not None else 1 for d in info["input_shape"]]
        if len(shape) == 0:
            shape = [1, 3, 224, 224]
        # two samples, or the graph's fixed batch size (then the forward chunks)
        shape[0] = _fixed_batch(info["input_shape"]) or 2
        x = torch.randn(*shape)
        with torch.no_grad():
            y_torch = predictor(x).float().cpu().numpy()
        session = _ort_session(path)
        y_ort = session.run(None, {info["input_name"]: x.numpy().astype(np.float32)})[0]
        diff = float(np.abs(y_torch - y_ort).max())
        if not np.isfinite(diff) or diff > atol:
            raise RuntimeError(
                f"converted module disagrees with onnxruntime on {path}: "
                f"max abs diff {diff:.3e} > {atol}"
            )
        _log.info(
            "%s",
            f"[onnx_predictor] {os.path.basename(path)} converted to torch "
            f"(constants moved {n_const}, identity aliases {n_ident}, "
            f"max |torch - onnxruntime| = {diff:.2e}, "
            f"{sum(p.numel() for p in predictor.parameters())} parameters)",
        )
    return predictor.to(device)


def load_onnx_predictor(path, device="cuda"):
    """What ``get_predictor`` calls for a ``.onnx`` path: the trainable module
    when the conversion succeeds, the runtime closure otherwise.

    Parameters
    ----------
    path : str
        Path of the ``.onnx`` file.
    device : str or torch.device
        Device for the module or the closure's outputs.

    Returns
    -------
    OnnxPredictor or callable
        :func:`load_onnx_as_torch` result, or the :func:`load_onnx_runtime`
        closure when ``PEAL_ONNX_CONVERT=0`` or the conversion raised
        ``ImportError``/``RuntimeError`` (a warning names the reason).
    """
    if os.environ.get("PEAL_ONNX_CONVERT", "1") == "0":
        return load_onnx_runtime(path, device=device)
    try:
        return load_onnx_as_torch(path, device=device)
    except (ImportError, RuntimeError) as exc:
        warnings.warn(
            f"[onnx_predictor] falling back to the onnxruntime closure for {path}; "
            f"this predictor can be analysed but NOT finetuned: {exc}"
        )
        return load_onnx_runtime(path, device=device)


def export_to_onnx(model, path, input_shape=None, device=None, opset_version=None):
    """Export a predictor to ONNX with a dynamic batch axis.

    ``input_shape`` is ``[C, H, W]`` (or the full ``[B, C, H, W]``; the batch
    axis is replaced). An :class:`OnnxPredictor` supplies its own shape when
    none is given. Returns ``path``.

    Parameters
    ----------
    model : torch.nn.Module
        Predictor to export; switched to eval mode.
    path : str
        Destination ``.onnx`` file.
    input_shape : list of int, optional
        See above; symbolic (``None``) dims are set to 1.
    device : torch.device, optional
        Device of the dummy input; defaults to the model's first parameter
        (CPU for parameterless models).
    opset_version : int, optional
        Passed to ``torch.onnx.export`` when given.

    Returns
    -------
    str
        ``path``.

    Raises
    ------
    ValueError
        If no ``input_shape`` is available.

    Notes
    -----
    Uses the legacy (``dynamo=False``) exporter with input ``"input"``, output
    ``"output"`` and a dynamic ``batch_size`` axis on both.
    """
    if input_shape is None and isinstance(model, OnnxPredictor):
        input_shape = model.input_shape
    if input_shape is None:
        raise ValueError(
            "export_to_onnx needs input_shape for a non-ONNX-derived model"
        )
    shape = [d if d is not None else 1 for d in input_shape]
    if len(shape) == 3:
        shape = [1] + shape
    shape[0] = 1
    model = model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    dummy = torch.randn(*shape, device=device)
    kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    if opset_version is not None:
        kwargs["opset_version"] = opset_version
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)
    except TypeError:  # torch < 2.5 has no dynamo flag
        torch.onnx.export(model, dummy, path, **kwargs)
    return path


def select_onnx_outputs(src, dst, indices, output_name="output"):
    """Write a copy of the classifier at ``src`` whose single output is
    ``logits[:, indices]`` (a ``Gather`` on the last axis), so an N-way model
    becomes the binary model PEAL analyses. Returns ``dst``.

    The corrected model DiDAE exports afterwards is binary as well: it is the
    finetuned two-class head the user asked to analyse, not the original N-way
    graph.

    Parameters
    ----------
    src : str
        Source ``.onnx`` file.
    dst : str
        Destination ``.onnx`` file (written and checked with
        ``onnx.checker``).
    indices : sequence of int
        At least two distinct output indices, in the order the new outputs
        should have.
    output_name : str
        Name of the new graph output (suffixed ``_selected`` when the old
        output already has that name).

    Returns
    -------
    str
        ``dst``.

    Raises
    ------
    ValueError
        If fewer than two distinct indices are given or an index is out of
        range for a statically shaped output.
    """
    import onnx
    from onnx import TensorProto, helper

    indices = [int(i) for i in indices]
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError(f"need at least two distinct output indices, got {indices}")
    model = onnx.load(src)
    graph = model.graph
    old_out = graph.output[0]
    old_shape = [
        (int(d.dim_value) if d.HasField("dim_value") else None)
        for d in old_out.type.tensor_type.shape.dim
    ]
    if len(old_shape) > 0 and old_shape[-1] is not None:
        bad = [i for i in indices if i < 0 or i >= old_shape[-1]]
        if bad:
            raise ValueError(
                f"output indices {bad} out of range for a model with {old_shape[-1]} outputs"
            )
    if (
        old_shape
        and old_shape[-1] == len(indices)
        and indices == list(range(len(indices)))
    ):
        # Already exactly these outputs in this order (e.g. a binary model, or
        # a model this function wrote before): nothing to select.
        import shutil

        shutil.copy(src, dst)
        return dst
    taken = {t.name for t in graph.initializer} | {n.name for n in graph.node}
    idx_name, k = f"{output_name}_selected_indices", 1
    while idx_name in taken:
        idx_name, k = f"{output_name}_selected_indices_{k}", k + 1
    graph.initializer.append(
        helper.make_tensor(idx_name, TensorProto.INT64, [len(indices)], indices)
    )
    used = {o for n in graph.node for o in n.output} | {i.name for i in graph.input}
    used |= taken
    gather_out = output_name
    j = 0
    while gather_out in used or gather_out == old_out.name:
        j += 1
        gather_out = f"{output_name}_selected" + (f"_{j}" if j > 1 else "")
    graph.node.append(
        helper.make_node(
            "Gather",
            [old_out.name, idx_name],
            [gather_out],
            axis=-1,
            name=f"peal_select_outputs_{k}" if k > 1 else "peal_select_outputs",
        )
    )
    new_shape = (
        old_shape[:-1] + [len(indices)] if len(old_shape) > 0 else [None, len(indices)]
    )
    new_out = helper.make_tensor_value_info(
        gather_out,
        old_out.type.tensor_type.elem_type,
        [d if d is not None else "batch_size" for d in new_shape],
    )
    del graph.output[:]
    graph.output.append(new_out)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    return dst


#: Ops that may sit between the final linear layer and the graph output without
#: changing which output belongs to which class (a Softmax keeps the argmax).
_PASS_THROUGH_OPS = ("Identity", "Softmax", "LogSoftmax")


def find_final_linear(path):
    """Locate the classifier's final linear layer in an ONNX graph.

    Walks back from the (single) graph output through pass-through ops and at
    most one constant-index ``Gather`` on the last axis (the output selection
    :func:`select_onnx_outputs` adds) to a ``Gemm``, or a ``MatMul`` optionally
    followed by an ``Add``, whose weight (and bias) are initializers.

    Parameters
    ----------
    path : str
        ``.onnx`` file.

    Returns
    -------
    dict or None
        ``None`` when no such layer is found; otherwise ``node`` (name of the
        Gemm / MatMul), ``op``, ``feature_tensor`` (its data input),
        ``weight`` / ``bias`` (initializer names, bias ``None`` if absent),
        ``add_node`` (the bias Add of a MatMul, or ``None``), ``weight_layout``
        (``"class_major"`` for ``[C, F]``, ``"feature_major"`` for ``[F, C]``),
        ``n_features``, ``n_classes`` and ``selected`` (the class indices the
        output Gather picks, or ``None`` for all classes).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    consts = {
        n.output[0]: numpy_helper.to_array(a.t)
        for n in graph.node
        if n.op_type == "Constant"
        for a in n.attribute
        if a.name == "value"
    }
    producer = {o: n for n in graph.node for o in n.output}
    if len(graph.output) != 1:
        return None
    tensor = graph.output[0].name
    selected = None
    for _ in range(8):
        node = producer.get(tensor)
        if node is None:
            return None
        if node.op_type in _PASS_THROUGH_OPS:
            tensor = node.input[0]
            continue
        if node.op_type == "Gather" and selected is None:
            idx_name = node.input[1]
            if idx_name in inits:
                idx = numpy_helper.to_array(inits[idx_name])
            elif idx_name in consts:
                idx = consts[idx_name]
            else:
                return None
            axis = next((a.i for a in node.attribute if a.name == "axis"), 0)
            if idx.ndim != 1 or axis not in (-1, 1):
                return None
            selected = [int(v) for v in idx.tolist()]
            tensor = node.input[0]
            continue
        break
    else:
        return None

    add_node, bias = None, None
    if node.op_type == "Add":
        a, b = node.input
        if b in inits and a in producer and producer[a].op_type == "MatMul":
            add_node, bias, node = node, b, producer[a]
        elif a in inits and b in producer and producer[b].op_type == "MatMul":
            add_node, bias, node = node, a, producer[b]
        else:
            return None
    if node.op_type == "Gemm":
        attrs = {a.name: a for a in node.attribute}
        if attrs.get("transA") and attrs["transA"].i:
            return None
        x, w = node.input[0], node.input[1]
        if w not in inits:
            return None
        bias = node.input[2] if len(node.input) > 2 and node.input[2] in inits else None
        wshape = list(inits[w].dims)
        trans_b = bool(attrs.get("transB") and attrs["transB"].i)
        layout = "class_major" if trans_b else "feature_major"
        for name in ("alpha", "beta"):
            if name in attrs and abs(attrs[name].f - 1.0) > 1e-6:
                return None
    elif node.op_type == "MatMul":
        x, w = node.input
        if w not in inits:
            return None
        wshape = list(inits[w].dims)
        layout = "feature_major"
    else:
        return None
    if len(wshape) != 2:
        return None
    n_features, n_classes = (
        (wshape[1], wshape[0]) if layout == "class_major" else (wshape[0], wshape[1])
    )
    if bias is not None and list(inits[bias].dims) not in ([n_classes], [1, n_classes]):
        return None
    if selected is not None and any(s < 0 or s >= n_classes for s in selected):
        return None
    return {
        "node": node.name,
        "op": node.op_type,
        "feature_tensor": x,
        "weight": w,
        "bias": bias,
        "add_node": add_node.name if add_node is not None else None,
        "weight_layout": layout,
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "selected": selected,
    }


def truncate_to_features(src, dst, feature_tensor):
    """Write a copy of ``src`` whose only output is ``feature_tensor`` (the
    input of the final linear layer found by :func:`find_final_linear`).
    Returns ``dst``."""
    import onnx
    from onnx import utils

    model = onnx.load(src)
    inits = {i.name for i in model.graph.initializer}
    inp = [i.name for i in model.graph.input if i.name not in inits][0]
    batch = _fixed_batch(inspect_onnx(src)["input_shape"])
    tmp = dst + ".full.onnx"
    onnx.save(model, tmp)
    try:
        utils.extract_model(tmp, dst, [inp], [feature_tensor], check_model=False)
    finally:
        os.remove(tmp)
    # extract_model can leave the feature output without a shape; make the batch
    # axis symbolic so any batch size runs -- unless the source fixes it (model-zoo
    # graphs with the batch baked into Reshape constants): then keep that number,
    # so the loaders still feed the truncated graph in chunks of that size
    m = onnx.load(dst)
    for vi in list(m.graph.input) + list(m.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) > 0:
            if batch is None:
                dims[0].dim_param = "batch_size"
            else:
                dims[0].dim_value = batch
    onnx.save(m, dst)
    return dst


def write_final_linear(src, dst, info, weight, bias):
    """Replace the final layer's weights for the selected classes.

    Parameters
    ----------
    src, dst : str
        Source and destination ``.onnx`` files.
    info : dict
        :func:`find_final_linear` of ``src``.
    weight : array-like, shape (k, n_features)
        New weight rows, one per selected class (``info["selected"]``, or all
        classes when nothing is selected), in output order.
    bias : array-like, shape (k,)
        New biases. A MatMul without bias gets an ``Add`` with a new bias
        initializer (zero for the classes that are not replaced).

    Returns
    -------
    str
        ``dst``.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    rows = info["selected"] or list(range(info["n_classes"]))
    if weight.shape != (len(rows), info["n_features"]) or bias.shape != (len(rows),):
        raise ValueError(
            f"expected weight {(len(rows), info['n_features'])} and bias {(len(rows),)}, "
            f"got {weight.shape} and {bias.shape}"
        )
    w_init = inits[info["weight"]]
    w = numpy_helper.to_array(w_init).astype(np.float32).copy()
    for k, c in enumerate(rows):
        if info["weight_layout"] == "class_major":
            w[c, :] = weight[k]
        else:
            w[:, c] = weight[k]
    w_init.CopyFrom(
        numpy_helper.from_array(
            w.astype(numpy_helper.to_array(w_init).dtype), w_init.name
        )
    )
    if info["bias"] is not None:
        b_init = inits[info["bias"]]
        b = numpy_helper.to_array(b_init).astype(np.float32).copy()
        shape = b.shape
        b = b.reshape(-1)
        for k, c in enumerate(rows):
            b[c] = bias[k]
        b_init.CopyFrom(numpy_helper.from_array(b.reshape(shape), b_init.name))
    else:
        # MatMul without bias: add one after it
        b = np.zeros(info["n_classes"], dtype=np.float32)
        for k, c in enumerate(rows):
            b[c] = bias[k]
        mm = next(n for n in graph.node if n.name == info["node"])
        old_out = mm.output[0]
        new_mid = old_out + "_peal_dfr_nobias"
        mm.output[0] = new_mid
        b_name = info["weight"] + "_peal_dfr_bias"
        graph.initializer.append(numpy_helper.from_array(b, b_name))
        idx = list(graph.node).index(mm)
        graph.node.insert(
            idx + 1,
            helper.make_node(
                "Add", [new_mid, b_name], [old_out], name=mm.name + "_peal_dfr_bias"
            ),
        )
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    return dst
