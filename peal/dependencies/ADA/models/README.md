# Generator model artifacts

Large model weights are not stored in Git. The versioned code expects a model
bundle containing the three matched RAEv2 variants:

- `cls_only`: timestep and DINOv2-L CLS conditioning;
- `cls_class`: timestep, ImageNet class, and DINOv2-L CLS conditioning;
- `class_only`: timestep and ImageNet class conditioning.

Each variant directory contains:

```text
ema.pt
config.yaml
metadata.json
```

The exported `ema.pt` contains only inference weights. Optimizer, scheduler,
and non-EMA training state are deliberately omitted. The top-level
`manifest.json` records checkpoint epoch/step, file size, SHA256, parameter
count, source experiment, and the required Stage-1 decoder assets.

On Hydra, the current collaborator snapshot is stored under:

```text
/home/space/pathomics/ADA/models/latest
```

See [docs/generator-models.md](../docs/generator-models.md) for export,
download, training, and sampling commands.
