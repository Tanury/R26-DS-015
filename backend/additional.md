# Additional Modal Hardware Suggestions

## Recommended Setup

Your proposed Modal setup is good:

```text
GPU: L40S
CPU: 8 cores
RAM: 64 GB
Storage: 150 GB
```

This is more than enough for the Phase 1 EEGNet baseline and gives comfortable margin for preprocessing, plots, checkpoints, and repeated experiments.

## Hardware Opinion

| Resource | Opinion |
|---|---|
| `L40S` GPU | Excellent. EEGNet is small, so training should be fast. L40S is stronger than strictly required, but useful for smooth experiments. |
| `8 CPU cores` | Useful mainly for EEG preprocessing, raw file loading, filtering, and future parallel preprocessing. |
| `64 GB RAM` | Very safe. EEG preprocessing can use memory when loading raw recordings and saving tensors. |
| `150 GB storage` | Good, but important files should still be saved in Modal Volumes because notebook container storage is not permanent. |

## Will More Hardware Increase Training Speed?

Yes, but only up to a point.

For EEGNet, the biggest speed gains come from:

1. GPU training with CUDA
2. Preprocessing all subjects once and reusing saved epochs
3. Using a reasonable batch size
4. Avoiding repeated raw EEG loading

Increasing CPU and RAM helps preprocessing more than model training. Increasing GPU power helps training, but EEGNet is lightweight, so the GPU may not be fully used.

## CUDA Support

The current code already supports CUDA automatically.

In `configs/training.yaml`:

```yaml
training:
  device: auto
```

The training code uses CUDA if available:

```python
cuda if torch.cuda.is_available() else cpu
```

So on Modal with an `L40S`, training should use the GPU automatically.

## Batch Processing

Batch processing is good and is already used through PyTorch `DataLoader`.

Current default:

```yaml
batch_size: 16
```

With `L40S + 64 GB RAM`, you can try:

```yaml
batch_size: 32
```

You may also test `64`, but start with `32`. If validation performance becomes unstable or memory usage increases too much, return to `16`.

## Parallel Processing

Parallel processing is useful mostly for preprocessing and data loading.

For Modal with `8 CPU cores`, you can later try:

```yaml
num_workers: 2
```

or:

```yaml
num_workers: 4
```

In notebooks, start with:

```yaml
num_workers: 0
```

This is safer for debugging. After everything works, increase it gradually.

## Best Practical Recommendation

Start with:

```text
GPU: L40S
CPU: 8 cores
RAM: 64 GB
Storage: 150 GB
Batch size: 16
num_workers: 0
```

Then test:

```text
Batch size: 32
num_workers: 2 or 4
```

## Final Opinion

The proposed setup is strong and comfortable. It may not make EEGNet dramatically faster than a smaller GPU because the model is lightweight, but it will make the full workflow smoother, especially preprocessing, repeated experiments, and future CWT or Transformer work.
