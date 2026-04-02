# NotJpegAi... but the idea is close enough!

A PyTorch-based neural image compression pipeline using the CompressAI MeanScaleHyperprior architecture. Supports end-to-end training, compression to a cleverly-named binary format (`.ramiro`), decompression back to PNG, and evaluation against JPEG baselines with rate-distortion curve generation.

## Usage

### Training

Train a neural compression model on a dataset of images:

```
python train.py \
  --dataset /path/to/train \
  --val_dataset /path/to/val \
  --epochs 100 \
  --output_dir checkpoints
```

Optional arguments:

- `--lambda` — Rate-distortion trade-off weight (default: 0.01)
- `--batch_size` — Batch size for training (default: 8)

### Compression

Compress a single image to the `.ramiro` binary format:

```
python compress.py \
  -i input.png \
  -o output.ramiro \
  -c checkpoints/best_model.pth
```

### Decompression

Decompress a `.ramiro` file back to a PNG image:

```
python decompress.py \
  -i output.ramiro \
  -o reconstructed.png \
  -c checkpoints/best_model.pth
```

### Evaluation

Benchmark a trained checkpoint against JPEG at multiple quality levels:

```
python evaluate.py \
  -i /path/to/test \
  -c checkpoints/best_model.pth \
  -rd rd_curve.png \
  -csv results.csv
```

This produces a rate-distortion plot comparing the neural codec (single data point) against JPEG baselines, along with a CSV of per-codec metrics.
