Not Jpeg Ai... but the idea is close enough

A PyTorch-based neural image compression pipeline using the CompressAI Mean-Scale Hyperprior architecture. Uses the cleverly-named binary format (`.ramiro`). Includes a graph-topology-based tool for artifact detection "Learned Geometric Boundary Topology" (LGBT).

Default settings are on the safer side as my computer runs an RTX 2060 with 16GB RAM.

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

### Structural Evaluation

Evaluate structural distortion using the Learned Geometric Boundary Topology (LGBT) metric, which measures false edges introduced by compression:

```
python -c "
from neural_compression.graph_metrics import evaluate_structural_integrity
from PIL import Image

orig = Image.open('original.png')
recon = Image.open('reconstructed.png')

result = evaluate_structural_integrity(orig, recon)
print(f'LGBT: {result[\"lgbt\"]:.4f}')
print(f'False edges: {result[\"false_edges\"]}')
print(f'True edges: {result[\"true_edges\"]}')
"
```

LGBT ranges from 0.0 (perfect structural preservation) to 1.0 (all edges are false). Lower values indicate the compression did not introduce spurious structural artifacts.
