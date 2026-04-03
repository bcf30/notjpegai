## Not Jpeg Ai... but the idea is close enough

A PyTorch-based neural image compression pipeline using the CompressAI Mean-Scale Hyperprior architecture. Uses the cleverly-named binary format (`.ramiro`). Includes a graph-topology-based tool for artifact detection "Learned Geometric Boundary Topology" (LGBT).

Default settings are on the safer side as my computer runs an RTX 2060 with 16GB RAM.

## Usage

train.py: train a new model
resume_training.py: train an existing model
compress.py: compress an image to the `.ramiro` format
decompress.py: decompress a `.ramiro` file to a PNG
evaluate.py: PSNR and MS-SSIM evaluation on image accuracy
graph_metrics.py: graph-topology based evaluation for image accuracy with focus on ringing artifacts. this is the "Learned Geometric Boundary Topology" tech

# Compress / Decompress — positional args, no flags
python compress.py input.jpg output.ramiro checkpoint.pth
python decompress.py input.ramiro output.png checkpoint.pth

# Evaluate — just two images
python evaluate.py original.jpg reconstructed.png
python graph_metrics.py original.jpg reconstructed.png

# Train / Resume — just edit the top of the file and run
python train.py
python resume_training.py
