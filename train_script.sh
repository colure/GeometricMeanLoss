#!/bin/zsh

# Train
uv run torchrun --nproc_per_node=1 --master_port=35768 train.py --logit=l1_dist --loss=GMLoss --output-dir=result/gm --class-proxy

# Evaluate
# uv run torchrun --nproc_per_node=1 --master_port=35768 train.py --logit=l1_dist --resume=result/gm/checkpoints/checkpoint_best_shot5.pth --test-only

# Train with episodes
# uv run torchrun --nproc_per_node=1 --master_port=35768 train.py --logit=l1_dist --loss=ProtoNet --output-dir=result/pn --class-aware-sampler='64,8' --batch-size=512

# t-SNE
# uv run torchrun --nproc_per_node=3 --master_port=35757 train.py --logit=l1_dist --resume=result/gm/checkpoints/best_shot5.pth --tsne --test-only