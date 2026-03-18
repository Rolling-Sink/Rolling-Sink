# following https://github.com/guandeh17/Self-Forcing and https://github.com/thu-ml/Causal-Forcing

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir-use-symlinks False \
    --local-dir wan_models/Wan2.1-T2V-1.3B

huggingface-cli download gdhe17/Self-Forcing \
    checkpoints/self_forcing_dmd.pt \
    --local-dir .

huggingface-cli download zhuhz22/Causal-Forcing \
    chunkwise/causal_forcing.pt --local-dir checkpoints
