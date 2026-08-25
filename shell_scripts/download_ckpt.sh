# Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.
#
# This file constitutes Licensed Materials under the Adobe Research License.
# Use is limited to noncommercial research purposes.
# See the LICENSE file at the project root for the complete license terms and disclaimer.

# Following https://github.com/guandeh17/Self-Forcing and https://github.com/thu-ml/Causal-Forcing

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir-use-symlinks False \
    --local-dir wan_models/Wan2.1-T2V-1.3B

huggingface-cli download gdhe17/Self-Forcing \
    checkpoints/self_forcing_dmd.pt \
    --local-dir .

huggingface-cli download zhuhz22/Causal-Forcing \
    chunkwise/causal_forcing.pt --local-dir checkpoints
