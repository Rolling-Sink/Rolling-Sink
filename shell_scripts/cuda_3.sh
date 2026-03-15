export CUDA_VISIBLE_DEVICES=3

PROMPTS_DIR="prompts/example"
OUT_DIR="videos/example"

for TXT_PATH in "${PROMPTS_DIR}"/*.txt; do
    echo "\nGenerating video for prompt: ${TXT_PATH} ⬇️\n"
    cat ${TXT_PATH}

    for SEED in $(seq 9 11); do
        python inference.py \
            --config_path configs/self_forcing_dmd.yaml \
            --output_folder ${OUT_DIR} \
            --checkpoint_path checkpoints/self_forcing_dmd.pt \
            --data_path ${TXT_PATH} \
            --seed ${SEED} \
            --use_ema

    done
done

### for checkpoints, you can choose either one from below:
# --checkpoint_path checkpoints/self_forcing_dmd.pt \
# --checkpoint_path checkpoints/chunkwise/causal_forcing.pt \
### Note: remove the '--use_ema' flag if using checkpoints/chunkwise/causal_forcing.pt
