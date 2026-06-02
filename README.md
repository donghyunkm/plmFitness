# PLM Fitness

Fine-tune ESM masked-language models for protein-fitness ranking with optional
PEFT adapters.

## CUDA 11.8 Conda Environment

These instructions target Linux or Windows systems with an NVIDIA GPU and a
CUDA 11.8-compatible NVIDIA driver.

Create and activate a Conda environment:

```bash
conda create -n plm-fitness-cu118 python=3.10 -y
conda activate plm-fitness-cu118
```

Install PyTorch 2.5.1 with the CUDA 11.8 runtime:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Install the remaining dependencies:

```bash
python -m pip install transformers peft pandas numpy scipy scikit-learn tqdm
```

Verify that PyTorch can use the GPU:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output includes:

```text
CUDA runtime: 11.8
CUDA available: True
```

## Training

Run training, validation, and test evaluation on CUDA:

```bash
python main.py \
  --model esm2 \
  --protein SYUA_HUMAN \
  --train_size 0.75 \
  --list_size 5 \
  --peft_type lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --device cuda
```

The test holdout is always a seeded, stratified 20% split of the selected
dataset. `--train_size` then selects a stratified subset from the remaining
eligible data, and that subset is split 80/20 into train/validation; for
example, `--train_size 0.75` gives roughly 48% train, 12% validation, 20%
test, with the rest unused. Training saves the best validation checkpoint and
then tests that checkpoint.

Use IA3 adapters instead of LoRA with:

```bash
python main.py \
  --model esm2 \
  --protein SYUA_HUMAN \
  --train_size 0.75 \
  --list_size 5 \
  --peft_type ia3 \
  --device cuda
```

Use `--peft_type none` to train without PEFT adapters. The LoRA-specific
`--lora_r` and `--lora_alpha` arguments are only used when `--peft_type lora`.

To skip training and test the latest matching checkpoint:

```bash
python main.py \
  --test \
  --model esm2 \
  --protein SYUA_HUMAN \
  --train_size 0.75 \
  --list_size 5 \
  --peft_type lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --device cuda
```

## Outputs

Best checkpoints and training logs are written under:

```text
checkpoints/{model}/{protein}/
```

Test predictions are written under:

```text
predictions/{model}/{protein}/
```

Each output name includes a timestamp so repeated runs remain separate.

## References

- [PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/)
- [Transformers installation](https://huggingface.co/docs/transformers/master/installation)
- [PEFT installation](https://huggingface.co/docs/peft/install)
