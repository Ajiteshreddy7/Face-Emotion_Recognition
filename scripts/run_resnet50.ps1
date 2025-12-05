<#
Run a resnet50 fine-tune using the project's training script.
Adjust `--batch-size` if you get OOM on GTX 1650 (try 8 or 4).
#>

Write-Host "Activate your venv first: .\.venv\Scripts\Activate.ps1 (or conda activate fer)"

python -m src.train `
  --data-dir data/fer2013 `
  --csv fer2013.csv `
  --model resnet50 `
  --image-size 224 `
  --batch-size 12 `
  --epochs 20 `
  --device cuda `
  --num-workers 4 `
  --pin-memory `
  --freeze-backbone `
  --unfreeze-top-n 2 `
  --output-dir outputs/resnet50_finetune

Write-Host "Training started. Monitor GPU with: nvidia-smi -l 2"
