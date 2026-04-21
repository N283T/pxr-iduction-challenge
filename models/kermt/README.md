# KERMT model weights

- `grover_base.pt` -- GROVER_base pretrained checkpoint (Tencent mirror, KERMT-compatible).
- Source: Google Drive ID `1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl`
- SHA256: `47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de`
- Size: ~193.6 MB
- torch.load top keys: `state_dict`, `args` (loaded with `weights_only=False`).
- Downloaded via `uvx gdown <file_id> -O <path>` on 2026-04-21 (gdown 6.0.0 dropped
  `--fuzzy`; passing the raw Google Drive ID still resolves through the confirm page).
- GROVER_large (ID `1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw`) is a follow-up if base succeeds.

## Continued-pretrain run (2026-04-21)

- Command: `bash track1_activity/scripts/run_kermt_pretrain.sh 30`
- Inputs: `data/kermt/pretrain_{train,val}.csv` (11,822 train / 1,314 val, 90/10 random split at seed 42)
- Config: 30 epochs, batch_size 32, ffn_hidden 256, `--self_attention --dist_coff 0.15`, max_lr 1e-4, final_lr 2e-5
- Backbone params frozen? No -- `--checkpoint_path` loads GROVER_base weights, then fine-tunes the full encoder + 2-task FFN head (base 48,931,556 params trainable).
- Wall-clock: ~31 min on RTX 5080
- **Best val_mae: 0.2607 at epoch 16**
- Final val_mae: 0.2652 (epoch 29). Mild overfit after epoch 20; KERMT saves best-val checkpoint to `model.pt`.
- Output: `models/kermt/pretrain/fold_0/model_0/model.pt` (195.8 MB, best checkpoint for embed extraction).
- Training log: `logs/kermt_pretrain_30ep.log` (gitignored).
- `overall_scaffold_balanced_test_mae=0.134542` reported at end is KERMT's internal scaffold-balanced split (not our 10% held-out val); ignore for downstream decisions.
