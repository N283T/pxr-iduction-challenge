# KERMT model weights

- `grover_base.pt` -- GROVER_base pretrained checkpoint (Tencent mirror, KERMT-compatible).
- Source: Google Drive ID `1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl`
- SHA256: `47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de`
- Size: ~193.6 MB
- torch.load top keys: `state_dict`, `args` (loaded with `weights_only=False`).
- Downloaded via `uvx gdown <file_id> -O <path>` on 2026-04-21 (gdown 6.0.0 dropped
  `--fuzzy`; passing the raw Google Drive ID still resolves through the confirm page).
- GROVER_large (ID `1bMg_ntUKEoOmHM0KoUi1XYJvzPBnHeWw`) is a follow-up if base succeeds.
