python src/eval.py \
  --checkpoint ./runs/a2_hybrid_pamap2/checkpoints/last.ckpt \
  --config config/base.yaml \
  --device cuda \
  --output_dir experiments/hybrid

python src/eval.py \
  --checkpoint ./runs/a2_hybrid_pamap2/checkpoints/last.ckpt \
  --config config/base.yaml \
  --device cuda \
  --output_dir experiments/hybrid \
  --missing_modality_test

python src/eval.py \
  --checkpoint ./runs/a2_early_pamap2/checkpoints/last.ckpt \
  --config config/base_early.yaml \
  --device cuda \
  --output_dir experiments/early

python src/eval.py \
  --checkpoint ./runs/a2_early_pamap2/checkpoints/last.ckpt \
  --config config/base_early.yaml \
  --device cuda \
  --output_dir experiments/early \
  --missing_modality_test

python src/eval.py \
  --checkpoint ./runs/a2_late_pamap2/checkpoints/last.ckpt \
  --config config/base_late.yaml \
  --device cuda \
  --output_dir experiments/late

python src/eval.py \
  --checkpoint ./runs/a2_late_pamap2/checkpoints/last.ckpt \
  --config config/base_late.yaml \
  --device cuda \
  --output_dir experiments/late \
  --missing_modality_test

python merge_results.py

python src/get_uncertainty_result.py --runs_root ./runs --device cuda

python src/analysis.py --experiment_dir experiments --output_dir analysis