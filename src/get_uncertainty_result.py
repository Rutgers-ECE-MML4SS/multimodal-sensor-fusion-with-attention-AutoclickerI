import os, json, glob, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import MultimodalFusionModule
from data import create_dataloaders
from uncertainty import CalibrationMetrics

def pick_checkpoint(run_dir: Path) -> Path:
    """results.json의 best_model_path > last.ckpt > 임의 .ckpt 순으로 선택"""
    results_json = run_dir / "results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                p = Path(json.load(f).get("best_model_path", ""))
            if p and p.exists():
                return p
        except Exception:
            pass
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    cands = sorted(glob.glob(str(run_dir / "checkpoints" / "*.ckpt")))
    if not cands:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/checkpoints")
    return Path(cands[0])

def eval_uncertainty_for_run(run_dir: Path, device: str, num_bins: int = 15):
    ckpt = pick_checkpoint(run_dir)
    print(f"[{run_dir.name}] checkpoint: {ckpt}")

    # 모델/설정 로드
    model = MultimodalFusionModule.load_from_checkpoint(ckpt).eval().to(device)
    cfg = model.config

    # 테스트 로더 구성 (ckpt에 저장된 hydra config 사용)
    _, _, test_loader = create_dataloaders(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        modalities=cfg.dataset.modalities,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )

    # 추론 수집
    all_conf, all_pred, all_lab = [], [], []
    nll_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for features, labels, mask in tqdm(test_loader, desc=f"Unc {run_dir.name}", leave=False):
            features = {k: v.to(device) for k, v in features.items()}
            labels, mask = labels.to(device), mask.to(device)

            logits = model(features, mask)
            nll_sum += F.cross_entropy(logits, labels).item()
            n_batches += 1

            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            all_conf.append(conf.cpu()); all_pred.append(pred.cpu()); all_lab.append(labels.cpu())

    conf = torch.cat(all_conf).numpy()
    pred = torch.cat(all_pred).numpy()
    lab  = torch.cat(all_lab).numpy()
    nll = float(nll_sum / max(n_batches, 1))
    ece = float(CalibrationMetrics.expected_calibration_error(
        torch.from_numpy(conf), torch.from_numpy(pred), torch.from_numpy(lab), num_bins=num_bins
    ))

    # 빈 중심 & 빈별 정확도
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    acc_bins = []
    for i in range(num_bins):
        lo, hi = edges[i], edges[i+1]
        m = (conf >= lo) & (conf <= hi if i == num_bins - 1 else conf < hi)
        acc_bins.append(float((pred[m] == lab[m]).mean()) if m.sum() else 0.0)

    # JSON 저장
    out_dir = Path("experiments") / run_dir.name.split('_')[1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dataset": cfg.dataset.name,
        "calibration_metrics": {
            "ece": ece,
            "nll": nll,
            "bins": [float(x) for x in centers],
            "accuracy_per_bin": [float(x) for x in acc_bins],
        },
    }
    with open(out_dir / "uncertainty.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"→ wrote {out_dir/'uncertainty.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="runs")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="평가할 run 폴더 이름들 (예: a2_early_pamap2 a2_late_pamap2). 미지정 시 runs_root/a2_*_pamap2 전체.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bins", type=int, default=15)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if args.runs:
        run_dirs = [runs_root / r for r in args.runs]
    else:
        run_dirs = [Path(p) for p in sorted(glob.glob(str(runs_root / "a2_*_pamap2")))]

    print(f"device = {args.device}")
    for rd in run_dirs:
        if rd.exists():
            eval_uncertainty_for_run(rd, device=args.device, num_bins=args.bins)
        else:
            print(f"[skip] {rd} not found")

if __name__ == "__main__":
    main()
