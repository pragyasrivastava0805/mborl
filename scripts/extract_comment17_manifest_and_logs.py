import csv, glob, json, os, subprocess

base = "/home/rchitale_google_com/mborl/exp/scalembrl"
git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd="/home/rchitale_google_com/mborl").decode().strip()

run_groups = [
    ("Original short-rollout COMBO (H=5, 50k)", "Original_H5_50k", f"{base}/combo_ablation_scene-play-singletask-task1-v0/*20261022T130232*-c3"),
    ("Experiment 1 COMBO (H=5, 100k)", "Exp1_H5_100k", f"{base}/combo_comment14_exp1_scene-play-singletask-task1-v0/*"),
    ("Original long-rollout COMBO (H=10, 50k)", "Original_H10_50k", f"{base}/combo_ablation_scene-play-singletask-task1-v0/*20261022T130232*-c10"),
    ("CQL reference (f=1.0)", "CQL_f1.0", f"{base}/combo_ablation_scene-play-singletask-task1-v0/*20261022T130232*-c8"),
    ("Phase 2 500k Reference (beta=0.5, H=10, 50k)", "Ref_500k_b0.5_H10", f"{base}/comment14_ref/scene_task1_s0_ckpt500k"),
    ("Experiment 2A Frozen Depths 1-5", "Exp2A_Frozen_d1to5", f"{base}/combo_comment14_exp2_scene-play-singletask-task1-v0/*d1to5*"),
    ("Experiment 2B Frozen Depths 6-10", "Exp2B_Frozen_d6to10", f"{base}/combo_comment14_exp2_scene-play-singletask-task1-v0/*d6to10*"),
]

target_steps = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
manifest_steps = [100000, 400000, 500000, 600000, 800000, 1000000]

manifest_rows = []
timeseries = {}

for full_label, short_key, pat in run_groups:
    timeseries[short_key] = {"label": full_label, "seeds": {}}
    for d in sorted(glob.glob(pat)):
        with open(f"{d}/flags.json") as f:
            fl = json.load(f)
        ag = fl["agent"]
        seed = int(fl["seed"])
        run_id = os.path.basename(d)
        ev_map = {}
        with open(f"{d}/eval.csv") as f:
            for r in csv.DictReader(f):
                ev_map[int(float(r["step"]))] = float(r["evaluation/success"])
        tr_map = {}
        with open(f"{d}/train.csv") as f:
            for r in csv.DictReader(f):
                st = int(float(r["step"]))
                if st in target_steps:
                    vq_pi = float(r["validation/critic/q_pi_mean"])
                    vq_da = float(r["validation/critic/q_data_mean"])
                    tr_map[st] = {
                        "step": st,
                        "eval_success": ev_map.get(st, None),
                        "val_q_pi_mean": vq_pi,
                        "val_q_data_mean": vq_da,
                        "val_dq_pi_minus_data": vq_pi - vq_da,
                        "val_q_max": float(r["validation/critic/q_max"]),
                        "val_q_min": float(r["validation/critic/q_min"]),
                        "val_td_loss": float(r["validation/critic/td_loss"]),
                        "val_regularizer": float(r["validation/critic/regularizer"]),
                        "train_q_model_mean": float(r.get("training/critic/q_model_mean", 0.0) or 0.0),
                        "train_q_data_mean": float(r["training/critic/q_data_mean"]),
                        "actor_lam_alpha": float(r["training/actor/lam"]),
                        "actor_entropy": float(r["training/actor/entropy"]),
                        "actor_std": float(r["training/actor/std"]),
                        "actor_mse_to_data": float(r["training/metrics/mse"]),
                    }
        keep_last = fl.get("keep_last_checkpoint_only", False)
        del_on_finish = fl.get("delete_checkpoints_on_finish", False)
        timeseries[short_key]["seeds"][str(seed)] = {
            "run_id": run_id,
            "dir": d,
            "beta": ag["beta"],
            "real_ratio": ag["real_ratio"],
            "rollout_steps": ag["rollout_steps"],
            "mb_batch_size": fl["mb_batch_size"],
            "keep_last_checkpoint_only": keep_last,
            "delete_checkpoints_on_finish": del_on_finish,
            "steps": [tr_map[s] for s in sorted(tr_map.keys())],
        }
        for mst in manifest_steps:
            if mst not in ev_map and mst not in tr_map:
                continue
            pkl_path = f"{d}/params_{mst}.pkl"
            exists = os.path.exists(pkl_path)
            t_info = tr_map.get(mst, {})
            manifest_rows.append({
                "config_group": full_label,
                "short_key": short_key,
                "run_id": run_id,
                "seed": seed,
                "beta": ag["beta"],
                "real_ratio": ag["real_ratio"],
                "rollout_horizon": ag["rollout_steps"],
                "starting_states": fl["mb_batch_size"],
                "step": mst,
                "checkpoint_path": pkl_path if exists else f"MISSING (deleted by keep_last={keep_last}, del_on_finish={del_on_finish})",
                "pkl_on_disk": exists,
                "code_version": git_rev,
                "logged_eval_success": ev_map.get(mst, None),
                "val_q_pi_mean": t_info.get("val_q_pi_mean"),
                "val_q_data_mean": t_info.get("val_q_data_mean"),
                "val_dq_pi_minus_data": t_info.get("val_dq_pi_minus_data"),
                "val_q_max": t_info.get("val_q_max"),
                "actor_lam_alpha": t_info.get("actor_lam_alpha"),
                "actor_mse_to_data": t_info.get("actor_mse_to_data"),
            })

os.makedirs("/home/rchitale_google_com/mborl/results/comment17", exist_ok=True)
with open("/home/rchitale_google_com/mborl/results/comment17/checkpoint_manifest.json", "w") as f:
    json.dump({"git_commit": git_rev, "manifest": manifest_rows, "timeseries": timeseries}, f, indent=2)

with open("/home/rchitale_google_com/mborl/results/comment17/checkpoint_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
    w.writeheader()
    w.writerows(manifest_rows)

print(f"Wrote {len(manifest_rows)} manifest entries to checkpoint_manifest.json and .csv")
