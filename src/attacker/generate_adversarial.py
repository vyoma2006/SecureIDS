# # generate_adversarial.py
# import argparse
# import os
# import pandas as pd

# from src.attacker.perturbation_utils import load_model_artifacts, prepare_inputs
# from src.attacker.fgsm_attack import fgsm_attack
# from src.attacker.evasion_metrics import compute_evasion_rate, compute_confidence_drop

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epsilons", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.3])
#     parser.add_argument("--classes", nargs="+", default=None)
#     # CHANGED: two separate paths instead of one --data_path
#     parser.add_argument("--x_path", default="data/processed/X_test.csv")
#     parser.add_argument("--y_path", default="data/processed/y_test.csv")
#     args = parser.parse_args()

#     model, scaler, label_encoder, feature_columns = load_model_artifacts()

#     # CHANGED: load X and y separately, no single df anymore
#     X_df = pd.read_csv(args.x_path)
#     y_df = pd.read_csv(args.y_path)
#     X_tensor, y_tensor = prepare_inputs(X_df, y_df, feature_columns, scaler, label_encoder)

#     os.makedirs("data/adversarial", exist_ok=True)
#     results = []

#     target_classes = args.classes or [c for c in label_encoder.classes_ if c != "Benign"]

#     for class_name in target_classes:
#         class_idx = label_encoder.transform([class_name])[0]
#         mask = (y_tensor == class_idx)
#         X_class, y_class = X_tensor[mask], y_tensor[mask]
#         if len(X_class) == 0:
#             continue

#         for eps in args.epsilons:
#             X_adv = fgsm_attack(model, X_class, y_class, eps)
#             evasion_rate, preds = compute_evasion_rate(model, X_adv, y_class)
#             conf_drop = compute_confidence_drop(model, X_class, X_adv, y_class)

#             adv_df = pd.DataFrame(X_adv.numpy(), columns=feature_columns)
#             out_path = f"data/adversarial/fgsm_{class_name}_{eps}.csv"
#             adv_df.to_csv(out_path, index=False)

#             results.append({
#                 "class": class_name, "epsilon": eps,
#                 "evasion_rate": evasion_rate, "confidence_drop": conf_drop,
#                 "n_samples": len(X_class),
#             })
#             print(f"{class_name} eps={eps}: evasion_rate={evasion_rate:.3f}")

#     results_df = pd.DataFrame(results)
#     os.makedirs("results/reports", exist_ok=True)
#     results_df.to_csv("results/reports/attacker_findings.csv", index=False)

# if __name__ == "__main__":
#     main()

# generate_adversarial.py
import argparse
import os
import json
import pandas as pd

from src.attacker.perturbation_utils import load_model_artifacts, prepare_inputs
from src.attacker.fgsm_attack import fgsm_attack
from src.attacker.evasion_metrics import compute_evasion_rate, compute_confidence_drop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.3])
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--x_path", default="data/processed/X_test.csv")
    parser.add_argument("--y_path", default="data/processed/y_test.csv")
    args = parser.parse_args()

    model, scaler, label_encoder, feature_columns = load_model_artifacts()

    X_df = pd.read_csv(args.x_path)
    y_df = pd.read_csv(args.y_path)
    X_tensor, y_tensor = prepare_inputs(X_df, y_df, feature_columns, scaler, label_encoder)

    os.makedirs("data/adversarial", exist_ok=True)
    results = []

    target_classes = args.classes or [c for c in label_encoder.classes_ if c != "Benign"]

    for class_name in target_classes:
        class_idx = label_encoder.transform([class_name])[0]
        mask = (y_tensor == class_idx)
        X_class, y_class = X_tensor[mask], y_tensor[mask]
        if len(X_class) == 0:
            continue

        for eps in args.epsilons:
            X_adv = fgsm_attack(model, X_class, y_class, eps)
            evasion_rate, preds = compute_evasion_rate(model, X_adv, y_class)
            conf_drop = compute_confidence_drop(model, X_class, X_adv, y_class)

            adv_df = pd.DataFrame(X_adv.numpy(), columns=feature_columns)
            out_path = f"data/adversarial/fgsm_{class_name}_{eps}.csv"
            adv_df.to_csv(out_path, index=False)

            results.append({
                "class": class_name, "epsilon": eps,
                "evasion_rate": evasion_rate, "confidence_drop": conf_drop,
                "n_samples": len(X_class),
            })
            print(f"{class_name} eps={eps}: evasion_rate={evasion_rate:.3f}")

    # Existing CSV summary
    results_df = pd.DataFrame(results)
    os.makedirs("results/reports", exist_ok=True)
    results_df.to_csv("results/reports/attacker_findings.csv", index=False)

    # NEW: JSON export to results/metrics/evasion_metrics.json
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/evasion_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved evasion metrics to results/metrics/evasion_metrics.json")

if __name__ == "__main__":
    main()