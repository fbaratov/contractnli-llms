import os
import json
import sys

def print_eval_metrics(response_dir: str) -> None:
    rows = []

    for root, dirs, files in os.walk(response_dir):
        if root.rstrip("/").endswith("/eval") or root.rstrip(os.sep).endswith(f"{os.sep}eval"):
            metrics_path = os.path.join(root, "aggregated_metrics.json")
            if not os.path.isfile(metrics_path):
                print(f"WARNING: No aggregated_metrics.json found in {root}")
                continue

            with open(metrics_path, "r") as f:
                metrics_dict = json.load(f)

            cls = metrics_dict["macro_label_micro_doc"]["class"]

            acc     = f"{round(cls['accuracy']['average'], 3):.3f} ± {round(cls['accuracy']['std'], 3):.3f}"
            f1_con  = f"{round(cls['f1_contradiction']['average'], 3):.3f} ± {round(cls['f1_contradiction']['std'], 3):.3f}"
            f1_ent  = f"{round(cls['f1_entailment']['average'], 3):.3f} ± {round(cls['f1_entailment']['std'], 3):.3f}"
            invalid = f"{round(metrics_dict['invalid_rate']['average'], 1):.1f} ± {round(metrics_dict['invalid_rate']['std'], 1):.1f}"
            inf_time = f"{round(metrics_dict['inference_time']['average'], 1):.1f}"
            root_list = root.split("/")
            short_path = f"{root_list[-4]}/{root_list[-2]}"

            rows.append((short_path, acc, f1_con, f1_ent, invalid, inf_time))


    if not rows:
        print("No eval directories found.")
        return
    
    rows.sort(key=lambda row: row[0])

    headers = ("Path", "Accuracy", "F1(C)", "F1(E)", "Invalid Rate", "Time (s)")
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cols):
        return " & ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))

    separator = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))

def print_common_metrics(response_dir):
    rows = []

    for root, dirs, files in os.walk(response_dir):
        if root.rstrip("/").endswith("/eval") or root.rstrip(os.sep).endswith(f"{os.sep}eval"):
            metrics_path = os.path.join(root, "aggregated_metrics.json")
            if not os.path.isfile(metrics_path):
                print(f"WARNING: No aggregated_metrics.json found in {root}")
                continue

            with open(metrics_path, "r") as f:
                metrics_dict = json.load(f)

            cls = metrics_dict["macro_label_micro_doc"]["class"]

            p_con  = f"{round(cls['precision_contradiction']['average'], 3):.3f} ± {round(cls['precision_contradiction']['std'], 3):.3f}"
            p_ent  = f"{round(cls['precision_entailment']['average'], 3):.3f} ± {round(cls['precision_entailment']['std'], 3):.3f}"
            p_nm  = f"{round(cls['precision_not_mentioned']['average'], 3):.3f} ± {round(cls['precision_not_mentioned']['std'], 3):.3f}"
            # precs = [p_con, p_ent, p_nm]

            r_con  = f"{round(cls['recall_contradiction']['average'], 3):.3f} ± {round(cls['recall_contradiction']['std'], 3):.3f}"
            r_ent  = f"{round(cls['recall_entailment']['average'], 3):.3f} ± {round(cls['recall_entailment']['std'], 3):.3f}"
            r_nm  = f"{round(cls['recall_not_mentioned']['average'], 3):.3f} ± {round(cls['recall_not_mentioned']['std'], 3):.3f}"
            # recs = [r_con, r_ent, r_nm]

            f1_con  = f"{round(cls['f1_contradiction']['average'], 3):.3f} ± {round(cls['f1_contradiction']['std'], 3):.3f}"
            f1_ent  = f"{round(cls['f1_entailment']['average'], 3):.3f} ± {round(cls['f1_entailment']['std'], 3):.3f}"
            f1_nm  = f"{round(cls['f1_not_mentioned']['average'], 3):.3f} ± {round(cls['f1_not_mentioned']['std'], 3):.3f}"
            # f1s = [f1_con, f1_ent, f1_nm]


            root_list = root.split("/")
            short_path = f"{root_list[-4]}/{root_list[-2]}"

            cont = [p_con, r_con, f1_con]
            ent = [p_ent, r_ent, f1_ent]
            nm = [p_nm, r_nm, f1_nm]

            rows.append([short_path] + cont + ent + nm) #+ precs + recs + f1s)


    if not rows:
        print("No eval directories found.")
        return
    
    rows.sort(key=lambda row: row[0])

    # headers = ("Path", "P(C)", "P(E)", "P(N)", "R(C)", "R(E)", "R(N)", "F1(C)", "F1(E)", "F1(N)")
    headers = ("Path", "P", "R", "F1", "P(E)", "R(E)", "F1(E)", "P(N)", "R(N)", "F1(N)")
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cols):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))

    separator = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))

if __name__=="__main__":
    print_eval_metrics(sys.argv[1])
    print_common_metrics(sys.argv[1])