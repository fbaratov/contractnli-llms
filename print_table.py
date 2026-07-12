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


            root_list = root.split("/")
            short_path = f"{root_list[-4]}/{root_list[-2]}"

            rows.append((short_path, acc, f1_con, f1_ent))


    if not rows:
        print("No eval directories found.")
        return
    
    rows.sort(key=lambda row: row[0])

    headers = ("Path", "Accuracy", "F1(C)", "F1(E)")
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(4)
    ]

    def fmt_row(cols):
        return " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cols))

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

            acc     = f"{round(cls['accuracy']['average'], 3):.3f} ± {round(cls['accuracy']['std'], 3):.3f}"
            f1_con  = f"{round(cls['f1_contradiction']['average'], 3):.3f} ± {round(cls['f1_contradiction']['std'], 3):.3f}"
            f1_ent  = f"{round(cls['f1_entailment']['average'], 3):.3f} ± {round(cls['f1_entailment']['std'], 3):.3f}"


            root_list = root.split("/")
            short_path = f"{root_list[-4]}/{root_list[-2]}"

            rows.append((short_path, acc, f1_con, f1_ent))


    if not rows:
        print("No eval directories found.")
        return
    
    rows.sort(key=lambda row: row[0])

    headers = ("Path", "Accuracy", "F1(C)", "F1(E)")
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(4)
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