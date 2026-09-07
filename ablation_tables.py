import json
import os
import sys

def get_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)

    cls = metrics_dict["macro_label_micro_doc"]["class"]

    acc     = f"{round(cls['accuracy']['average'], 3):.3f} ± {round(cls['accuracy']['std'], 3):.3f}"
    f1_con  = f"{round(cls['f1_contradiction']['average'], 3):.3f} ± {round(cls['f1_contradiction']['std'], 3):.3f}"
    f1_ent  = f"{round(cls['f1_entailment']['average'], 3):.3f} ± {round(cls['f1_entailment']['std'], 3):.3f}"
    invalid = f"{round(metrics_dict['invalid_rate']['average'], 1):.1f} ± {round(metrics_dict['invalid_rate']['std'], 1):.1f}"
    inf_time = f"{round(metrics_dict['inference_time']['average'], 1):.1f}"

    return acc, f1_con, f1_ent, invalid, inf_time

def find_files(main_dir, subdir, filename) -> list:
    """
    Scours main_dir for all file paths that contain the relevant subdir and have the correct filename
    """
    file_paths = []
    for root, dirs, files in os.walk(main_dir):
        if subdir in root.split(os.sep) and filename in files:
            file_paths.append(os.path.join(root, filename))
    return file_paths

def make_latex_table(data: dict, caption: str = "Caption", label: str = "tab:placeholder") -> str:
    """
    data: dict mapping setting name -> [acc, f1_con, f1_ent, invalid, inf_time]
    e.g. {"Reasoning": [0.812, 0.75, 0.79, 3.2, 1.05], ...}
    """

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, float) else str(x)

    def escape(s):
        return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    rows = []
    for setting, vals in data.items():
        acc, f1_con, f1_ent, invalid, inf_time = vals
        row = (
            f"         {escape(str(setting))} &\n"
            f"         {fmt(acc)} & {fmt(f1_con)} & {fmt(f1_ent)}\n"
            f"         & {fmt(invalid)} & {fmt(inf_time)} \\\\"
        )
        rows.append(row)

    rows_str = "\n         %\n".join(rows)

    table = f"""\\begin{{table}}[h]
    \\centering
    \\begin{{tabular}}{{r||ccc|c|c}}
         \\textbf{{Setting}} & \\textbf{{Accuracy}} & \\textbf{{F1(C)}} & \\textbf{{F1(E)}} & \\textbf{{Invalid Rate (\\%)}} & \\textbf{{Inf time (s)}}  \\\\
         \\hline\\hline
{rows_str}
    \\end{{tabular}}
    \\caption{{{caption}}}
    \\label{{{label}}}
\\end{{table}}"""

    return table

def main(main_dir, subdir, filename):
    file_paths = find_files(main_dir, subdir, filename)

    blank = ["-"] * 5
    try:
        thinking = get_metrics([fp for fp in file_paths if "think" in fp][0])
    except:
        thinking = blank

    try:
        struct = get_metrics([fp for fp in file_paths if "struct" in fp][0])
    except:
        struct = blank

    try:
        both = get_metrics([fp for fp in file_paths if "both" in fp][0])
    except:
        both = blank

    data = {
        "Reasoning": thinking,
        "Structure": struct,
        "Both": both 
    }

    table = make_latex_table(data)
    print(table)

if __name__=="__main__":
    main_dir = sys.argv[1]
    subdir = sys.argv[2]
    filename = sys.argv[3]
    main(main_dir, subdir, filename)