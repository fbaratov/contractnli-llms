import click
import json

# \begin{table}[h]
#     \centering
#     \begin{tabular}{c||c|ccc}
#           & & & Results & \\
         
#          Model & & Entailment & Contradiction & Invalid \\
#          \hline
#          \hline
#          DeepSeek-R1 8B & E & 1 & 0 & \\
#                         & C & 0 & 0 & 1187 \\
#          \hline
#          Gemma3 4B & E & 764 & 91 & \\
#                    & C & 130  & 115 & 88 \\
#          \hline
#          Gemma3 27B & E & 898 & 54 &  \\
#                     & C & 64  & 166 & 6 \\
#          \hline
#          GPT-OSS 20B & E & 810 & 39 & \\
#                      & C & 157 & 181 & 1 \\
#          \hline
#          LLaMA3.1 8B & E & 515 & 95  & \\
#                      & C & 158 & 67 & 353 \\
#          \hline
#          Qwen3 30B   & E & 820 & 39   & \\
#                      & C & 52  & 172  & 105 \\
#     \end{tabular}
#     \caption{Confusion matrices on test set with Seed 0 and temperature 0}
#     \label{tab:seed0temp0confusion}
# \end{table}


confusion = {
    "start" : """
\\begin{table}[h]
    \centering
    \\begin{tabular}{c||c|ccc}
          & & & Results & \\\\
         
         Model & & Entailment & Contradiction & Invalid \\\\
         \hline
""",
    "end": """\end{tabular}
    \caption{Your caption here}
    \label{tab:generic_label}
\end{table}""",

    "row": """
%
\hline
         {model} & E & {int(round(data["sum"]["true_e"]["average"],0))} & {int(round(data["sum"]["false_e"]["average"],0))} & \\\\\\\\
                 & C & {int(round(data["sum"]["false_c"]["average"],0))} & {int(round(data["sum"]["true_c"]["average"],0))} & {int(round(data["sum"]["none_prediction"]["average"],0))} \\\\\\\\"""
}

metrics_binary = {
    "start" : """
\\begin{table}[h]
    \centering
    \\begin{tabular}{l|c||ccc}
        & Model & Accuracy & F1(C) & F1(E) \\\\
        \hline
        \hline
        Reproduction & Span NLI BERT & $\\textbf{0.891}\pm0.009$ & $0.500\pm0.028$ & $0.804\pm0.005$ \\\\
        \hline
        Narendra 2024 & GPT-4 & 0.87 & 0.70 & 0.91 \\\\
                      & Mixtral & {0.90} & {0.74} & {0.93} \\\\
        \hline
        Ollama""",

    "row" : """
& {model} 
& ${round(data["macro_label_micro_doc"]["class_binary"]["accuracy"]["average"] ,3)} \pm {round(data["macro_label_micro_doc"]["class_binary"]["accuracy"]["std"] ,3)} $
& ${round(data["macro_label_micro_doc"]["class_binary"]["f1_contradiction"]["average"] ,3)} \pm {round(data["macro_label_micro_doc"]["class_binary"]["f1_contradiction"]["std"] ,3)} $
& ${round(data["macro_label_micro_doc"]["class_binary"]["f1_entailment"]["average"] ,3)} \pm {round(data["macro_label_micro_doc"]["class_binary"]["f1_entailment"]["std"] ,3)} $ \\\\\\\\ """,

    "end" : """

\end{tabular}
    \caption{Placeholder }
    \label{tab:placeholder}
\end{table}
"""
}



def to_table(models, rows, template):
    table = template["start"]
    for model, data in zip(models, rows):
        row = eval("f'''" + template["row"] + "'''")
        table += row
    table += "\n" + template["end"]

    return table

@click.command()
@click.option("-f", type=click.Path(exists=True), multiple=True)
@click.option("--template", type=str)
@click.option("-o", type=click.Path())
def main(f, template, o):
    rows = []
    table = None

    for fp in f:
        with open(fp, "r") as row_file:
            row = json.load(row_file)
            rows.append(row)
    print(f"Files opened")
    table = to_table(f, rows, eval(template))
    print("Table converted")
    with open(o, "w") as out_file:
        out_file.write(table)
    print(f"File written to {o}")


if __name__=="__main__":
    main()