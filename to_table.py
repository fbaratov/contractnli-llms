import click
import json

path_to_model = {
    "llama3.2_3b/": "LLaMA3.2 3B",
    "gemma3/": "Gemma3 4B",
    "mistral_7b/": "Mistral 7B",
    "llama3.1_8b/": "LLaMA3.1 8B",
    "gpt-oss_20b/": "GPT-OSS 20B",
    "gemma3_27b/": "Gemma3 27B",
    "qwen3_30b/": "Qwen3 30B",
}

def to_model(fp: str) -> str:
    for k,v in path_to_model.items():
        if k in fp:
            return v

def rearrange(f, rows):
    rearranged_rows, models = [],[]
    for k,v in path_to_model.items():
        for fp, row in zip(f, rows):
            if k in fp:
                models.append(v)
                rearranged_rows.append(row)
    
    return models, rearranged_rows

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
    models = []


    for fp in f:
        with open(fp, "r") as row_file:
            row = json.load(row_file)
            rows.append(row)
    print(f"Files opened")
    models, rows = rearrange(f, rows)
    table = to_table(models, rows, eval(template))
    print("Table converted")
    with open(o, "w") as out_file:
        out_file.write(table)
    print(f"File written to {o}")


if __name__=="__main__":
    main()