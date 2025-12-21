import matplotlib.pyplot as plt

def boxplot_analysis(correct: dict, wrong: dict, save_file: str, ylabel: str ="Probability"):

    d = list(correct.values()) + list(wrong.values())
    labels = [
        f"Correct {k}" for k in correct.keys()
        ] + [
        f"Wrong {k}" for k in wrong.keys()
        ]
    
        

    fig = plt.figure(figsize=(10,7)) 
    plt.boxplot(d, labels=labels)
    
    ax = plt.gca()
    
    fig.savefig(save_file)