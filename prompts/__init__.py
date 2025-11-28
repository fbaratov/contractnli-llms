import os

def load_txt_files_to_dict(directory):
    txt_files_dict = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
            file_key = os.path.splitext(filename)[0]  # Remove '.txt'
            txt_files_dict[file_key] = file_content
            
    return txt_files_dict

prompts = load_txt_files_to_dict("./prompts")