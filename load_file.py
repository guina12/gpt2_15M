from datasets import load_dataset
import os

def save_openwebtext_to_txt(output_path="openwebtext.txt"):
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    else: 
        try:
            ds_dict = load_dataset("elriggs/openwebtext-100k")
            ds = ds_dict.get('train') if 'train' in ds_dict else next(iter(ds_dict.values()))
        except Exception:
            return ""

        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in ds:
                text = item["text"].replace("\r\n", "\n").strip()
                f_out.write(text + "\n\n")
        
        with open(output_path, mode = 'r', encoding = 'utf-8') as f:
           raw_text = f.read()
        
        return raw_text