from datasets import load_dataset

ds = load_dataset("elriggs/openwebtext-100k")
print(ds)
print(ds["train"][0])
