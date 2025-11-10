from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

REPO = "shunk031/Rico"
print("Configs:", get_dataset_config_names(REPO))

CFG = "ui-screenshots-and-view-hierarchies"
try:
    print("Splits:", get_dataset_split_names(REPO, CFG))
except Exception as e:
    print("Could not list splits for", CFG, e)

ds = load_dataset(REPO, name=CFG, split="train")
print("Columns:", ds.column_names)
print("Row 0 keys:", ds[0].keys())
