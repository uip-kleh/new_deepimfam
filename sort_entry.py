import os
import pandas as pd
import json
from new_deepimfam.component import Common

if __name__ == "__main__":
    config_path = "new_deepimfam/config.yaml"
    common = Common(config_path=config_path, index=None)

    results = []

    entry_path = os.path.join(common.data_direcotry, "aaindex_entry.txt")
    base_path = os.path.join(common.storage_path, "mizuta_NORM_INDEX_256")
    with open(entry_path, "r") as f:
        for entry in f.read().splitlines():
            history_path = os.path.join(entry, "results/FOCAL_LOSS/history.csv")
            history_path = os.path.join(base_path, history_path)
            print(history_path)
            acc = pd.read_csv(history_path)["val_accuracy"].values[-1]

            results.append([acc, entry]) 

    results = sorted(results)
    results.reverse()
    fname = os.path.join(common.results, "sorted_results.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
