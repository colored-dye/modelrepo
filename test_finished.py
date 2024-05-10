import argparse
import csv
import os

from modelstore import ModelStore


def model_exists(root_dir: str,
                 repo_id: str):
    model_store = ModelStore.from_file_system(root_directory=root_dir, create_directory=True)
    try:
        model_ids = model_store.list_models(repo_id)
    except:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir")
    parser.add_argument("--csv_file")
    args = parser.parse_args()

    with open(args.csv_file, "r", encoding='utf-8') as fp, open("todo.csv", "w", encoding='utf-8') as todo_fp:
        reader = csv.reader(fp, lineterminator='\n')
        header = next(reader)

        writer = csv.writer(todo_fp, lineterminator='\n')
        writer.writerow(header)
        for row in reader:
            if not model_exists(args.root_dir, row[0]):
                writer.writerow(row)
