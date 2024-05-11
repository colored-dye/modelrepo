import os
import csv
import shutil
import subprocess
import argparse
import time
import tarfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tempfile

from modelstore import ModelStore
from transformers import AutoModel, TFAutoModel
import joblib
from huggingface_hub import hf_hub_download

import utils


from collections import namedtuple
RepoSibling = namedtuple("RepoSibling", ['rfilename', 'size', 'blob_id', "lfs"])
header = ['id', 'author', 'sha', 'last_modified', 'created_at', 'private', 'gated', 'disabled', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'mask_token',
          'card_data', 'widget_data', 'model_index', 'config', 'transformers_info', 'siblings', 'spaces', 'safetensors', 'lastModified', 'cardData', 'transformersInfo', '_id', 'modelId', 'download_size']


def hfhub_download(model_info, download_dir, hf_token, max_connections):
    repo_id = model_info[0]
    print(f"Downloading {repo_id} to {download_dir}.")

    siblings_idx = header.index("siblings")
    siblings = model_info[siblings_idx]
    siblings = eval(siblings)


    def download_from_repo(repo_id, sib):
        file = sib.rfilename
        hf_hub_download(repo_id, file, resume_download=True, local_dir=download_dir, local_dir_use_symlinks=False, token=hf_token)

    
    with ThreadPoolExecutor(max_workers=max_connections) as pool:
        handles = []
        # with ThreadPoolExecutor(max_workers=max_connections) as pool:
        for sib in siblings:
            handles.append(pool.submit(download_from_repo, repo_id, sib))
        while len(handles) > 0:
            finished = []
            for h in handles:
                if h.running():
                    try:
                        None
                    except KeyboardInterrupt:
                        print("Ctrl-C")
                        pool.shutdown(wait=False, cancel_futures=True)
                else:
                    finished.append(h)
            for f in finished:
                handles.remove(f)

    return


def tar_repo(src_dir, target_path):
    print(f"Packing archive.")
    start = time.time()
    # ret = subprocess.run(["tar", "-czf", target_path,
    #                 "-C", src_dir, "./",
    #                 "--transform", "flags=r;s,^\./,transformers/,", "--show-transformed-names"],
    #                 stdout=log_fp, stderr=log_fp)
    
    with tarfile.open(target_path, "w:gz") as tar:
        files = os.listdir(src_dir)
        for f in tqdm(files, desc="Tar"):
            tar.add(os.path.join(src_dir, f), os.path.join("transformers/", f))
    end = time.time()
    print(f"Time used: {(end-start)//60} min {int(end-start)%60} s")

    print(f"Archive complete.")


def upload_from_cache(repo_id: str,
                      basename: str,
                      model_dir: str,
                      root_dir: str,
                      backend: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        temp_archive = os.path.join(tmpdir, f"{basename}.tar.gz")
        tar_repo(model_dir, temp_archive)
    
        if backend == "torch":
            model = AutoModel.from_pretrained(model_dir)
        elif backend == "tf":
            model = TFAutoModel.from_pretrained(model_dir)
        elif backend == "sklearn":
            model_files = os.listdir(model_dir)
            model_files = [f for f in model_files if f.endswith(".joblib")]
            if len(model_files) == 0:
                raise ValueError("No available model!")
            elif len(model_files) > 1:
                raise ValueError(f"Multiple models: {model_files}")
            model_file = os.path.join(model_dir, model_files[0])
            model = joblib.load(model_file)

        model_store = ModelStore.from_file_system(root_directory=root_dir, create_directory=True)
        meta_data = model_store.upload(repo_id, model=model)
        stored_path = meta_data['storage']['path']
        shutil.move(temp_archive, stored_path)
    return


def get_backends_from_storage(model_dir: str):
    files = os.listdir(model_dir)
    backends = []
    has_torch = False
    has_tf = False
    has_sklearn = False
    for file in files:
        if "pytorch_model" in file:
            has_torch = True
        elif "tf_model" in file:
            has_tf = True
        elif file.startswith("sklearn") and file.endswith(".joblib"):
            has_sklearn = True
    if has_torch:
        backends.append("torch")
    if has_tf:
        backends.append("tf")
    if has_sklearn:
        backends.append("sklearn")
    return backends


def model_exists(root_dir: str,
                 repo_id: str):
    model_store = ModelStore.from_file_system(root_directory=root_dir, create_directory=True)
    try:
        model_ids = model_store.list_models(repo_id)
    except:
        return False
    return True


def repo_download_and_upload(model_info: list,
                root_dir: str,
                cache_dir: str,
                hf_token: str,
                max_connections: int,):
    repo_id = model_info[0]
    if model_exists(root_dir, repo_id):
        print(f"{repo_id} already uploaded.")
        return
    
    if "/" in repo_id:
        owner, model_name = repo_id.split("/")
        basename = owner + "-" + model_name
        download_dir = os.path.join(cache_dir, owner, model_name)
    else:
        basename = repo_id
        download_dir = os.path.join(cache_dir, repo_id)
    
    os.makedirs(download_dir, exist_ok=True)

    # Download to cache
    hfhub_download(model_info, download_dir, hf_token, max_connections)
    
    # Upload to model repository.
    # A model might support multiple backends.
    backends = get_backends_from_storage(download_dir)
    print(f"Backends: {backends}")
    for backend in backends:
        upload_from_cache(repo_id, basename, download_dir, root_dir, backend)

    # Remove download cache.
    shutil.rmtree(download_dir)

    return


def test_load(root_dir: str,
              repo_id: str,
              backend: str):
    print(f"Test load {repo_id}.")
    model_store = ModelStore.from_file_system(root_directory=root_dir, create_directory=True)

    try:
        model_id = model_store.list_versions(repo_id)[0]
        # model_pack = utils.load(model_store, repo_id, model_id, backend)
        model_pack = model_store.load(repo_id, model_id)

        for i in model_pack:
            print(i.__class__)
        
        print(f"Test load OK.")
    except:
        print(f"Test load failed.")
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--max_connections", type=int, required=True)
    parser.add_argument("--backend", choices=["torch", "tf", "sklearn"])
    args = parser.parse_args()
    
    with open(args.csv_file, "r", encoding='utf-8') as fp:
        reader = csv.reader(fp, lineterminator='\n')
        next(reader)    # Skip header row.
        for row in reader:
            repo_download_and_upload(root_dir=args.root_dir,
                                    model_info=row,
                                    cache_dir=args.cache_dir,
                                    hf_token=args.hf_token,
                                    max_connections=args.max_connections)
            
            # repo_id = row[0]
            # test_load(root_dir=args.root_dir,
            #           repo_id=repo_id,
            #           backend=args.backend)

