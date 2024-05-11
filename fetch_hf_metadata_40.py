import os

import requests.adapters
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from huggingface_hub import HfApi
from tqdm import tqdm
import csv
import requests
import json

from collections import namedtuple
RepoSibling = namedtuple("RepoSibling", ['rfilename', 'size', 'blob_id', "lfs"])

HF_TOKEN = os.environ['HF_TOKEN']

criterion = "likes"

header = ['id', 'author', 'sha', 'last_modified', 'created_at', 'private', 'gated', 'disabled', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'mask_token',
          'card_data', 'widget_data', 'model_index', 'config', 'transformers_info', 'siblings', 'spaces', 'safetensors', 'lastModified', 'cardData', 'transformersInfo', '_id', 'modelId']


def is_backend_valid(tags):
    if "pytorch" in tags or "tf" in tags or "sklearn" in tags:
        return True
    return False


skipped_files = ["README.md", ".gitattributes"]
def get_download_size(repo_id, siblings):
    url = f"https://hf-mirror.com/{repo_id}/resolve/main"
    # siblings = eval(siblings)
    from concurrent.futures import ThreadPoolExecutor
    # from requests.packages.urllib3.retry import Retry

    def worker(sib):
        # session = requests.Session()
        # adapter = requests.adapters.HTTPAdapter(max_retries=Retry(
        #     total=10, backoff_factor=1, status_forcelist=[429,500,502,503,504],
        #     method_whitelist=["HEAD"],
        # ))
        # session.mount("http://", adapter)
        # session.mount("https://", adapter)
        
        link = f"{url}/{sib.rfilename}"
        r = requests.head(link, headers={"Accept-Encoding": "identity"})
        if r.status_code in (401, 403):
            r = requests.head(link, headers={"Authorization": f"Bearer {HF_TOKEN}", "Accept-Encoding": "identity"})
        if 'x-error-code' in r.headers and r.headers['x-error-code'].lower() == 'gatedrepo':
            return 0
        if 'location' not in r.headers:
            cl = int(r.headers['Content-Length'])
        else:
            loc = r.headers['location']
            r_redirect = requests.head(loc, headers={"Accept-Encoding": "identity"})
            if 'content-length' not in r_redirect.headers:
                print(link)
                print(r.headers)
                print(r_redirect.headers)
            cl = int(r_redirect.headers['Content-Length'])
        print(link, cl)
        return cl

    with ThreadPoolExecutor(16) as pool:
        handles = []
        for sib in siblings:
            if sib.rfilename in skipped_files:
                continue
            h = pool.submit(worker, sib)
            handles.append(h)
    
    size = 0
    for h in handles:
        if h.done():
            # print(h.result)
            size += h.result()
            
    return size


def one_trial(ids, task, criterion, topK, limit):
    api = HfApi(endpoint="https://hf-mirror.com")
    models = api.list_models(filter=task, sort=criterion, direction=-1, limit=limit)
    model_infos = []

    for model_info in models:
        model_info = model_info.__dict__

        tags = model_info['tags']
        if not is_backend_valid(tags):
            continue

        id = model_info['id']
        if id in ids:
            continue

        repo_id = model_info['id']
        siblings = model_info['siblings']
        # dl_size = get_download_size(repo_id, siblings)
        # if dl_size == 0:
        #     print(f"Auth needed for {repo_id}. Skipped.")
        #     continue

        model_info = list(model_info.values())
        # model_info.append(dl_size)

        model_infos.append(model_info)
        ids.add(id)
        if len(model_infos) == topK:
            break
    return model_infos


if __name__ == "__main__":
    with open("tasks_40.json", "r") as fp:
        repo_ids = json.load(fp)

    metadata_dir = f"metadata/40/"
    os.makedirs(metadata_dir, exist_ok=True)

    api = HfApi(endpoint="https://hf-mirror.com")
    with open(os.path.join(metadata_dir, "40.csv"), "w") as fp:
        writer = csv.writer(fp, lineterminator='\n')
        writer.writerow(header)
        for id in repo_ids:
            model_info = api.model_info(id)
            model_info = model_info.__dict__
            writer.writerow(model_info.values())

