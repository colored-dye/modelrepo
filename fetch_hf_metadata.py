import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

from huggingface_hub import HfApi, list_models, ModelFilter
from tqdm import tqdm
import csv
import requests

from collections import namedtuple
RepoSibling = namedtuple("RepoSibling", ['rfilename', 'size', 'blob_id', "lfs"])

HF_TOKEN = "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"

criterion = "likes"

header = ['id', 'author', 'sha', 'last_modified', 'created_at', 'private', 'gated', 'disabled', 'downloads', 'likes', 'library_name', 'tags', 'pipeline_tag', 'mask_token',
          'card_data', 'widget_data', 'model_index', 'config', 'transformers_info', 'siblings', 'spaces', 'safetensors', 'lastModified', 'cardData', 'transformersInfo', '_id', 'modelId', 'download_size']


tasks = {
    'nlp': {
        "text-generation": 30,
        "question-answering": 20,
        "text-classification": 20,
        "translation": 20,
        "summarization": 20,
        "token-classification": 10,
        "zero-shot-classification": 10,
        "feature-extraction": 10,
        "text2text-generation": 10,
        "fill-mask": 10,
        "sentence-similarity": 10,
        "table-question-answering": 10,
    },
    'vision': {
        "depth-estimation": 10,
        "image-classification": 10,
        "object-detection": 10,
        "image-segmentation": 10,
        "text-to-image": 10,
        "image-to-text": 10,
        "image-to-image": 10,
        "unconditional-image-generation": 10,
        "text-to-video": 1,
        "zero-shot-image-classification": 10,
        "mask-generation": 10,
        "zero-shot-object-detection": 10,
        "text-to-3d": 0,
        "image-to-3d": 6,
        "image-feature-extraction": 10,
    },
    'audio': {
        "text-to-speech": 10,
        "text-to-audio": 10,
        "automatic-speech-recognition": 10,
        "audio-to-audio": 10,
        "audio-classification": 10,
        "voice-activity-detection": 10,
    },
    'multi-modal': {
        "image-text-to-text": 5,
        "visual-question-answering": 10,
        "document-question-answering": 10,
    },
}


def is_backend_valid(tags):
    if "pytorch" in tags or "tf" in tags or "sklearn" in tags:
        return True
    return False


skipped_files = ["README.md", ".gitattributes"]
def get_download_size(repo_id, siblings):
    url = f"https://hf-mirror.com/{repo_id}/resolve/main"
    size = 0
    # siblings = eval(siblings)
    for sib in siblings:
        if sib.rfilename in skipped_files:
            continue
        link = f"{url}/{sib.rfilename}"
        r = requests.head(link, headers={"Accept-Encoding": "identity"})
        if r.status_code in (401, 403):
            r = requests.head(link, headers={"Authorization": f"Bearer {HF_TOKEN}", "Accept-Encoding": "identity"})
        if 'location' in r.headers:
            loc = r.headers['location']
            r = requests.head(loc, headers={"Accept-Encoding": "identity"})
        cl = int(r.headers['Content-Length'])
        print(link, cl)
        size += cl
    return size


def one_trial(task, criterion, topK, limit):
    models = list_models(filter=task, sort=criterion, direction=-1, limit=limit)
    ids = set()
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
        dl_size = get_download_size(repo_id, siblings)
        if dl_size < 10000:
            print(f"Auth needed for {repo_id}. Skipped.")
            continue

        model_info = list(model_info.values())
        model_info.append(dl_size)

        model_infos.append(model_info)
        ids.add(id)
        if len(model_infos) == topK:
            break
    return model_infos


if __name__ == "__main__":
    total = 0
    for task in tasks.values():
        for topK in task.values():
            total += topK

    metadata_dir = f"metadata/{criterion}"
    os.makedirs(metadata_dir, exist_ok=True)

    for field, task_dict in tasks.items():
        file = os.path.join(metadata_dir, f"{field}.csv")
        with open(file, "w", encoding='utf-8') as fp:
            writer = csv.writer(fp, lineterminator='\n')
            writer.writerow(header)

            for task, topK in tqdm(task_dict.items(), desc=field):
                model_infos = one_trial(task, criterion, topK, 2*topK)
                loop = 2
                while len(model_infos) < topK:
                    model_infos = one_trial(task, criterion, topK, topK*(2**(loop)))
                    loop += 1
                    print(task, len(model_infos))

                writer.writerows(model_infos)
