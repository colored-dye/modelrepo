# Mission

运行脚本,将列表中的模型下载到本地,加载到模型仓库中.

Total download size: 4851.775 GiB.


# ***不要使用代理***! (下载数据量太大)


# Dependencies

仅测试过python 3.10/3.11.

## Conda

```bash
conda create -n modelrepo python==3.10 -y
conda activate modelrepo
pip install modelstore transformers torch tensorflow tf-keras scikit-learn tqdm tensorflow_probability
```

## Venv

Bash:
```bash
python -m venv modelrepo
source modelrepo/bin/activate
pip install modelstore transformers torch tensorflow tf-keras scikit-learn tqdm tensorflow_probability
```

Powershell:
```powershell
python -m venv modelrepo
modelrepo\Scripts\activate.ps1
pip install modelstore transformers torch tensorflow tf-keras scikit-learn tqdm tensorflow_probability
```

# How to run

1. 更新脚本参数: `upload.sh`(Bash) / `upload.ps1`(Powershell). 修改:
    - `CSV_FILE`: 模型列表文件. 从`jobs/`选择一个.
    - `ROOT_DIR`: 模型仓库根目录.
2. 个别模型可能需要额外的许可和身份认证,因此需要到Huggingface上申请访问权限,并获取自己账号的Access token. 修改:
    - `HF_TOKEN`: `hf_xxxx`形式的字符串.
3. 运行脚本.
    - Powershell: `.\upload.ps1`
    - Bash: `bash upload.sh`
4. 检查模型是否全部加载到模型仓库中.

    有些模型不能加载到模型仓库中.可能的原因:
    1. 程序因为网络问题或人为终止而中断.
    2. 模型仓库不支持该模型.

    可以运行`python test_finished.py --root_dir <ROOT_DIR> --csv_file <CSV_FILE>`检查初始的任务列表是否全部完成,输出`todo.csv`文件,其中包含未加载到模型仓库中的模型.
    该文件可以重新作为步骤(1)的输入.

    **由于第二种原因而未能成功加载的模型可以直接跳过**.
