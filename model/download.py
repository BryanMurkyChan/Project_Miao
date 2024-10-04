#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站

from huggingface_hub import snapshot_download

model_name = "infgrad/stella-base-zh-v3-1792d"
# while True 是为了防止断联
while True:
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=True,  # 在local-dir指定的目录中都是一些“链接文件”
            ignore_patterns=["*.bin"],  # 忽略下载哪些文件
            local_dir=DOWNLOAD_PATH,
            token="*************",   # huggingface的token
            resume_download=True        )
        break
    except:
        pass

