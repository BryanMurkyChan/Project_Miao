from modelscope import snapshot_download
import os
DOWNLOAD_PATH = os.path.join(os.path.dirname(__file__))
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", cache_dir=DOWNLOAD_PATH)
