import os

from huggingface_hub import HfApi

if __name__ == "__main__":
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = "DatPySci/pretrain-sharpen"
    api.upload_folder(
        repo_id=repo_id,
        folder_path="OLMo-150M/OLMo-150M-constant-3e3/step60000-hf",
        repo_type="model",
        path_in_repo="OLMo-150M/OLMo-150M-constant-3e3",
    )
