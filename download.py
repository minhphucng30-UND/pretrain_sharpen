from huggingface_hub import snapshot_download

if __name__=='__main__':
    snapshot_download(
        repo_id="DatPySci/pretrain-sharpen",
        allow_patterns=["OLMo-150M/**"],
        local_dir="",
    )