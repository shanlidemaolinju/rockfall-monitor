from modelscope import snapshot_download

# 自动下载SAM 3.1基础版（3.4GB，适合8G显存）
model_dir = snapshot_download(
    'AI-ModelScope/sam3.1-base',
    cache_dir='./models',
    revision='master'
)

print(f"SAM 3.1模型已下载到：{model_dir}")
print("请记住这个路径，后面会用到")