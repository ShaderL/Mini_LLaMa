# Mini_LLaMa
手搓LLaMa

# Steps
1.创建新虚拟环境
```sh
Windows
python -m venv .venv
.\.venv\Scripts\activate
```
```sh
python -m venv .venv
source .venv/bin/activate
```

2.安装依赖
```sh
pip install -r requirements.txt
python -c "import torch; import transformers; import datasets; import yaml; print('all imports ok')"
```

3.修改 huggingface Cache 路径！！！
```sh
export HF_HOME=/data/xiangyouLiu/huggingface_cache
```

3.修改 torch 至当前 GPU 配套 CUDA 版本


4.（可选）修改 configs/default.yaml 训练数据集相关配置


5.运行
```sh
python cli.py --help
python cli.py train --help
python cli.py generate --help
```
