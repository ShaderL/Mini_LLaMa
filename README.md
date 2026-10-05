# Mini_LLaMa
手搓LLaMa

# Steps
1.创建新虚拟环境
```sh
python -m venv .venv
.\.venv\Scripts\activate
```

2.安装依赖
```sh
pip install -r requirements.txt
python -c "import torch; import transformers; import datasets; import yaml; print('all imports ok')"
```

3.修改 torch 至当前 GPU 配套 CUDA 版本

4.（可选）修改 configs/default.yaml 训练数据集相关配置

5.运行
```sh
python cli.py --help
python cli.py train --help
python cli.py generate --help
```
