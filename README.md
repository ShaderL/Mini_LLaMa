# Mini_LLaMa
手搓LLaMa

# Steps
1.创建新虚拟环境
python -m venv .venv
.\.venv\Scripts\activate

2.安装依赖
pip install -r requirements.txt
python -c "import torch; import transformers; import datasets; import yaml; print('all imports ok')"

3.运行
python cli.py --help
python cli.py train --help
python cli.py generate --help
