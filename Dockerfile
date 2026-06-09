# 使用官方 Python 镜像
FROM python:3.12-slim

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 安装依赖（包括 fastapi, uvicorn, spacy）
RUN uv sync --frozen --no-dev

# 下载 spaCy 模型
RUN uv run python -m spacy download en_core_web_sm

# 复制应用代码
COPY main.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "fastapi", "run", "main.py", "--port", "8000", "--host", "0.0.0.0"]