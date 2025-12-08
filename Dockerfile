# 使用一个较小的Python官方镜像作为基础，推荐使用具体版本号而非latest
FROM python:3.12-slim AS builder

# 设置工作目录
WORKDIR /app

# 复制依赖列表文件
COPY requirements.txt .

## 1. 安装系统构建依赖和 PortAudio 开发库
#RUN apt-get update && apt-get install -y \
#    --no-install-recommends \
#    gcc g++ \
#    portaudio19-dev \  # 这是解决 pyaudio 编译问题的关键包
#    && rm -rf /var/lib/apt/lists/*  # 清理缓存以减小镜像体积

# 安装构建依赖和运行时依赖
# 先安装构建vosk等可能需要的工具，然后安装Python依赖，最后清理缓存以减小镜像体积
RUN apt-get update && apt-get install -y \
    --no-install-recommends gcc g++ \
#    && pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 第二阶段构建，得到一个更小的最终镜像
FROM python:3.12-slim

# 设置容器内的工作目录
WORKDIR /app

# 从构建阶段复制已安装的Python包
COPY --from=builder /usr/local /usr/local

# 复制你的应用程序代码（先复制依赖关系更稳定的文件，利用Docker缓存层）
COPY . .

# 将模型文件复制到容器中（请确保你的模型路径与此一致）
COPY model/ /app/model/

# 确保容器运行时使用我们安装的包
ENV PATH=/usr/local/bin:$PATH

# 对外暴露服务运行的端口
EXPOSE 8000

# 设置环境变量，例如语言环境，防止输出乱码
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# 设置健康检查，确保容器服务正常
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs')"]

# 使用非root用户运行应用以提升安全性[7](@ref)
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# 设置容器启动时的默认命令
CMD ["uvicorn", "asr_service:app", "--host", "0.0.0.0", "--port", "8000"]