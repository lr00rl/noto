FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# 安装Python和依赖
RUN apt-get update && apt-get install -y python3 python3-pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 复制代码
COPY embedding_server.py .
COPY start_server.py .

EXPOSE 8000

CMD ["python3", "start_server.py"]
