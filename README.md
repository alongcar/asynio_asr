### 构建与运行 
##### 构建 Docker 镜像
在包含 Dockerfile和 requirements.txt的目录下，执行以下命令。注意最后有一个点（.），表示使用当前目录作为构建上下文。
docker build -t my-asr-service .
##### 运行 Docker 容器
镜像构建成功后，使用以下命令运行容器：
docker run -d --name asr-container -p 8000:8000 my-asr-service
-d：让容器在后台运行。
--name asr-container：为容器起一个名字。
-p 8000:8000：将宿主机的 8000 端口映射到容器的 8000 端口。