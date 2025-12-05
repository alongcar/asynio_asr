#!/usr/bin/env python3
"""
基于Vosk的多用户并发语音识别服务
支持实时WebSocket音频流，线程池处理，热词配置
"""

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional
from queue import Queue, Empty

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from vosk import Model, KaldiRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoskASRServer:
    """Vosk语音识别服务器核心类"""

    def __init__(
            self,
            model_path: str,
            sample_rate: float = 16000.0,
            max_workers: int = 4,
            hotwords: Optional[list] = None
    ):
        """
        初始化语音识别服务器

        Args:
            model_path: Vosk模型路径
            sample_rate: 音频采样率
            max_workers: 工作线程数
            hotwords: 热词列表，提升特定词汇识别率
        """
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.max_workers = max_workers

        # 加载Vosk模型
        logger.info(f"加载Vosk模型: {model_path}")
        self.model = Model(model_path)

        # 热词配置
        self.hotwords = hotwords or []
        if self.hotwords:
            logger.info(f"加载热词: {', '.join(self.hotwords[:5])}...")

        # 工作线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"初始化线程池，工作线程数: {max_workers}")

        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        self.session_lock = threading.Lock()

        # 任务队列
        self.task_queue = Queue()

        # 启动工作线程
        self._start_workers()

    def _create_recognizer(self) -> KaldiRecognizer:
        """创建并配置识别器实例"""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)

        # 应用热词配置
        if self.hotwords:
            try:
                # 创建语法规则
                grammar = json.dumps(self.hotwords)
                recognizer.SetGrammar(grammar)
            except Exception as e:
                logger.warning(f"设置热词失败: {e}")

        return recognizer

    def _worker_task(self):
        """工作线程任务"""
        # 每个线程创建自己的识别器实例
        recognizer = self._create_recognizer()

        while True:
            try:
                # 从队列获取任务
                task = self.task_queue.get()
                if task is None:  # 终止信号
                    break

                session_id, audio_data = task

                with self.session_lock:
                    session = self.sessions.get(session_id)
                    if not session:
                        continue

                # 处理音频数据
                if recognizer.AcceptWaveform(audio_data):
                    # 获取最终结果
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()

                    if text:
                        with self.session_lock:
                            if session_id in self.sessions:
                                self.sessions[session_id]["results"].append(text)
                                self.sessions[session_id]["last_result"] = text
                else:

                    # 获取部分结果
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    print("正在实时识别中：", partial_text)

                    if partial_text:
                        with self.session_lock:
                            if session_id in self.sessions:
                                self.sessions[session_id]["partial"] = partial_text

            except Exception as e:
                logger.error(f"工作线程处理错误: {e}")
            finally:
                self.task_queue.task_done()

    def _start_workers(self):
        """启动工作线程"""
        for i in range(self.max_workers):
            self.thread_pool.submit(self._worker_task)
        logger.info(f"已启动 {self.max_workers} 个工作线程")

    def create_session(self) -> str:
        """创建新的识别会话"""
        session_id = str(uuid.uuid4())

        with self.session_lock:
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": asyncio.get_event_loop().time(),
                "results": [],
                "partial": "",
                "last_result": ""
            }

        logger.info(f"创建新会话: {session_id}")
        return session_id

    def process_audio(self, session_id: str, audio_data: bytes):
        """处理音频数据"""
        with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"会话不存在: {session_id}")
                return

        # 将任务放入队列
        self.task_queue.put((session_id, audio_data))

    def get_session_results(self, session_id: str) -> Dict:
        """获取会话结果"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            if not session:
                return {"error": "Session not found"}

            # 返回副本，避免并发修改
            return {
                "id": session["id"],
                "results": session["results"].copy(),
                "partial": session["partial"],
                "last_result": session["last_result"]
            }

    def close_session(self, session_id: str):
        """关闭会话"""
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"关闭会话: {session_id}")


# 全局服务器实例
asr_server: Optional[VoskASRServer] = None

# 创建FastAPI应用
app = FastAPI(title="Vosk语音识别服务")


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global asr_server

    # 配置热词（麻醉问诊相关术语）
    hotwords = [
        "是", "否", "有", "没有", "不知道",
        "头痛", "头晕", "恶心", "呕吐", "心慌",
        "麻醉", "手术", "病史", "过敏", "药物",
        "丙泊酚", "七氟烷", "罗库溴铵", "舒芬太尼",
        "全麻", "局麻", "椎管内麻醉", "神经阻滞"
    ]

    # 初始化语音识别服务器
    # 注意：需要先下载Vosk中文模型，例如: vosk-model-small-cn-0.22
    model_path = "model/vosk-model-small-cn-0.22"  # 修改为你的模型路径

    asr_server = VoskASRServer(
        model_path=model_path,
        sample_rate=16000.0,
        max_workers=4,  # 根据CPU核心数调整
        hotwords=hotwords
    )


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global asr_server
    if asr_server:
        # 发送终止信号给工作线程
        for _ in range(asr_server.max_workers):
            asr_server.task_queue.put(None)
        asr_server.thread_pool.shutdown(wait=True)
        logger.info("服务器资源已清理")


@app.websocket("/ws/asr")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，处理实时音频流"""
    global asr_server

    await websocket.accept()
    session_id = None

    try:
        # 创建新会话
        session_id = asr_server.create_session()
        await websocket.send_json({"type": "session_created", "session_id": session_id})

        logger.info(f"WebSocket连接建立，会话ID: {session_id}")

        while True:
            # 接收音频数据
            data = await websocket.receive_bytes()

            if not data:
                continue

            # 处理音频数据
            asr_server.process_audio(session_id, data)

            # 获取并发送最新的部分结果
            session_data = asr_server.get_session_results(session_id)
            if session_data.get("partial"):
                await websocket.send_json({
                    "type": "partial_result",
                    "text": session_data["partial"]
                })

            # 检查是否有最终结果
            if session_data.get("last_result"):
                last_result = session_data["last_result"]
                await websocket.send_json({
                    "type": "final_result",
                    "text": last_result
                })
                # 清除last_result，避免重复发送
                asr_server.sessions[session_id]["last_result"] = ""

    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开，会话ID: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        if session_id:
            asr_server.close_session(session_id)


@app.post("/api/session")
async def create_session():
    """创建新的识别会话（HTTP API）"""
    global asr_server
    session_id = asr_server.create_session()
    return {"session_id": session_id}


@app.post("/api/recognize/{session_id}")
async def recognize_audio(session_id: str, audio_data: bytes):
    """识别音频数据（HTTP API）"""
    global asr_server
    asr_server.process_audio(session_id, audio_data)

    # 等待处理完成
    await asyncio.sleep(0.1)  # 短暂等待

    results = asr_server.get_session_results(session_id)
    return results


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """获取识别结果（HTTP API）"""
    global asr_server
    return asr_server.get_session_results(session_id)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话（HTTP API）"""
    global asr_server
    asr_server.close_session(session_id)
    return {"message": "Session closed"}


@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """服务首页，包含简单的测试页面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vosk语音识别服务</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { display: flex; flex-direction: column; gap: 20px; }
            .section { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:disabled { background: #ccc; }
            .result { background: #f5f5f5; padding: 10px; border-radius: 4px; min-height: 100px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Vosk语音识别服务</h1>

            <div class="section">
                <h2>WebSocket实时识别</h2>
                <div>
                    <button id="startRecording">开始录音</button>
                    <button id="stopRecording" disabled>停止录音</button>
                </div>
                <div>
                    <h3>识别结果:</h3>
                    <div id="results" class="result"></div>
                </div>
            </div>

            <div class="section">
                <h2>服务状态</h2>
                <div>
                    <p>WebSocket连接状态: <span id="wsStatus">未连接</span></p>
                    <p>会话ID: <span id="sessionId">-</span></p>
                </div>
            </div>
        </div>

        <script>
            let mediaRecorder = null;
            let audioChunks = [];
            let ws = null;
            let sessionId = null;

            document.getElementById('startRecording').onclick = async () => {
                try {
                    // 连接WebSocket
                    ws = new WebSocket(`ws://${window.location.host}/ws/asr`);

                    ws.onopen = () => {
                        document.getElementById('wsStatus').textContent = '已连接';
                        document.getElementById('startRecording').disabled = true;
                        document.getElementById('stopRecording').disabled = false;
                    };

                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'session_created') {
                            sessionId = data.session_id;
                            document.getElementById('sessionId').textContent = sessionId;
                        } else if (data.type === 'partial_result') {
                            document.getElementById('results').textContent = `实时识别: ${data.text}`;
                        } else if (data.type === 'final_result') {
                            const current = document.getElementById('results').textContent;
                            document.getElementById('results').textContent = 
                                (current.startsWith('实时识别:') ? '' : current + '\\n') + data.text;
                        }
                    };

                    ws.onclose = () => {
                        document.getElementById('wsStatus').textContent = '已断开';
                        document.getElementById('startRecording').disabled = false;
                        document.getElementById('stopRecording').disabled = true;
                    };

                    // 获取麦克风权限并开始录音
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            // 将音频数据发送到服务器
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(event.data);
                            }
                        }
                    };

                    mediaRecorder.start(100); // 每100ms发送一次数据

                } catch (error) {
                    console.error('录音失败:', error);
                    alert('无法访问麦克风: ' + error.message);
                }
            };

            document.getElementById('stopRecording').onclick = () => {
                if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                    mediaRecorder.stop();
                    mediaRecorder.stream.getTracks().forEach(track => track.stop());
                }

                if (ws) {
                    ws.close();
                }

                document.getElementById('startRecording').disabled = false;
                document.getElementById('stopRecording').disabled = true;
            };
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    # 启动服务
    logger.info("启动Vosk语音识别服务...")
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有接口
        port=8000,  # 服务端口
        log_level="info"
    )
