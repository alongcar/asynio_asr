import json
import logging
import threading
import time
from typing import Dict, List, Optional

from vosk import KaldiRecognizer, Model


logger = logging.getLogger(__name__)


class ASRService:
    def __init__(self, model: Model, sample_rate: int, hotwords: Optional[List[str]] = None) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.hotwords = hotwords or []

        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        self.session_recognizers: Dict[str, KaldiRecognizer] = {}
        self.session_lock = threading.Lock()
        # 新增：每个会话独立的处理锁，避免同一识别器被并发调用
        self.session_process_locks: Dict[str, threading.Lock] = {}

    def _create_recognizer(self) -> KaldiRecognizer:
        """创建并配置识别器实例（移除对 SetGrammar 的滥用）"""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        # 不再调用 SetGrammar 以避免将识别限制为热词
        return recognizer

    def open_session(self, session_id: str) -> None:
        """打开一个新的识别会话"""
        recognizer = self._create_recognizer()
        with self.session_lock:
            self.session_recognizers[session_id] = recognizer
            # 为每个会话创建独立的处理锁
            self.session_process_locks[session_id] = threading.Lock()
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": time.time(),
                "results": [],
                "partial": "",
                "last_result": "",
                "recognizer": recognizer,
            }
        logger.info(f"打开会话: {session_id}")

    def process_audio(self, session_id: str, audio_data: bytes) -> Optional[str]:
        """处理一段音频数据，返回最终文本（若有）"""
        with self.session_lock:
            session = self.sessions.get(session_id)
            recognizer = self.session_recognizers.get(session_id)
            process_lock = self.session_process_locks.get(session_id)

        if not session or not recognizer or not process_lock:
            return None

        # 使用独立处理锁保护识别器调用，避免并发
        with process_lock:
            try:
                if recognizer.AcceptWaveform(audio_data):
                    # 获取最终结果
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        with self.session_lock:
                            if session_id in self.sessions:
                                self.sessions[session_id]["results"].append(text)
                                self.sessions[session_id]["last_result"] = text
                        return text
                else:
                    # 获取部分结果
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        with self.session_lock:
                            if session_id in self.sessions:
                                self.sessions[session_id]["partial"] = partial_text
            except Exception as e:
                logger.warning(f"识别处理失败: {e}")
        return None

    def get_partial(self, session_id: str) -> str:
        with self.session_lock:
            session = self.sessions.get(session_id)
            return (session or {}).get("partial", "")

    def get_results(self, session_id: str) -> List[str]:
        with self.session_lock:
            session = self.sessions.get(session_id)
            return list((session or {}).get("results", []))

    def close_session(self, session_id: str) -> None:
        """关闭会话，调用 FinalResult 获取最后一段识别结果并清理资源和锁"""
        # 先获取处理锁（不持有 session_lock），避免死锁
        with self.session_lock:
            process_lock = self.session_process_locks.get(session_id)

        # 为避免与正在进行的识别并发，先获取处理锁
        if process_lock:
            process_lock.acquire()

        try:
            with self.session_lock:
                if session_id in self.sessions:
                    recognizer = self.session_recognizers.get(session_id)
                    session = self.sessions.get(session_id)
                    if recognizer and session:
                        try:
                            final = json.loads(recognizer.FinalResult())
                            text = final.get("text", "").strip()
                            if text:
                                session["results"].append(text)
                                session["last_result"] = text
                        except Exception:
                            # FinalResult 失败不影响清理
                            pass
                    # 清理识别器与锁
                    if session_id in self.session_recognizers:
                        del self.session_recognizers[session_id]
                    if session_id in self.session_process_locks:
                        del self.session_process_locks[session_id]
                    del self.sessions[session_id]
                    logger.info(f"关闭会话: {session_id}")
        finally:
            # 释放处理锁（如果仍持有）
            if process_lock:
                try:
                    process_lock.release()
                except RuntimeError:
                    pass
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
from contextlib import asynccontextmanager
from queue import Queue, Empty, Full

import os
import pyaudio
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
            max_workers: Optional[int] = None,
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
        if max_workers is None:
            env_workers = os.getenv("ASR_MAX_WORKERS")
            if env_workers:
                try:
                    max_workers = max(1, int(env_workers))
                except Exception:
                    max_workers = (os.cpu_count() or 1) + 1
            else:
                max_workers = (os.cpu_count() or 1) + 1
        self.max_workers = max_workers

        # 添加音频播放功能
        self.audio_player = None
        self.playback_enabled = True  # 可配置，是否开启播放
        self.playback_stream = None
        self.setup_audio_playback(int(sample_rate))

        # 加载Vosk模型
        logger.info(f"加载Vosk模型: {model_path}")
        self.model = Model(model_path)

        # 热词配置
        self.hotwords = hotwords or []
        if self.hotwords:
            logger.info(f"加载热词: {', '.join(self.hotwords[:5])}...")

        # 工作线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"初始化线程池，工作线程数: {self.max_workers}")

        # 会话管理
        self.sessions: Dict[str, Dict] = {}
        self.session_recognizers: Dict[str, KaldiRecognizer] = {}
        self.session_lock = threading.Lock()
        self.session_process_locks: Dict[str, threading.Lock] = {}
        self.min_chunk_bytes = int(self.sample_rate * 0.2) * 2

        # 任务队列
        self.task_queue = Queue(maxsize=self.max_workers * 50)

        # 启动工作线程
        self._start_workers()

    def setup_audio_playback(self, sample_rate: int):
        """设置音频播放"""
        try:
            self.audio_player = pyaudio.PyAudio()
            self.playback_stream = self.audio_player.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            print(f"🔊 音频播放器已初始化，采样率: {sample_rate}Hz")
        except Exception as e:
            print(f"⚠️ 无法初始化音频播放器: {e}")

    def debug_play_audio(self, audio_data: bytes, session_id: str):
        """调试功能：播放接收到的音频"""
        if not self.playback_enabled or not self.playback_stream:
            return

        try:
            # 只在前几个数据块播放，避免过多输出
            if hasattr(self, f'played_{session_id}'):
                if getattr(self, f'played_{session_id}') > 1000:  # 每个会话只播放前3个数据块
                    return
            else:
                setattr(self, f'played_{session_id}', 0)

            setattr(self, f'played_{session_id}', getattr(self, f'played_{session_id}') + 1)

            # 播放音频
            self.playback_stream.write(audio_data)
            print(f"▶️ 播放会话 {session_id[:8]}... 的音频数据块")

        except Exception as e:
            print(f"❌ 音频播放失败: {e}")

    def _create_recognizer(self) -> KaldiRecognizer:
        """创建并配置识别器实例"""
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        return recognizer

    def _worker_task(self):
        """工作线程任务"""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                session_id, audio_data = task
                with self.session_lock:
                    session = self.sessions.get(session_id)
                    recognizer = self.session_recognizers.get(session_id)
                    process_lock = self.session_process_locks.get(session_id)
                if not session or not recognizer or not process_lock:
                    continue
                with process_lock:
                    if recognizer.AcceptWaveform(audio_data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            with self.session_lock:
                                if session_id in self.sessions:
                                    self.sessions[session_id]["results"].append(text)
                                    self.sessions[session_id]["last_result"] = text
                    else:
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
        recognizer = self._create_recognizer()
        with self.session_lock:
            self.session_recognizers[session_id] = recognizer
            self.session_process_locks[session_id] = threading.Lock()
            self.sessions[session_id] = {
                "id": session_id,
                "created_at": asyncio.get_event_loop().time(),
                "results": [],
                "partial": "",
                "last_result": "",
                "recognizer": recognizer,
                "buffer": bytearray()
            }
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def process_audio(self, session_id: str, audio_data: bytes):
        """处理音频数据"""
        with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"会话不存在: {session_id}")
                return

        print("开始播放音频....")
        self.debug_play_audio(audio_data, session_id)

        load = 0.0
        try:
            load = self.task_queue.qsize() / float(self.task_queue.maxsize or 1)
        except Exception:
            pass
        target_ms = 300 if load >= 0.8 else (200 if load >= 0.5 else 150)
        min_bytes = int(self.sample_rate * (target_ms / 1000.0)) * 2

        with self.session_lock:
            buf = self.sessions[session_id].setdefault("buffer", bytearray())
            buf.extend(audio_data)
            if len(buf) < min_bytes:
                return
            chunk = bytes(buf)
            self.sessions[session_id]["buffer"] = bytearray()
        try:
            self.task_queue.put_nowait((session_id, chunk))
        except Full:
            try:
                _ = self.task_queue.get_nowait()
                self.task_queue.task_done()
                self.task_queue.put_nowait((session_id, chunk))
                print("⚠️ 背压: 队列已满，丢弃最旧任务以保留最新音频块")
            except Empty:
                pass

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
            recognizer = self.session_recognizers.get(session_id)
            session = self.sessions.get(session_id)
        if recognizer and session:
            try:
                final = json.loads(recognizer.FinalResult())
                text = final.get("text", "").strip()
                if text:
                    with self.session_lock:
                        session["results"].append(text)
                        session["last_result"] = text
            except Exception:
                pass
        with self.session_lock:
            if session_id in self.session_recognizers:
                del self.session_recognizers[session_id]
            if session_id in self.session_process_locks:
                del self.session_process_locks[session_id]
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"关闭会话: {session_id}")


# 全局服务器实例
asr_server: Optional[VoskASRServer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_server
    hotwords = [
        "是", "否", "有", "没有", "不知道",
        "头痛", "头晕", "恶心", "呕吐", "心慌",
        "麻醉", "手术", "病史", "过敏", "药物",
        "丙泊酚", "七氟烷", "罗库溴铵", "舒芬太尼",
        "全麻", "局麻", "椎管内麻醉", "神经阻滞"
    ]
    model_path = "model/vosk-model-small-cn-0.22"
    asr_server = VoskASRServer(
        model_path=model_path,
        sample_rate=16000.0,
        max_workers=4,
        hotwords=hotwords
    )
    try:
        yield
    finally:
        if asr_server:
            for _ in range(asr_server.max_workers):
                asr_server.task_queue.put(None)
            asr_server.thread_pool.shutdown(wait=True)
            try:
                if asr_server.playback_stream:
                    asr_server.playback_stream.close()
            except Exception:
                pass
            try:
                if asr_server.audio_player:
                    asr_server.audio_player.terminate()
            except Exception:
                pass

# 创建FastAPI应用
app = FastAPI(title="Vosk语音识别服务", lifespan=lifespan)


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

        # 音频统计
        audio_chunks = 0
        total_audio_bytes = 0

        while True:
            # 接收音频数据
            data = await websocket.receive_bytes()
            audio_chunks += 1
            total_audio_bytes += len(data)

            print(f"📥 收到音频数据块 #{audio_chunks}:")
            print(f"   ├ 数据大小: {len(data)} 字节")
            print(f"   ├ 采样点数: {len(data) // 2} (16位PCM)")
            print(f"   ├ 音频时长: {len(data) / 2 / 16000 * 1000:.1f}ms (16kHz)")
            print(f"   ├ 前10字节: {bytes(data[:10]).hex()}")

            # # 检查音频内容是否静音
            # if len(data) >= 2:
            #     audio_data = np.frombuffer(data[:100], dtype=np.int16)  # 检查前50个样本
            #     max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0
            #     print(f"   └ 前50样本最大振幅: {max_amplitude} ({(max_amplitude / 32767 * 100):.1f}%)")
            #
            #     if max_amplitude < 100:  # 阈值可调整
            #         print("   ⚠️ 警告: 数据可能为静音或音量过低")

            if not data or len(data) < 10:
                print("   ❌ 错误: 收到的数据过短或为空")
                continue

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




if __name__ == "__main__":
    # 启动服务
    logger.info("启动Vosk语音识别服务...")
    uvicorn.run(
        app,
        host="0.0.0.0",  # 监听所有接口
        port=8000,  # 服务端口
        log_level="info"
    )
