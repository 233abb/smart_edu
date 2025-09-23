import uvicorn
from fastapi import FastAPI
from starlette.responses import RedirectResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import json

from configuration import config
from server import ChatService  # 导入新的服务类
from pydantic import BaseModel


class Question(BaseModel):
    message: str


# 应用上下文，用于存放全局服务实例
app_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 应用启动: 初始化ChatService...")
    app_context["service"] = ChatService()
    print(">>> ChatService初始化成功。")
    yield
    print(">>> 应用关闭...")
    app_context.clear()


app = FastAPI(lifespan=lifespan)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=config.WEB_STATIC_DIR), name="static")


@app.get("/")
def read_root():
    return RedirectResponse("/static/index.html")


# ==========================================================
# =================== 全新流式API端点 ======================
# ==========================================================
@app.post("/api/chat")
async def stream_chat(question: Question):
    service: ChatService = app_context["service"]

    async def event_generator():
        # 调用服务中的流式处理方法
        async for event in service.stream_chat_pipeline(question.message):
            # 将每个事件字典转换为JSON字符串，并添加分隔符
            yield json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


if __name__ == '__main__':
    uvicorn.run('web.app:app', host="0.0.0.0", port=8888)