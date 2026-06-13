import uuid, requests, time, os
from dotenv import load_dotenv
from ds_api.adaptor import QwenAdapter

load_dotenv()

QWEN_TOKEN = os.environ.get("QWEN_TOKEN", "")
QWEN_COOKIES = os.environ.get("QWEN_COOKIES", "")
QWEN_BASE_URL = "https://chat.qwen.ai"
DEFAULT_QWEN_MODEL = "qwen3.7-plus"

adaptor = QwenAdapter(token=QWEN_TOKEN, cookies=QWEN_COOKIES)
res = adaptor.chat("4f782e51-cc30-4717-a40c-6f8ebca89818", prompt="Explain Beta in trading", model_type="qwen3.7-plus")
print(res)
