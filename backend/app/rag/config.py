import os
from dotenv import load_dotenv

load_dotenv()
CLAUDE_VISION_MODEL = "claude-sonnet-4-5"
GPT_VISION_MODEL  = "gpt-4o"
FINANCE_CLASSIFIER_MODEL = "gpt-4.1-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")
