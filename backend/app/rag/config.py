import os
from dotenv import load_dotenv

load_dotenv()
# Model configurations
CLAUDE_VISION_MODEL = "claude-sonnet-4-5"
GPT_VISION_MODEL  = "gpt-4.1"
FINANCE_CLASSIFIER_MODEL = "gpt-4.1"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# LangSmith Configuration
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT") 
# ✅ IMPORTANT: Set as environment variable so langsmith SDK picks it up
if LANGCHAIN_ENDPOINT:
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")

# LangSmith is optional - warn if not configured
if not LANGCHAIN_API_KEY:
    print("⚠️  LANGCHAIN_API_KEY not set - LangSmith tracing disabled")
else:
    print(f"🔧 LangSmith endpoint: {LANGCHAIN_ENDPOINT}")