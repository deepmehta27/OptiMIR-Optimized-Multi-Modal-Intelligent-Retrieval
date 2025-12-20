from fastapi import FastAPI

app = FastAPI(title="OptiMIR Backend", version="0.1.0")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "OptiMIR backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
