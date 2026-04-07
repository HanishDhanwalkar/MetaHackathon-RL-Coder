import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from inference import get_completion

app = FastAPI()

class IDEContext(BaseModel):
    content: str

@app.post("/predict")
async def predict(data: IDEContext):
    # Triggers the RL Agent in inference.py
    return get_completion(data.content)

# Serve the static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Port 7860 is required for HF Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)