from fastapi import FastAPI
from inference.generate import generate_text

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to MiniGPT API"}

@app.post("/generate")
def generate(prompt: str):
    return {"generated_text": generate_text(prompt)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
