import os
from openai import OpenAI
from src.server import CodeAutocompleteEnv
from src.models import CodeAction

# Mandatory Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
env = CodeAutocompleteEnv()

def run_baseline():
    task = "line-completion"
    print(f"[START] task={task} env=cra-v1 model={MODEL_NAME}")
    
    obs = env.reset()
    # 1. Generate Completion
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Complete this code: {obs.code_context}"}]
    )
    completion = response.choices[0].message.content
    
    # 2. Step Environment
    action = CodeAction(completion=completion)
    new_obs, reward, done, info = env.step(action)
    
    # 3. Required Output Format
    print(f"[STEP] step=1 action={completion} reward={reward:.2f} done={str(done).lower()} error=null")
    print(f"[END] success={str(reward > 0.5).lower()} steps=1 rewards={reward:.2f}")

if __name__ == "__main__":
    run_baseline()