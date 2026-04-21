import os
import json
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI()

# IMPORTANT: Allows your Netlify site to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your Netlify URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GenAI Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ResumeRequest(BaseModel):
    master_profile: str
    job_description: str

def get_gemini_response(master_profile: str, job_description: str):
    system_instr = (
        "You are an expert ATS-optimised resume writer. Return a strictly structured "
        "JSON object. CRITICAL: Summary 2 lines max. Skills 4-5 categories max. "
        "Projects 2 max, 2 bullets each (120 chars max). No filler. No markdown."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instr,
        response_mime_type="application/json",
        temperature=0.1
    )

    user_prompt = f"""
    Candidate Profile: {master_profile}
    Job Description: {job_description}

    Return JSON structure:
    {{
      "summary": "max 2 lines",
      "skills": {{ "Category": "item1, item2" }},
      "projects": [
        {{
          "name": "project name",
          "tech": "stack",
          "bullets": ["bullet 1", "bullet 2"]
        }}
      ],
      "achievements": ["achievement 1", "achievement 2"]
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=user_prompt,
        config=config
    )
    return json.loads(response.text)

@app.post("/generate")
async def generate_endpoint(data: ResumeRequest):
    try:
        resume_data = get_gemini_response(data.master_profile, data.job_description)
        return resume_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)