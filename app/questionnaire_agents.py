from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
from settings import OPENAI_API_KEY

app = FastAPI()

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://inference.cloudrift.ai/v1",
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    max_tokens=1000,
    temperature=0.2
)

prompt = PromptTemplate(
    input_variables=["descriptors"],
    template="""
Given the movie plot descriptors: {descriptors}
Generate exactly 10 concise questions for a user questionnaire to understand their taste profile. Return only the questions as a valid JSON array.
"""
)

chain = prompt | llm

@app.post("/generate_questions")
async def generate_questions(request: Request):
    descriptors = "sibling rivalry, romance, comedy, rat race, mistaken identity, coming-of-age, dystopian future, haunted house, treasure hunt, underdog sports team"
    response = await chain.ainvoke({"descriptors": descriptors})
    questions = json.loads(response.content)
    return {"questions": questions}
