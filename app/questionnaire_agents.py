from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import json
import pandas as pd
from settings import OPENAI_API_KEY

app = FastAPI()

llm = ChatOpenAI(api_key="rift_E2U5eUWwadYltd9no5szmKhUS1awp7Q0S0UNfehaKTnryAyKtCIq6D52sJE", base_url="https://inference.cloudrift.ai/v1", model="meta-llama/Llama-4-Maverick-17B-128E-Instruct", max_tokens=1000, temperature=0.2)

questions_prompt = PromptTemplate(input_variables=["descriptors"], template="""
Given the movie plot descriptors: {descriptors}
Generate exactly 10 concise questions for a user questionnaire to understand their taste profile. Return only the questions as a valid JSON array, no explanation, no markdown.
""")

mock_response_prompt = PromptTemplate(input_variables=["questions"], template="""
Given these questions: {questions}
Generate realistic example user responses. Return only the responses as a valid JSON array, no explanation, no markdown.
""")

case_file_prompt = PromptTemplate(input_variables=["responses", "descriptors"], template="""
Given the movie plot descriptors: {descriptors} and the user responses: {responses}, summarize the client's overall mood and what type of movie they want to watch. Return only a JSON object with 'mood' and 'recommended_descriptors' fields. No explanation, no markdown.
""")

@app.post("/generate_case_file")
async def generate_case_file(request: Request):
    df = pd.read_csv('movie_subset_descriptors_cleaned.csv')
    descriptors_text = ', '.join(df['descriptors'].dropna().head(1).values)

    chain1 = questions_prompt | llm
    questions_response = await chain1.ainvoke({"descriptors": descriptors_text})
    questions = json.loads(questions_response.content.strip())

    chain2 = mock_response_prompt | llm
    mock_responses_response = await chain2.ainvoke({"questions": json.dumps(questions)})
    responses = json.loads(mock_responses_response.content.strip())

    chain3 = case_file_prompt | llm
    case_file_response = await chain3.ainvoke({"responses": json.dumps(responses), "descriptors": descriptors_text})
    case_file = json.loads(case_file_response.content.strip())

    return {"questions": questions, "responses": responses, "case_file": case_file}
