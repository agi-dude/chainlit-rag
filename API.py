from fastapi import FastAPI
from pydantic import BaseModel

from graphrag.query.cli import run_global_search, run_local_search

import os
import json
from llm import Client
import chromadb

import prompts


def chroma(query, CHROMA_PATH, COLLECTION_NAME, CLIENT):
    print('    - Querying ChromaDB')
    persistentClient = chromadb.PersistentClient(CHROMA_PATH, settings=chromadb.Settings(anonymized_telemetry=False))
    collection = persistentClient.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_texts=[query],  # Chroma will embed this for you
        n_results=5  # how many results to return
    )

    client = CLIENT
    response = client.chat(messages=[
        {'role': 'user', 'content': 'Summarize these documents, and extract all the important information: {}'.format(str(results['documents']))}
    ])

    return response


def graph_rag(query, GRAPHRAG_INDEX_LOCATION, GRAPHRAG_ROOT):
    print('    - Querying GraphRAG')
    global_results = run_global_search(GRAPHRAG_INDEX_LOCATION, GRAPHRAG_ROOT,
                                       query=query, response_type="One page report", community_level=2)

    # local_results = run_local_search(GRAPHRAG_INDEX_LOCATION, GRAPHRAG_ROOT,
    #                                 query=query, response_type="One page report", community_level=2)

    return "", global_results


def summerize(query, agent1, agent2, agent3, CLIENT):
    client = CLIENT
    response = client.chat(messages=[
        {'role': 'user', 'content': prompts.summerize_prompt.format(
            query=query,
            output1=agent1,
            output2=agent2,
            output3=agent3
        )}
    ])

    return response


def research_query(query, CHROMA_PATH, COLLECTION_NAME, CLIENT, GRAPHRAG_INDEX_LOCATION, GRAPHRAG_ROOT):
    local_results, global_results = graph_rag(query, GRAPHRAG_INDEX_LOCATION, GRAPHRAG_ROOT)
    chroma_results = chroma(query, CHROMA_PATH, COLLECTION_NAME, CLIENT)

    print('    - Generating final response...')
    return summerize(query, local_results, global_results, chroma_results, CLIENT)


class SettingsChange(BaseModel):
    key: str
    value: str


class Query(BaseModel):
    prompt: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "running"}


@app.get('/settings')
def get_settings():
    if os.path.isfile('settings.json'):
        with open('settings.json') as settings_file:
            settings = json.load(settings_file)

        return settings

    else:
        return "File not found!"


@app.put('/settings')
def set_settings(item: SettingsChange):
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)

    settings[item.key] = item.value

    with open('settings.json', 'w') as settings_file:
        json.dump(settings, settings_file)

    return settings


@app.post('/research')
def research(query: Query):
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)

    CLIENT = Client(settings['Provider'], settings["API"], settings["Chat Model"], settings["OpenAI Host"])
    response = research_query(query.prompt, settings['ChromaDB Root'],
                              settings['ChromaDB Collection'], CLIENT, settings['GraphRAG Root'],
                              settings['GraphRAG Input'])

    return response
