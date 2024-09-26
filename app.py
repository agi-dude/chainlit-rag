import PIL.Image
from graphrag.query.cli import run_global_search, run_local_search

from chainlit.input_widget import Select, TextInput
import chainlit as cl

# from vllm import LLM
# from vllm.sampling_params import SamplingParams

import os
import json
from llm import Client
import chromadb
#from lmdeploy import pipeline, TurbomindEngineConfig
#from lmdeploy.vl import load_image
#import torch
#from PIL import Image
#from transformers import AutoModel, AutoTokenizer

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


@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("chat_settings")
    CLIENT = Client(settings['Provider'], settings["API"], settings["Chat Model"], settings["OpenAI Host"])

    if len(message.elements) < 1:
        do_research = CLIENT.chat(messages=[
            {'role': 'user', 'content': prompts.research_selector.format(message.content)}
        ])

        if 'RESEARCH' in do_research:
            print('    - Starting research...')
            research = research_query(message.content, settings['ChromaDB Root'],
                                      settings['ChromaDB Collection'], CLIENT, settings['GraphRAG Root'],
                                      settings['GraphRAG Input']
            )

            print('    - Sending...')

            await cl.Message(
                content=research
            ).send()

        else:
            print('    - Responding (No research)')
            response = CLIENT.chat(messages=cl.chat_context.to_openai())

            await cl.Message(
                content=response,
            ).send()

    else:
        if settings['Vision Model'] == 'InternVL2':
            print('    - Starting InternVL2')
            image = load_image(message.elements[0].path)
            backend_config = TurbomindEngineConfig(model_format='awq')
            pipe = pipeline('OpenGVLab/InternVL2-Llama3-76B-AWQ', backend_config=backend_config, log_level='INFO')
            response = pipe((message.content, image))
            await cl.Message(content=response.text).send()

        elif settings['Vision Model'] == 'Mini-CPM':
            print('    - Starting Mini-CPM')
            model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.int8)
            model = model.to(device='cuda')

            tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
            model.eval()

            prompt = []
            for file in message.elements:
                prompt.append(Image.open(file.path).convert('RGB'))

            prompt.append(message.content)
            msgs = [{'role': 'user', 'content': prompt}]

            res = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,  # if sampling=False, beam_search will be used by default
                temperature=0.7,
                # system_prompt='' # pass system_prompt if needed
            )
            await cl.Message(content=res).send()

        elif settings['Vision Model'] == 'Pixtral':
            model_name = "mistralai/Pixtral-12B-2409"

            sampling_params = SamplingParams(max_tokens=4192)

            llm = LLM(model=model_name, tokenizer_mode="mistral")

            prompt = message.content
            image = Image.open(message.elements[0].path)

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image}}]
                },
            ]

            outputs = llm.chat(messages, sampling_params=sampling_params)

            print(outputs[0].outputs[0].text)


@cl.on_chat_start
async def start():
    print("    - \u001b[35mNew session started!\u001b[0m")

    if os.path.isfile('settings.json'):
        with open('settings.json') as settings_file:
            default_settings = json.load(settings_file)

    else:
        default_settings = {"Vision Model": "InternVL2", "Provider": "openai", "Chat Model": "dolphin-mistral", "OpenAI Host": "http://localhost:11434", "API": "sk-0000000000", "GraphRAG Root": ".", "GraphRAG Input": "./output/artifacts", "ChromaDB Root": "./.chroma", "ChromaDB Collection": "collection1"}

    settings = await cl.ChatSettings(
        [
            Select(
                id="Vision Model",
                label="Vision Model",
                values=["InternVL2", "Mini-CPM", "Pixtral"],
                initial_index=["InternVL2", "Mini-CPM", "Pixtral"].index(default_settings['Vision Model']),
            ),
            TextInput(
                id="Provider",
                label='LLM Provider (openai, anthropic, azure, ollama)',
                initial=default_settings["Provider"]
            ),
            TextInput(
                id="Chat Model",
                label="Chat Model",
                initial=default_settings["Chat Model"]
            ),
            TextInput(
                id="OpenAI Host",
                label='API Base',
                initial=default_settings["OpenAI Host"]
            ),
            TextInput(
                id="API",
                label='API Key',
                initial=default_settings["API"]
            ),
            TextInput(
                id="GraphRAG Root",
                label="GraphRAG Root Directory",
                initial=default_settings["GraphRAG Root"]
            ),
            TextInput(
                id="GraphRAG Input",
                label="GraphRAG Database",
                initial=default_settings["GraphRAG Input"]
            ),
            TextInput(
                id="ChromaDB Root",
                label="ChromaDB Root Directory",
                initial=default_settings["ChromaDB Root"]
            ),
            TextInput(
                id="ChromaDB Collection",
                label="ChromaDB collection name",
                initial=default_settings["ChromaDB Collection"]
            )
        ]
    ).send()


print('''
░▒▓███████▓▒░ ░▒▓██████▓▒░ ░▒▓██████▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒▒▓███▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░


Started!\n\n''')


@cl.on_settings_update
async def setup_agent(settings):
    print('    - Settings updated!')
    with open('settings.json', 'w') as settings_file:
        settings_file.write(json.dumps(settings))


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
