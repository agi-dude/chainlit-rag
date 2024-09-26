import json
import os
from typing import Union
from fastapi import FastAPI
from app import research_query

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


@app.post('/settings')
def set_settings(key: Union[str, None] = None, value: Union[str, None] = None):
    with open('settings.json') as settings_file:
        settings = json.load(settings_file)

    settings[key] = value

    with open('settings.json', 'w') as settings_file:
        json.dump(settings, settings_file)
