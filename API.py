import json
import os
from typing import Union
from fastapi import FastAPI
from app import research_query
from pydantic import BaseModel


class SettingsChange(BaseModel):
    key: str
    value: str


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
