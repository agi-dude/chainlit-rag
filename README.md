<div align="center">
    <h1>Chainlit RAG Application</h1>
</div>

---
<details open>
<summary><b>📕 Table of Contents</b></summary>
  
- 💡 [What is this repo?](#-what-is-this-repo)


- 📷 [Screenshots](#-screenshots)


- 🚀 [Get Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Initial Setup](#initial-setup)
  - [Setup Database](#setup-database)


- 🚗 [Usage](#-usage)
  - [Adding more files](#add-more-files)


- 🙌 [Contributing](#-contributing)

</details>

----
## 💡 What is this repo?
This is a hybrid RAG application designed to enhance text generation by integrating powerful retrieval mechanisms. By combining Microsoft's 
GraphRAG and traditional RAG techniques, we acheive state-of-the-art results. We also provide a webUI based on ChainLit for seamless integration, 
extensibility, and ease of deployment.

## 📷 Screenshots
![screenshot2.png](static%2Fscreenshot2.png)![screenshotSettings.png](static%2FscreenshotSettings.png)
![screenshotLight.png](static%2FscreenshotLight.png)

## 🚀 Getting Started
### Prerequisites

- Docker >= 24.0.0 & Docker Compose >= v2.26.1
  > If you have not installed Docker on your local machine (Windows, Mac, or Linux), see [Install Docker Engine](https://docs.docker.com/engine/install/).
- Python >= 3.9.0
- Conda
  > If you do not have Conda installed, then follow the steps [here](https://docs.anaconda.com/miniconda/miniconda-install/), to install miniconda 
  > on your machine

### Initial Setup
1. Initialize a new conda enviroment
  ```bash
$ conda create python==3.11 -n chainlit_rag
$ conda activate chainlit_rag
```

2. Clone this repository, and install dependencies
```bash
$ git clone https://github.com/agi-dude/chainlit-rag
$ cd chainlit-rag
$ pip install -r requirements.txt
```

3. Configure GraphRAG. Open the `settings.yaml` file located in the main directory, and then change these lines:
```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY} # Change to your openai api key if you are using openAI models
  type: openai_chat # or azure_openai_chat
  model: dolphin-mistral:latest # Change to your model
  ...
  api_base: http://localhost:11434/v1 # By default, it's configured to use Ollama. You can change it to `https://api.openai.com/v1` if you want to use openai models
  ...
embeddings:
  ...
  llm:
    api_key: ${GRAPHRAG_API_KEY} # Change to your openai api key if you are using openAI models
    type: openai_embedding # or azure_openai_embedding
    model: mxbai-embed-large:latest # Change to your model
    api_base: http://192.168.10.102:11434/v1 # By default, it's configured to use Ollama. You can change it to `https://api.openai.com/v1` if you want to use openai models
```

### Setup database
1. Create the path  `input/pdfs` in the root folder of this project and place your pdf files into it.
2. Run the `loader.py`
```bash
$ python loader.py -c -n # This might take some time (~1 hour or more for large datasets), because it has to index everything, so be patient!
```

## 🚗 Usage
1. Start the server by running `app.py`
```bash
$ python app.py
```

2. Open https://localhost:8000 in your browser.
3. Press the settings button to change your settings.
  > ![Settings.png](static%2FSettings.png)

### Add more files
1. To add more documents to the database, first add them into `input/pdf`. After that, run `loader.py` without `-n`:
```bash
$ python loader.py -c
```

## 🙌 Contributing
Feel free to fork the project, make some updates, and submit pull requests. Any contributions are welcomed!
