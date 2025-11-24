# Document-aware Q/A bot

## Setup

1. **Clone the repository**
- ``git clone https://github.com/vpakarinen2/qa-doc-bot-langchain.git``
- ``cd qa-doc-bot-langchain``

2. **Create virtual environment (Windows, Python 3.11)**
- ``python3.11 -m venv .venv``
- ``.venv\Scripts\Activate.ps1`` (source .venv/bin/activate on Linux)

3. **Install dependencies**

- ``python -m pip install --upgrade pip``
- ``pip install -r requirements.txt``

## Usage
1. **Run the Python script**

- ``python qa_doc_bot.py``

## CLI arguments

### Options

- `-q`, `--question`  
  - Description: The question to ask the model.

- `-m`, `--model-name`  
  - Description: Hugging Face model id (e.g. Qwen/Qwen3-4B-Thinking-2507).
 
- `-d`, `--doc`
  - Description: Path to the document (.pdf or .txt).

- `--trust-remote-code`
  - Description: Allow execution of custom remote code.

## Author

Ville Pakarinen (@vpakarinen2)
