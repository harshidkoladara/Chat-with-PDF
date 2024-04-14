# Project Setup Guide

## Step 1: Install Ollama

1. Visit [Ollama's website](https://ollama.com/download).
2. Download the appropriate Ollama version for your system and follow the installation instructions.
3. Open your command prompt or terminal.
4. Run the following command to pull 'llama2':
    ```
    ollama pull llama2
    ```
5. Start the Ollama server by running:
    ```
    ollama serve
    ```

## Step 2: Setting Up the Project

1. Create a virtual environment:
    ```
    python -m venv env
    ```
2. Activate the virtual environment:
    - **Windows**: 
        ```
        .\env\Scripts\activate
        ```
    - **Linux/macOS**:
        ```
        source env/bin/activate
        ```
3. Install dependencies from the requirements.txt file:
    ```
    pip install -r requirements.txt
    ```
4. Run the project using Uvicorn:
    ```
    uvicorn app:app --reload --workers 4 --host 0.0.0.0 --port 8000
    ```

Your project should now be up and running!
