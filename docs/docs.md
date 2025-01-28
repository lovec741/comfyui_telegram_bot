# User Documentation
The user documentation is in the `/docs/user/` folder as an HTML website. 

# Developer Documentation
## Overview
This project doesn't include any complex algorithms or datastructures.

It consists of 4 main parts:
1. A Telegram bot that handles client requests asynchronously
    - uses the `python-telegram-bot` library
    - uses the `deque` datastructure to manage the generation queue
2. A ComfyUI client that communicates over HTTP and Websockets.
    - uses the `websockets` and `requests` libraries
3. Prompt enhancement using external services
    - detects any services added in the `/services/` folder
    - one implemented service `anthropic` that uses the Anthropic AI Claude LLM (uses the `anthropic` library)
4. Configuration using YAML config files
    - uses the `PyYAML` library

Other smaller components:
- Placeholder images created using the `pillow` library
- Parsing of the generation parameters using regex (using the builtin `re` library)
- Logging using the builtin `logging` library

## Project structure

**__main__.py**
- launches the project

**__init__.py**
- starts a logger and loads the config

**telegram_bot.py**
- connects all the other components together
- defines the Telegram bot
    - it is an async application that handles:
        - generation queue management (uses deque datastructure)
            - adding generations
            - cancelling generations
            - viewing queue status
        - help and usage information

**image_gen.py**
- parsing of generation requests
- preparation of ComfyUI workflow files
- communication with ComfyUI server
    - HTTP
        - request generation
        - cancel generation
        - get final image
    - Websockets
        - get generation progress and preview
- sends information back to user using callbacks

**config.py**
- defines structure of the config files
- defines how the config information is to be interpreted as python objects

**prompt_enhance.py**
- handles the loading of prompt enhancement services
- defines the prompt enhancement service base class

**/services/**
- folder for implemented prompt enhancement services
    - they are identified by the filename

**/services/anthropic.py**
- implemented prompt enhancement service `anthropic` for the Anthropic AI Claude LLM

## Generation process
- User requests a generation by sending a message on Telegram
- The Telegram bot (in **telegram_bot.py**) receives it, the generation parameters are parsed (in **image_gen.py**) and it is added to the queue
- When the generation reaches the front of the queue 3 things happen:
    1. the prompt is sent to be enhanced if the user requested it (in **prompt_enhance.py**)
    2. the placeholder image is created (in **image_gen.py**) and then sent to the user
    3. the generation is started (in **image_gen.py**)
- The communication with ComfyUI then looks like this (in **image_gen.py**):
    1. The workflow file is created from the parameters
    2. The generation is requested through an HTTP GET request
    3. A websocket connection is established and generation progress sent back to the user using callbacks
        - When generation previews are received they are sent back to the user
        - If a `cancel` flag is set the generation is cancelled over HTTP and the websocket connection is closed
    4. When the generation is finished the websocket connection is closed and the final image is retrieved over HTTP and sent back to the user using a callback