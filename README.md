---
license: apache-2.0
title: Search engine
sdk: streamlit
emoji: 🔥
colorFrom: purple
colorTo: pink
short_description: Search engine with llm
---
# LangChain Search Chat — Streamlit app

A small Streamlit chat application that uses LangChain agents and community tools to search the web (DuckDuckGo), fetch summaries from Wikipedia, and retrieve research papers from arXiv — powered by Groq LLM (via `langchain_groq`).

This repository contains a minimal example demonstrating how to wire up:
- A Groq-based chat model (ChatGroq)
- LangChain agent orchestration (ZERO_SHOT_REACT_DESCRIPTION)
- Community API wrappers and tools for Wikipedia and arXiv
- DuckDuckGo web search tool
- A Streamlit frontend with an interactive chat UI and streaming callbacks

---

## Features

- Ask general questions and let the agent decide whether to:
  - Search the web (DuckDuckGo)
  - Fetch a concise summary from Wikipedia
  - Retrieve an arXiv paper summary
- Streamed assistant responses in the Streamlit chat UI
- Enter Groq API key in-app (or use environment variable)

---

## Quick demo

1. Start the app:
   - streamlit run app.py
2. Enter your Groq API key in the sidebar or set it via environment variable.
3. Type a question in the chat input (e.g. "What is reinforcement learning?") and see the agent search and answer.

---

## Prerequisites

- Python 3.10+ (or compatible)
- A Groq API key (for the LLM)
- Internet access (for DuckDuckGo, Wikipedia, arXiv APIs)

---

## Installation

1. Clone the repository:
   - git clone https://github.com//Hanaakp/Search-engine-withLangchain-tools-agents.git
   - cd Search-engine-withLangchain-tools-agents

2. Create and activate a virtual environment (recommended):
   - python -m venv .venv
   - source .venv/bin/activate   # macOS / Linux
   - .venv\Scripts\activate      # Windows

3. Install dependencies:
   - pip install streamlit python-dotenv
   - pip install langchain_groq langchain_community langchain_classic
   - (Or use a requirements file if provided: pip install -r requirements.txt)

Note: package names and availability may change; pin versions as needed.

---

## Configuration

The app loads environment variables using python-dotenv. You can either:

- Create a `.env` file in the project root:
  - GROQ_API_KEY=your_groq_api_key_here

- Or enter the API key at runtime in the Streamlit sidebar when the app launches.

The example code also accepts the key via Streamlit sidebar input (password field).

---

## Running the app

Save the example script as `app.py` (or ensure the repository file exists) and run:

- streamlit run app.py

Open the local URL printed by Streamlit (typically http://localhost:8501).

---

## How it works (overview of the code)

- The app initializes three tools:
  - Wikipedia: `WikipediaAPIWrapper` + `WikipediaQueryRun` (returns short summaries)
  - arXiv: `ArxivAPIWrapper` + `ArxivQueryRun` (returns paper metadata/abstract)
  - DuckDuckGo: `DuckDuckGoSearchRun` (general web search)
- The Groq LLM is instantiated via `ChatGroq`, with the API key provided in the sidebar (or from env).
- A LangChain agent (ZERO_SHOT_REACT_DESCRIPTION) is initialized with the tools and LLM.
- User messages are stored in `st.session_state["messages"]`. On chat input:
  - The user's message is appended to the session messages.
  - The agent is run (agent.run(...)) with a Streamlit callback handler to stream responses into the chat UI.
- The agent decides which tool(s) to call based on the user's prompt and the tools' descriptions, composes an answer, and returns it to the chat.

---

## Important code notes

- The example uses `StreamlitCallbackHandler` so the assistant can show intermediate reasoning/thoughts in the UI (controlled by `expand_new_thoughts`).
- `top_k_results` and `doc_content_chars_max` are set low for concise responses; adjust as you like.
- Make sure your LLM model name and API key are compatible with your Groq plan / SDK.

---

## Dependencies (suggested)

- streamlit
- python-dotenv
- langchain_groq
- langchain_community
- langchain_classic

Pin versions in a `requirements.txt` for reproducible environments.

---

## Troubleshooting

- "No API key" or authentication errors:
  - Ensure the Groq API key is correct and either set in `.env` as `GROQ_API_KEY` or entered in the sidebar.
- Long responses are cut or slow:
  - Increase or tune the model choice or the streaming callback behavior.
- Tools not returning results:
  - Confirm network connectivity and that the community tool wrappers haven't changed (APIs sometimes change).

---
