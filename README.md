## Installation

1. install Ollama, [reference link](https://ollama.com/download)
```
please install 0.11.11 version from Ollama github releases.

For latest version, I detect slow inference inssue.
```
2. pull qwen3:8b
```
ollama run qwen3:8b
```
3. install uv
```aiignore
pip install uv
```
4. clone repo and prepare env
```
git clone git@github.com:ReverseFelicity/mindgames-stars.git
cd mindgames-star
uv sync
```
5. test agent
```aiignore
cd src
uv run stars_agent_track2_v7.py

```