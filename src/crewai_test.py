from crewai import Agent, LLM, Crew, Task
from crewai.tools import tool
from langchain_ollama import OllamaLLM


@tool("calculator")
def calculator(a: float, b: float) -> float:
    """calculator"""
    return a + b

# ollama_llm = OllamaLLM(model="qwen3:8b",
#                        num_predict=128,
#                        temperature=0.1)

llm = LLM(
    model="ollama/qwen3:8b",
    base_url="http://localhost:11434",
    max_tokens=128,
)

writer = Agent(
    role="assistant",
    goal="answer question about math",
    backstory="an assistant。",
    llm=llm,
    tools=[calculator],
    allow_delegation=False,
    verbose=True
)


if __name__ == "__main__":

    print(writer.kickoff(messages="what's the  value of 1233 + 131131313"))

    # print(llm.call("hi"))