from langchain_core.prompts import ChatPromptTemplate


class PromptRegistry:
    """Quản lý toàn bộ prompt theo từng task"""

    _prompts = {
        "qa": ChatPromptTemplate.from_template("""
You are a technical assistant answering questions from knowledge context.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
If unknown, say "I don't know."
"""),

        "code_explain": ChatPromptTemplate.from_template("""
You are a senior software engineer.

Code:
{context}

Explain:
- What the code does
- Key logic
- Important patterns
- Possible improvements
"""),

        "paper_review": ChatPromptTemplate.from_template("""
You are a research assistant.

Paper Context:
{context}

Question:
{question}

Give a deep technical explanation of the method, contribution, and intuition.
"""),

        "debug": ChatPromptTemplate.from_template("""
You are a debugging expert.

System Knowledge:
{context}

Question:
{question}

Diagnose the problem and suggest fixes.
"""),
    }

    @classmethod
    def get(cls, task: str):
        if task not in cls._prompts:
            raise ValueError(f"Prompt task '{task}' not registered")
        return cls._prompts[task]