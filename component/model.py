from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
load_dotenv()
class Model:
  def __init__(self):
    self.__llm_onl = ChatGoogleGenerativeAI(
      model = 'gemini-2.5-flash-lite',
      google_api_key = os.getenv('GEMINI_API_KEY'),
      temperature = 0
    )
    self.__llm_off = ChatOllama(
            model="qwen2.5:1.5b-instruct", 
            base_url="http://localhost:11434",
            temperature = 0
    )
    self.ans = ""
    self.__llm = None

  def select_base_model(self, query="Hello"):
    try:
        answer = self.__llm_onl.invoke(query)

        if not answer.content or answer.content.strip() == "":
            self.__llm = self.__llm_off
        else:
            self.__llm = self.__llm_onl

    except Exception:
        self.__llm = self.__llm_off

  def model(self):
     self.select_base_model()
     return self.__llm
  
  # mock example
  def answer(self, question: str):
    self.select_base_model()
    return self.__llm.invoke(question)
  
if __name__ == '__main__':
  model = Model()
  print(model.answer("3 + 5 * 4 is equal to (brief answer)"))