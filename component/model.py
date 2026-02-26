# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Model:
#     def __init__(self):
#         # ==== Load ALL Gemini Keys Dynamically ====
#         self.gemini_keys = self._load_gemini_keys()

#         if len(self.gemini_keys) == 0:
#             raise ValueError("No GEMINI_API_KEY_x found in .env")

#         self.key_index = 0  # dùng cho round-robin

#         # ==== Local model (fallback) ====
#         self.__llm_off = ChatOllama(
#             model="qwen2.5:1.5b-instruct",
#             base_url="http://localhost:11434",
#             temperature=0
#         )

#         self.__llm = None
#         self.model_name = None

#         # init model
#         self.select_base_model()

#     # ---------------------------------------------------
#     # Load keys: GEMINI_API_KEY_1 ... GEMINI_API_KEY_N
#     # ---------------------------------------------------
#     def _load_gemini_keys(self):
#         keys = []
#         for k, v in os.environ.items():
#             if k.startswith("GEMINI_API_KEY_"):
#                 keys.append(v)

#         keys.sort()  # optional: deterministic order
#         print(f"[INFO] Loaded {len(keys)} Gemini keys")
#         return keys

#     # ---------------------------------------------------
#     # Create Gemini model from current key
#     # ---------------------------------------------------
#     def _create_online_model(self):
#         key = self.gemini_keys[self.key_index]

#         return ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash-lite",
#             google_api_key=key,
#             temperature=0
#         )

#     # ---------------------------------------------------
#     # Rotate key (round-robin)
#     # ---------------------------------------------------
#     def _rotate_key(self):
#         self.key_index = (self.key_index + 1) % len(self.gemini_keys)

#     # ---------------------------------------------------
#     # Select model (Gemini if possible, else Ollama)
#     # ---------------------------------------------------
#     def select_base_model(self, query="ping"):
#         for _ in range(len(self.gemini_keys)):
#             try:
#                 llm = self._create_online_model()
#                 answer = llm.invoke(query)

#                 if answer.content and answer.content.strip():
#                     self.__llm = llm
#                     self.model_name = f"Gemini-Key-{self.key_index+1}"
#                     return

#             except Exception as e:
#                 print(f"[WARN] Key {self.key_index+1} failed → rotating")

#             self._rotate_key()

#         # If ALL keys fail → fallback local
#         print("[INFO] All Gemini keys failed → Using Ollama")
#         self.__llm = self.__llm_off
#         self.model_name = "Ollama"

#     # ---------------------------------------------------
#     def model(self):
#         return self.__llm

#     # ---------------------------------------------------
#     def answer(self, question: str):
#         try:
#             self.select_base_model(question)
#             ans = self.__llm.invoke(question)

#             print(f"Answer: {ans.content}")
#             print(f"Model name: {self.model_name}")

#         except Exception:
#             print("[ERROR] Online failed → fallback Ollama")
#             self.__llm = self.__llm_off
#             ans = self.__llm.invoke(question)

#             print(f"Answer: {ans.content}")
#             print("Model name: Ollama (forced fallback)")


# if __name__ == '__main__':
#     model = Model()
#     model.answer("What company does Jeff Bezos own?")
    
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()


class Model:
    def __init__(self):
        self.gemini_keys = self._load_gemini_keys()

        if len(self.gemini_keys) == 0:
            raise ValueError("No GEMINI_API_KEY_x found in .env")

        self.key_index = 0

        # Local fallback model
        self.__llm_off = ChatOllama(
            model="qwen2.5:1.5b-instruct",
            base_url="http://localhost:11434",
            temperature=0
        )

        self.__llm = None
        self.model_name = None

        self.select_base_model()

    # ---------------------------------------------------
    def _load_gemini_keys(self):
        keys = []
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY_"):
                keys.append(v)

        keys.sort()
        print(f"[INFO] Loaded {len(keys)} Gemini keys")
        return keys

    # ---------------------------------------------------
    def _create_online_model(self):
        key = self.gemini_keys[self.key_index]

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=key,
            temperature=0
        )

    # ---------------------------------------------------
    def _rotate_key(self):
        self.key_index = (self.key_index + 1) % len(self.gemini_keys)

    # ---------------------------------------------------
    def select_base_model(self, query="ping"):
        """
        Try each Gemini key. If all fail → fallback Ollama.
        """
        for _ in range(len(self.gemini_keys)):
            try:
                llm = self._create_online_model()

                # lightweight health-check
                test = llm.invoke("hello")

                if test.content:
                    self.__llm = llm
                    self.model_name = f"Gemini-Key-{self.key_index+1}"
                    print(f"[INFO] Using {self.model_name}")
                    return

            except Exception:
                print(f"[WARN] Key {self.key_index+1} failed → rotating")

            self._rotate_key()

        print("[INFO] All Gemini keys failed → Using Ollama")
        self.__llm = self.__llm_off
        self.model_name = "Ollama"

    # ---------------------------------------------------
    def get_llm(self, tools=None):
        """
        RETURN LLM (optionally bind tools)
        This is the function LangGraph should call.
        """
        self.select_base_model()

        if tools:
            print(f"[INFO] Binding tools to {self.model_name}")
            return self.__llm.bind_tools(tools)

        return self.__llm

    # ---------------------------------------------------
    def safe_invoke(self, messages, tools=None):
        """
        Used if you want manual invoke outside LangGraph.
        Includes auto key-rotation on failure.
        """
        try:
            llm = self.get_llm(tools)
            return llm.invoke(messages)

        except Exception:
            print("[WARN] Invoke failed → rotating key and retrying")
            self._rotate_key()
            self.select_base_model()
            llm = self.get_llm(tools)
            return llm.invoke(messages)