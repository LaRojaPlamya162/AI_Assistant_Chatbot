# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
# import os
# from dotenv import load_dotenv

# load_dotenv()


# class Model:
#     def __init__(self):
#         self.gemini_keys = self._load_gemini_keys()

#         if len(self.gemini_keys) == 0:
#             raise ValueError("No GEMINI_API_KEY_x found in .env")

#         self.key_index = 0

#         # Local fallback model
#         self.__llm_off = ChatOllama(
#             model="qwen2.5:1.5b-instruct",
#             base_url="http://localhost:11434",
#             temperature=0
#         )

#         self.__llm = None
#         self.model_name = None

#         self.select_base_model()

#     # ---------------------------------------------------
#     def _load_gemini_keys(self):
#         keys = []
#         for k, v in os.environ.items():
#             if k.startswith("GEMINI_API_KEY_"):
#                 keys.append(v)

#         keys.sort()
#         print(f"[INFO] Loaded {len(keys)} Gemini keys")
#         return keys

#     # ---------------------------------------------------
#     def _create_online_model(self):
#         key = self.gemini_keys[self.key_index]

#         return ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash-lite",
#             google_api_key=key,
#             temperature=0
#         )

#     # ---------------------------------------------------
#     def _rotate_key(self):
#         self.key_index = (self.key_index + 1) % len(self.gemini_keys)

#     # ---------------------------------------------------
#     def select_base_model(self, query="ping"):
#         """
#         Try each Gemini key. If all fail → fallback Ollama.
#         """
#         for _ in range(len(self.gemini_keys)):
#             try:
#                 llm = self._create_online_model()

#                 # lightweight health-check
#                 test = llm.invoke("hello")

#                 if test.content:
#                     self.__llm = llm
#                     self.model_name = f"Gemini-Key-{self.key_index+1}"
#                     print(f"[INFO] Using {self.model_name}")
#                     return

#             except Exception:
#                 print(f"[WARN] Key {self.key_index+1} failed → rotating")

#             self._rotate_key()

#         print("[INFO] All Gemini keys failed → Using Ollama")
#         self.__llm = self.__llm_off
#         self.model_name = "Ollama"

#     # ---------------------------------------------------
#     def get_llm(self, tools=None):
#         """
#         RETURN LLM (optionally bind tools)
#         This is the function LangGraph should call.
#         """
#         self.select_base_model()

#         if tools:
#             print(f"[INFO] Binding tools to {self.model_name}")
#             return self.__llm.bind_tools(tools)

#         return self.__llm

#     # ---------------------------------------------------
#     def safe_invoke(self, messages, tools=None):
#         """
#         Used if you want manual invoke outside LangGraph.
#         Includes auto key-rotation on failure.
#         """
#         try:
#             llm = self.get_llm(tools)
#             return llm.invoke(messages)

#         except Exception:
#             print("[WARN] Invoke failed → rotating key and retrying")
#             self._rotate_key()
#             self.select_base_model()
#             llm = self.get_llm(tools)
#             return llm.invoke(messages)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from threading import Lock

load_dotenv()


class Model:
    _instance = None
    _lock = Lock()  # thread-safe singleton

    # ---------------------------------------------------
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("[INFO] Creating Singleton Model instance...")
                    cls._instance = super(Model, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    # ---------------------------------------------------
    def __init__(self):
        # Prevent __init__ chạy lại nhiều lần
        if self._initialized:
            return

        print("[INFO] Initializing Model (ONLY ONCE)")
        self._initialized = True

        self.gemini_keys = self._load_gemini_keys()

        if len(self.gemini_keys) == 0:
            raise ValueError("No GEMINI_API_KEY_x found in .env")

        self.key_index = 0

        # Local fallback model (khởi tạo 1 lần duy nhất)
        self.__llm_off = ChatOllama(
            model="qwen2.5:1.5b-instruct",
            base_url="http://localhost:11434",
            temperature=0
        )

        self.__llm = None
        self.model_name = None

        # Chỉ select model đúng 1 lần
        self.select_base_model(first_init=True)

    # ---------------------------------------------------
    def _load_gemini_keys(self):
        keys = []
        for k, v in os.environ.items():
            if k.startswith("GEMINI_API_KEY_"):
                keys.append(v)

        keys.sort()
        print(f"[INFO] Loaded {len(keys)} Gemini keys (ONCE)")
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
        print(f"[INFO] Rotated to key {self.key_index+1}")

    # ---------------------------------------------------
    def select_base_model(self, first_init=False):
        """
        Only health-check when:
        - first init
        - or previous invoke failed
        """
        if self.__llm is not None and not first_init:
            return  # tránh check lại mỗi lần get_llm()

        for _ in range(len(self.gemini_keys)):
            try:
                llm = self._create_online_model()

                if first_init:
                    # Chỉ test đúng 1 lần lúc start app
                    test = llm.invoke("hello")
                    if not test.content:
                        raise RuntimeError("Empty response")

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
        Called MANY times but model NEVER reloads.
        """
        if tools:
            return self.__llm.bind_tools(tools)

        return self.__llm

    # ---------------------------------------------------
    def safe_invoke(self, messages, tools=None):
        """
        Only rotate key when real failure happens.
        """
        try:
            llm = self.get_llm(tools)
            return llm.invoke(messages)

        except Exception:
            print("[WARN] Invoke failed → rotating key")

            self._rotate_key()
            self.__llm = None
            self.select_base_model()

            llm = self.get_llm(tools)
            return llm.invoke(messages)