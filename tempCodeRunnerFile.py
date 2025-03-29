from langchain_google_genai import GoogleGenerativeAI

llm = GoogleGenerativeAI()
print(llm.list_models())  # This will print available models
