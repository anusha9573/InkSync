import os
import pdfplumber
import pytesseract
import streamlit
import torch
import dotenv
import fpdf
import langchain
import langchain_community
import langchain_ollama
import PIL
import torchvision

# Print versions
print("OS Version:", os.name)
print("pdfplumber Version:", pdfplumber.__version__)
print("pytesseract Version:", pytesseract.get_tesseract_version())
print("Streamlit Version:", streamlit.__version__)
print("Torch Version:", torch.__version__)
print("Dotenv Version:", dotenv.__version__)
print("FPDF Version:", fpdf.FPDF_VERSION)
print("LangChain Version:", langchain.__version__)
print("LangChain Community Version:", langchain_community.__version__)
print("LangChain Ollama Version:", langchain_ollama.__version__)
print("PIL Version:", PIL.__version__)
print("Torchvision Version:", torchvision.__version__)
