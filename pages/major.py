# import os

# import pdfplumber
# import pytesseract
# import streamlit as st
# import torch
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_ollama import ChatOllama, OllamaEmbeddings
# from PIL import Image
# from torchvision import models, transforms

# # Fix OpenMP conflict
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # Streamlit configuration
# st.set_page_config(page_title="PDF/Image Chatbot", page_icon=":scroll:")

# base_url = "https://59d4-34-125-25-52.ngrok-free.app"


# # Load pre-trained model for image feature extraction
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# model.eval()
# # Image transformation
# transform = transforms.Compose(
#     [
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )


# # Extract text from all pages of PDFs
# def extract_pdf_text(pdf_docs):
#     text = []
#     for pdf in pdf_docs:
#         with pdfplumber.open(pdf) as pdf_file:
#             text.extend(
#                 [page.extract_text() for page in pdf_file.pages if page.extract_text()]
#             )
#     return "\n".join(text) if text else "No readable text found."


# # Extract text from images using OCR
# def extract_image_text(image_files):
#     return "\n".join(
#         [pytesseract.image_to_string(Image.open(img)) for img in image_files]
#     )


# # Extract image features efficiently
# def extract_image_features(image_files):
#     features = []
#     for img in image_files:
#         image = Image.open(img).convert("RGB")
#         image = transform(image).unsqueeze(0)
#         with torch.no_grad():
#             features.append(
#                 model(image).squeeze().tolist()[:5]
#             )  # Reduce memory footprint
#     return features


# # Create conversation chain
# def get_conversational_chain():
#     prompt = PromptTemplate(
#         template="""
#        Given the extracted image features and document context, provide a detailed explanation of the image.
#        Context: {context}
#        Image Features: {image_features}
#        Question: {question}
#        Answer:
#        """,
#         input_variables=["context", "image_features", "question"],
#     )
#     return LLMChain(llm=ChatOllama(base_url=base_url, model="qwen2.5"), prompt=prompt)


# # Split text into chunks
# def chunk_text(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#     return splitter.split_text(text)


# # Create FAISS vector store
# def create_vector_store(text_chunks):
#     vector_store = FAISS.from_texts(
#         text_chunks,
#         OllamaEmbeddings(base_url=base_url, model="nomic-embed-text"),
#     )
#     vector_store.save_local("faiss_index")
#     return vector_store


# # Search FAISS store
# def search_vector_store(query, vector_store):
#     return "\n".join(
#         [r.page_content for r in vector_store.similarity_search(query, k=7)]
#     )


# # Main function
# def main():
#     st.header("📚 Visual Data Interpreter System 🤖")
#     if "vector_store" not in st.session_state:
#         st.session_state.vector_store = None
#     user_question = st.text_input("Ask a Question about the uploaded files: ✍️")
#     with st.sidebar:
#         st.image("Robot.jpg")
#         st.title("📁 Upload Files")
#         uploaded_files = st.file_uploader(
#             "Upload PDF & Images",
#             type=["pdf", "jpg", "jpeg", "png"],
#             accept_multiple_files=True,
#         )
#         if st.button("Process Files"):
#             if not uploaded_files:
#                 st.error("Please upload at least one file.")
#             else:
#                 with st.spinner("Processing..."):
#                     pdfs = [
#                         file for file in uploaded_files if file.name.endswith(".pdf")
#                     ]
#                     images = [
#                         file
#                         for file in uploaded_files
#                         if file.name.endswith((".jpg", "jpeg", "png"))
#                     ]
#                     pdf_text = extract_pdf_text(pdfs)
#                     image_text = extract_image_text(images)
#                     image_features = extract_image_features(images)
#                     combined_text = (
#                         pdf_text + "\n\n[Image Extracted Text]\n" + image_text
#                     )
#                     text_chunks = chunk_text(combined_text)
#                     st.session_state.vector_store = create_vector_store(text_chunks)
#                     st.session_state.image_features = image_features
#                     st.success("Processing complete!")
#     if user_question:
#         try:
#             if st.session_state.vector_store is None:
#                 st.session_state.vector_store = FAISS.load_local(
#                     "faiss_index",
#                     OllamaEmbeddings(base_url=base_url, model="nomic-embed-text"),
#                 )
#             context = search_vector_store(user_question, st.session_state.vector_store)
#             image_features = st.session_state.get(
#                 "image_features", "No image features extracted."
#             )
#             chain = get_conversational_chain()
#             response = chain.run(
#                 context=context, image_features=image_features, question=user_question
#             )
#             st.write("### 🤖 Visual Data Interpreter Response:")
#             st.write(response)
#         except Exception as e:
#             st.error(f"Error: {e}")
#             st.warning("Please upload and process files before asking a question.")


# if __name__ == "__main__":
#     main()


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pdfplumber
import pytesseract
import streamlit as st
import torch
from dotenv import load_dotenv
from fpdf import FPDF
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from PIL import Image
from torchvision import models, transforms

# Streamlit Page Config
st.set_page_config(page_title="📚 Multilingual Visual Data Interpreter", page_icon="🌍")

# Ollama API Setup
base_url = "https://4732-34-125-25-52.ngrok-free.app"

# Load ResNet Model for Image Processing
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Language Selection
languages = {
    "English": "English",
    "French": "French",
    "Spanish": "Spanish",
    "German": "German",
    "Chinese": "Chinese",
    "Hindi": "Hindi",
    "Arabic": "Arabic",
}
selected_language = st.sidebar.selectbox("🌍 Choose Language", list(languages.keys()))

SAVE_DIR = "downloaded_documents"

# Create directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def save_response_as_pdf(response, uploaded_files):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Save text response
    pdf.multi_cell(0, 10, response)

    # Folder to save files
    save_folder = "saved_files"
    os.makedirs(save_folder, exist_ok=True)

    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        file_path = os.path.join(save_folder, file.name)

        # Save file locally
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        # Insert only images (JPG, PNG, etc.), not PDFs
        if file_extension in [".jpg", ".jpeg", ".png"]:
            pdf.image(file_path, x=10, w=100)

    # Save final PDF
    pdf_path = os.path.join(save_folder, "output.pdf")
    pdf.output(pdf_path)

    return pdf_path


# Extract Text from PDFs
def extract_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            text.extend(
                [page.extract_text() for page in pdf_file.pages if page.extract_text()]
            )
    return "\n".join(text) if text else "No readable text found."


# Extract Text from Images (OCR)
def extract_image_text(image_files):
    return "\n".join(
        [pytesseract.image_to_string(Image.open(img)) for img in image_files]
    )


# Extract Image Features
def extract_image_features(image_files):
    features = []
    for img in image_files:
        image = Image.open(img).convert("RGB")
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features.append(
                model(image).squeeze().tolist()[:5]
            )  # Reduce memory footprint
    return features


# Create Conversational Chain with Language Support
def get_conversational_chain(target_language):
    prompt = PromptTemplate(
        template=f"""
        Given the extracted image features and document context, provide a detailed explanation of the image in {target_language}.
        Context: {{context}}
        Image Features: {{image_features}}
        Question: {{question}}
        Answer:
        """,
        input_variables=["context", "image_features", "question"],
    )
    return LLMChain(llm=ChatOllama(base_url=base_url, model="qwen2.5"), prompt=prompt)


# Split Text into Chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    return splitter.split_text(text)


# Create FAISS Vector Store
def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(
        text_chunks,
        OllamaEmbeddings(base_url=base_url, model="nomic-embed-text"),
    )
    vector_store.save_local("faiss_index")
    return vector_store


# Search FAISS Store
def search_vector_store(query, vector_store):
    return "\n".join(
        [r.page_content for r in vector_store.similarity_search(query, k=7)]
    )


# Main Streamlit UI
def main():
    st.header("📚 Multilingual Visual Data Interpreter 🤖")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    user_question = st.text_input("Ask a Question about the uploaded files: ✍️")

    with st.sidebar:
        st.image("Robot.jpg")
        st.title("📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Upload PDF & Images",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if st.button("Process Files"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    pdfs = [
                        file for file in uploaded_files if file.name.endswith(".pdf")
                    ]
                    images = [
                        file
                        for file in uploaded_files
                        if file.name.endswith((".jpg", "jpeg", "png"))
                    ]

                    pdf_text = extract_pdf_text(pdfs)
                    image_text = extract_image_text(images)
                    image_features = extract_image_features(images)

                    combined_text = (
                        pdf_text + "\n\n[Image Extracted Text]\n" + image_text
                    )
                    text_chunks = chunk_text(combined_text)

                    st.session_state.vector_store = create_vector_store(text_chunks)
                    st.session_state.image_features = image_features
                    st.success("Processing complete!")

    if user_question:
        try:
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.load_local(
                    "faiss_index",
                    OllamaEmbeddings(base_url=base_url, model="nomic-embed-text"),
                )

            context = search_vector_store(user_question, st.session_state.vector_store)
            image_features = st.session_state.get(
                "image_features", "No image features extracted."
            )

            chain = get_conversational_chain(selected_language)
            response = chain.run(
                context=context, image_features=image_features, question=user_question
            )

            st.write(f"### 🤖 Response in {selected_language}:")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Please upload and process files before asking a question.")

    if st.button("Download PDF"):
        pdf_path = save_response_as_pdf(response, uploaded_files)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed PDF",
                data=f,
                file_name="Processed_Document.pdf",
                mime="application/pdf",
            )


if __name__ == "__main__":
    main()
