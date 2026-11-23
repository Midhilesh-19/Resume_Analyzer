from pypdf import PdfReader
from docx import Document
import os

def extract_text(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":

        reader = PdfReader(file_path)

        extracted_text = ""

        for page in reader.pages:
            extracted_text += page.extract_text() or ""

        return extracted_text

    elif extension == ".docx":

        doc = Document(file_path)

        extracted_text = ""

        for paragraph in doc.paragraphs:
            extracted_text += paragraph.text + "\n"

        return extracted_text

    else:
        raise ValueError("Only PDF and DOCX files are supported")


# Provide your file path

