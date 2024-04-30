import os
import json
import PyPDF2
from dotenv import load_dotenv

load_dotenv()
TEXTBOOK_DIRECTORY = os.getenv('TEXTBOOK_DIRECTORY')
DATA_DIRECTORY = os.getenv('DATA_DIRECTORY')
textbook_config = 'textbook_config_full_chapters.json'

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

with open(textbook_config, 'r') as file:
    textbooks = json.load(file)

for book_name, book_info in textbooks.items():
    chapters = book_info['chapters']
    pdf_path = os.path.join(TEXTBOOK_DIRECTORY, f"{book_name}.pdf")

    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for chapter in chapters:
            for chapter_number, pages in chapter.items():
                output_dir = os.path.join(DATA_DIRECTORY, "PDF", book_name)
                ensure_dir(output_dir)

                output_file_path = os.path.join(output_dir, f"{chapter_number}.pdf")

                pdf_writer = PyPDF2.PdfWriter()

                for page in range(pages[0] - 1, pages[1]):
                    pdf_writer.add_page(pdf_reader.pages[page])

                with open(output_file_path, 'wb') as output_pdf:
                    pdf_writer.write(output_pdf)

print("PDF chapters have been successfully created.")
