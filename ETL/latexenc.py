"""
    Util script if you want to performa Latex conversions on an individual text file.
"""

from pylatexenc.latexencode import UnicodeToLatexEncoder
import os
import sys
import re
from dotenv import load_dotenv

encoder = UnicodeToLatexEncoder(
    replacement_latex_protection='braces-all',
    unknown_char_policy='ignore'
)

load_dotenv()
textbook_directory = os.getenv('TEXTBOOK_DIRECTORY')
data_directory = os.getenv('DATA_DIRECTORY')
chapter_directory = f'{data_directory}/Deep Learning/14.txt'

def unicode_to_latex(chapter_content):
    chapter_content_latex = encoder.unicode_to_latex(chapter_content) # Convert the unicode characters to LaTeX
    chapter_content_latex = re.sub(r'\\ensuremath\{(.+?)\}', r'\1', chapter_content_latex) # Remove all instances of \ensuremath{}

    # Manually fix incorrect LaTeX conversions
    chapter_content_latex = re.sub(r'\{\\textquoteright\}', '`', chapter_content_latex)
    chapter_content_latex = re.sub(r'\{ff\}', 'ff', chapter_content_latex)
    chapter_content_latex = re.sub(r'\{fi\}', 'fi', chapter_content_latex)
    chapter_content_latex = re.sub(r'\{ffi\}', 'ffi', chapter_content_latex) 
    chapter_content_latex = re.sub(r'\{\\textbullet\}', '', chapter_content_latex)
    return chapter_content_latex


with open(chapter_directory, 'r', encoding='utf-8') as file:
    chapter_content = file.read()

chapter_content_latex = unicode_to_latex(chapter_content)

with open(chapter_directory, 'w', encoding='utf-8') as file:
    file.write(chapter_content_latex)