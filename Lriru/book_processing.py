# book_processing.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from multiprocessing import Pool
import os
import shutil
import ebooklib.epub
from bs4 import BeautifulSoup
import re
from langchain_community.document_loaders import UnstructuredPDFLoader


def parallel_processing(func, data):
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(func, data)
        
    if all(results):
        print("All tasks were completed successfully!")
    else:
        print("Some tasks were not completed successfully.")
        
        
def get_book_href():
    driver = webdriver.Chrome()
    driver.get('https://vetbooks.ir/category/animal-based-categories/small-animal/')

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        if not driver.find_elements(By.CLASS_NAME, "load-more-posts-button-w"):
            break
        
    href_elements = driver.find_elements(By.CSS_SELECTOR, '.archive-item-media a')

    href_list = [element.get_attribute('href') for element in href_elements if element.get_attribute('href')]
    driver.quit()
    
    return [(webdriver.Chrome(), href) for href in href_list]


def click_download_button(driver, href):
    driver.get(href)
    driver.find_element(By.CLASS_NAME, 'su-button.su-button-style-flat').click()
    

def move_files(download_folder, book_files):
    main_directory = "Veterinary_Literature"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)
    
    for book_file in book_files:
        source_path = os.path.join(download_folder, book_file)
        target_path = os.path.join(main_directory, book_file)
        shutil.move(source_path, target_path)


def download_book():
    download_folder = r'C:\Users\user\Downloads'
    before_download = os.listdir(download_folder)
    parallel_processing(click_download_button, get_book_href())
    after_download = os.listdir(download_folder)
    
    move_files(download_folder, [file for file in after_download if file not in before_download])
    

def find_files_recursive(extension):
    directory = r'C:\Users\user\Downloads\Veterinary_Literature'
    file_list = os.listdir(directory)
    file_paths = [os.path.join(directory, file) for file in file_list if file.lower().endswith(extension)]
    
    return file_paths


def sanitize_filename(filename):
    invalid_chars = '\\/:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename


def extract_epub_contents(epub_file_path):
    book = ebooklib.epub.read_epub(epub_file_path)
    
    directory = r'C:\Users\user\Downloads\Veterinary_Literature'
        
    book_title = sanitize_filename(book.get_metadata('DC', 'title')[0][0])
    book_directory = os.path.join(directory, book_title)
    if not os.path.exists(book_directory):
        os.makedirs(book_directory)
    
    for image_item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        with open(os.path.join(book_directory, f"{sanitize_filename(image_item.get_name())}"), "wb") as f:
            f.write(image_item.get_content())
            
    for document_item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        with open(os.path.join(book_directory, f"{sanitize_filename(document_item.get_name())}"), "w", encoding='utf-8') as f:
            content = document_item.get_content().decode('utf-8')
            f.write(content)
            
    return book_title


def epub_extraction():
    epub_file_path = find_files_recursive('.epub')
    parallel_processing(extract_epub_contents, epub_file_path)


def html_content_extraction(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return [element['src'] if element.name == 'img' and element.has_attr('src') else element.get_text().strip() for element in soup.find_all(True)]


def preprocess_text(text):
    return re.sub(r'figure.*?\n', '', re.sub(r"[^a-zA-Z0-9\s,.'\"-%–]", '', text.lower()))


def process_pdf(pdf_file):
    loader = UnstructuredPDFLoader(pdf_file)
    page = loader.load()
    return page[0].page_content


def text_splitter():
    pass