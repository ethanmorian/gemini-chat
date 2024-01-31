import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_images(page):
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)')

    with webdriver.Chrome(options=chrome_options) as driver:
        try:
            url = base_url.format(page)
            driver.implicitly_wait(3)
            driver.get(url)
            
            for _ in range(3):
                driver.execute_script("window.scrollBy(0, window.innerHeight);")
            
            while True:
                links = [link.get_attribute('href') for link in driver.find_elements(By.CSS_SELECTOR, 'a.thumbnail_thumb__Bxb6Z.linkAnchor')]
                
                if len(links) >= 40:
                    break
                
                driver.execute_script("window.scrollBy(0, window.innerHeight);")

            for link in tqdm(links, desc=f'Downloading img on Page {page}', leave=False):
                try:
                    driver.implicitly_wait(3)
                    driver.get(link)
                    driver.find_element(By.CSS_SELECTOR, 'div._3osy73V_eD._1Hc_ju_IXp > button').click()
                    product_name = driver.find_element(By.XPATH, '//th[text()="모델명"]/following-sibling::td').text
                    image_elements = driver.find_elements(By.CSS_SELECTOR, 'a.se-module-image-link.__se_image_link.__se_link > img')
                    image_urls = [img.get_attribute('data-src') for img in image_elements]

                    path = f'./naver/{pet_type}/{product_name}'

                    if not os.path.exists(path):
                        os.makedirs(path)
                    else:
                        with open('os_error_urls.txt', 'a') as error_file:
                            error_file.write(f'{link}\n')

                    for index, img_url in enumerate(image_urls):
                        img_data = requests.get(img_url).content
                        with open(f'{path}/image_{index + 1}.jpg', 'wb') as handler:
                            handler.write(img_data)
                except Exception as e:
                    pass
        except Exception as e:
            with open('error_urls.txt', 'a') as error_file:
                error_file.write(f'{url}\n')

if __name__ == "__main__":
    pet_type = 'dog'
    total_pages = 10900
    base_url = 'https://search.shopping.naver.com/search/all?adQuery=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%82%AC%EB%A3%8C&frm=NVSHATC&npayType=2&origQuery=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%82%AC%EB%A3%8C&pagingIndex={}&pagingSize=40&productSet=checkout&query=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%82%AC%EB%A3%8C&sort=rel&timestamp=&viewType=image'
    
    # pet_type = 'dog'
    # total_pages = 2500
    # base_url = 'https://search.shopping.naver.com/search/all?adQuery=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%98%81%EC%96%91%EC%A0%9C&frm=NVSHATC&npayType=2&origQuery=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%98%81%EC%96%91%EC%A0%9C&pagingIndex={}&pagingSize=40&productSet=checkout&query=%EA%B0%95%EC%95%84%EC%A7%80%20%EC%98%81%EC%96%91%EC%A0%9C&sort=rel&timestamp=&viewType=list'
    
    # pet_type = 'cat'
    # total_pages = 5900
    # base_url = 'https://search.shopping.naver.com/search/all?adQuery=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%82%AC%EB%A3%8C&frm=NVSHATC&npayType=2&origQuery=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%82%AC%EB%A3%8C&pagingIndex={}&pagingSize=40&productSet=checkout&query=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%82%AC%EB%A3%8C&sort=rel&timestamp=&viewType=image'
    
    # pet_type = 'cat'
    # total_pages = 1000
    # base_url = 'https://search.shopping.naver.com/search/all?adQuery=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%98%81%EC%96%91%EC%A0%9C&frm=NVSHATC&npayType=2&origQuery=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%98%81%EC%96%91%EC%A0%9C&pagingIndex={}&pagingSize=40&productSet=checkout&query=%EA%B3%A0%EC%96%91%EC%9D%B4%20%EC%98%81%EC%96%91%EC%A0%9C&sort=rel&timestamp=&viewType=list'

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(download_images, range(1, total_pages+1))