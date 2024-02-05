import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_images(page, driver):
    try:
        url = base_url.format(page)
        driver.get(url)

        for _ in range(3):
            driver.execute_script("window.scrollBy(0, window.innerHeight);")

        links = []
        while len(links) < 40:
            links = [link.get_attribute('href') for link in driver.find_elements(By.CSS_SELECTOR, 'a.thumbnail_thumb__Bxb6Z.linkAnchor')]
            driver.execute_script("window.scrollBy(0, window.innerHeight);")

        for link in tqdm(links, desc=f'Downloading img on Page {page}', leave=False):
            try:
                driver.get(link)
                driver.find_element(By.CSS_SELECTOR, 'div._3osy73V_eD._1Hc_ju_IXp > button').click()
                product_name = driver.find_element(By.XPATH, '//th[text()="모델명"]/following-sibling::td').text

                path = f'./naver/{pet_type}/{product_name}'

                if os.path.exists(path):
                    with open('os_error_urls.txt', 'a') as error_file:
                        error_file.write(f'{link}\n')
                    continue

                os.makedirs(path)

                image_elements = driver.find_elements(By.CSS_SELECTOR, 'a.se-module-image-link.__se_image_link.__se_link > img')
                image_urls = [img.get_attribute('data-src') for img in image_elements]

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
    
    num_workers = 8
    pages_per_worker = total_pages // num_workers
    
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--incognito')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)')

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        drivers = [webdriver.Chrome(options=chrome_options) for _ in range(num_workers)]
        futures = [executor.submit(download_images, i * pages_per_worker + 1, drivers[i]) for i in range(num_workers)]

        for future in futures:
            future.result()

        for driver in drivers:
            driver.quit()