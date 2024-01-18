# This script crawled the Vimeo search results to obtain URLs and playback
# times for the unmonitored videos that we subsequently streamed

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep
import re

log = open('vimeo_raw_unmonitored.txt', 'a')
re_clip_id = r"clip_id&quot;:[0-9]+"
re_time = r">[0-9][0-9]:[0-9][0-9]<"
set_preferences = False
content_filter_xpath = '/html/body/div[1]/footer/section[2]/div[2]/a'
unrated_xpath = '//*[@id="mc_unrated_content"]'
profanity_xpath = '//*[@id="mc_bad_words"]'
drugs_xpath = '//*[@id="mc_drugs_alcohol"]'
violence_xpath = '//*[@id="mc_violence"]'
apply_xpath = '//*[@id="submit_content_filter"]'
options = Options()
options.add_argument("--headless")
driver = webdriver.Firefox(options = options)

#for genre in ['kids', 'sports', 'highlights', 'news', 'story',
#              'music', 'concert', 'nature', 'class', 'lesson']:
for genre in ['fun', 'game', 'documentary', 'travel']:
    for i in range(1,450):
        target = "https://vimeo.com/search/page:" + str(i) + "?duration=medium&q=" + genre
        print('getting ' + target)
        try:
            driver.get(target)
            sleep(2)
        except:
            print('*** failed to get target ***')
            continue
        if not set_preferences:
            sleep(3)
            last_height = driver.execute_script("return document.body.scrollHeight")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            content_filter_button = driver.find_element(By.XPATH, content_filter_xpath)
            ActionChains(driver).move_to_element(content_filter_button).perform()
            ActionChains(driver).click(content_filter_button).perform()
            sleep(3)
            unrated_button = driver.find_element(By.XPATH, unrated_xpath)
            ActionChains(driver).click(unrated_button).perform()
            profanity_button = driver.find_element(By.XPATH, profanity_xpath)
            ActionChains(driver).click(profanity_button).perform()
            drugs_button = driver.find_element(By.XPATH, drugs_xpath)
            ActionChains(driver).click(drugs_button).perform()
            violence_button = driver.find_element(By.XPATH, violence_xpath)
            ActionChains(driver).click(violence_button).perform()
            apply_button = driver.find_element(By.XPATH, apply_xpath)
            ActionChains(driver).click(apply_button).perform()
            set_preferences = True
            sleep(3)
        clip_id_list = re.findall(re_clip_id, driver.page_source)
        time_list = re.findall(re_time, driver.page_source)
        for clip_id, time in zip(clip_id_list, time_list):
            minutes_to_seconds = int(time[1:3]) * 60
            seconds = int(time[4:6])
            string = 'https://vimeo.com/' + clip_id[14:] + ',' + str(minutes_to_seconds + seconds) + '\n'
            log.write(string)

log.close()
driver.quit()
