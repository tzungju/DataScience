from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
 
driver = webdriver.Chrome()
driver.get("https://www.cmoney.tw/finance/f00032.aspx?s=1101")
elem = driver.find_element_by_id("qStockId")
elem.clear()
elem.send_keys("2332")
elem.send_keys(Keys.RETURN)
elem = driver.find_element_by_css_selector(".tb.tb2")
#print(driver.page_source)
time.sleep(1)
print(elem.text)

time.sleep(5)
driver.quit()