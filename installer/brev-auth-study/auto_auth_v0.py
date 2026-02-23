from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# 1. Setup Headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")  # No GUI needed
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

# 2. Paste your Brev Login URL here
url = "PASTE_YOUR_NVIDIA_URL_HERE"

try:
    print("Navigating to NVIDIA login...")
    driver.get(url)

    # 3. Wait for and click the "Agree" button (Cookie Banner)
    print("Looking for 'Agree' button...")
    agree_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Agree')]"))
    )
    agree_button.click()
    print("✅ Clicked Agree.")

    # 4. Wait for the banner to disappear and click "Continue"
    time.sleep(2) 
    print("Looking for 'Continue' button...")
    continue_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Continue')]"))
    )
    continue_button.click()
    print("✅ Clicked Continue.")

    print("\nCheck your other terminal. Brev should now be logged in!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    # Take a debug screenshot if it fails
    driver.save_screenshot("debug_error.png")
    print("Saved debug_error.png to see what went wrong.")

finally:
    driver.quit()