# Issues and Ideas to use the Brev CLI provision instances

The brev CLI needs to be authenticated therefore you must run:

```bash
brev login
```

Rendering the Nvidia Login Website for Brev:

```bash
chromium-browser --headless --disable-gpu --screenshot --window-size=1280,800 "[URL]"
```

This shows the first issue.

We cannot use ```lynx``` and simple ```curl``` commands because the NVIDIA's cookie consent banner. This overlay is a JavaScript-driven "modal" that blocks the rest of the page. Until the "Agree" button is clicked, the "Continue" button underneath remains inactive or hidden from the terminal's view.

![Login Page](./images/screenshot.png)

We can Solve this by

- Method 1: Injecting a JavaScript Click
- Method 2: Using Puppeteer if node.js is prefered
- Method 3: Using Selenium in Python

## Method 1 Injecting Javascript click

This is the worst method and might not work.

```bash
chromium-browser --headless --disable-gpu --remote-debugging-port=9222 "[URL]" & 
sleep 5 # Wait for the page to load
curl -X POST http://localhost:9222/json/new -d '{"url": "javascript:document.querySelectorAll(\"button\").forEach(b => { if(b.innerText.includes(\"Agree\")) b.click(); });"}'
```

## Method 2: Using Pupppeteer

This requres Node.js and get over the overlay not the rest of the steps.

```javascript
const puppeteer = require('puppeteer');
(async () => {
  const browser = await puppeteer.launch({headless: true});
  const page = await browser.newPage();
  await page.goto('[YOUR_URL_HERE]');

  // Click "Agree" on the cookie banner
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const agreeBtn = buttons.find(b => b.innerText.includes('Agree'));
    if (agreeBtn) agreeBtn.click();
  });

  // Wait a second for the banner to vanish, then click "Continue"
  await new Promise(r => setTimeout(r, 1000));
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const contBtn = buttons.find(b => b.innerText.includes('Continue'));
    if (contBtn) contBtn.click();
  });

  console.log("Logged in! Check your brev terminal.");
  await browser.close();
})();
```


## Method 3: using Selenium 

This is probably the best way to to get over the Overlay, it will not do the rest of the steps requried to login but it is a good start.

```python
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
```