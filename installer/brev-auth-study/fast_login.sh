#!/bin/bash

# 1. Start brev login in the background and capture output to a temporary file
# We use 'script' to trick brev into thinking it's a real TTY to get the URL
TEMP_LOG="brev_output.log"
rm -f $TEMP_LOG

echo "🚀 Starting Brev Login..."
script -q -c "brev login --skip-browser" /dev/null > $TEMP_LOG &
BREV_PID=$!

# 2. Wait for the URL to appear in the log file
echo "⏳ Waiting for NVIDIA Auth URL..."
MAX_RETRIES=20
COUNT=0
URL=""

while [ $COUNT -lt $MAX_RETRIES ]; do
    URL=$(grep -oP 'https://api.ngc.nvidia.com/login\?code=[^ ]+' $TEMP_LOG | head -1)
    if [ ! -z "$URL" ]; then
        echo "🔗 URL Found: $URL"
        break
    fi
    sleep 1
    ((COUNT++))
done

if [ -z "$URL" ]; then
    echo "❌ Failed to capture URL. Check $TEMP_LOG"
    kill $BREV_PID
    exit 1
fi

# 3. Pass the URL to the Python Auto-Auth script
# Note: We pass the URL as a command line argument to Python
echo "🤖 Triggering Python Selenium Auto-Auth..."
python3 <<EOF
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)
url = "$URL"

try:
    driver.get(url)
    # Click Agree
    WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Agree')]"))).click()
    print("✅ Clicked Agree.")
    
    time.sleep(2)
    
    # Click Continue
    WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Continue')]"))).click()
    print("✅ Clicked Continue.")
    
    time.sleep(5) # Give it a moment to finish the handshake
except Exception as e:
    print(f"❌ Error during Python execution: {e}")
finally:
    driver.quit()
EOF

# 4. Clean up
wait $BREV_PID
rm $TEMP_LOG
echo "🎉 Login process complete!"