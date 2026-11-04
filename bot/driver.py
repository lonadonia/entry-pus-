"""Shared Selenium WebDriver factory."""
import os

import undetected_chromedriver as uc


def make_driver(headless: bool = True) -> uc.Chrome:
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")

    chrome_bin = os.environ.get("CHROME_BIN")
    if chrome_bin:
        options.binary_location = chrome_bin

    # Prefer the pinned, version-matched chromedriver from the image over
    # undetected_chromedriver's auto-download (see bot/scraper.py).
    kwargs = {"options": options, "use_subprocess": True}
    driver_path = os.environ.get("CHROMEDRIVER_PATH")
    if driver_path and os.path.exists(driver_path):
        kwargs["driver_executable_path"] = driver_path

    driver = uc.Chrome(**kwargs)
    driver.set_page_load_timeout(30)
    return driver
