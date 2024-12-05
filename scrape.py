import requests
from bs4 import BeautifulSoup

def scrape_url_with_post_session(url, data=None):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": url,  # Set the referer to make it look like the request is coming from the site itself
    }

    try:
        # Use a session to maintain cookies and headers
        with requests.Session() as session:
            session.headers.update(headers)
            
            # Perform a GET request first to mimic a browser visiting the page
            session.get(url, timeout=10)
            
            # Perform the POST request
            response = session.post(url, data=data, timeout=10)
            response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx

            # Parse the response content
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            return text.strip()

    # except requests.exceptions.HTTPError as e:
    #     print(f"HTTP Error: {e}")
    # except requests.exceptions.ConnectionError:
    #     print("Connection Error")
    # except requests.exceptions.Timeout:
    #     print("Request Timeout")
    # except requests.exceptions.RequestException as e:
    #     print(f"General Error: {e}")
    except Exception as e:
        # print(f"An error occurred: {e}")
        return ""

# Example usage
# if __name__ == "__main__":
#     url = "https://misbar.com/factcheck/2020/09/22/لا-رسوم-على-فحص-المغادرين-عبر-معبر-رفح#dec2d5e42563f746d8707d7911005d82"  # Replace with the target URL
#     form_data = {"key": "value"}  # Replace with actual data if needed

#     content = scrape_url_with_post_session(url, data=form_data)
#     print("Scraped Content:")
#     # save the content to a file
#     with open("scraped_content.txt", "w") as file:
#         file.write(content)