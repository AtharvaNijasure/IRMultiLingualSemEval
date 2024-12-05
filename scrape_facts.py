import pandas as pd
import json
from scrape import scrape_url_with_post_session
from tqdm import tqdm

df_fact_checks = pd.read_csv("../data/fact_checks.csv")

## sample 5% of the data
df_fact_checks = df_fact_checks.sample(frac=0.05, random_state=42)

df_fact_checks['instances'] = df_fact_checks['instances'].apply(lambda x: x[1:-1].split(", ") if x else [])

factcheck_ids = df_fact_checks['fact_check_id'].tolist()
urls = df_fact_checks['instances'].apply(lambda x: x[1][1:-2] if x else "")

print("number of fact checks:", len(factcheck_ids))
print("number of urls:", len(urls))
print("fact check ids:", factcheck_ids[:5])
print("urls:", urls[:5])

scraped_content = []
count = 0

for factcheck_id, url in tqdm(zip(factcheck_ids, urls), total=len(factcheck_ids), desc="Scraping fact checks"):
    form_data = {"key": "value"}
    content = scrape_url_with_post_session(url, data=form_data)

    if(len(content) > 0):
        count += 1
        print(f"Scraped {count} fact checks")
    
    content = {
        "fact_check_id": factcheck_id,
        "url": url,
        "content": content
    }
    scraped_content.append(content)

    if len(scraped_content) % 100 == 0:
        ## save to jsonl file
        with open(f"../data/scraped_content.jsonl", "w") as file:
            for item in scraped_content:
                file.write(json.dumps(item) + "\n")