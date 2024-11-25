import os 
import json
import pandas as pd
import sys 
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

selected_companies = ["AAPL", "KO", "V", "XOM", "TSLA"]
company_names = {
    "AAPL": "Apple Inc.",
    "KO": "The Coca-Cola Company",
    "V": "Visa Inc.",
    "XOM": "Exxon Mobil Corporation",
    "TSLA": "Tesla, Inc."
}
tweet_dir = "./tweet/raw/"
SYSTEM_PROMPT = "Forget all your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience adept at analysing the effect of tweets on stock price."

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16)
model.to("cuda")
model.eval()

def get_tweets(ticker, date_str):
        #print(f"Getting tweets for {ticker} on {date_str}")
        tweets = []
        tweet_path = os.path.join(tweet_dir, ticker, date_str)

        if os.path.exists(tweet_path):
            with open(tweet_path) as f:
                lines = f.readlines()
                for line in lines:
                    tweet_obj = json.loads(line)
                    tweets.append(tweet_obj['text'])
        return tweets

for company in selected_companies:
    
    daily_tweets = {}
    
    for date_str in sorted(os.listdir(f"{tweet_dir}{company}")):
        daily_tweets[date_str] = get_tweets(company, date_str)
    print(f"Number of days of {company} tweets:", len(daily_tweets))

    company_name = company_names[company]

    results = pd.DataFrame(columns=["ticker", "date_of_tweets", "prediction", "confidence"])

    results_dic = {
            "ticker": [],
            "date_of_tweets": [],
            "prediction": [],
            "confidence": []
        }

    cnt = 1

    for date, tweets in daily_tweets.items():

        current_tweet = " ".join(tweets)

        prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"

        messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        last_logits = logits[0, -1, :]

        # Calculate probabilities
        probabilities = torch.softmax(last_logits, dim=-1)

        # Get the predicted token and the corresponding probability
        predicted_token = torch.argmax(probabilities).item()
        predicted_probability = probabilities[predicted_token].item()

        #Decode the predicted token
        predicted_text = tokenizer.decode(predicted_token)

        if predicted_text != "1" and predicted_text != "0":
            predicted_text = "1"
            predicted_probability = 0.5

        results_dic["ticker"].append(company)
        results_dic["date_of_tweets"].append(date)
        results_dic["prediction"].append(predicted_text)
        results_dic["confidence"].append(predicted_probability)

        if cnt%100 == 0:
            print(f"Predicted {cnt} days of sentiments for {company}")

    results = pd.DataFrame(results_dic)
    results.to_csv(f"./sentiments/{company}_sentiment.csv", index=False)

        
        
        





