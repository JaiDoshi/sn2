import os
import re
import pandas as pd
import json
import numpy as np
import time
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_TOKEN")
genai.configure(api_key=gemini_api_key)


selected_companies = ["AAPL", "KO", "V", "XOM", "TSLA"]
tweet_dir = "./tweet/raw/"


def get_tweets(ticker, date_str):
        print(f"Getting tweets for {ticker} on {date_str}")
        tweets = []
        tweet_path = os.path.join(tweet_dir, ticker, date_str)

        if os.path.exists(tweet_path):
            with open(tweet_path) as f:
                lines = f.readlines()
                for line in lines:
                    tweet_obj = json.loads(line)
                    tweets.append(tweet_obj['text'])
        return tweets

# apple_daily_tweets = {}
# for date_str in sorted(os.listdir(f"{tweet_dir}{selected_companies[0]}")):
#     apple_daily_tweets[date_str] = get_tweets(selected_companies[0], date_str)
# print("Number of days of Apple tweets: ", len(apple_daily_tweets))

coca_daily_tweets = {}
for date_str in sorted(os.listdir(f"{tweet_dir}{selected_companies[1]}")):
    coca_daily_tweets[date_str] = get_tweets(selected_companies[1], date_str)
print("Number of days of Coca Cola tweets: ", len(coca_daily_tweets))

visa_daily_tweets = {}
for date_str in sorted(os.listdir(f"{tweet_dir}{selected_companies[2]}")):
    visa_daily_tweets[date_str] = get_tweets(selected_companies[2], date_str)
print("Number of days of Visa tweets: ", len(visa_daily_tweets))

exxon_daily_tweets = {}
for date_str in sorted(os.listdir(f"{tweet_dir}{selected_companies[3]}")):
    exxon_daily_tweets[date_str] = get_tweets(selected_companies[3], date_str)
print("Number of days of Exxon tweets: ", len(exxon_daily_tweets))

tesla_daily_tweets = {}
for date_str in sorted(os.listdir(f"{tweet_dir}{selected_companies[4]}")):
    tesla_daily_tweets[date_str] = get_tweets(selected_companies[4], date_str)
print("Number of days of Tesla tweets: ", len(tesla_daily_tweets))

generation_config = {
  "temperature": 0.3,
  "max_output_tokens": 1,
  "response_mime_type": "text/plain",
}

gen_model = genai.GenerativeModel( 
      model_name='models/gemini-1.5-flash',
      generation_config=generation_config, 
      system_instruction= "Forget all your previous instructions. Pretend you are a financial expert. You are a financial expert with stock recommendation experience adept at analysing the effect of tweets on stock price.",
      safety_settings={}, tools=None
 )

results = pd.DataFrame(columns=["ticker", "date_of_tweets", "prediction", "confidence"])

for date, tweets in apple_daily_tweets.items():
    ticker = "AAPL"
    current_tweet = " ".join(tweets)
    company_name = f"Apple Inc. ({ticker})"
    prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"
    response = gen_model.generate_content([prompt])
    results = pd.concat([results, pd.DataFrame([{
        "ticker": ticker,
        "date_of_tweets": date,
        "prediction": response.text,
        "confidence": np.exp(response.candidates[0].avg_logprobs)
    }])], ignore_index=True)
    results.to_csv("gemini_sentiment_predictions.csv", index=False)
    time.sleep(4)
    print(date, response.text, np.exp(response.candidates[0].avg_logprobs))

for date, tweets in coca_daily_tweets.items():
    ticker = "KO"
    current_tweet = " ".join(tweets)
    company_name = f"The Coca-Cola Company ({ticker})"
    prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"
    response = gen_model.generate_content([prompt])
    results = pd.concat([results, pd.DataFrame([{
        "ticker": ticker,
        "date_of_tweets": date,
        "prediction": response.text,
        "confidence": np.exp(response.candidates[0].avg_logprobs)
    }])], ignore_index=True)
    results.to_csv("gemini_sentiment_predictions_others.csv", index=False)
    time.sleep(4)
    print(date, response.text, np.exp(response.candidates[0].avg_logprobs))

for date, tweets in visa_daily_tweets.items():
    ticker = "V"
    current_tweet = " ".join(tweets)
    company_name = f"Visa Inc. ({ticker})"
    prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"
    response = gen_model.generate_content([prompt])
    results = pd.concat([results, pd.DataFrame([{
        "ticker": ticker,
        "date_of_tweets": date,
        "prediction": response.text,
        "confidence": np.exp(response.candidates[0].avg_logprobs)
    }])], ignore_index=True)
    results.to_csv("gemini_sentiment_predictions_others.csv", index=False)
    time.sleep(4)
    print(date, response.text, np.exp(response.candidates[0].avg_logprobs))

for date, tweets in exxon_daily_tweets.items():
    ticker = "XOM"
    current_tweet = " ".join(tweets)
    company_name = f"Exxon Mobil Corporation ({ticker})"
    prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"
    response = gen_model.generate_content([prompt])
    results = pd.concat([results, pd.DataFrame([{
        "ticker": ticker,
        "date_of_tweets": date,
        "prediction": response.text,
        "confidence": np.exp(response.candidates[0].avg_logprobs)
    }])], ignore_index=True)
    results.to_csv("gemini_sentiment_predictions_others.csv", index=False)
    time.sleep(4)
    print(date, response.text, np.exp(response.candidates[0].avg_logprobs))

for date, tweets in tesla_daily_tweets.items():
    ticker = "TSLA"
    current_tweet = " ".join(tweets)
    company_name = f"Tesla, Inc. ({ticker})"
    prompt = f"Predict the next-day price movement of the stock for {company_name} based on the following tweets. Your prediction should be binary, 1 for positive price movement and 0 for negative price movement. Only give the prediction as your response, nothing else. Here are the tweets from today: {current_tweet}\n"
    response = gen_model.generate_content([prompt])
    results = pd.concat([results, pd.DataFrame([{
        "ticker": ticker,
        "date_of_tweets": date,
        "prediction": response.text,
        "confidence": np.exp(response.candidates[0].avg_logprobs)
    }])], ignore_index=True)
    results.to_csv("gemini_sentiment_predictions_others.csv", index=False)
    time.sleep(4)
    print(date, response.text, np.exp(response.candidates[0].avg_logprobs))

results.to_csv("gemini_sentiment_predictions_others.csv", index=False)

print("All predictions done")