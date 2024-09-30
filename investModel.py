import requests
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  
import dask
from dask import delayed, compute
from math import ceil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def fetch_recent_news(query, api_key):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')
    url = (f'https://newsapi.org/v2/everything?q={query}  Minerva Foods OR BEEF3'
           f'&from={start_date_str}&to={end_date_str}&apiKey={api_key}')
    response = requests.get(url)
    try:
        news_data = response.json()
    except ValueError:
        print(f"Erro ao decodificar a resposta JSON: {response.text}")
        return None
    if response.status_code != 200:
        print(f"Erro na requisição: {news_data.get('message', 'Sem mensagem de erro')}")
        return None
    articles = news_data.get('articles', [])
    if not articles:
        print('Nenhuma notícia encontrada.')
        return None
    return articles

# Função para carregar o modelo e calcular embeddings
def fetch_tensorflow_sentiment(news_texts):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    # Filtrar títulos não nulos
    filtered_texts = [text for text in news_texts if text is not None]
    
    if not filtered_texts:
        print("Nenhum título válido encontrado para análise.")
        return []
    
    news_texts_tensor = tf.convert_to_tensor(filtered_texts, dtype=tf.string)
    embeddings = model(news_texts_tensor)
    return embeddings.numpy()

# Função para analisar o sentimento das notícias
def analyze_sentiment(embeddings):
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    positive_ref = model(["good", "positive", "great", "excellent", "happy"]).numpy()
    negative_ref = model(["bad", "negative", "poor", "terrible", "sad"]).numpy()
    
    sentiments = []
    
    for emb in embeddings:
        positive_similarity = cosine_similarity([emb], positive_ref).mean()
        negative_similarity = cosine_similarity([emb], negative_ref).mean()
        
        if positive_similarity > negative_similarity:
            sentiments.append("Positivo")
        elif negative_similarity > positive_similarity:
            sentiments.append("Negativo")
        else:
            sentiments.append("Neutro")
    
    return sentiments

@delayed
def analyze_news_batch(news_batch):
    titles = [article.get('title', 'Sem título') for article in news_batch]
    embeddings = fetch_tensorflow_sentiment(titles)
    
    # Obtenção dos sentimentos com base nos embeddings
    sentiment_labels = analyze_sentiment(embeddings)
    
    sentiments = []
    for article, sentiment in zip(news_batch, sentiment_labels):
        title = article.get('title', 'Sem título')
        published_at = article.get('publishedAt', 'Sem data')
        sentiments.append({
            'title': title,
            'published_at': published_at,
            'sentiment': sentiment
        })
    return sentiments

def analyze_news_in_batches(news_articles, batch_size=5):
    num_batches = ceil(len(news_articles) / batch_size)
    batches = [news_articles[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    delayed_results = [analyze_news_batch(batch) for batch in batches]
    news_sentiments = compute(*delayed_results, scheduler='processes')
    all_sentiments = [sentiment for batch in news_sentiments for sentiment in batch]
    return all_sentiments

# Função para obter as cotações da empresa Minerva (BEEF3) nos últimos 30 dias
def fetch_stock_data(ticker='BEEF3.SA'):
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="1mo")
    hist['Date'] = hist.index
    hist = hist[['Date', 'Open', 'Close', 'Volume']]
    hist['Variation'] = (hist['Close'] - hist['Open']) / hist['Open']
    return hist

# Função para obter dados de indicadores macroeconômicos
def fetch_macroeconomic_data(indicator='NY.GDP.MKTP.CD'):
    url = f'http://api.worldbank.org/v2/country/BR/indicator/{indicator}?format=json'
    response = requests.get(url)
    try:
        data = response.json()
    except ValueError:
        print(f"Erro ao decodificar a resposta JSON: {response.text}")
        return None
    if response.status_code != 200:
        print(f"Erro na requisição: {data.get('message', 'Sem mensagem de erro')}")
        return None
    return data[1]  


# Função para obter dados de sentimento social do Twitter
def fetch_twitter_sentiments(query, api_key):
    url = f'https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=created_at'
    headers = {'Authorization': f'Bearer {api_key}'}
    response = requests.get(url, headers=headers)
    try:
        data = response.json()
    except ValueError:
        print(f"Erro ao decodificar a resposta JSON: {response.text}")
        return None
    if response.status_code != 200:
        print(f"Erro na requisição: {data.get('message', 'Sem mensagem de erro')}")
        return None
    return data.get('data', [])


def main():
    news_query = 'Minerva BEEF3'
    google_news_api_key = '***************'  
    twitter_api_key = '**********************'  
    news_articles = fetch_recent_news(news_query, google_news_api_key)
    if not news_articles:
        print('Não foi possível obter as notícias.')
        return

    news_sentiments = analyze_news_in_batches(news_articles)
    
    # Obter dados de ações da Minerva (BEEF3) nos últimos 30 dias
    stock_data = fetch_stock_data()
    
    macroeconomic_data = fetch_macroeconomic_data()
    
    twitter_sentiments = fetch_twitter_sentiments('Minerva BEEF3', twitter_api_key)
    
    for news in news_sentiments:
        # Extrair a data de publicação e adicionar 1 dia
        news_date = datetime.strptime(news['published_at'][:10], '%Y-%m-%d') + timedelta(days=1)
        
        stock_on_news_date = stock_data[stock_data['Date'] == news_date.strftime('%Y-%m-%d')]
        
        if not stock_on_news_date.empty:
            variation = stock_on_news_date['Variation'].values[0]
            print(f"\nNotícia: {news['title']}")
            print(f"Data de publicação: {news['published_at']}")
            print(f"Sentimento: {news['sentiment']}")
            print(f"Variação da ação (BEEF3) no dia seguinte: {variation:.2%}")
        else:
            print(f"\nNotícia: {news['title']}")
            print(f"Data de publicação: {news['published_at']}")
            print(f"Sentimento: {news['sentiment']}")
            print("Variação da ação (BEEF3) no dia seguinte: Dados não disponíveis")
    
    # Exibir dados macroeconômicos
    if macroeconomic_data:
        print("\nDados Macroeconômicos:")
        for record in macroeconomic_data:
            print(record)

    # Exibir dados de sentimentos do Twitter
    if twitter_sentiments:
        print("\nSentimentos do Twitter:")
        for tweet in twitter_sentiments:
            print(f"Tweet: {tweet['text']}")
            print(f"Data de criação: {tweet['created_at']}")



if __name__ == '__main__':
    main()
