import sys
from collections import Counter
from pprint import pprint
from itertools import chain
import unicodedata

tokenizer_name = 'mecab'

def get_nfc_text(string):
    return unicodedata.normalize('NFC', string)


def get_query_results(query, tokenize=False, max_len=1000):
    responses = news_search.get_news(query, max_len=max_len)
    newslist = []
    # texts_agg, tokens_agg = [], []
    for res in responses:
        if res['link'].startswith("https://news.naver.com"):
            try:
                body_text = news_search.get_naver_news_body(res['link'])
                body_text = get_nfc_text(body_text)
                if tokenize:
                    tokens = tok.pos(body_text)
                    res['tokens'] = tokens
                res['body'] = body_text
                newslist.append(res)
                # texts_agg.append(body_text)
                # tokens_agg += tokens
            except Exception as e:
                print(e)
    print(f"number of retrieved news: {len(newslist)}")
    print(query)
    return newslist


def main():
    queries = ['아이유']
    for query in queries:
        res = get_query_results(query, 1000)


if __name__ == '__main__':
    main()
