# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 13:10
# @Author  : 10867
# @FileName: embedding_many.py
# @Software: PyCharm
import os
import random
import time
import asyncio
from threading import Lock

lock = Lock()

import openai
import requests
import tiktoken
import pandas as pd
from openai.error import RateLimitError, APIConnectionError

tokenizer = tiktoken.get_encoding("cl100k_base")

url = 'http://127.0.0.1:5000/embedding/many'


def test_embedding_many_api():
    with open('刘慈欣 - 三体.txt', 'r') as f:
        content_list = f.readlines()
        content_len = [len(c) for c in content_list]
        content_len.sort()

        items = []
        current_id = 0
        for i, text in enumerate(content_list):
            current_id += 1
            items.append(
                {'id': current_id, 'text': text}
            )

            if len(items) == 200:
                resp = requests.post(
                    url=url,
                    json={
                        "collectionName": "3body",
                        "storageType"   : "csv",
                        "items"         : items
                    }
                )
                print(resp)
                time.sleep(1)
                items = []

        requests.post(
            url=url,
            json={
                "collectionName": "movies",
                "storageType"   : "csv",
                "items"         : items
            }
        )


openai_key_pool = [
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
    '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',
]


def create_embedding(x):
    print(x)
    while True:
        _open_key = random.choice(openai_key_pool)
        openai.api_key = _open_key
        try:
            result = openai.Embedding.create(
                input=x,
                engine='text-embedding-ada-002'
            )['data'][0]['embedding']
            return result
        except (RateLimitError, APIConnectionError) as e:
            print('%s: openid kye - %s; time: - %s' % (str(e), openai.api_key, (time.time() * 1000)))
            time.sleep(1.5)

        time.sleep(0.5)


def embedding_text(text: str = '', csv_name: str = '', text_set: set = ()):
    text.replace('\n', '')
    tests = text.split('。')
    df = pd.DataFrame(tests, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df = df[(df.n_tokens <= 500) & (df.n_tokens > 2)]
    df = df[df.text.notna()]

    # 过滤掉已存在的文本
    df = df[~df.text.isin(text_set)]
    if df.shape[0] == 0:
        return

    df['embeddings'] = df.text.apply(lambda x: create_embedding(x))

    with lock:
        df.to_csv(f'{csv_name}.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    csv_name = '3body'
    with open('刘慈欣 - 三体.txt', 'r') as f:
        content_list = f.readlines()
        content_len = [len(c) for c in content_list]
        content_len.sort()

        text_set = set()
        if os.path.exists(f'../csvs/{csv_name}.csv'):
            try:
                df = pd.read_csv(f'../csvs/{csv_name}.csv', header=None)
                df = df.rename(columns={0: 'text', 1: 'n_token', 2: 'embeddings'})
                text_set = set(df.text.values)
            except ValueError:
                pass

        for i, text in enumerate(content_list):
            embedding_text(text, csv_name, text_set)
