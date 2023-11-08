# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 17:36
# @Author  : 10867
# @FileName: question_answer.py
# @Software: PyCharm
import time
from typing import Union, List, Optional

import jwt
import openai
import tenacity
import tiktoken
import pandas as pd
from flask import request, jsonify
from weaviate.gql.get import GetBuilder
from weaviate.data import replication as weaviate_replication
from weaviate import exceptions as weaviate_exceptions
from weaviate.exceptions import UnexpectedStatusCodeException, WeaviateBaseError

from . import app, loop_executor, auth, weaviate_client, format_logger

tokenizer = tiktoken.get_encoding("cl100k_base")
max_tokens = app.config.get('MAX_TOKENS')


class WeaviateModel:
    _instance = {}
    _first_init = False

    client = weaviate_client

    def __new__(cls, class_name: str, schema_class: Union[dict, str] = None):
        if not cls._instance.get(class_name):
            cls._instance[class_name] = super().__new__(cls)
        return cls._instance.get(class_name)

    def __init__(self, class_name: str, schema_class: Union[dict, str] = None):
        if self._first_init:
            return

        self._first_init = True
        self.class_name = class_name

        if not self.schema_is_exist() and schema_class:
            self.create_schema(schema_class)

    def schema_is_exist(self):
        try:
            self.client.schema.get(self.class_name)
            return True
        except UnexpectedStatusCodeException as e:
            if e.status_code == 404:
                return False

    def create_schema(self, schema_class: Union[dict, str]):
        self.client.schema.create_class(schema_class)

    def data_is_exist(self, uuid: str):
        result = self.client.data_object.get_by_id(uuid, class_name=self.class_name)
        return result

    def create_data(self, data: Union[dict, str], **kwargs):
        object_uuid: str = self.client.data_object.create(
            data,
            self.class_name,
            **kwargs
        )
        return object_uuid

    def delete_data(self, uuid: str):
        self.client.data_object.delete(
            uuid,
            class_name=self.class_name,
            consistency_level=weaviate_replication.ConsistencyLevel.ALL,
        )

    def query_data(self, properties: Union[List[str], str, None]) -> GetBuilder:
        return self.client.query.get(self.class_name, properties)

    def query_id_data(self, uuid: str) -> Optional[dict]:
        return self.client.data_object.get_by_id(uuid=uuid, class_name=self.class_name)

    def update_data(self, data: dict, uuid: str):
        self.client.data_object.update(
            data,
            class_name=self.class_name,
            uuid=uuid,
            consistency_level=weaviate_replication.ConsistencyLevel.ALL
        )


class WeaviateModelTenant:

    def __init__(self, tenant_id):
        self.tenant_id = tenant_id

        self._TextModel = None
        self._TextRelationModel = None

    @property
    def TextModel(self):
        class_name = f'{app.config.get("WEAVIATE_CLASS_PREFIX")}{self.tenant_id}'
        schema_class = {
            'class'     : class_name,
            'properties': [
                {'name': 'text', 'dataType': ['text'], 'indexInverted': False},
                {'name': 'collection_name', 'dataType': ['string']},
                {'name': 'source_id', 'dataType': ['string']}
            ]
        }
        return WeaviateModel(class_name, schema_class)

    @property
    def TextRelationModel(self):
        class_name = f'{app.config.get("WEAVIATE_CLASS_PREFIX")}{self.tenant_id}TextRelation'
        schema_class = {
            'class'     : class_name,
            'properties': [
                {'name': 'sub_ids', 'dataType': ['text'], 'indexInverted': False},
                {'name': 'status', 'dataType': ['string']},
            ]
        }
        return WeaviateModel(class_name, schema_class)

    def delete_data(self, text_id: str):
        '''删除问答上下文'''
        # 1、查询子句
        where_content = {
            'path'       : ["source_id"],
            'operator'   : 'Equal',
            'valueString': text_id
        }
        result = self.TextModel.query_data(['source_id']).with_where(where_content).with_additional(['id']).do()

        # 2、删除所有子句
        for item in result['data']['Get'][self.TextModel.class_name]:
            self.TextModel.delete_data(item['_additional']['id'])

        # 3、删除关系记录
        self.TextRelationModel.delete_data(text_id)

    def get_question_context(self, collection_name: str, near_vector: list) -> str:
        '''查询问答上下文'''
        where_content = {
            'path'       : ["collection_name"],
            'operator'   : 'Equal',
            'valueString': collection_name
        }
        near_vector_content = {
            "vector"   : near_vector,
            'certainty': app.config.get('WEAVIATE_QUERY_NEAR_CERTAINTY')
        }
        result = self.TextModel.query_data(['text']) \
            .with_where(where_content) \
            .with_near_vector(near_vector_content) \
            .with_limit(app.config.get('WEAVIATE_QUERY_NEAR_LIMIT')) \
            .do()

        context_list = [i['text'] for i in result['data']['Get'][self.TextModel.class_name]]
        context = split_context(context_list)

        return context


class OpenaiModel:

    @classmethod
    @tenacity.retry(
        stop=tenacity.stop_after_delay(app.config.get('OPENAI_API_TIMEOUT')),
        retry_error_callback=lambda _: [],
        wait=tenacity.wait_random(min=0, max=2)
    )
    def create_embedding(cls, x):
        format_logger('', 'Debug', 'create_embedding', x, '')
        openai.api_key = app.config.get('OPENAI_API_KEY')
        result = openai.Embedding.create(
            input=x,
            engine='text-embedding-ada-002',
            timeout=60
        )['data'][0]['embedding']
        return result

    @classmethod
    @tenacity.retry(
        stop=tenacity.stop_after_delay(app.config.get('OPENAI_API_TIMEOUT')),
        wait=tenacity.wait_random(min=0, max=2)
    )
    def create_completion(cls, prompt):
        openai.api_key = app.config.get('OPENAI_API_KEY')
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=app.config.get('OPENAI_COMPLETION_MAX_TOKEN'),
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"],
            timeout=app.config.get('OPENAI_API_TIMEOUT'),
        )
        return response

    @classmethod
    @tenacity.retry(
        stop=tenacity.stop_after_delay(app.config.get('OPENAI_API_TIMEOUT')),
        wait=tenacity.wait_random(min=0, max=2)
    )
    def create_chat_completion(cls, messages: list):
        openai.api_key = app.config.get('OPENAI_API_KEY')
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return response


def tokenizer_text(text: str) -> list:
    text.replace('\n', '')
    tests = text.split('。')

    df = pd.DataFrame(tests, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df = df[(df.n_tokens <= 500) & (df.n_tokens > 2)]
    df = df[df.text.notna()]

    return df.text.to_list()


def split_context(text_list: list, max_token=None):
    if not max_token:
        max_token = app.config.get('QUESTION_CONTEXT_MAX_TOKEN')

    returns = []
    cur_len = 0

    for text in text_list:
        cur_len += len(text)

        if cur_len > max_token:
            break

        if isinstance(text, str):
            returns.append(text)

    return "。".join(returns)


def create_embedding_task(tenant_id: str, collection_name: str, text: str, text_id: str):
    tenant_model = WeaviateModelTenant(tenant_id=tenant_id)
    try:
        tenant_model.TextRelationModel.create_data({'sub_ids': '', 'status': 'creating'}, uuid=text_id)

        text_list: list = tokenizer_text(text)
        text_id_list = []
        for text in text_list:
            vector: list = OpenaiModel.create_embedding(text)
            data = {'text': text, 'collection_name': collection_name, 'source_id': text_id}
            _id = tenant_model.TextModel.create_data(data=data, vector=vector)
            text_id_list.append(_id)

    except Exception as e:
        data = {'sub_ids': '', 'status': 'created'}
        tenant_model.TextRelationModel.update_data(data, uuid=text_id)
        format_logger(
            category='',
            exception=f'create_embedding_task: ({e.__class__.__name__} {e})'
        )
    else:
        sub_ids = ';'.join(text_id_list)
        data = {'sub_ids': sub_ids, 'status': 'created'}
        tenant_model.TextRelationModel.update_data(data, uuid=text_id)


@app.route("/embedding/delete", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def embedding_delete():
    data = request.get_json()
    storage_type = data.get('storageType')
    collection_name = data.get('collectionName')
    text_id = data.get('id')

    #  get tenant from token
    tenant_id = request.headers.get('__tenant', 'test').replace('-', '')

    tenant_model = WeaviateModelTenant(tenant_id=tenant_id)
    result = tenant_model.TextRelationModel.query_id_data(uuid=text_id)
    if not result:
        return f'{text_id} not exist', 400

    status = result.get('properties').get('status')
    if status != 'created':
        return f'{text_id} status is {status}', 400

    tenant_model.delete_data(text_id)

    return "deleted"


@app.route("/embedding/update", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def embedding_update():
    data = request.get_json()
    storage_type = data.get('storageType')
    collection_name = data.get('collectionName')
    text_id = data.get('id')
    text = data.get('text')

    #  get tenant from token
    tenant_id = request.headers.get('__tenant', 'test').replace('-', '')

    tenant_model = WeaviateModelTenant(tenant_id=tenant_id)
    result = tenant_model.TextRelationModel.query_id_data(uuid=text_id)
    if not result:
        return f'{text_id} not exist', 400

    status = result.get('properties').get('status')
    if status != 'created':
        return f'{text_id} status is {status}', 400

    # 删除数据
    tenant_model.delete_data(text_id)
    # 创建数据
    _future = loop_executor.submit(create_embedding_task, tenant_id, collection_name, text, text_id)

    return "updating"


@app.route("/embedding/one", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def embedding_one():
    data = request.get_json()
    storage_type = data.get('storageType')
    collection_name = data.get('collectionName')
    text_id = data.get('id')
    text = data.get('text')

    #  get tenant from token
    tenant_id = request.headers.get('__tenant', 'test').replace('-', '')

    _future = loop_executor.submit(create_embedding_task, tenant_id, collection_name, text, text_id)

    return "ok"


@app.route("/embedding/many", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def embedding_many():
    data = request.get_json()
    storage_type = data.get('storageType')
    collection_name = data.get('collectionName')
    items = data.get('items')

    #  get tenant from token
    tenant_id = request.headers.get('__tenant', 'test').replace('-', '')

    for item in items:
        text = item.get('text')
        text_id = item.get('id')

        _future = loop_executor.submit(create_embedding_task, tenant_id, collection_name, text, text_id)

    return "ok"


@app.route("/completion", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def completion():
    data = request.get_json()
    question_text = data.get('questionText')
    context_source = data.get('contextSource')
    collection_name = context_source.get('collectionName')
    storage_type = context_source.get('storageType')
    sorry_text = context_source.get('sorryText', "I don't know.")

    #  get tenant from token
    tenant_id = request.headers.get('__tenant', 'test').replace('-', '')

    openai.api_key = app.config.get('OPENAI_API_KEY')
    question_embeddings = OpenaiModel.create_embedding(question_text)

    if not question_embeddings:
        log_msg = f'Question: {question_text}, create question embeddings timeout'
        format_logger(category=request.endpoint, message=log_msg)
        return '', 500

    tenant_model = WeaviateModelTenant(tenant_id=tenant_id)
    context = tenant_model.get_question_context(collection_name, question_embeddings)
    log_msg = f'Question: {question_text}, Context：{context}'
    format_logger(level='Debug', category=request.endpoint, message=log_msg)

    if not context:
        return jsonify({'answer': sorry_text})

    prompt = f"Answer the question based on the context below, " \
             f"and if the question can't be answered based on the context, " \
             f"say \"{sorry_text}\"\n" \
             f"Context: {context}\n。" \
             f"Question: {question_text}\n" \
             f"Answer:"
    response = OpenaiModel.create_completion(prompt)
    choices = response.get('choices', [])
    return jsonify({'answer': choices[0]['text'].rsplit('。', 1)[0]}) if choices else sorry_text


@app.route("/chat/completion", methods=['POST'])
@auth.token_auth('pt_ai', ["internal_service"])
def chat_completion():
    data = request.get_json()
    question_text = data.get('questionText')
    context_source = data.get('contextSource', {})
    sorry_text = context_source.get('sorryText', "I don't know.")

    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question_text},
    ]
    response = OpenaiModel.create_chat_completion(message)
    choices = response.get('choices', [])
    return jsonify({'answer': choices[0]['message']['content'].rsplit('。', 1)[0]}) if choices else sorry_text


@app.route("/iotest", methods=['GET'])
def test_io():
    print('start')
    time.sleep(10)
    print('end')
    return 'ok'
