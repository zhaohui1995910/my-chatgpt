import json
import asyncio
import logging
import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor

import jwt
import weaviate
from flask import Flask, request
from flask.logging import default_handler
from werkzeug.exceptions import HTTPException
from cachetools import cached, TTLCache
from flask_pyoidc import OIDCAuthentication
from flask_pyoidc.provider_configuration import ProviderConfiguration, ClientMetadata, ProviderMetadata
from dotenv import load_dotenv
from healthcheck import HealthCheck

app = Flask(__name__)
# 控制台日志格式
default_handler.setFormatter(logging.Formatter(''))

# 健康检查
health = HealthCheck()
app.add_url_rule("/health/live", "live", view_func=lambda: health.run())
app.add_url_rule("/health/ready", "ready", view_func=lambda: health.run())

# token 缓存 1小时
tolenCache = TTLCache(maxsize=100, ttl=60 * 60)

# thread loop
loop_executor = ThreadPoolExecutor(max_workers=10)

# init config
load_dotenv('.env', override=False)  # 不覆盖系统变量
app.config.from_pyfile('../config.py')
app.logger.info('OPENAI_API_KEY: %s' % app.config.get('OPENAI_API_KEY'))

# create opdc auth
client_metadata = ClientMetadata(
    app.config.get("APP_CLIENT_ID"),
    app.config.get("APP_CLIENT_SECRET")
)
provider_metadata = ProviderMetadata(
    app.config.get("DANGQUYUN_IIDENTITY_SERVER_ISSUER"),
    authorization_endpoint=app.config.get("DANGQUYUN_IIDENTITY_SERVER_AUTH_ENDPOINT"),
    token_endpoint=app.config.get("DANGQUYUN_IIDENTITY_SERVER_TOKEN_ENDPOINT"),
    introspection_endpoint=app.config.get("DANGQUYUN_IIDENTITY_SERVER_INTROSPECTION_ENDPOINT")
)
provider_config = ProviderConfiguration(
    provider_metadata=provider_metadata,
    client_metadata=client_metadata,
    auth_request_params={'scope': app.config.get("OIDC_SCOPES")}
)

auth = OIDCAuthentication({'pt_ai': provider_config})

weaviate_client = weaviate.client.Client(app.config.get('WEAVIATE_URL'))


def format_logger(correlation_id='', level='Error', category='', message='', exception=''):
    level_map = {'Debug': 'debug', 'Information': 'info', 'Warning': 'warngin', 'Error': 'error'}
    log_data = {
        'Timestamp'    : datetime.datetime.now().isoformat()[:-3] + 'Z',
        'CorrelationId': correlation_id,
        'LogLevel'     : level,
        'Category'     : category,
        'Message'      : message,
        'Exception'    : exception
    }
    app.logger.__getattribute__(level_map[level])(json.dumps(log_data, ensure_ascii=False))


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e

    # 记录异常信息
    format_logger(
        correlation_id='',
        level='Error',
        category=request.endpoint,
        message='',
        exception=traceback.format_exc(),
    )
    # 返回错误响应
    return 'An error occurred: ' + str(e), 500


def thread_async(func):
    def make_decorater(*args):
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)

        func_rsult = _loop.run_until_complete(func(*args))

        return func_rsult

    return make_decorater


def create_app():
    # Oidc
    auth.init_app(app)
    return app


@app.route("/")
@auth.token_auth('pt_ai', ["internal_service"])
def home():
    return "Hello, Flask!"


@cached(tolenCache)
def get_accesstoken():
    token_response = auth.clients['pt_ai'].client_credentials_grant()
    return token_response.get('access_token')


@app.route("/accesstoken/test")
def get_accesstoken_test():
    token = get_accesstoken()
    print(jwt.decode(token, algorithms="RS256", options={"verify_signature": False}))
    return token


__all__ = [
    'create_app',
    'question_answer',
    'get_accesstoken',
]
