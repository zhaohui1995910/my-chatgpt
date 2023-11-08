import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
MAX_TOKENS = 500
QUESTION_CONTEXT_MAX_TOKEN = 1000
OPENAI_COMPLETION_MAX_TOKEN = 1000

OPENAI_API_TIMEOUT = 30

APP_CLIENT_ID = 'pt_ai'
APP_CLIENT_SECRET = 'permission_manage'

DANGQUYUN_IIDENTITY_SERVER = os.environ.get('DANGQUYUN_IIDENTITY_SERVER')
DANGQUYUN_IIDENTITY_SERVER_ISSUER = f'{DANGQUYUN_IIDENTITY_SERVER}/useridentity'
DANGQUYUN_IIDENTITY_SERVER_TOKEN_ENDPOINT = f'{DANGQUYUN_IIDENTITY_SERVER}/useridentity/connect/token'
DANGQUYUN_IIDENTITY_SERVER_AUTH_ENDPOINT = f'{DANGQUYUN_IIDENTITY_SERVER}/useridentity/connect/auth'
DANGQUYUN_IIDENTITY_SERVER_INTROSPECTION_ENDPOINT = f'{DANGQUYUN_IIDENTITY_SERVER}/useridentity/connect/introspect'

OIDC_SCOPES = ["internal_service"]
OIDC_REDIRECT_URI = "http://www.baidu.com"

WEAVIATE_URL = os.environ.get('WEAVIATE_URL')
WEAVIATE_CLASS_PREFIX = 'PtAi'
WEAVIATE_QUERY_NEAR_CERTAINTY = 0.9
WEAVIATE_QUERY_NEAR_LIMIT = 10
