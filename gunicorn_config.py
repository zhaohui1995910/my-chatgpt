from gunicorn.glogging import Logger as GunicornBaseLogger


class CustomGunicornLogger(GunicornBaseLogger):

    def access(self, resp, req, environ, request_time):
        if "/health" not in req.path:
            super().access(resp, req, environ, request_time)


debug = False
preload_app = False
logger_class = 'gunicorn_config.CustomGunicornLogger'
loglevel = 'info'
bind = '0.0.0.0:5000'
pidfile = 'gunicorn.pid'
errorlog = '-'
accesslog = '-'
timeout = 600

# 启动的进程数
workers = 2
worker_class = 'gevent'
