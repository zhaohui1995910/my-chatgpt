# -*- coding: utf-8 -*-
# @Time    : 2023/3/14 17:36
# @Author  : 10867
# @FileName: main.py
# @Software: PyCharm
from app import *

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
