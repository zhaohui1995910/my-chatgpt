# -*- coding: utf-8 -*-
# @Time    : 2023/3/21 13:02
# @Author  : 10867
# @FileName: weaviate_test.py
# @Software: PyCharm
import weaviate

client = weaviate.Client("http://weaviate.weaviate.svc.cluster.local")

# class_demo = {
#     "class"     : "PtAitest",
#     "properties": [
#         {
#             "name"    : "text",
#             "dataType": ["string"]
#         },
#         {
#             "name"    : "collection_name",
#             "dataType": ["string"]
#         },
#     ]
# }
# client.schema.create_class(class_demo)
# client.schema.get()

# data = {'text': 'abc', 'collection_name': 'collection_name_test'}
# data_uuid = client.data_object.create(
#     data,
#     'PtAitest',
#     vector=[1.1, 2.3, 3.4, 4.4, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]
# )

# near_vector = {"vector": [100.1, 100.3, 11.4, 14.4, 45.5, 25.5, 15.5, 55.5, 65.5, 75.5], 'certainty': 0.5}
# client.query.get('PtAitest', ['text']).with_near_vector(near_vector).with_limit(2).do()
# near_vector = {"vector": [100.1, 100.3, 11.4, 14.4, 45.5, 25.5, 15.5, 55.5, 65.5, 75.5], 'certainty': 0.9}
# client.query.get('PtAitest', ['text']).with_near_vector(near_vector).with_limit(2).do()
