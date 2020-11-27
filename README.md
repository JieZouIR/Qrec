# Qrec
This is for the paper: Towards Question-based Recommender Systems

Environments: Python 3.6; the evaluation package we used is pytrec_eval.

You can see the inputted parameter setting in the config folder.

Inputted files format:

(1) TagMeResults files: This is the entities used for questions. each line is an pid (the lower cases of original product id from Amazon +'txt') followed by the extracted entities of the product separated by comma. We used ‘TagMe’to extract entities from the product description and reviews in this paper. 

(2) train and test files: they are ratings data, in the form of “indexed_user_id indexed_product_id rating”. indexed_user_id and indexed_product_id are indexed ids from the original Amazon user id and product id. Since it is in the Matrix Market Exchange Formats, you also need add two extra head lines in the beginning, before the ratings. The second line is ’number_of_users number_of_products number_of_data’.

(3) 'dataset-home.txt’ & 'dataset-pet.txt’: they are mapping files between original product id and indexed product id. Specifically, it is a dict which can loaded by pickle. The dict contains five variables: ‘NUM_USERS’: number of users in the data. ‘NUM_ITEMS’: number of products in the data. ‘item_id_index’: a dict mapping from original product id to indexed product id. ‘item_id_index_txt’: a dict mapping from lower cases of original_product_id+'txt' to indexed product id. ‘items2’: a python list of lower cases of original_product_id+'txt' from the used Amazon data. 

Please cite the following paper if you use it in any way:

Jie Zou, Yifan Chen, and Evangelos Kanoulas. 2020. Towards Question-based Recommender Systems. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’20), July 25–30, 2020, Virtual Event, China. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3397271.3401180

