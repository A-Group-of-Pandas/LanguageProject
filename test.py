from model import Model
from coco import COCO
from query_database import query_database

dataset = COCO(database_dir='database')

# testing the data
query = input("What would you like to search for? \n")
k = int(input("How many images would you like to see? \n"))
query_database(query, k=k)