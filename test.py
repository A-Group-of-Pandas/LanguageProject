from model import Model
from coco import COCO
from query_database import Query

querier = Query()
# testing the data
query = input("What would you like to search for? ")
k = int(input("How many images would you like to see? "))
querier.query_database(query, k=k)

