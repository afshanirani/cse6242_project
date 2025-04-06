from tabpy.tabpy_tools.client import Client
import json
import sqlite3

client = Client('http://localhost:9004/')

def retrieveSchools(state=None, UserZipCode=None, desiredZipCode = None, maximumRadius = None, urbanization=None,major=None,carnegieClass= None,SATScore = None, ACTScore = None,familyIncome = None,schoolSize=None,degree=None):

    #algorithm here

    #return score for each school in an array
    

    return [2] * len(state)


client.deploy('retrieveSchools',
              retrieveSchools,
              'Returns a list of dictionaries with each dictionary containing information about a school',
              override=True)