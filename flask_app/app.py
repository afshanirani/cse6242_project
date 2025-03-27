import json
import sqlite3
from flask import Flask, render_template

degree_to_pcip = {
    "Agriculture, Agriculture Operations, And Related Sciences": "PCIP01",
    "Natural Resources And Conservation": "PCIP03",
    "Architecture And Related Services": "PCIP04",
    "Area, Ethnic, Cultural, Gender, And Group Studies": "PCIP05",
    "Communication, Journalism, And Related Programs": "PCIP09",
    "Communications Technologies/Technicians And Support Services": "PCIP10",
    "Computer And Information Sciences And Support Services": "PCIP11",
    "Personal And Culinary Services": "PCIP12",
    "Education": "PCIP13",
    "Engineering": "PCIP14",
    "Engineering Technologies And Engineering-Related Fields": "PCIP15",
    "Foreign Languages, Literatures, And Linguistics": "PCIP16",
    "Family And Consumer Sciences/Human Sciences": "PCIP19",
    "Legal Professions And Studies": "PCIP22",
    "English Language And Literature/Letters": "PCIP23",
    "Liberal Arts And Sciences, General Studies And Humanities": "PCIP24",
    "Library Science": "PCIP25",
    "Biological And Biomedical Sciences": "PCIP26",
    "Mathematics And Statistics": "PCIP27",
    "Military Technologies And Applied Sciences": "PCIP29",
    "Multi/Interdisciplinary Studies": "PCIP30",
    "Parks, Recreation, Leisure, And Fitness Studies": "PCIP31",
    "Philosophy And Religious Studies": "PCIP38",
    "Theology And Religious Vocations": "PCIP39",
    "Physical Sciences": "PCIP40",
    "Science Technologies/Technicians": "PCIP41",
    "Psychology": "PCIP42",
    "Homeland Security, Law Enforcement, Firefighting And Related Protective Services": "PCIP43",
    "Public Administration And Social Service Professions": "PCIP44",
    "Social Sciences": "PCIP45",
    "Construction Trades": "PCIP46",
    "Mechanic And Repair Technologies/Technicians": "PCIP47",
    "Precision Production": "PCIP48",
    "Transportation And Materials Moving": "PCIP49",
    "Visual And Performing Arts": "PCIP50",
    "Health Professions And Related Programs": "PCIP51",
    "Business, Management, Marketing, And Related Support Services": "PCIP52",
    "History": "PCIP54",
}

carnegieClassification_map = {
    "Not applicable": -2.0,
    "(Not classified)": 0.0,
    "Two-year, very small": 1.0,
    "Two-year, small": 2.0,
    "Two-year, medium": 3.0,
    "Two-year, large": 4.0,
    "Two-year, very large": 5.0,
    "Four-year, very small, primarily nonresidential": 6.0,
    "Four-year, very small, primarily residential": 7.0,
    "Four-year, very small, highly residential": 8.0,
    "Four-year, small, primarily nonresidential": 9.0,
    "Four-year, small, primarily residential": 10.0,
    "Four-year, small, highly residential": 11.0,
    "Four-year, medium, primarily nonresidential": 12.0,
    "Four-year, medium, primarily residential": 13.0,
    "Four-year, medium, highly residential": 14.0,
    "Four-year, large, primarily nonresidential": 15.0,
    "Four-year, large, primarily residential": 16.0,
    "Four-year, large, highly residential": 17.0,
    "Exclusively graduate/professional": 18.0,
}

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('college.db')
    conn.row_factory = sqlite3.Row
    return conn

#Parameter Descriptions
#degree: Students desired degree, must be a key present in the degree_to_pcip map defined in the route function
#carnegieClass: students desired school size/setting, must be a valid carnegie classification as outlined in the data dictionary (ex. "Four-year, small, primarily nonresidential")
@app.route('/<state>/<ZipCode>/<urbanization>/<carnegieClass>/<SATScore>/<ACTScore>/<familyIncome>/<schoolSize>/<degree>')
def index(state=None, ZipCode=None, urbanization=None,major=None,carnegieClass= None,SATScore = None, ACTScore = None,familyIncome = None,schoolSize=None,degree=None):
    
    degreePcip = degree_to_pcip[degree]
    carnegieClassNum = carnegieClassification_map[carnegieClass]
    print(carnegieClassNum)
    conn = get_db_connection()
    # Construct the query safely using f-string or .format()
    query = f'SELECT * FROM College_Data WHERE {degreePcip} != 0.0 and CCSIZSET = ? LIMIT 10'

    # Execute the query with the value parameterized
    rows = conn.execute(query, (carnegieClassNum,)).fetchall()
    result_list = [dict(row) for row in rows]
    json_string = json.dumps(result_list, indent=2)

    conn.close()
    return json_string