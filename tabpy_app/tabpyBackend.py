from tabpy.tabpy_tools.client import Client
import json
import sqlite3
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

client = Client('http://localhost:9004/')

def retrieveSchools(userZIP, searchRadius, desiredUrbanization, desiredSchoolSize, familyIncome,userSAT, userACT,
                    major,tuitionBudget, yearsRepay,weights,school_name,stabbr,latitude, longitude,tuition_in,
                    tuition_out,ccsizset,curroper, locale,md_earn10,md_earn11,md_earn6, md_earn7,md_earn8, md_earn9,
                    npt41_priv, npt41_pub, npt42_priv,npt42_pub, npt43_priv, npt43_pub,npt44_priv, npt44_pub,npt45_priv,
                    npt45_pub,Pcip01,Pcip03,Pcip04,Pcip05,Pcip09,Pcip10,Pcip11,Pcip12,Pcip13,Pcip14,Pcip15,Pcip16,
                    Pcip19,Pcip22,Pcip23,Pcip24,Pcip25,Pcip26,Pcip27,Pcip29,Pcip30,Pcip31,Pcip38,Pcip39,Pcip40,Pcip41,
                    Pcip42,Pcip43,Pcip44,Pcip45,Pcip46,Pcip47,Pcip48,Pcip49,Pcip50,Pcip51,Pcip52,Pcip54,sat_avg,act_cmmid):


    #algorithm here

    # Preprocessing:

    # Recreate the df:
    df = pd.DataFrame({
        "school_name": school_name,
        "stabbr": stabbr,
        "latitude": latitude,
        "longitude": longitude,
        "tuition_in": tuition_in,
        "tuition_out": tuition_out,
        "ccsizset": ccsizset,
        "curroper": curroper,
        "locale": locale,
        "md_earn10": md_earn10,
        "md_earn11": md_earn11,
        "md_earn6": md_earn6,
        "md_earn7": md_earn7,
        "md_earn8": md_earn8,
        "md_earn9": md_earn9,
        "npt41_priv": npt41_priv,
        "npt41_pub": npt41_pub,
        "npt42_priv": npt42_priv,
        "npt42_pub": npt42_pub,
        "npt43_priv": npt43_priv,
        "npt43_pub": npt43_pub,
        "npt44_priv": npt44_priv,
        "npt44_pub": npt44_pub,
        "npt45_priv": npt45_priv,
        "npt45_pub": npt45_pub,
        "Pcip01": Pcip01,
        "Pcip03": Pcip03,
        "Pcip04": Pcip04,
        "Pcip05": Pcip05,
        "Pcip09": Pcip09,
        "Pcip10": Pcip10,
        "Pcip11": Pcip11,
        "Pcip12": Pcip12,
        "Pcip13": Pcip13,
        "Pcip14": Pcip14,
        "Pcip15": Pcip15,
        "Pcip16": Pcip16,
        "Pcip19": Pcip19,
        "Pcip22": Pcip22,
        "Pcip23": Pcip23,
        "Pcip24": Pcip24,
        "Pcip25": Pcip25,
        "Pcip26": Pcip26,
        "Pcip27": Pcip27,
        "Pcip29": Pcip29,
        "Pcip30": Pcip30,
        "Pcip31": Pcip31,
        "Pcip38": Pcip38,
        "Pcip39": Pcip39,
        "Pcip40": Pcip40,
        "Pcip41": Pcip41,
        "Pcip42": Pcip42,
        "Pcip43": Pcip43,
        "Pcip44": Pcip44,
        "Pcip45": Pcip45,
        "Pcip46": Pcip46,
        "Pcip47": Pcip47,
        "Pcip48": Pcip48,
        "Pcip49": Pcip49,
        "Pcip50": Pcip50,
        "Pcip51": Pcip51,
        "Pcip52": Pcip52,
        "Pcip54": Pcip54,
        "sat_avg": sat_avg,
        "act_cmmid": act_cmmid
    })


    # Filter out closed institutions and graduate schools, 2-year schools
    df = df[(df['ccsizset'] not in (1, 2, 3, 4, 5, 18)) & (df['CURROPER'] == 1)]

    # Change values of Carnegie Class (1=very small, 2=small, 3=medium, 4=large, 4.5=very large
    df['ccsizset'] = df['ccsizset'].replace(
        {6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 5: 4.5, 15: 4, 16: 4, 17: 4})
    # Change values of Locale city size (1=Distant Rural, 4 = Distant Town, 7=Small Suburb, 8=Midsize Suburb, 9= Large Suburb, 12=Small City, 13=Midsize City, 14=Large City)
    df['locale'] = df['locale'].replace(
        {11: 14, 12: 13, 13: 12, 21: 9, 22: 8, 23: 7, 31: 4, 32: 4, 33: 4, 41: 1, 42: 1, 43: 1})

    # Inflate tuition to 2025
    df['tuition_in'] *= 1.1006
    df['TUITION_OUT'] *= 1.1006

    # Inflate earnings from 2021/2022 dollars to 2025 dollars
    df['md_earn6'] *= 1.1006
    df['md_earn7'] *= 1.1887
    df['md_earn8'] *= 1.1006
    df['md_earn9'] *= 1.1887
    df['md_earn10'] *= 1.1006
    df['md_earn11'] *= 1.1887

    # Get the projected earnings for years 2-20 after graduating
    df['earnings'] = df.apply(perform_polynomial, axis=1)

    # Coalesce coa for public and private (might need to coalesce with more)
    df['npt41_pub'] = df['npt41_pub'].fillna(df['npt41_priv'])
    df['npt42_pub'] = df['npt41_pub'].fillna(df['npt42_priv'])
    df['npt43_pub'] = df['npt41_pub'].fillna(df['npt43_priv'])
    df['npt44_pub'] = df['npt41_pub'].fillna(df['npt44_priv'])
    df['npt45_pub'] = df['npt41_pub'].fillna(df['npt45_priv'])

    # Code that needs to be run each time

    # Radius calculations
    if userZIP is None:
        df['radius_e'] = 0
    else:
        # Get user coordinates based on inputted zip code
        geo = Nominatim()
        location = geo.geocode({'postalcode': userZIP, 'country': 'US'})
        desired_lat = location.latitude
        desired_lon = location.longitude

        df['miles_away'] = geodesic((desired_lat, desired_lon), (df['latitude'], df['longitude']))

        df['radius_e'] = (df['miles_away'] - searchRadius) ** 2 if df['miles_away'] > searchRadius else 0

    # desiredUrbanization calculations
    if desiredUrbanization is None:
        df['urban_e'] = 0
    else:
        df['urban_e'] = (desiredUrbanization - df['locale']) ** 2

    # SAT score calculation
    if userSAT is None:
        df['sat_e'] = 0
    else:
        df['sat_e'] = (df['sat_avg'] - userSAT) if df['sat_avg'] > userSAT else 0

    # ACT score calculation
    if userACT is None:
        df['act_e'] = 0
    else:
        df['act_e'] = (df['act_cmmid'] - userACT) ** 2 if df['act_cmmid'] > userACT else 0

    # Major calculations (assumes major will come in as a number from the mapping)
    if major is None:
        df['major_e'] = 0
    else:
        major_col = get_major_col(major)
        df['major_e'] = (1 - df[major_col]) ** 2

    # School size calculations (ignore 0 and -2)
    if desiredSchoolSize is None or df['ccsizset'] in (0,-2):
        df['cc_e'] = 0
    else:
        df['cc_e'] = (desiredSchoolSize - df['ccsizset']) ** 2

    # Tuition calculations
    if tuitionBudget is None:
        df['tuition_e'] = 0
    else:
        df['tuition_e'] = (tuitionBudget - df['tuition_in']) ** 2 if df['tuition_in'] > tuitionBudget else 0

    # Years to repayment calculations
    if yearsRepay is None:
        df['repay_e'] = 0
    else:
        # Calculate years to repayment
        if familyIncome == '0-30000':
            coa = df['npt41_pub'] * 4
        elif familyIncome == '30001-48000':
            coa = df['npt42_pub'] * 4
        elif familyIncome == '48001-75000':
            coa = df['npt43_pub'] * 4
        elif familyIncome == '75001-111000':
            coa = df['npt44_pub'] * 4
        else:
            coa = df['npt45_pub'] * 4

        # Calculate years to repay given total coa and projected earnings
        for i, rpmt in enumerate(df['earnings']):
            coa -= (rpmt[i] * .1)
            if coa <= 0:
                df['years'] = i + 2
                break
        # If still haven't broken even, set years to 21
        if coa > 0:
            df['years'] = 21

        df['repay_e'] = (yearsRepay - df['years']) ** 2 if df['years'] > yearsRepay else 0


    # Perform the actual Euclidean distance calculation:
    cols = [df['major_e'], df['cc_e'], df['radius_e'], df['urban_e'], df['sat_e'], df['act_e'], df['tuition_e'], df['repay_e']]
    #weights = weights
    for i in range(len(cols)):
        df['score'] += cols[i] * weights[i]

    df['score'] = sqrt(df['score'])

    # return score for each school

    return df['score']


def perform_polynomial(row):
    """
    Perform the whole polynomial regression to get years 2-20 of earnings after graduating
    :param row: row from the dataframe
    :return: list of earnings from year 2-20
    """
    # Get only the earning rows
    row_earn = row[
        ['md_earn6', 'md_earn7', 'md_earn8', 'md_earn9', 'md_earn10', 'md_earn11']]
    earn_cols = [f'MD_EARN_WNE_P{i}' for i in range(6, 12)]

    # Check if empty and populate empty earnings
    if row_earn.empty:
        return None
    for i, col in enumerate(earn_cols):
        if pd.isnull(row_earn[col]):
            if i == 0:
                row_earn[col] = row_earn[earn_cols].bfill(axis=1).iloc[:, 0]
            else:
                row_earn[col] = row_earn[earn_cols].fill(axis=1).iloc[:, 0]

    # Train the polynomial model
    X = np.arange(2, 8).reshape(-1, 1)
    y = row_earn.to_numpy().reshape(-1, 1)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_lin = LinearRegression()
    poly_lin.fit(X_poly, y)

    # Predict results based on future X
    X_future = np.arange(8, 21).reshape(-1, 1)
    X_poly_future = poly.transform(X_future)

    y_pred = poly_lin.predict(X_poly_future)

    ys = y.flatten().tolist() + y_pred.flatten().tolist()
    # Convert predictions back into a dataframe
    # df_pred = pd.DataFrame(y_pred.tolist(), columns=[f'MD_EARN_WNE_P{i}' for i in range(12,21)])

    return ys

def get_major_col(major):
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
    return degree_to_pcip[major]

client.deploy('retrieveSchools',
              retrieveSchools,
              'Returns a list of dictionaries with each dictionary containing information about a school',
              override=True)