from tabpy.tabpy_tools.client import Client
import json
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

client = Client('http://localhost:9004/')

def retrieveSchools(userZIP, searchRadius, desiredUrbanization, desiredSchoolSize, familyIncome,userSAT, userACT,
                    major,tuitionBudget, yearsRepay,weight1,weight2,weight3,school_name,stabbr,latitude, longitude,tuition_in,
                    tuition_out,ccsizset,curroper, locale,md_earn10,md_earn11,md_earn6, md_earn7,md_earn8, md_earn9,
                    npt41_priv, npt41_pub, npt42_priv,npt42_pub, npt43_priv, npt43_pub,npt44_priv, npt44_pub,npt45_priv,
                    npt45_pub,PCIP01,PCIP03,PCIP04,PCIP05,PCIP09,PCIP10,PCIP11,PCIP12,PCIP13,PCIP14,PCIP15,PCIP16,
                    PCIP19,PCIP22,PCIP23,PCIP24,PCIP25,PCIP26,PCIP27,PCIP29,PCIP30,PCIP31,PCIP38,PCIP39,PCIP40,PCIP41,
                    PCIP42,PCIP43,PCIP44,PCIP45,PCIP46,PCIP47,PCIP48,PCIP49,PCIP50,PCIP51,PCIP52,PCIP54,sat_avg,actcmmid):


    ## Preprocessing:

    # Recreate the df:
    df = pd.DataFrame({
        "school_name": school_name, "stabbr": stabbr, "latitude": latitude, "longitude": longitude, "tuition_in": tuition_in,
        "tuition_out": tuition_out, "ccsizset": ccsizset, "curroper": curroper, "locale": locale, "md_earn10": md_earn10,
        "md_earn11": md_earn11, "md_earn6": md_earn6, "md_earn7": md_earn7, "md_earn8": md_earn8, "md_earn9": md_earn9,
        "npt41_priv": npt41_priv, "npt41_pub": npt41_pub, "npt42_priv": npt42_priv, "npt42_pub": npt42_pub, "npt43_priv": npt43_priv,
        "npt43_pub": npt43_pub, "npt44_priv": npt44_priv, "npt44_pub": npt44_pub, "npt45_priv": npt45_priv, "npt45_pub": npt45_pub,
        "PCIP01": PCIP01, "PCIP03": PCIP03, "PCIP04": PCIP04, "PCIP05": PCIP05, "PCIP09": PCIP09, "PCIP10": PCIP10,
        "PCIP11": PCIP11, "PCIP12": PCIP12, "PCIP13": PCIP13, "PCIP14": PCIP14, "PCIP15": PCIP15, "PCIP16": PCIP16,
        "PCIP19": PCIP19, "PCIP22": PCIP22, "PCIP23": PCIP23, "PCIP24": PCIP24, "PCIP25": PCIP25, "PCIP26": PCIP26,
        "PCIP27": PCIP27, "PCIP29": PCIP29, "PCIP30": PCIP30, "PCIP31": PCIP31, "PCIP38": PCIP38, "PCIP39": PCIP39,
        "PCIP40": PCIP40, "PCIP41": PCIP41, "PCIP42": PCIP42, "PCIP43": PCIP43, "PCIP44": PCIP44, "PCIP45": PCIP45,
        "PCIP46": PCIP46, "PCIP47": PCIP47, "PCIP48": PCIP48, "PCIP49": PCIP49, "PCIP50": PCIP50, "PCIP51": PCIP51,
        "PCIP52": PCIP52, "PCIP54": PCIP54, "sat_avg": sat_avg, "actcmmid": actcmmid
    })

    ## Data cleaning and prep:

    # Filter out closed institutions and graduate schools, 2-year schools
    #df = df[(~df['ccsizset'].isin([1, 2, 3, 4, 5, 18])) & (df['curroper'] == 1)]
    df.loc[(~df['ccsizset'].isin([1, 2, 3, 4, 5, 18])) & (df['curroper'] == 1), 'score'] = 999999

    # Change values of Carnegie Class (1=very small, 2=small, 3=medium, 4=large, 4.5=very large
    df['ccsizset'] = df['ccsizset'].replace(
        {6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 5: 4.5, 15: 4, 16: 4, 17: 4})
    # Change values of Locale city size (1=Distant Rural, 4 = Distant Town, 7=Small Suburb, 8=Midsize Suburb, 9= Large Suburb, 12=Small City, 13=Midsize City, 14=Large City)
    df['locale'] = df['locale'].replace(
        {11: 14, 12: 13, 13: 12, 21: 9, 22: 8, 23: 7, 31: 4, 32: 4, 33: 4, 41: 1, 42: 1, 43: 1})

    # Inflate tuition to 2025
    df['tuition_in'] *= 1.1006
    df['tuition_out'] *= 1.1006

    # Inflate earnings from 2021/2022 dollars to 2025 dollars
    df['md_earn6'] *= 1.1006
    df['md_earn7'] *= 1.1887
    df['md_earn8'] *= 1.1006
    df['md_earn9'] *= 1.1887
    df['md_earn10'] *= 1.1006
    df['md_earn11'] *= 1.1887

    # Combine net price column with either public and private NPT and inflate to 2025 dollars
    df['npt41'] = df['npt41_pub'].fillna(df['npt41_priv'])
    df['npt42'] = df['npt41_pub'].fillna(df['npt42_priv'])
    df['npt43'] = df['npt41_pub'].fillna(df['npt43_priv'])
    df['npt44'] = df['npt41_pub'].fillna(df['npt44_priv'])
    df['npt45'] = df['npt41_pub'].fillna(df['npt45_priv'])

    df = df.drop(columns = ['npt41_pub', 'npt41_priv', 'npt42_pub', 'npt42_priv', 'npt43_pub', 'npt43_priv',
                            'npt44_pub', 'npt44_priv', 'npt45_pub', 'npt45_priv'])

    df['npt41'] *= 1.1006
    df['npt42'] *= 1.1006
    df['npt43'] *= 1.1006
    df['npt44'] *= 1.1006
    df['npt45'] *= 1.1006

    # Impute NAs using median
    for col in ['sat_avg', 'actcmmid', 'ccsizset', 'locale', 'tuition_in', 'tuition_out', 'md_earn6', 'md_earn7',
                'md_earn8', 'md_earn9', 'md_earn10', 'md_earn11']:
        df[col] = df[col].fillna(df[col].median())

    # Run polynomial model to get the projected earnings for years 2-20 after graduating
    df['earnings'] = df.apply(perform_polynomial, axis=1)

    ## Process user inputs:
    urban_dict = {"Distant Rural": 1, "Distant Town": 4, "Small Suburb": 7, "Midsize Suburb": 8, "Large Suburb":9, "Small City":12, "Midsize City":13, "Large City":14}
    desiredUrbanization = urban_dict[desiredUrbanization[0]]

    cc_dict = {"Very Small":1, "Small":2, "Medium":3, "Large":4}
    desiredSchoolSize = cc_dict[desiredSchoolSize[0]]

    userSAT = userSAT[0]
    userACT = userACT[0]
    tuitionBudget = tuitionBudget[0]
    searchRadius = searchRadius[0]
    userZIP = userZIP[0]
    familyIncome = familyIncome[0]
    major = major[0]
    yearsRepay = yearsRepay[0]
    weight1 = weight1[0]
    weight2 = weight2[0]
    weight3= weight3[0]

    # Normalize needed algorithm columns and user inputs
    user_norm = {'sat_avg': userSAT, 'actcmmid': userACT, 'ccsizset':desiredSchoolSize, 'locale':desiredUrbanization,
                 'tuition_in': tuitionBudget, 'tuition_out': tuitionBudget,'miles_away': searchRadius}
    cols = ['sat_avg', 'actcmmid', 'ccsizset', 'locale', 'tuition_in']

    for col in cols:
        normalize_col_and_user(df, user_norm, col)




    ## Algorithm calculations:

    # Radius calculations
    if userZIP is None:
        df['radius_e'] = 0
        userStabbr = ''
    else:
        # Get user coordinates based on inputted zip code
        geo = Nominatim(user_agent="xyz")
        location = geo.geocode({'postalcode': userZIP, 'country': 'US'})
        desired_lat = location.latitude
        desired_lon = location.longitude
        userStabbr = location.raw.get('state_code')

        df['latitude'] = df['latitude'].fillna(0.0)
        df['longitude'] = df['longitude'].fillna(0.0)

        df['miles_away'] = df.apply(lambda row: geodesic((desired_lat, desired_lon), (row['latitude'], row['longitude'])).miles, axis=1)

        # Normalize radius and miles_away
        normalize_col_and_user(df, user_norm, 'miles_away')

        df['radius_e'] = np.where(df['miles_away'] > searchRadius, (df['miles_away'] - searchRadius) ** 2, 0)

    # desiredUrbanization calculations
    if desiredUrbanization is None:
        df['urban_e'] = 0
    else:
        df['urban_e'] = (user_norm['locale'] - df['locale']) ** 2

    # SAT score calculation
    if userSAT is None:
        df['sat_e'] = 0
    else:
        df['sat_e'] = np.where(df['sat_avg'] > user_norm['sat_avg'], (df['sat_avg'] - user_norm['sat_avg']) **2, 0)

    # ACT score calculation
    if userACT is None:
        df['act_e'] = 0
    else:
        df['act_e'] = np.where(df['actcmmid'] > user_norm['actcmmid'], (df['actcmmid'] - userACT) ** 2, 0)

    # Major calculations (converts major to corresponding column using get_major_col)
    if major is None:
        df['major_e'] = 0
    else:
        major_col = get_major_col(major)
        df['major_e'] = (1 - df[major_col]) ** 2

    # School size calculations (ignore 0 and -2)
    if desiredSchoolSize is None or df['ccsizset'].isin([0,-2]).any():
        df['cc_e'] = 0
    else:
        df['cc_e'] = (user_norm['ccsizset'] - df['ccsizset']) ** 2

    # Tuition calculations
    if tuitionBudget is None:
        df['tuition_e'] = 0
    else:
        df['tuition_e'] = np.where(df['stabbr'] == userStabbr,
            np.where(df['tuition_in'] > tuitionBudget, (user_norm['tuition_in'] - df['tuition_in']) ** 2, 0),
            np.where(df['tuition_out'] > tuitionBudget, (user_norm['tuition_out'] - df['tuition_out']) ** 2, 0))

    # Years to repayment calculations
    if yearsRepay is None:
        df['repay_e'] = 0
    else:
        # Calculate years to repayment
        if familyIncome == '$0-30K':
            df['coa'] = df['npt41'] * 4
        elif familyIncome == '$30K-$48K':
            df['coa'] = df['npt42'] * 4
        elif familyIncome == '$48K-$75K':
            df['coa'] = df['npt43'] * 4
        elif familyIncome == '$75K-$110K':
            df['coa'] = df['npt44'] * 4
        else:
            df['coa'] = df['npt45'] * 4

        # Calculate years to repay given total coa and projected earnings
        df['years'] = df.apply(lambda row: compute_years_to_repay(row['earnings'], row['coa']), axis=1)

        # Normalize years and yearsRepay
        df['years'] = (df['years'] - 1) / 20
        yearsRepay = (yearsRepay - 1) / 20

        df['repay_e'] = np.where(df['years'] > yearsRepay, (yearsRepay - df['years']) ** 2,0)


    # Perform the actual Euclidean distance calculation:
    weight_cols = {
        'Desired Major': 'major_e',
        'Desired School Size': 'cc_e',
        'Search Radius': 'radius_e',
        'Desired Degree of Urbanization': 'urban_e',
        'User SAT Score': 'sat_e',
        'User ACT Score': 'act_e',
        'Tuition Budget': 'tuition_e',
        'Desired Years to Repay': 'repay_e'
    }
    weights = {key: 1 for key in weight_cols}
    # Add extra weighting to user's top factors
    weights[weight1] = 4
    weights[weight2] = 3
    weights[weight3] = 2

    df['score'] = 0
    # Add each column score * column weight to the total score
    for key, col in weight_cols.items():
        df['score'] += df[col] * weights[key]

    df['score'] = np.sqrt(df['score'])

    df['score'] = df['score'].fillna(999999)

    # return score for each school
    return df['score'].to_list()


def perform_polynomial(row):
    """
    Perform the whole polynomial regression to get years 2-20 of earnings after graduating
    :param row: row from the dataframe
    :return: list of earnings from year 2-20
    """
    # Get only the earning rows
    # row_earn = row[
    #     ['md_earn6', 'md_earn7', 'md_earn8', 'md_earn9', 'md_earn10', 'md_earn11']]
    # earn_cols = [f'md_earn{i}' for i in range(6, 12)]
    #
    # # Check if empty and populate empty earnings
    # if row_earn.empty:
    #     return None
    # for i, col in enumerate(earn_cols):
    #     if pd.isnull(row_earn[col]):
    #         if i == 0:
    #             row_earn[col] = row_earn[earn_cols].bfill(axis=1).iloc[:, 0]
    #         else:
    #             row_earn[col] = row_earn[earn_cols].fill(axis=1).iloc[:, 0]
    earn_cols = [f'md_earn{i}' for i in range(6, 12)]
    row_earn = row[earn_cols]

    # Convert to DataFrame (1 row) so you can apply axis=1 operations
    row_earn_df = pd.DataFrame([row_earn])

    # Fill missing values across the row (horizontally)
    row_earn_filled = row_earn_df.bfill(axis=1).ffill(axis=1)

    # Now use the filled values for your polynomial regression
    earnings = row_earn_filled.values.flatten().tolist()

    if all(pd.isnull(earnings)):
        return None  # Still all NaNs? Return None
    else:
        # Return filled earnings list for regression
        return earnings

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

def normalize_col_and_user(df, user_norm, col):
    """
    Normalize the column using min-max normalization and updates the user's input with the same scale
    :param df: college pandas dataframe
    :param user_norm: dictionary of column name: user variable
    :param col: column name to normalize
    """
    cmin = df[col].min(skipna=True)
    cmax = df[col].max(skipna=True)

    if cmax == cmin:
        df[col] = 0.5  # or 0.0, depending on your use case
        user_norm[col] = 0.5
    else:
        df[col] = (df[col] - cmin) / (cmax - cmin)
        user_norm[col] = (user_norm[col] - cmin) / (cmax - cmin)

def compute_years_to_repay(earnings, coa):
    if not earnings or not isinstance(earnings, (list, tuple)):
        return 21

    coa_left = coa
    for i, rpmt in enumerate(earnings):
        coa_left -= rpmt * 0.1
        if coa_left <= 0:
            return i + 2
    # If still haven't broken even, set years to 21
    return 21





client.deploy('retrieveSchools',
              retrieveSchools,
              'Returns a list of dictionaries with each dictionary containing information about a school',
              override=True)