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

def retrieveSchools(df, state=None, UserZipCode=None, desiredZipCode=None, maximumRadius=None, urbanization=None,
                    major=None, SATScore=None, ACTScore=None, familyIncome=None, schoolSize=None, tuitionBudget=None, yearsToRepay=None):
    #algorithm here

    # Preprocessing:

    # Filter out closed institutions and graduate schools
    df = df[df['CCSIZSET'] != 18 & df['CURROPER'] == 1]

    # Change values of Carnegie Class (1=very small, 2=small, 3=medium, 4=large, 4.5=very large
    df['CCSIZSET'] = df['CCSIZSET'].replace(
        {0: -2, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 5: 4.5, 15: 4, 16: 4, 17: 4})
    # Change values of Locale city size (1=Distant Rural, 4 = Distant Town, 7=Small Suburb, 8=Midsize Suburb, 9= Large Suburb, 12=Small City, 13=Midsize City, 14=Large City)
    df['LOCALE'] = df['LOCALE'].replace(
        {11: 14, 12: 13, 13: 12, 21: 9, 22: 8, 23: 7, 31: 4, 32: 4, 33: 4, 41: 1, 42: 1, 43: 1})

    # Inflate tuition to 2025
    df['TUITION_IN'] *= 1.1006
    df['TUITION_OUT'] *= 1.1006

    # Inflate earnings from 2021/2022 dollars to 2025 dollars
    df['MD_EARN_WNE_P6'] *= 1.1006
    df['MD_EARN_WNE_P7'] *= 1.1887
    df['MD_EARN_WNE_P8'] *= 1.1006
    df['MD_EARN_WNE_P9'] *= 1.1887
    df['MD_EARN_WNE_P10'] *= 1.1006
    df['MD_EARN_WNE_P11'] *= 1.1887

    # Get the projected earnings for years 2-20 after graduating
    df['earnings'] = df.apply(perform_polynomial, axis=1)

    # Coalesce coa for public and private (might need to coalesce with more)
    df['NPT41_PUB'] = df['NPT41_PUB'].fillna(df['NPT41_PRIV'])
    df['NPT42_PUB'] = df['NPT41_PUB'].fillna(df['NPT42_PRIV'])
    df['NPT43_PUB'] = df['NPT41_PUB'].fillna(df['NPT43_PRIV'])
    df['NPT44_PUB'] = df['NPT41_PUB'].fillna(df['NPT44_PRIV'])
    df['NPT45_PUB'] = df['NPT41_PUB'].fillna(df['NPT45_PRIV'])

    # Code that needs to be run each time

    # Radius calculations
    if desiredZipCode is None:
        df['radius_e'] = 0
    else:
        # Get user coordinates based on inputted zip code
        # nomi = pgeocode.Nominatim('us')
        # coord = nomi.query_postal_code(desiredZipCode)
        # desired_lat = coord['latitude']
        # desired_lon = coord['longitude']

        geo = Nominatim()
        location = geo.geocode({'postalcode': desiredZipCode, 'country': 'US'})
        desired_lat = location.latitude
        desired_lon = location.longitude

        df['miles_away'] = geodesic((desired_lat, desired_lon), (df['LATITUDE'], df['LONGITUDE']))

        df['radius_e'] = (df['miles_away'] - maximumRadius) ** 2 if df['miles_away'] > maximumRadius else 0

    # Urbanization calculations
    if urbanization is None:
        df['urban_e'] = 0
    else:
        df['urban_e'] = (urbanization - df['LOCALE']) ** 2

    # SAT score calculation
    if SATScore is None:
        df['sat_e'] = 0
    else:
        df['sat_e'] = (df['SAT_AVG'] - SATScore) if df['SAT_AVG'] > SATScore else 0

    # ACT score calculation
    if ACTScore is None:
        df['act_e'] = 0
    else:
        df['act_e'] = (df['ACTCMMID'] - ACTScore) ** 2 if df['ACTCMMID'] > ACTScore else 0

    # Major calculations
    if major is None:
        df['major_e'] = 0
    else:
        df['major_e'] = (1 - df[f'PCIP{major}']) ** 2

    # School size calculations
    if schoolSize is None:
        df['cc_e'] = 0
    else:
        df['cc_e'] = (schoolSize - df['CCSIZSET']) ** 2

    # Tuition calculations (inflate from 2022 to 2025 dollars
    if tuitionBudget is None:
        df['tuition_e'] = 0
    else:
        df['tuition_e'] = (tuitionBudget - df['TUITION_IN']) ** 2 if df['TUITION_IN'] > tuitionBudget else 0

    # Years to repayment calculations
    if yearsToRepay is None:
        df['repay_e'] = 0
    else:
        # Calculate years to repayment
        if familyIncome == '0-30000':
            coa = df['NPT41_PUB'] * 4
        elif familyIncome == '30001-48000':
            coa = df['NPT42_PUB'] * 4
        elif familyIncome == '48001-75000':
            coa = df['NPT43_PUB'] * 4
        elif familyIncome == '75001-111000':
            coa = df['NPT44_PUB'] * 4
        else:
            coa = df['NPT45_PUB'] * 4

        # Calculate years to repay given total coa and projected earnings
        for i, rpmt in enumerate(df['earnings']):
            coa -= (rpmt[i] * .1)
            if coa <= 0:
                df['years'] = i + 2
                break
        # If still haven't broken even, set years to 21
        if coa > 0:
            df['years'] = 21

        df['repay_e'] = (yearsToRepay - df['years']) ** 2 if df['years'] > yearsToRepay else 0

    """ Different format, cleaner
    calculations = {
    'urban_e': lambda df: 0 if urbanization is None else (urbanization - df['LOCALE']) ** 2,
    'sat_e': lambda df: 0 if SATScore is None else (df['SAT_AVG'] - SATScore if df['SAT_AVG'] > SATScore else 0),
    'grad_e': lambda df: 0 if grad_rate is None else abs(df['GRAD_RATE'] - grad_rate),
    }
    for col, func in calculations.items():
        df[col] = func(df)
    """

    # Perform the actual Euclidean distance calculation (NEED TO UPDATE THIS:
    cols = df['major_e'], df['cc_e'], df['radius_e'], df['urban_e'], df['test_e']
    for col in cols:
        df['score'] += df['major_weight'] * col

    df['score'] = sqrt(df['score'])

    # return score for each school in an array

    return df['score']


def perform_polynomial(row):
    """
    Perform the whole polynomial regression to get years 2-18 of earnings after graduating
    :param row:
    :return:
    """
    # Get only the earning rows
    row_earn = row[
        ['MD_EARN_WNE_P6', 'MD_EARN_WNE_P7', 'MD_EARN_WNE_P8', 'MD_EARN_WNE_P9', 'MD_EARN_WNE_P10', 'MD_EARN_WNE_P11']]
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



client.deploy('retrieveSchools',
              retrieveSchools,
              'Returns a list of dictionaries with each dictionary containing information about a school',
              override=True)