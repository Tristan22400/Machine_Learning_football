import http.client
import json

import numpy
import time
from datetime import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import footballdata as foo
import requests


connection = http.client.HTTPConnection('api.football-data.org')
headers = { 'X-Auth-Token': 'd8e7492561bf49edbb652f367822715f' }
connection.request('GET', '/v2/competitions/PL/matches', None, headers )
response = json.loads(connection.getresponse().read().decode())

connection = http.client.HTTPConnection('api.football-data.org')
headers = { 'X-Auth-Token': 'd8e7492561bf49edbb652f367822715f' }
connection.request('GET', '/v2/competitions/PL/teams', None, headers )
response1 = json.loads(connection.getresponse().read().decode())
Team=list(map(lambda x:x["name"],response1["teams"]))

def selection_league():
  print(foo.MatchHistory.available_leagues())
  country_name="ENG"
  number_season=2
  #input("Entrez un nombre de saison d'analyse : ")
  season="21/22"
  #input("Entrez la dernière saison à sélectionner sous le format 21/22 : ")
  int1,int2=season.split("/")
  return int1,int2,number_season,country_name

def treatement_country(country_name):
  print(foo.MatchHistory.available_leagues())
  if country_name=='ENG':
    league_name1="ENG-Premier League"
    league_name2="ENG-Championship"
  elif country_name=="ESP":
    league_name1="ESP-La Liga"
    league_name2="ESP-La Liga 2"
  elif country_name=="FRA":
    league_name1="FRA-Ligue 1"
    league_name2="FRA-Ligue 2"
  elif country_name=="GER":
    league_name1="GER-Bundesliga"
    league_name2="GER-Bundesliga 2"
  elif country_name=="ITA":
    league_name1="ITA-Serie A"
    league_name2="ITA-Serie B"
  elif country_name=="POR":
    league_name1="POR-Liga 1"
    league_name2=None
  elif country_name=="NED":
    league_name1="NED-Eredivisie"
    league_name2=None
  elif country_name=="SCO":
    league_name1="SCO-Division 1"
    league_name2="SCO-Division 2"
  elif country_name=="BEL":
    league_name1="BEL-Jupiler League"
    league_name2=None
  elif country_name=="GRE":
    league_name1="GRE-Ethniki Katigoria"
    league_name2=None
  elif country_name=="TUR":
    league_name1="TUR-Ligi 1"
    league_name2=None
  else:
    return "le championnat choisie n'est pas étudié par notre site"
  return league_name1,league_name2

def load_data():
  int1,int2,number_season,country_name=selection_league()
  league_name1,league_name2=treatement_country(country_name)
  League=[]
  for k in range(int(number_season)):
    int1bis=int(int1)-k
    int2bis=int(int2)-k
    season=str(int1bis)+str(int2bis)
    print(season)
    League.append(foo.MatchHistory(league_name1, season).read_games())
    League.append(foo.MatchHistory(league_name2, season).read_games())

  Match_Data_Original=pd.concat(League)
  Match_Data = Match_Data_Original.iloc[:,  [0,1,2,3,4,5]]
  date_time_obj=datetime.date.today()
  date_time_obj=pd.to_datetime(date_time_obj)
  Match_Data['date']=pd.to_datetime(Match_Data['date'], format='%y-%m-%d')
  Match_Data["date"] = Match_Data["date"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
  Match_Data["t"] = (date_time_obj - Match_Data["date"]).astype('timedelta64[D]')
  Match_Data.columns = ['Date','Time','HomeTeam','AwayTeam','FTHG','FTAG','t']
  return Match_Data
