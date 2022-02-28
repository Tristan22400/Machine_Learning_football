#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 21:42:15 2020

@author: philipwinchester
"""
import numpy as np
import pandas as pd
import json
import datetime
from collections import defaultdict
from scipy.stats import poisson
import matplotlib.pyplot as plt

def NMod(Vector,n=1):
    # Takes vector and returns n*mod
    return n*np.sqrt(np.inner(Vector, Vector))

def tau(x,y,lamb,mu,rho):
    # Defining tau function
    if x == 0 and y == 0:
        return 1 - (lamb*mu*rho)
    elif x == 0 and y == 1:
        return 1 + (lamb*rho)
    elif x == 1 and y == 0:
        return 1 + (mu*rho)
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1

def phi(t,eps = 0):
    # Define the weight function
    return np.exp(-eps*t)


def MatchLL(x,y,ai, aj, bi, bj, gamma, rho, t,eps):
    # A function which calculates the log likelihood of some game
    lamb = ai*bj*gamma
    mu = aj*bi
    return phi(t,eps)*(np.log(tau(x, y, lamb, mu, rho)) - lamb + x*np.log(lamb) - mu + y*np.log(mu))

def LL(Match_Data, Parameters, Teams):
  # Function which calculates the LL for all the games
  # This can also be made quicker if we avoid the for loop
  LL = 0

  # Fixing gamma and rho, as these are constant for all games
  gamma = Parameters[2*len(Teams)]
  rho = Parameters[2*len(Teams)+1]
  eps = Parameters[2*len(Teams)+2]

  for k in range(0,len(Match_Data.index)):
    # Finding index for the home and away team
    IndexHome = Teams.index(Match_Data['HomeTeam'][k])
    IndexAway = Teams.index(Match_Data['AwayTeam'][k])

    # Finding relevant Parameters and other variables
    ai = Parameters[IndexHome]
    aj = Parameters[IndexAway]
    bi = Parameters[IndexHome + len(Teams)]
    bj = Parameters[IndexAway + len(Teams)]
    t = Match_Data['t'][k]
    x =  Match_Data['FTHG'][k]
    y =  Match_Data['FTAG'][k]

    #Adding the LL from game k to the total
    LL = LL + MatchLL(x,y,ai, aj, bi, bj, gamma, rho, t, eps)

  return LL

# Functions for alpha derivative are below

def GradAlphaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t,eps)*bj*(-gamma-mu*gamma*rho/(1-lamb*mu*rho))

def GradAlphaHomeZeroOne(ai, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  return phi(t,eps)*bj*(-gamma+gamma*rho/(1+lamb*rho))

def GradAlphaHomeNotZero(ai, bj, gamma, x,t,eps):
  return phi(t,eps)*(x/ai-bj*gamma)

def GradAlphaHome(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the home attacking strenth from some game
  if x == 0 and y == 0:
    return GradAlphaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 0 and y == 1:
    return GradAlphaHomeZeroOne(ai, bj, gamma, rho,t,eps)
  else:
    return GradAlphaHomeNotZero(ai, bj, gamma, x,t,eps)

def GradAlphaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t,eps)*bi*(-1-lamb*rho/(1-lamb*mu*rho))

def GradAlphaAwayOneZero(aj, bi, rho,t,eps):
  mu = aj*bi
  return phi(t,eps)*bi*(-1+rho/(1+mu*rho))


def GradAlphaAwayNotZero(aj, bi, y,t,eps):
  return phi(t,eps)*(y/aj-bi)

def GradAlphaAway(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the away attacking strenth from some game
  if x == 0 and y == 0:
    return GradAlphaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 1 and y == 0:
    return GradAlphaAwayOneZero(aj, bi, rho,t,eps)
  else:
    return GradAlphaAwayNotZero(aj, bi, y,t,eps)

# Functions for beta derivative are below

def GradBetaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t,eps)*aj*(-1-lamb*rho/(1-lamb*mu*rho))

def GradBetaHomeOneZero(aj, bi, rho,t,eps):
  mu = aj*bi
  return phi(t,eps)*aj*(-1+rho/(1+mu*rho))

def GradBetaHomeNotZero(aj, bi, y,t,eps):
  return phi(t,eps)*(y/bi-aj)

def GradBetaHome(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the home defense strenth from some game
  if x == 0 and y == 0:
    return GradBetaHomeZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 1 and y == 0:
    return GradBetaHomeOneZero(aj, bi, rho,t,eps)
  else:
    return GradBetaHomeNotZero(aj, bi, y,t,eps)

def GradBetaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  mu = aj*bi
  return phi(t,eps)*ai*(-gamma-mu*gamma*rho/(1-lamb*mu*rho))


def GradBetaAwayZeroOne(ai, bj, gamma, rho,t,eps):
  lamb = ai*bj*gamma
  return phi(t,eps)*ai*(-gamma+rho*gamma/(1+lamb*rho))

def GradBetaAwayNotZero(ai, bj, gamma,x,t,eps):
  return phi(t,eps)*(x/bj-ai*gamma)

def GradBetaAway(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the away defense strenth from some game
  if x == 0 and y == 0:
    return GradBetaAwayZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 0 and y == 1:
    return GradBetaAwayZeroOne(ai, bj,gamma, rho,t,eps)
  else:
    return GradBetaAwayNotZero(ai, bj, gamma, x,t,eps)

# Functions for gamma derivative are below

def GradGammaZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamd = ai*bj*gamma
  mu = aj*bi
  return phi(t,eps)*ai*bj*(-1-mu*rho/(1-lamd*mu*rho))

def GradGammaZeroOne(ai, bj, gamma, rho,t,eps):
  lamd = ai*bj*gamma
  return phi(t,eps)*ai*bj*(-1+rho/(1+lamd*rho))

def GradGammaNotZero(ai, bj, gamma, x,t,eps):
  return phi(t,eps)*(-ai*bj+x/gamma)

def GradGamma(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the gamma param from some game
  if x == 0 and y == 0:
    return GradGammaZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 0 and y == 1:
    return GradGammaZeroOne(ai, bj, gamma, rho,t,eps)
  else:
    return GradGammaNotZero(ai, bj, gamma, x,t,eps)

# Functions for rho derivative are below

def GradRhoZeroZero(ai, aj, bi, bj, gamma, rho,t,eps):
  lamd = ai*bj*gamma
  mu = aj*bi
  return -phi(t,eps)*lamd*mu/(1-lamd*mu*rho)

def GradRhoZeroOne(ai,bj, gamma, rho,t,eps):
  lamd = ai*bj*gamma
  return phi(t,eps)*lamd/(1+lamd*rho)

def GradRhoOneZero(aj,bi, rho,t,eps):
  mu = aj*bi
  return phi(t,eps)*mu/(1+mu*rho)

def GradRhoOneOne (rho,t,eps):
  return -phi(t,eps)/(1-rho)

def GradRho(ai, aj, bi, bj, gamma, rho,t,x,y,eps):
  # Funtion which determines the addition to the gradient of the gamma param from some game
  if x == 0 and y == 0:
    return GradRhoZeroZero(ai, aj, bi, bj, gamma, rho,t,eps)
  elif x == 0 and y == 1:
    return GradRhoZeroOne(ai,bj, gamma, rho,t,eps)
  elif x == 1 and y == 0:
    return GradRhoOneZero(aj,bi, rho,t,eps)
  elif x == 1 and y == 1:
    return GradRhoOneOne(rho,t,eps)
  else:
    return 0

def GradEps(ai, aj, bi, bj, gamma, rho,t,x,y, eps):
    lamb = ai*bj*gamma
    mu = aj*bi
    # return -t*phi(t,eps)*(np.log(tau(x, y, lamb, mu, rho)) - lamb + x*np.log(lamb) - mu + y*np.log(mu))
    return 0

def GradAdder(Match_Data, Parameters, GradientVector,i, gamma, rho, Teams):
  # Function which takes the df of mathches, the current Parameters and calcualtes the addition to gradient vector for the i'th match
  # Returns the resulting gradient vector

  # Finding index for the home and away team
  IndexHome = Teams.index(Match_Data['HomeTeam'][i])
  IndexAway = Teams.index(Match_Data['AwayTeam'][i])

  # Finding relevant Parameters and other variables
  ai = Parameters[IndexHome]
  aj = Parameters[IndexAway]
  bi = Parameters[IndexHome + len(Teams)]
  bj = Parameters[IndexAway + len(Teams)]
  eps = Parameters[2*len(Teams)+2]
  t = Match_Data['t'][i]
  x =  Match_Data['FTHG'][i]
  y =  Match_Data['FTAG'][i]

  # Adding onto the Gradient vector
  GradientVector[IndexHome] = GradientVector[IndexHome] + GradAlphaHome(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[IndexAway] = GradientVector[IndexAway] + GradAlphaAway(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[IndexHome + len(Teams)] = GradientVector[IndexHome + len(Teams)] + GradBetaHome(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[IndexAway + len(Teams)] = GradientVector[IndexAway + len(Teams)] + GradBetaAway(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[2*len(Teams)] = GradientVector[2*len(Teams)] + GradGamma(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[2*len(Teams) + 1] = GradientVector[2*len(Teams) + 1] + GradRho(ai, aj, bi, bj, gamma, rho,t,x,y,eps)
  GradientVector[2*len(Teams) + 2] = GradientVector[2*len(Teams) + 2] + GradEps(ai, aj, bi, bj, gamma, rho, t, x, y, eps)

  return GradientVector

def GradientVectorFinder(Match_Data, Parameters, Teams):
  # Function whcih takes the match data, current Parameters and returns the Gradient Vector

  # Building the gradient vector
  GradientVector = np.zeros(len(Teams)*2+3)

  # Setting gamma and rho
  gamma = Parameters[2*len(Teams)]
  rho = Parameters[2*len(Teams)+1]
  eps = Parameters[2*len(Teams)+2]

  # Running through all the matches, every i makes an addition to the gradient vector
  for i in range(0,len(Match_Data.index)):
    GradientVector = GradAdder(Match_Data, Parameters, GradientVector,i, gamma, rho, Teams)

  return GradientVector

def NormalisingTheGradientVector(GradientVector,n, Teams):
  # Function which takes the GradientVector and normalises it such that the average of the alpha gradients is 0.

  AlphaGradValues = GradientVector[0:len(Teams)]
  AverageAlphaGradValues = np.mean(AlphaGradValues) # This is the average of paramaters in notes. But in our corrections, we want to add the gradint. Hence, there should be a net 0 efferct on the everage of the alphas from the gradint, as they already add up to one.
  Normaliser = np.concatenate((AverageAlphaGradValues*np.ones(len(Teams)), np.zeros(len(Teams)+3)))

  return (GradientVector - Normaliser)/NMod(GradientVector - Normaliser,n)

def NormalisingTheGradientVector2(GradientVector, Teams):
  # Function which takes the GradientVector and normalises it such that the average of the alpha gradients is 0.

  AlphaGradValues = GradientVector[0:len(Teams)]
  AverageAlphaGradValues = np.mean(AlphaGradValues) # This is the average of paramaters in notes. But in our corrections, we want to add the gradint. Hence, there should be a net 0 efferct on the everage of the alphas from the gradint, as they already add up to one.
  Normaliser = np.concatenate((AverageAlphaGradValues*np.ones(len(Teams)), np.zeros(len(Teams)+3)))
  
  GradientVectorNorm = GradientVector - Normaliser
  m = max(abs(GradientVectorNorm))*100
  for k in range(len(GradientVectorNorm-1)):
      GradientVectorNorm[k]=GradientVectorNorm[k]/m
  
  return GradientVectorNorm




def Optimise(Match_Data, Teams,Max = 200, m = 10):
  # Takes some match data and returns returns the parameters which maximise the log liklihood function.
  # This is done with a gradient ascent alogorithm
  # The default maximum step size is is 1/200, can be changed in the Max variable
  # The default is that we start with a step size of 1/10, which then goes to 1/20 etc... this can be changed in m

  # Setting all Parameters equal to 1 at first
  Parameters = np.ones(2*len(Teams)+3)

  # Setting gamma equal to 1.3 and rho equal to -0.05
  Parameters[2*len(Teams)] = 1.3
  Parameters[2*len(Teams)+1] = -0.05

  Mult = 1
  Step = m

  count = 0
  # Doing itertaitons until we have added just one of the smallets gradient vecor we want to add
  while Step <= Max:

    count = count + 1
    #print("count is " + str(count))

    # Finding gradient
    GradientVector = GradientVectorFinder(Match_Data, Parameters, Teams)

    # Normalising (Avergage of alhpas is 1), and adjusting the length
    GradientVectorNormalised = NormalisingTheGradientVector(GradientVector,Step, Teams)
    #print("step is " + str(Step))

    PresentPoint = Parameters
    StepToPoint = Parameters + GradientVectorNormalised
    LLLoop = 0
    LLOld = LL(Match_Data, PresentPoint, Teams)
    LLNew = LL(Match_Data, StepToPoint, Teams)
    # Adding GradientVectorNormalised until we have maxemised the LL
    while LLNew > LLOld:
      PresentPoint = StepToPoint
      StepToPoint = PresentPoint + GradientVectorNormalised
      LLLoop = LLLoop + 1
      LLOld = LLNew
      LLNew = LL(Match_Data, StepToPoint, Teams)

    #print("LLLoop is " + str(LLLoop))

    # If there has only been one itteration (or zero), we increase the step size
    if LLLoop < 2:
      Mult = Mult + 1
      Step = Mult*m

    Parameters = PresentPoint

  Alpha = Parameters[0:len(Teams)]
  Beta = Parameters[len(Teams):(len(Teams)*2)]
  Gamma = Parameters[len(Teams)*2]
  Rho = Parameters[len(Teams)*2+1]
  eps = Parameters[len(Teams)*2+2]
  d = {'Team': Teams, 'Alpha': Alpha, 'Beta': Beta, 'Gamma': Gamma*np.ones(len(Teams)), 'Rho': Rho*np.ones(len(Teams))}
  Results = pd.DataFrame(data=d)

  return Results

def LoadData(CurrentDate):
    with open('season-1718_json.json') as f:
        data = json.load(f)

    Match_Data_Original = pd.DataFrame(data)
    Match_Data = Match_Data_Original[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'Date']]

    date_time_obj = datetime.datetime.strptime(CurrentDate, "%d/%m/%y")
    date_time_obj.date()

    Match_Data["Date"] = Match_Data["Date"].apply(lambda x: datetime.datetime.strptime(x, "%d/%m/%y"))
    Match_Data["t"] = (date_time_obj - Match_Data["Date"]).astype('timedelta64[D]')
    return Match_Data

def Optimise2(Match_Data, Teams, Parameters = None):
  # Takes some match data and returns the parameters which maximise the log liklihood function.
  # This is done with a gradient ascent alogorithm
  # The default maximum step size is is 1/200, can be changed in the Max variable
  # The default is that we start with a step size of 1/10, which then goes to 1/20 etc... this can be changed in m
  got = 0
  if Parameters == None:
      got = 1
      # Setting all Parameters equal to 1 at first
      Parameters = np.ones(2*len(Teams)+3)
    
      # Setting gamma equal to 1.3 and rho equal to -0.05
      Parameters[2*len(Teams)] = 1.3
      Parameters[2*len(Teams)+1] = -0.05
      Parameters[2*len(Teams)+2] = 0.0065

  count = 0
  cont = 1
  start = 1
  # Doing itertaitons until we have added just one of the smallets gradient vecor we want to add
  while cont == 1:

    count = count + 1
    #print("count is " + str(count))

    # Finding gradient
    GradientVector = GradientVectorFinder(Match_Data, Parameters, Teams)

    # Normalising (Avergage of alhpas is 1), and adjusting the length
    GradientVectorNormalised = NormalisingTheGradientVector2(GradientVector,Teams)
    #print("step is " + str(Step))
    if start == 1 and got == 1:
        GradientVectorNormalised = GradientVectorNormalised*10

    PresentPoint = Parameters
    StepToPoint = Parameters + GradientVectorNormalised
    LLLoop = 0
    LLOld = LL(Match_Data, PresentPoint, Teams)
    LLNew = LL(Match_Data, StepToPoint, Teams)
    # Adding GradientVectorNormalised until we have maxemised the LL
    # while LLNew > LLOld:
    if LLNew > LLOld:
      PresentPoint = StepToPoint
      StepToPoint = PresentPoint + GradientVectorNormalised
      LLLoop = LLLoop + 1
      #LLOld = LLNew
      #LLNew = LL(Match_Data, StepToPoint, Teams)

    # If there has only been one itteration (or zero), we increase the step size
    if LLLoop == 0:
        if start == 0:
            cont = 0
        start = 0

    Parameters = PresentPoint

  Alpha = Parameters[0:len(Teams)]
  Beta = Parameters[len(Teams):(len(Teams)*2)]
  Gamma = Parameters[len(Teams)*2]
  Rho = Parameters[len(Teams)*2+1]
  eps = Parameters[len(Teams)*2+2]
  d = {'Team': Teams, 'Alpha': Alpha, 'Beta': Beta, 'Gamma': Gamma*np.ones(len(Teams)), 'Rho': Rho*np.ones(len(Teams)), 'eps': eps}
  Results = pd.DataFrame(data=d)

  return Results

def ProbMatrix(HomeTeam, AwayTeam, Parameters, gamma, rho, Teams,Max = 10,RealMadridAttackChange=0, RealMadridDefenceChange = 0):
      # Function which takes two teams and returns a scoreline probability matrix.
      # Parameters is the set of parameters we have after running the Optimise function
      # Max is the maximum number of goals we assume any team can score in a game.     
      
      HomeIndex = Teams.index(HomeTeam)
      AwayIndex = Teams.index(AwayTeam)
      
      # Finding relevant Parameters
      ai = Parameters['Alpha'][HomeIndex]
      aj = Parameters['Alpha'][AwayIndex]
      bi = Parameters['Beta'][HomeIndex]
      bj = Parameters['Beta'][AwayIndex]
      
      lamb = ai*bj*gamma
      mu = aj*bi
      
      # Change parameters if Real Madrid
      if HomeTeam == 'Real Madrid' or AwayTeam == 'Real Madrid':
          if not(RealMadridAttackChange ==0):          
              if HomeTeam == 'Real Madrid':
                  lamb += RealMadridAttackChange
              else:
                  mu += RealMadridAttackChange             
          elif not(RealMadridDefenceChange ==0):
              if HomeTeam == 'Real Madrid':
                  mu += RealMadridDefenceChange
              else:
                  lamb += RealMadridDefenceChange
              # Check greater than 0
              mu = max(0,mu)
              lamb = max(0,lamb)
                   
      # Making the scoreline probability matrix, without the tau function at first
      Result = np.outer(poisson.pmf(np.arange(0,Max +1), lamb), poisson.pmf(np.arange(0,Max +1), mu))
      
      # Adding the tau function
      Result[0,0] = Result[0,0]*(1-lamb*mu*rho)
      Result[1,0] = Result[1,0]*(1+mu*rho)
      Result[0,1] = Result[0,1]*(1+lamb*rho)
      Result[1,1] = Result[1,1]*(1-rho)
      
      # Making sure probabilites add to one
      Result = Result/np.sum(Result)
      
      return(Result)
  
def HG(n,Max):
    return np.floor(n/(Max+1))

def AG(n,Max, HomeG):
    return n - HomeG*(Max+1)

def SimulateMatch(HomeTeam, AwayTeam, Parameters, gamma, rho, Teams,Max = 10,RealMadridAttackChange=0, RealMadridDefenceChange = 0):
            
    PMatrix = ProbMatrix(HomeTeam, AwayTeam, Parameters, gamma, rho,Teams, Max,RealMadridAttackChange, RealMadridDefenceChange )
    RandomNumber = np.random.uniform()
    c = np.cumsum(PMatrix)
    n = np.argmax(c>RandomNumber) # Checking which bin we are in
    HomeG = HG(n,Max)
    AwayG = AG(n,Max, HomeG)
    return [HomeG, AwayG]

def Prob(PMatrix, HomeTeam, AwayTeam):
    AW = np.sum(np.triu(PMatrix,1))   
    Draw = np.trace(PMatrix)
    HW = np.sum(PMatrix) - Draw - AW
    return HomeTeam + ': ' + str(HW) + ' Draw: ' + str(Draw) + ' ' + AwayTeam + ': ' + str(AW)

def simulate_one_season(Teams, Parameters, gamma, rho,Max = 10):

    # Update result data in dicts - faster than updating a DataFrame.
    points = defaultdict(int)
    wins = defaultdict(int)
    draws = defaultdict(int)
    losses = defaultdict(int)
    goals_for = defaultdict(int)
    goals_against = defaultdict(int)

    games_played = 0
    # Simulate all the games in a season
    for home_team in Teams:
        for away_team in Teams:
            if home_team == away_team:
                continue
            games_played += 1
                
            
            GameResult = SimulateMatch(home_team, away_team, Parameters, gamma, rho, Teams,Max)
            home_goals = GameResult[0]
            away_goals = GameResult[1]
            
            # Update the points and win/draw/loss statistics.
            if home_goals > away_goals:
                points[home_team] += 3
                wins[home_team] += 1
                losses[away_team] += 1
            elif home_goals == away_goals:
                points[home_team] += 1
                points[away_team] += 1
                draws[home_team] += 1
                draws[away_team] += 1
            else:
                points[away_team] += 3
                wins[away_team] += 1
                losses[home_team] += 1
            
            # Update the goals.
            goals_for[home_team] += home_goals
            goals_against[home_team] += away_goals
            
            goals_for[away_team] += away_goals
            goals_against[away_team] += home_goals
           
    # Return the table as a DataFrame (needs to be sorted on points and goal diff).
    
    # Build the empty table
    empty_rows = np.zeros((len(Teams),7), dtype=int)
    season_table_values = pd.DataFrame(empty_rows, columns=['points', 'wins', 'draws', 'losses', 'goals for', 'goals against', 'goal diff'])
    season_table_teams = pd.DataFrame(Teams,columns=['team'])
    season_table = pd.concat([season_table_teams, season_table_values], axis=1)
    
    # Fill in the table
    for team in Teams:
        values_list = [points[team], wins[team], draws[team], losses[team], goals_for[team], goals_against[team]]
        season_table.loc[season_table.team == team, ['points', 'wins', 'draws', 'losses', 'goals for', 'goals against']] = values_list

    # Calculate the goal diff.
    season_table.loc[:, 'goal diff']= season_table.loc[:, 'goals for'] - season_table.loc[:, 'goals against']
    
    return season_table.sort_values(['points', 'goal diff'], ascending=[False, False])

def simulate_n_seasons(Teams, Parameters, gamma, rho, calendar,Team1,Team2=None,n=100,match_added=None,Max = 10):
    #
    
    team_position=[]
    GoalsFor = []
    GoalsAgainst= []
    Losses = []
    Wins = []
    Draws = []
    Points = []
    Name=[]
    
    team_position2=[]
    GoalsFor2 = []
    GoalsAgainst2= []
    Losses2 = []
    Wins2 = []
    Draws2 = []
    Points2 = []
    Name2=[]
      
    for i in range(n):
        
        season_table = simulate_one_season(Teams, Parameters, gamma, rho,Max)
        rank_team=int(np.where(season_table['team']==str(Team1))[0])
        team_position += [rank_team + 1 ]# First index is 0, therefore + 1.
        
        index_team=season_table[season_table['team']==str(Team1)].index.values.astype(int)[0]
        
        GoalsFor += [season_table['goals for'][index_team]]
        GoalsAgainst +=[ season_table['goals against'][index_team]]
        Losses +=[ season_table['losses'][index_team]]
        Draws +=[ season_table['draws'][index_team]]
        Wins += [season_table['wins'][index_team]]
        Points +=[season_table['points'][index_team]]
        Name+=[str(Team1)]
        
        if Team2!= None:
           rank_team2=int(np.where(season_table['team']==str(Team2))[0])
           team_position2 += [rank_team2 + 1 ]# First index is 0, therefore + 1.
        
           index_team2=season_table[season_table['team']==str(Team2)].index.values.astype(int)[0]
         
           GoalsFor2 += [season_table['goals for'][index_team2]]
           GoalsAgainst2 +=[ season_table['goals against'][index_team2]]
           Losses2 +=[ season_table['losses'][index_team2]]
           Draws2 +=[ season_table['draws'][index_team2]]
           Wins2 += [season_table['wins'][index_team2]]
           Points2 +=[season_table['points'][index_team2]] 
           Name2+=[str(Team2)]
        
    
    recap_team1=pd.DataFrame(list(zip(team_position,Points,Wins,Draws,Losses,GoalsFor,GoalsAgainst,Name)), columns = ['Classement','Points','Victoire','Nul','Défaite','Buts marqués','Buts encaissés','Nom'])
    recap_team2=pd.DataFrame(list(zip(team_position2,Points2,Wins2,Draws2,Losses2,GoalsFor2,GoalsAgainst2,Name2)), columns = ['Classement','Points','Victoire','Nul','Défaite','Buts marqués','Buts encaissés','Nom'])
    return recap_team1,recap_team2

def probabilites(recap_team):
    lenght=len(recap_team)
    ranking_proba=[]
    proba_vector=[0 for k in range(20)]
    proba_3lastplace=0
    proba_4firstplace=0
    
    for k in range(lenght):
        
        ranking=recap_team.iloc[k,0]
        ranking_proba+=[ranking]
        proba_vector[ranking-1]+=1/lenght
        if ranking>17:
            proba_3lastplace+=1/lenght
        if ranking<5:
            proba_4firstplace+=1/lenght
    """plt.hist(ranking_proba, range = (1,(recap_team['Victoire'][0]+recap_team['Nuls'][0]+recap_team['Défaite'][0])/2), color = 'blue',edgecolor = 'red')
    plt.xlabel('Classement')
    plt.ylabel('Probabilités')
    plt.title('Probabilité de classement à la fin du championnat')"""
        
    return np.array(proba_vector),proba_3lastplace,proba_4firstplace


def probabiliteteamvsteam(recap_team1,recap_team2):
    lenght1=len(recap_team1)
    lenght2=len(recap_team2)
    proba1above2=0
    if lenght1==lenght2:
        for k in range(lenght1):
            ranking1=recap_team1.iloc[k,0]
            ranking2=recap_team2.iloc[k,0]
            
            if ranking1>ranking2:
                proba1above2+=1/lenght1
        return proba1above2
    else:
        return None

def simulate_end_season(Teams, Parameters, gamma, rho, calendar,match_added=None,Max=10):
    #match_added de la forme dataframe avec colonnes teamH, teamA, Fthg,Ftag
    
    points = defaultdict(int)
    wins = defaultdict(int)
    draws = defaultdict(int)
    losses = defaultdict(int)
    goals_for = defaultdict(int)
    goals_against = defaultdict(int)
    
    for home_team in Teams:
        for away_team in Teams:
            if home_team==away_team:
                pass
            
            describe_match=calendar
            if match_added!= None:
                if home_team==match_added['teamH'] and away_team==match_added['teamA']:
                    home_goals=match_added['Fthg']
                    away_goals=match_added['Ftag']
            
            elif (describe_match['status']=='POSTPONED' or describe_match['status']=='SCHEDULED' or describe_match['status']=='CANCELED'):
                GameResult = SimulateMatch(home_team, away_team, Parameters, gamma, rho, Teams,Max)
                home_goals = GameResult[0]
                away_goals = GameResult[1]
                
                # Update the points and win/draw/loss statistics.
                
            else:
                home_goals=describe_match['Home Goals']
                away_goals=describe_match['Away Goals']
                
            if home_goals > away_goals:
                    points[home_team] += 3
                    wins[home_team] += 1
                    losses[away_team] += 1
            elif home_goals == away_goals:
                    points[home_team] += 1
                    points[away_team] += 1
                    draws[home_team] += 1
                    draws[away_team] += 1
            else: 
                    points[away_team] += 3
                    wins[away_team] += 1
                    losses[home_team] += 1   
            goals_for[home_team] += home_goals
            goals_against[home_team] += away_goals
             
            goals_for[away_team] += away_goals
            goals_against[away_team] += home_goals
             
    empty_rows = np.zeros((len(Teams),7), dtype=int)
    season_table_values = pd.DataFrame(empty_rows, columns=['points', 'wins', 'draws', 'losses', 'goals for', 'goals against', 'goal diff'])
    season_table_teams = pd.DataFrame(Teams,columns=['team'])
    season_table = pd.concat([season_table_teams, season_table_values], axis=1)
    
    # Fill in the table
    for team in Teams:
        values_list = [points[team], wins[team], draws[team], losses[team], goals_for[team], goals_against[team]]
        season_table.loc[season_table.team == team, ['points', 'wins', 'draws', 'losses', 'goals for', 'goals against']] = values_list

    # Calculate the goal diff.
    season_table.loc[:, 'goal diff']= season_table.loc[:, 'goals for'] - season_table.loc[:, 'goals against']
    
    return season_table.sort_values(['points', 'goal diff'], ascending=[False, False])
   

    
def importance_match_victory(Teams, Parameters, gamma, rho, calendar,match_addedH,match_addedA,n=10000):
    """ le match added home correspond à la victoire de l'équipe à domicile tandis que le match added A correspond à la victoire de l'équipe à l'extérieur"""
    recap_with_match=simulate_n_seasons(Teams, Parameters, gamma, rho, calendar, n, match_addedH['teamH'],match_addedH)
    recap_without_match=simulate_n_seasons(Teams, Parameters, gamma, rho, calendar, n, match_addedA['teamH'],match_addedA)
    
    
    ranking_proba_match,proba_3lastplace_match,proba_4firstplace_match=probabilites(recap_with_match)
    ranking_proba,proba_3lastplace,proba_4firstplace=probabilites(recap_without_match)
    
    ranking_difference=ranking_proba_match-ranking_proba
    proba_4firstdifference=proba_4firstplace_match-proba_4firstplace
    proba_3lastdifference= proba_3lastplace_match-proba_3lastplace
    
    return ranking_difference,proba_3lastdifference,proba_4firstdifference
    
import time
import matplotlib.pyplot as plt
Match_Data = LoadData("01/08/18")

Teams = sorted(list(set(Match_Data['HomeTeam'])))
Res = Optimise2(Match_Data, Teams)




t = time.time()
df,df1 = simulate_n_seasons(Teams, Res, Res['Gamma'][0],  Res['Rho'][0], 'Valencia','Alaves',n=100)
elapsed = time.time() - t
proba,probalast,probafirst=probabilites(df1)
pbteam1=probabiliteteamvsteam(df,df1)
proba1,probalast1,probafirst1=probabilites(df)

def histogramme(recap_team,Team):
    
    fig, ax = plt.subplots(figsize=(4,1.7))

    
    ax.hist(recap_team["Classement"], bins=np.arange(0,20)+0.5, ec="k")
    plt.xlabel('classement ')
    plt.ylabel('Probabilités')
    plt.title('Simulation du classement '+str(Team))