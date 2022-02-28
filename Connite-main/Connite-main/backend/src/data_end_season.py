# from collections import defaultdict
# from .parameters import Params
# from .parameters import Team
# from .data import response
# from parameters import SimulateMatch

# import pandas as pd
# import numpy as np

# nb_simulations = 10

# def simulate_end_season(Parameters, Team,response):
    
#     # Update result data in dicts - faster than updating a DataFrame.
#     points=defaultdict(int)
#     wins=defaultdict(int)
#     losses=defaultdict(int)
#     draws=defaultdict(int)
#     goals_for=defaultdict(int)
#     goals_against=defaultdict(int)
#     Max=10

#     games_played = 0
#     # Simulate all the games in a season
#     for match in response["matches"]:
#       home_team=match['homeTeam']['name']
#       away_team=match['awayTeam']['name']
#       if match['status']=='FINISHED':
#        home_goals=match['score']['fullTime']['homeTeam']
#        away_goals=match['score']['fullTime']['awayTeam']
#       else :
#         GameResult = SimulateMatch(home_team, away_team, Parameters, Team)
#         home_goals = GameResult[0]
#         away_goals =GameResult[1]

#       # Update the points and win/draw/loss statistics.
#       if home_goals > away_goals:
#         points[home_team] += 3
#         wins[home_team] += 1
#         losses[away_team] += 1
#       elif home_goals == away_goals:
#         points[home_team] += 1
#         points[away_team] += 1
#         draws[home_team] += 1
#         draws[away_team] += 1
#       else:
#         points[away_team] += 3
#         wins[away_team] += 1
#         losses[home_team] += 1

#       # Update the goals.
#       goals_for[home_team] += home_goals
#       goals_against[home_team] += away_goals
            
#       goals_for[away_team] += away_goals
#       goals_against[away_team] += home_goals

#     # Return the table as a DataFrame (needs to be sorted on points and goal diff).
    
#     # Build the empty table
#     empty_rows = np.zeros((len(Team),7), dtype=int)
#     season_table = pd.DataFrame(empty_rows, index=Team,columns=['points', 'wins', 'draws', 'losses', 'goals for', 'goals against', 'goal diff'])
    
    
    
#     # Fill in the table
#     for team in Team:
#         values_list = [points[team], wins[team], draws[team], losses[team], goals_for[team], goals_against[team]]
#         season_table.loc[team, ['points', 'wins', 'draws', 'losses', 'goals for', 'goals against']] = values_list

#     # Calculate the goal diff.
#     season_table.loc[:, 'goal diff']= season_table.loc[:, 'goals for'] - season_table.loc[:, 'goals against']
    
#     season_table=season_table.sort_values(['points', 'goal diff'], ascending=[False, False])
#     rank=[k for k in range(1,len(Team)+1)]
  
#     season_table.insert(1,"rank",rank,True)
#     return season_table

# season_table=simulate_end_season(Params,Team,response)
# season_table

# def simulate_n_seasons(Team, Parameters, response,n=nb_simulations,Max = 10):
    
#     team_position=np.zeros((len(Team),n))
#     GoalsFor =np.zeros((len(Team),n))
#     GoalsAgainst= np.zeros((len(Team),n))
#     Losses = np.zeros((len(Team),n))
#     Wins = np.zeros((len(Team),n))
#     Draws = np.zeros((len(Team),n))
#     Points = np.zeros((len(Team),n))
#     Name=[]
    
#     for i in range(n):
#       season_table = simulate_end_season(Parameters,Team,response)
      
#       for (k,team) in enumerate(Team):
#         rank_team=season_table.loc[team,'rank']
#         team_position[k,i]+= [rank_team ]# First index is 0, therefore + 1.
        
        
#         GoalsFor[k,i]= season_table.loc[team,'goals for']
#         GoalsAgainst[k,i]= season_table.loc[team,'goals against']
#         Losses[k,i]= season_table.loc[team,'losses']
#         Draws[k,i]= season_table.loc[team,'draws']
#         Wins[k,i]= season_table.loc[team,'wins']
#         Points[k,i]=season_table.loc[team,'points']
#         if i==0:
#           Name+=[str(team)]     
       
      
#     return Name,team_position,GoalsFor,GoalsAgainst,Points,Wins,Draws,Losses



# def probabilites(Name,team_position,team_wanted):
#     for (k,team) in enumerate(Name):
#       if team==team_wanted:
#         index_team_wanted=k
#     ranking_proba=[]
#     proba_vector=[0 for k in range(20)]
#     proba_3lastplace=0
#     proba_4firstplace=0
#     proba_middletable=0
#     proba_europe = 0
#     proba_first=0
#     number_season_simulated=len(team_position[0])
#     for k in range(number_season_simulated):
#         ranking=int(team_position[index_team_wanted,k])
#         ranking_proba+=[ranking]
#         proba_vector[ranking-1]+=1/number_season_simulated
#         if ranking>17:
#             proba_3lastplace+=1/number_season_simulated
#         if ranking<5:
#             proba_4firstplace+=1/number_season_simulated
#         if ranking<10:
#           proba_middletable+=1/number_season_simulated
#         if ranking==1:
#           proba_first+=1/number_season_simulated
#         if ranking<6:
#           proba_europe+=1/number_season_simulated

        
#     return np.array(proba_vector),proba_first,proba_4firstplace,proba_europe,proba_middletable,(1-proba_3lastplace)

# def mean_final(Name, GoalsFor, GoalsAgainst, Points, Wins, Draws, Losses, nb_simulations):
#   mean_n_season = []
#   for i in range(len(Name)):
#     meanPoints = round(1/nb_simulations*Points.sum(axis=1)[i],2)
#     meanWins = round(1/nb_simulations*Wins.sum(axis=1)[i],2)
#     meanDraws = round(1/nb_simulations*Draws.sum(axis=1)[i],2)
#     meanLosses = round(1/nb_simulations*Losses.sum(axis=1)[i],2)
#     meanGoals_scored = round(1/nb_simulations*GoalsFor.sum(axis=1)[i],2)
#     mean_Goals_conceded = round(1/nb_simulations*GoalsAgainst.sum(axis=1)[i],2)
#     newteam = {"Team":Name[i],"Points": meanPoints,"Wins": meanWins,"Draws": meanDraws, "Losses":meanLosses, "Goals_scored": meanGoals_scored, "Goals_conceded": mean_Goals_conceded }
#     mean_n_season.append(newteam)
#   new_mean_n_season = sorted(mean_n_season, key=lambda d: d["Points"]) 
#   return new_mean_n_season[::-1]

# Name,team_position,GoalsFor,GoalsAgainst,Points,Wins,Draws,Losses=simulate_n_seasons(Team, Params, response,10)
# final_ranking = mean_final(Name,GoalsFor,GoalsAgainst,Points,Wins,Draws,Losses,nb_simulations)


















# final_ranking = []
# arsenal = {"Team": "Arsenal FC", "Points": 86,"Wins": 17, "Draws":10,"Losses": 11, "Goals_scored":90,"Goals_conceded":65}
# manchester = {"Team": "Manchester United FC", "Points": 80,"Wins": 15, "Draws":10,"Losses": 13, "Goals_scored":85,"Goals_conceded":65}
# chelsea = {"Team": "Chelsea FC", "Points": 77,"Wins": 14, "Draws":10,"Losses": 14, "Goals_scored":70,"Goals_conceded":48}


# final_ranking.append(arsenal)
# final_ranking.append(manchester)
# final_ranking.append(chelsea)

final_ranking = [{'Team': 'Manchester City FC', 'Points': 94.24, 'Wins': 29.72, 'Draws': 5.08, 'Losses': 3.2, 'Goals_scored': 89.13, 'Goals_conceded': 22.49}, {'Team': 'Liverpool FC', 'Points': 84.97, 'Wins': 25.54, 'Draws': 8.35, 'Losses': 4.11, 'Goals_scored': 94.9, 'Goals_conceded': 31.33}, {'Team': 'Chelsea FC', 'Points': 76.86, 'Wins': 21.98, 'Draws': 10.92, 'Losses': 5.1, 'Goals_scored': 75.72, 'Goals_conceded': 28.54}, {'Team': 'Arsenal FC', 'Points': 67.4, 'Wins': 20.2, 'Draws': 6.8, 'Losses': 11.0, 'Goals_scored': 60.69, 'Goals_conceded': 41.49}, {'Team': 'Tottenham Hotspur FC', 'Points': 61.45, 'Wins': 18.04, 'Draws': 7.33, 'Losses': 12.63, 'Goals_scored': 52.52, 'Goals_conceded': 48.19}, {'Team': 'Manchester United FC', 'Points': 61.45, 'Wins': 17.13, 'Draws': 10.06, 'Losses': 10.81, 'Goals_scored': 60.44, 'Goals_conceded': 51.08}, {'Team': 'West Ham United FC', 'Points': 60.71, 'Wins': 17.81, 'Draws': 7.28, 'Losses': 12.91, 'Goals_scored': 62.69, 'Goals_conceded': 49.39}, {'Team': 'Wolverhampton Wanderers FC', 'Points': 55.84, 'Wins': 15.62, 'Draws': 8.98, 'Losses': 13.4, 'Goals_scored': 33.09, 'Goals_conceded': 31.14}, {'Team': 'Leicester City FC', 'Points': 52.89, 'Wins': 14.73, 'Draws': 8.7, 'Losses': 14.57, 'Goals_scored': 66.13, 'Goals_conceded': 65.57}, {'Team': 'Brighton & Hove Albion FC', 'Points': 50.78, 'Wins': 11.38, 'Draws': 16.64, 'Losses': 9.98, 'Goals_scored': 38.89, 'Goals_conceded': 40.51}, {'Team': 'Southampton FC', 'Points': 48.89, 'Wins': 11.65, 'Draws': 13.94, 'Losses': 12.41, 'Goals_scored': 51.38, 'Goals_conceded': 58.29}, {'Team': 'Aston Villa FC', 'Points': 48.34, 'Wins': 13.9, 'Draws': 6.64, 'Losses': 17.46, 'Goals_scored': 52.99, 'Goals_conceded': 57.73}, {'Team': 'Crystal Palace FC', 'Points': 43.91, 'Wins': 10.16, 'Draws': 13.43, 'Losses': 14.41, 'Goals_scored': 52.67, 'Goals_conceded': 58.43}, {'Team': 'Leeds United FC', 'Points': 41.09, 'Wins': 9.94, 'Draws': 11.27, 'Losses': 16.79, 'Goals_scored': 48.97, 'Goals_conceded': 72.74}, {'Team': 'Brentford FC', 'Points': 38.66, 'Wins': 10.14, 'Draws': 8.24, 'Losses': 19.62, 'Goals_scored': 42.53, 'Goals_conceded': 63.09}, {'Team': 'Everton FC', 'Points': 33.91, 'Wins': 8.72, 'Draws': 7.75, 'Losses': 21.53, 'Goals_scored': 42.33, 'Goals_conceded': 70.26}, {'Team': 'Burnley FC', 'Points': 33.08, 'Wins': 5.64, 'Draws': 16.16, 'Losses': 16.2, 'Goals_scored': 31.44, 'Goals_conceded': 50.03}, {'Team': 'Newcastle United FC', 'Points': 32.27, 'Wins': 6.57, 'Draws': 12.56, 'Losses': 18.87, 'Goals_scored': 40.24, 'Goals_conceded': 73.45}, {'Team': 'Watford FC', 'Points': 28.84, 'Wins': 7.44, 'Draws': 6.52, 'Losses': 24.04, 'Goals_scored': 37.48, 'Goals_conceded': 70.48}, {'Team': 'Norwich City FC', 'Points': 27.09, 'Wins': 6.36, 'Draws': 8.01, 'Losses': 23.63, 'Goals_scored': 25.6, 'Goals_conceded': 75.6}]

