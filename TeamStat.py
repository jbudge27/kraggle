# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:50:18 2018

@author: Jeff Budge

Team_Stats object and functions.
"""

import numpy as np
#import csv
import statslib

class TeamStat(object):
    
    """
    Object initialization.
    folder is the name of the folder with all the data csv files in it.
    team_id is the ID of the team you want to look for.
    
    Loads in all games, both season and tourney, along with coaching info and conference info (not that that should be too relevant, methinks.)
    """
    def __init__(self, folder, team_id, verbose=False):
        file_loc = statslib.loadDataDict(folder)
        self.file_loc = folder
        self.data_loc = file_loc
        self.team_id = team_id
        
        if verbose:
            print 'Getting regular season data...'
        self.season_games, self.season_wins, self.season_games_played = statslib.getTeamGames(team_id, file_loc)
        if verbose:
            print 'Getting tournament data...'
        self.tourney_games, self.tourney_wins, self.tourney_games_played = statslib.getTeamGames(team_id, file_loc, True)
        if verbose:
            print 'Getting peripheral data (coaches, conference)...'
        self.team_name = statslib.getTeamFromID(team_id, file_loc)
        self.coaches = statslib.getCoaches(team_id, file_loc)
        self.conf = statslib.getConferences(team_id, file_loc)
        if verbose:
            print 'Getting stats structure...'
        self.stats = self.getStats()
        self.tourney_stats = self.getStats(True)
        
    """
    Your stats-getting function. Returns a gigantic matrix, 34 x games played, with the following structure:
    0 Win/Loss (0:L, 1:W) | 1 Season | 2 Day of Season | 3 Team Score | 4 Opp. Score 
    | 5 Home/Away for team (0:N, 1:A, 2:H) | 6 Num. Overtimes | 7:19 Team stats (13 cols) 
    | 20:32 Opp. stats (13 cols) | 33 Opp. TeamID
    
    Stat structure:
    7 20 | 8 21 | 9 22 | 10 23 | 11 24 | 12 25 | 13 26 | 14 27 | 15 28 | 16 29 | 17 30 | 18 31 | 19 32
    FGM  | FGA  | FGM3 | FGA3  |  FTM  |  FTA  |  OR   |  DR   |  Ast  |  TO   | Stl   |  Blk  |  PF
    
    All values are floats.
    """
    def getStats(self, tourney=False):
        gp = self.tourney_games_played if tourney else self.season_games_played
        stats = np.zeros((34, gp))
        games = self.tourney_games if tourney else self.season_games
        wins = self.tourney_wins if tourney else self.season_wins
        for i in range(gp):
            stats[0, i] = wins[i]
            stats[1:3, i] = games[i][0:2]
            if wins[i]:
                stats[3,i] = float(games[i][3])
                stats[4,i] = float(games[i][5])
                stats[7:20,i] = games[i][8:21]
                stats[20:33,i] = games[i][21:]
                if games[i][6] == 'N':
                    stats[5,i] = 0
                elif games[i][6] == 'A':
                    stats[5,i] = 1
                else:
                    stats[5,i] = 2
                stats[33,i] = float(games[i][4])
            else:
                stats[3,i] = float(games[i][5])
                stats[4,i] = float(games[i][3])
                stats[7:20,i] = games[i][21:]
                stats[20:33,i] = games[i][8:21]
                if games[i][6] == 'N':
                    stats[5,i] = 0
                elif games[i][6] == 'A':
                    stats[5,i] = 2
                else:
                    stats[5,i] = 1
                stats[33,i] = float(games[i][2])
            stats[6,i] = float(games[i][7])
            
        return stats.T
        
    def getStatsByYear(self, year, tourney=False):
        stats = self.tourney_stats if tourney else self.stats
        return stats[stats[:,1] == year, :]

    """
    This function derives some more high-level stuff for messing around with.
    Returns a matrix with the following fields:
    0 Team FG% | 1 Team 3FG% | 2 Team FT% | 3 Team RB% | 4 Team Ast% | 5 Opp FG% | 6 Opp 3FG%
    | 7 Opp FT% | 8 Opp RB% | 9 Opp Ast% | 10 Pt Diff | 11 OR Diff | 12 DR Diff 
    | 13 Ast Diff | 14 TO Diff | 15 Stl Diff | 16 Blk Diff | 17 PF Diff
    | 18 Team TS% | 19 Opp TS% | 20 Team FTR | 21 Opp FTR
    """
    def getDerivedStats(self, tourney=False):
        gp = self.tourney_games_played if tourney else self.season_games_played
        s = self.stats
        stats = np.zeros((gp, 22))
        for i in range(gp):
            tft = 1 if s[i, 12] == 0 else s[i, 12]
            oft = 1 if s[i, 25] == 0 else s[i, 25]
            stats[i, 0] = s[i, 7] / s[i, 8]
            stats[i, 1] = s[i, 9] / s[i, 10]
            stats[i, 2] = s[i, 11] / tft
            stats[i, 3] = (s[i, 13] + s[i, 14]) / (sum(s[i, 13:14] + s[i, 26:27]))
            stats[i, 4] = s[i, 15] / s[i, 7]
            stats[i, 5] = s[i, 20] / s[i, 21]
            stats[i, 6] = s[i, 22] / s[i, 23]
            stats[i, 7] = s[i, 24] / oft
            stats[i, 8] = 1.0 - stats[i, 3]
            stats[i, 9] = s[i, 28] / s[i, 20]
            stats[i, 10] = s[i, 3] - s[i, 4]
            stats[i, 11:18] = s[i, 13:20] - s[i, 26:33]
            stats[i, 18] = s[i, 3] / (2.0*s[i, 8] + .44*s[i, 12])
            stats[i, 19] = s[i, 4] / (2.0*s[i, 21] + .44*s[i, 25])
            stats[i, 20] = s[i, 12] / s[i, 8]
            stats[i, 21] = s[i, 25] / s[i, 21]
        return stats
        
    def getDerivedStatsByYear(self, year, tourney=False):
        stats = self.tourney_stats if tourney else self.stats
        der_stats = self.getDerivedStats(tourney)
        return der_stats[stats[:,1] == year, :]
        
    """
    Runs through the Events file and grabs the logs for the specified game.
    
    Game must come from the TeamStat object's self.season_games or else it will break.
    """
        
    def getGameLog(self, game):
        game_log = []
        win_id = game[2]
        lose_id = game[4]
        day = game[1]
        season = game[0]
        events = statslib.loadPlayData(self.file_loc + "/play_by_play", season)
        for g in events:
            if g[2] == day:
                if g[3] == win_id:
                    if g[4] == lose_id:
                        game_log.append(g)
        return game_log
        
    def getGamesByYear(self, year, tourney=False):
        games = []
        in_games = self.tourney_games if tourney else self.season_games
        for g in in_games:
            if g[0] == str(year):
                games.append(g)
        return games
        
    def getGameNumber(self, year, day, tourney=False):
        in_games = self.getGamesByYear(year, tourney)
        for idx, g in enumerate(in_games):
            if g[0] == str(int(year)):
                if g[1] == str(int(day)):
                    return idx
        return 0
        
    def gamesInSeason(self, year, tourney=False):
        in_games = self.getGamesByYear(year, tourney)
        return len(in_games)


    """
    Runs through the derived and basic stats and gets a few of my own making.
    I want to see if these are predictive of anything.
    """
    
    def getJeffStats(self, year, tourney=False):
        teamStats = self.getGamesByYear(year, tourney)
        jeff_stats = np.zeros((len(teamStats),3))
        
        for i in range(len(teamStats)):
            print '.',
            pace = 0.0
            poss = 1.0
            prod_poss = 0.0
            unique_players = []
            game = self.getGameLog(teamStats[i])
            for g in game:
                if g[-3] == str(self.team_id):
                    if "timeout" not in g[-1] or "sub" not in g[-1]:
                        if float(g[-4]) < 2400.0: #exclude overtime stuff
                            pace += float(g[-4]) - 1200.0
                        unique_players.append(int(g[-2]))
                    if "made" in g[-1] or "reb" in g[-1] or g[-1] == "assist" or g[-1] == "block" or g[-1] == "steal":
                        prod_poss += 1
                        poss += 1
                    elif "miss" in g[-1] or "foul" in g[-1] or g[-1] == "turnover":
                        prod_poss -= 1
                        poss += 1
                        
                    
            jeff_stats[i, 0] = pace / float(g[-4]) #fitness - change in pace over course of game
            jeff_stats[i, 1] = len(np.unique(np.array(unique_players))) #depth - how many players did something
            jeff_stats[i, 2] = prod_poss / poss #productive possession percentage - number of possessions used productively
        return jeff_stats
        
    def getAverageRank(self, year):
        rankings = statslib.getRanking(self.team_id, year, self.data_loc)
        games = self.getGamesByYear(year)
        gp = 0
        rks = []
        for g in games:
            av_rk = 0.0
            num_rks = 0
            for r in rankings:
                if int(g[1]) > int(r[1]) and int(r[1]) >= gp:
                    num_rks += 1
                    av_rk += float(r[4])
                    gp = int(r[1])
            if num_rks > 0:
                rks.append([float(g[1]), av_rk / num_rks])
            else:
                rks.append([float(g[1]), 0.0])
        rks = np.array(rks)
        firstrank = rks[rks[:, 1] != 0.0, 1]
        rks[rks[:, 1] == 0.0, 1] = firstrank[0]
        return rks
                    
        