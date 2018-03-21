# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:18:54 2018

@author: Jeff Budge

Helper library for loading csv files and getting data.
"""

import glob
import csv
import numpy as np
#import pickle

def loadDataDict(folder):
    fname = folder + "/*.csv"
    filenames = glob.glob(fname)
    data = {}
    
    for f in filenames:
        if "Teams" in f:
            data['teams'] = f
        elif "TeamCoaches" in f:
            data['coach'] = f
        elif "TeamConferences" in f:
            data['conf'] = f
        elif "SecondaryTourneyCompact" in f:
            data['nit'] = f
        elif "RegularSeasonDetailed" in f:
            data['season'] = f
        elif "NCAATourneySlots" in f:
            data['tourney_slots'] = f
        elif "NCAATourneySeeds" in f:
            data['tourney_seeds'] = f
        elif "NCAATourneyDetailed" in f:
            data['tourney'] = f
        elif "Conferences" in f:
            data['conf_names'] = f
        elif "Massey" in f:
            data['rankings'] = f
    return data
    
class DataLib(object):
    
    def __init__(self, folder):
        self.folder = folder
        fname = folder + "/*.csv"
        filenames = glob.glob(fname)
        data = {}
        
        for f in filenames:
            if "Teams" in f:
                data['teams'] = f
            elif "TeamCoaches" in f:
                data['coach'] = f
            elif "TeamConferences" in f:
                data['conf'] = f
            elif "SecondaryTourneyCompact" in f:
                data['nit'] = f
            elif "RegularSeasonDetailed" in f:
                data['season'] = f
            elif "NCAATourneySlots" in f:
                data['tourney_slots'] = f
            elif "NCAATourneySeeds" in f:
                data['tourney_seeds'] = f
            elif "NCAATourneyDetailed" in f:
                data['tourney'] = f
            elif "Conferences" in f:
                data['conf_names'] = f
            elif "Massey" in f:
                data['rankings'] = f
                
        self.data = data
        self.loadRankings()
        self.loadIDs()
        self.loadGames()
        
    def loadRankings(self):
        ranks = []
        with open(self.data['rankings'], 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            for row in rd:
                ranks.append(row)
        self.ranks = ranks
        
    def getRanking(self, team_id, year):
        ret = []
        for row in self.ranks:
            if row[0] == str(year):
                if row[3] == str(team_id):
                    ret.append(row)
        return ret
        
    def loadIDs(self):
        ids = []
        with open(self.data['teams'], 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            for row in rd:
                ids.append(row)
        self.ids = ids
        
    def getIDFromTeam(self, team_name):
        for row in self.ids:
            if row[1] == team_name:
                return int(row[0])
        return 0
        
    def getTeamFromID(self, team_id):
        for row in self.ids:
            if row[0] == str(team_id):
                return row[1]
        return 0
        
    def loadGames(self):
        season_games = []
        tourney_games = []
        with open(self.data['season'], 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            for row in rd:
                season_games.append(row)
        with open(self.data['tourney'], 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            for row in rd:
                tourney_games.append(row)
        self.season_games = season_games
        self.tourney_games = tourney_games
        
    def getTeamGames(self, team_id, tourney=False, year=None):
        wins = []
        games = []
        gp = 0
        src = self.tourney_games if tourney else self.season_games
        for row in src:
            if year is None:
                if (row[2] == str(team_id) or row[4] == str(team_id)):
                    games.append(row)
                    if row[2] == str(team_id):
                        wins.append(1)
                    else:
                        wins.append(0)
            else:
                if (row[2] == str(team_id) or row[4] == str(team_id)) and row[0] == str(year):
                    games.append(row)
                    if row[2] == str(team_id):
                        wins.append(1)
                    else:
                        wins.append(0)
        gp = len(games)
        return games, wins, gp
        
    def getTeam(self, team_id, tourney=False, year=None):
        team = {}
        team['id'] = team_id
        games, wins, gp = self.getTeamGames(team_id, tourney, year)
        #Snag the stats for this team
        stats = np.zeros((34, gp))
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
        team['stats'] = stats.T
        
        #Derived stats
        s = team['stats']
        dstats = np.zeros((gp, 24))
        for i in range(gp):
            tft = 1 if s[i, 12] == 0 else s[i, 12]
            oft = 1 if s[i, 25] == 0 else s[i, 25]
            dstats[i, 0] = s[i, 7] / s[i, 8]
            dstats[i, 1] = s[i, 9] / s[i, 10]
            dstats[i, 2] = s[i, 11] / tft
            dstats[i, 3] = (s[i, 13] + s[i, 14]) / (sum(s[i, 13:15] + s[i, 26:28]))
            dstats[i, 4] = s[i, 15] / s[i, 7]
            dstats[i, 5] = s[i, 20] / s[i, 21]
            dstats[i, 6] = s[i, 22] / s[i, 23]
            dstats[i, 7] = s[i, 24] / oft
            dstats[i, 8] = 1.0 - dstats[i, 3]
            dstats[i, 9] = s[i, 28] / s[i, 20]
            dstats[i, 10] = s[i, 3] - s[i, 4]
            dstats[i, 11:18] = s[i, 13:20] - s[i, 26:33]
            dstats[i, 18] = s[i, 3] / (2.0*s[i, 8] + .44*s[i, 12])
            dstats[i, 19] = s[i, 4] / (2.0*s[i, 21] + .44*s[i, 25])
            dstats[i, 20] = s[i, 12] / s[i, 8]
            dstats[i, 21] = s[i, 25] / s[i, 21]
            dstats[i, 22] = s[i, 8] - s[i, 13] + s[i, 16] + .44*s[i, 12]
            dstats[i, 23] = s[i, 21] - s[i, 26] + s[i, 29] + .44*s[i, 25]
        team['dstats'] = dstats
        
        #Get the averaged Massey Ordinals - that is, ranks from polls and shtuff
        rks = []
        if year is not None:
            gp = 0
            rts = self.getRanking(team_id, year)
            for g in games:
                av_rk = 0.0
                num_rks = 0
                for r in rts:
                    if int(g[1]) > int(r[1]) and int(r[1]) >= gp:
                        num_rks += 1
                        av_rk += float(r[4])
                        gp = int(r[1])
                if num_rks > 0:
                    rks.append([float(g[1]), av_rk / num_rks])
                else:
                    rks.append([float(g[1]), 0.0])
            rks = np.array(rks)
            if len(rks) > 0:
                firstrank = rks[rks[:, 1] != 0.0, 1]
                rks[rks[:, 1] == 0.0, 1] = firstrank[0]
        team['ranks'] = rks
        team['tourney'] = tourney
        return team
        
        
def loadPlayData(folder, year):
    if int(year) < 2010:
        print 'Year before 2010'
        return 0
    fname = folder + "/*.csv"
    filenames = glob.glob(fname)
    plays = []
    
    for f in filenames:
        if "Events_" + str(year) in f:
            with open(f, 'rb') as csvfile:
                rd = csv.reader(csvfile, delimiter=',')
                for row in rd:
                        plays.append(row)
            return plays
    return 0
    
def getRanking(team_id, year, data):
    ranks = []
    with open(data['rankings'], 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[0] == str(year):
                if row[3] == str(team_id):
                    ranks.append(row)
    return ranks
    
def loadPlayerData(folder, year, team_id=None):
    if int(year) < 2010:
        print 'Year before 2010'
        return 0
    fname = folder + "/*.csv"
    filenames = glob.glob(fname)
    players = []
    
    for f in filenames:
        if "Players_" + str(year) in f:
            with open(f, 'rb') as csvfile:
                rd = csv.reader(csvfile, delimiter=',')
                for row in rd:
                    if team_id is not None:
                        if str(team_id) == row[2]:
                            players.append(row)
            return players
    return 0
    
def getIDFromTeam(team_name, data):
    
    with open(data['teams'], 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[1] == team_name:
                return int(row[0])
    return 0
    
#365 teams from 1101-1464
def getTeamFromID(team_id, data):
    
    with open(data['teams'], 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[0] == str(team_id):
                return row[1]
    return 0
    
def getCoaches(team_id, data):
    coaches = []
    
    with open(data['coach'], 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[1] == str(team_id):
                coaches.append(row)
    return coaches
    
def getConferences(team_id, data):
    confs = []
    
    with open(data['conf'], 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[1] == str(team_id):
                confs.append(row)
    return confs
    
def getTeamGames(team_id, data, tourney=False):
    games = []
    wins = []
    gp = 0
    fname = data['tourney'] if tourney else data['season']
    team_id = str(team_id)
    
    with open(fname, 'rb') as csvfile:
        rd = csv.reader(csvfile, delimiter=',')
        for row in rd:
            if row[2] == team_id or row[4] == team_id:
                games.append(row)
                if row[2] == team_id:
                    wins.append(1)
                else:
                    wins.append(0)
                    
    gp = len(games) #for building arrays with the data
    
    return games, wins, gp

            
        

