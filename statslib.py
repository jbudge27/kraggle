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
        
    def loadRankings(self):
        ranks = []
        with open(self.data['rankings'], 'rb') as csvfile:
            rd = csv.reader(csvfile, delimiter=',')
            for row in rd:
                ranks.append(row)
        self.ranks = ranks
        
        
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

            
        

