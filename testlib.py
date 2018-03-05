# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:49:48 2018

@author: Jeff Budge

Matchup generator, using stats from the season to predict winner of a game.
"""

from TeamStat import TeamStat
import numpy as np
import scipy as sp
import statslib as slib
import csv


def genProbabilities(t1, t2, year):
    #grab data folder and csv filenames
    folder = '../kraggle_data'
    #data = slib.loadDataDict(folder)
    home = TeamStat(folder, t1)
    away = TeamStat(folder, t2)
    
    #get defensive stats
    hg = home.gamesInSeason(year)
    homeRaw = home.getStatsByYear(year)
    homeDer = home.getDerivedStatsByYear(year)
    homeRank = home.getAverageRank(year)
    homeImpact = np.zeros((hg,))
    hOppImpact = np.zeros_like(homeImpact)
    hSoS = np.zeros_like(homeImpact)
    for g in range(hg):
        oppteam_id = homeRaw[g, 33]
        oppteam = TeamStat(folder, int(oppteam_id))
        oppteamStats = oppteam.getDerivedStatsByYear(year)
        oppteamRank = oppteam.getAverageRank(year)
        oppRank = oppteamRank[oppteam.getGameNumber(year, homeRaw[g, 2]),1]
        opp_avfgp = oppteamStats[:,4].mean()
        homeImpact[g] = homeDer[g,9] - opp_avfgp
        hOppImpact[g] = homeDer[g,4]
        hSoS[g] = homeRank[g, 1] - oppRank
        
    homeScore = np.array([homeImpact.mean(), hOppImpact.std(),
    (np.average(homeRank[:,1], weights=np.exp(np.arange(hg))) / 182.5) - 1,
    np.gradient(homeRank[:,1]).std() / 50.0,
    hSoS.mean() / 100.0])
    #mn = homeScore.mean()
    #mstd = homeScore.std()
    #homeScore = (homeScore - mn) / mstd
        
    ag = away.gamesInSeason(year)
    awayRaw = away.getStatsByYear(year)
    awayDer = away.getDerivedStatsByYear(year)
    awayRank = away.getAverageRank(year)
    awayImpact = np.zeros((ag,))
    aOppImpact = np.zeros_like(awayImpact)
    aSoS = np.zeros_like(awayImpact)
    for g in range(ag):
        oppteam_id = awayRaw[g, 33]
        oppteam = TeamStat(folder, int(oppteam_id))
        oppteamStats = oppteam.getDerivedStatsByYear(year)
        oppteamRank = oppteam.getAverageRank(year)
        oppRank = oppteamRank[oppteam.getGameNumber(year, awayRaw[g, 2]),1]
        opp_avfgp = oppteamStats[:,4].mean()
        awayImpact[g] = awayDer[g,9] - opp_avfgp
        aOppImpact[g] = awayDer[g,4]
        aSoS[g] = awayRank[g, 1] - oppRank

    awayScore = np.array([awayImpact.mean(), aOppImpact.std(),
    (np.average(awayRank[:,1], weights=np.exp(np.arange(ag))) / 182.5) - 1,
    np.gradient(awayRank[:,1]).std() / 50.0,
    aSoS.mean() / 100.0])
    #awayScore = (awayScore - mn) / mstd
    
    hFinal = (-homeScore[0] + homeScore[2])*(homeScore[1] + homeScore[3]) / homeScore[4]
    aFinal = (-awayScore[0] + awayScore[2])*(awayScore[1] + awayScore[3]) / awayScore[4]
    
    return [hFinal, aFinal], [homeScore, awayScore]
    
"""
Matchups array key as follows:
|0 Strong seed |1 Weak seed| 2 Winner (1 if Strong, 0 if Weak)
"""
def getMatchups(year):
    folder = '../kraggle_data'
    data = slib.loadDataDict(folder)
    ret = []
    seeds = {}
    first_four = 0
    with open(data['tourney_seeds']) as cf:
        rd = csv.reader(cf, delimiter=',')
        for row in rd:
            if row[0] == str(year):
                seeds[row[1]] = row[2]
    with open(data['tourney_slots']) as cf:
        rd = csv.reader(cf, delimiter=',')
        for row in rd:
            if row[0] == str(year):
                if len(row[1]) != 3:
                    if len(row[2]) != 3:
                        ret.append(row[1:4])
                    else:
                        ret.append([row[1], seeds[row[2]], seeds[row[3]]])
                else:
                    strong = TeamStat(folder, seeds[row[2]])
                    sg = strong.getStatsByYear(year, True)
                    for n in range(strong.gamesInSeason(year, True)):
                        if sg[n, 33] == float(seeds[row[3]]):
                            seeds[row[1]] = seeds[row[2]] if sg[n, 3] > sg[n, 4] else seeds[row[3]]
    tlen = len(ret)
    rounds = [32, 48, 56, 60, 62, 63]
    games = np.zeros((tlen, 3)).astype(int)
    print "32"
    for i in range(rounds[0]):
        strong = TeamStat(folder, ret[i][1])
        sg = strong.getStatsByYear(year, True)
        for n in range(strong.gamesInSeason(year, True)):
            if sg[n, 33] == float(ret[i][2]):
                games[i, :] = np.array([int(ret[i][1]), int(ret[i][2]), sg[n, 3] > sg[n, 4]])
    for rnds in range(1, len(rounds)):
        print str(rounds[rnds])
        for i in range(rounds[rnds-1], rounds[rnds]):
            for n in range(len(ret)):
                if ret[n][0] == ret[i][1]:
                    if games[n, 2]:
                        games[i, 0] = games[n, 0]
                    else:
                        games[i, 0] = games[n, 1]
                if ret[n][0] == ret[i][2]:
                    if games[n, 2]:
                        games[i, 1] = games[n, 0]
                    else:
                        games[i, 1] = games[n, 1]
            strong = TeamStat(folder, games[i, 0])
            sg = strong.getStatsByYear(year, True)
            for n in range(strong.gamesInSeason(year, True)):
                if sg[n, 33] == float(games[i, 1]):
                    games[i, 2] = sg[n, 3] > sg[n, 4]
#    for i in range(32 + first_four, tlen):
#        strong = TeamStat(folder, games[i, 0])
#        sg = strong.getStatsByYear(year, True)
#        for n in range(strong.gamesInSeason(year, True)):
#            if sg[n, 33] == float(games[i, 0]):
#                games[i, :] = np.array([int(ret[i][1]), int(ret[i][2]), sg[n, 3] > sg[n, 4]])
    return games
        
                
    
    
def getLogScore(predictions, labels):
    n = len(labels)
    return -1/n*sum(labels*np.log10(predictions[:,0]) + (1.0 - labels)*np.log10(predictions[:,1]))
