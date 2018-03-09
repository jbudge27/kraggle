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
from pylab import shape
import csv



def scale(x, mx, mn):
    a = 0.0
    b = 1.0
    return (b - a)*(x - mn) / (mx - mn) + a
    
def unscale(x, a, b):
    mx = 1.0
    mn = 0.0
    return (x-a)*(mx - mn) / (b - a) + mn    


def getTeamType(stats, pt_diff, verbose):
    team_types = np.array([[ 0.18947928,  0.69989005,  0.27564864,  0.22650642,  0.10083039,  0.18585862,  -0.15739254],
                           [ 0.09000703,  0.76764929,  0.42093012,  0.44655541,  0.10403649,  0.14024649,  -0.40459465],
                           [-0.17506678,  0.78111676,  0.54828041,  0.51704579, -0.07191937, -0.1188948,  -0.39758784]])
    tt = np.corrcoef(stats, np.reshape(pt_diff, (1, shape(stats)[1])))[shape(stats)[0], 0:-1]
    dists = np.zeros((shape(team_types)[0],))
    for i in range(len(dists)):
        dists[i] = np.linalg.norm(team_types[i] - tt)
    if dists.min() == dists[0]:
        if verbose:
            print "Shooting Team"
        return [51.106, 90.868, -1.703, 13.345, 1.464, -27.938, -6.804]
    elif dists.min() == dists[1]:
        if verbose:
            print "Balanced Team"
        return [52.244, 86.026, .822, 21.130, .161, -23.314, -14.710]
    elif dists.min() == dists[2]:
        if verbose:
            print "Defensive Team"
        return [44.059, 83.952, 3.521, 23.963, -1.635, -20.134, -10.562]
    else:
        print "THIS IS WEIRD..."
    
    
"""
Stats structure:
| 0 TS% Impact | 1 Opp TS% | 2 Team Average Rank | 3 Rank Differential

Also returns the number of games in the set as 'hg'.
"""
def getDefStats(t1, year, verbose=False):
    folder = '../kraggle_data'
    home = TeamStat(folder, t1)
    
    if verbose:
        print "Getting defensive stats",
        
    hg = home.gamesInSeason(year)
    homeRaw = home.getStatsByYear(year)
    homeDer = home.getDerivedStatsByYear(year)
    homeRank = home.getAverageRank(year)
    stats = np.zeros((hg, 4))
    stats[:, 2] = homeRank[:, 1]
    
    for g in range(hg):
        if verbose:
            print '.',
        oppteam_id = homeRaw[g, 33]
        oppteam = TeamStat(folder, int(oppteam_id))
        oppteamStats = oppteam.getDerivedStatsByYear(year)
        oppteamRank = oppteam.getAverageRank(year)
        oppRank = oppteamRank[oppteam.getGameNumber(year, homeRaw[g, 2]),1]
        opp_avfgp = oppteamStats[:,18].mean()
        stats[g, 0] = homeDer[g,19] - opp_avfgp
        stats[g, 1] = homeDer[g,18]
        stats[g, 3] = homeRank[g, 1] - oppRank
        
    if verbose:
        print "Done."
    return stats, hg
    
"""
Your scaled stats glossary:
| 0 Opp TS% | 1 Team TS% | 2 DR Diff. | 3 Ast Diff. | 4 Opp FT% | 5 TS% Impact
| 6 Rank Diff. |
""" 
def getScaledStats(t1, year, verbose=False):
    folder = '../kraggle_data'
    home = TeamStat(folder, t1).getDerivedStatsByYear(year)
    
    #get defensive stats
    homeStats, hg = getDefStats(t1, year, verbose)
    
    return np.array([home[:,19], home[:, 18], scale(home[:, 12], 10, -10), 
                scale(home[:, 13], 30, -30), home[:, 7], scale(homeStats[:, 0], .5, -.5), 
                scale(homeStats[:, 3], 333, -333)])
                
def simulateScore(hStats, aStats, pt_diff, iters = 100, verbose = False):
    results = np.zeros((iters,))
    corrs = getTeamType(hStats, pt_diff, verbose)
    gameStats = hStats
    gameStats[0, :] = aStats[0, :]
    gameStats[1, :] = hStats[1, :] + unscale(aStats[5, :].mean(), .5, -.5)
    gameStats[4, :] = aStats[4, :]
    avs = np.mean(hStats, axis=1)
    sigma = np.std(hStats, axis=1)
    avs[6] = aStats[6, -1]
    sigma[6] = 0
    for i in range(iters):
        sts = np.zeros((len(corrs),))
        for n in range(len(corrs)):
            sts[n] = np.random.normal(loc=avs[n], scale=sigma[n])
        results[i] = sum(sts*corrs)
    return results
    
def genProbabilities(t1, t2, year, verbose=False):
    #grab data folder and csv filenames
    folder = '../kraggle_data'
    home = TeamStat(folder, t1).getDerivedStatsByYear(year)
    away = TeamStat(folder, t2).getDerivedStatsByYear(year)
    hStats = getScaledStats(t1, year, verbose)
    aStats = getScaledStats(t2, year, verbose)
    if verbose:
        print "Simulating games..."
    homeScore = simulateScore(hStats, aStats, home[:,10], 100, verbose)
    awayScore = simulateScore(aStats, hStats, away[:,10], 100, verbose)
    
    hperc = sum(homeScore - awayScore > 0) / 100.0
    
    return [hperc, 1.0 - hperc], [hStats, aStats], [homeScore, awayScore]
    
"""
Matchups array key as follows:
|0 Strong seed |1 Weak seed| 2 Winner (1 if Strong, 0 if Weak)
"""
def getMatchups(year):
    folder = '../kraggle_data'
    data = slib.loadDataDict(folder)
    ret = []
    seeds = {}
    #Create a dict that takes the seeds and associates them with team ids
    with open(data['tourney_seeds']) as cf:
        rd = csv.reader(cf, delimiter=',')
        for row in rd:
            if row[0] == str(year):
                seeds[row[1]] = row[2]
    #take the bracket structure and get the teams playing each other
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
    #now that we have all the games loaded, run the tournament using tourney results for that year
    print "32"
    for i in range(rounds[0]):
        strong = TeamStat(folder, ret[i][1])
        sg = strong.getStatsByYear(year, True)
        for n in range(strong.gamesInSeason(year, True)):
            if sg[n, 33] == float(ret[i][2]):
                games[i, :] = np.array([int(ret[i][1]), int(ret[i][2]), sg[n, 3] > sg[n, 4]])
    #the first round is done, now to find out who moves on. This is every other round in the tournament
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
    return games
        
                
    
    
def getLogScore(predictions, labels):
    n = len(labels)
    return -1/n*sum(labels*np.log10(predictions[:,0]) + (1.0 - labels)*np.log10(predictions[:,1]))
