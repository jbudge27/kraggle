# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:49:48 2018

@author: Jeff Budge

Matchup generator, using stats from the season to predict winner of a game.
"""

from TeamStat import TeamStat
from Network import Network
from joblib import Parallel, delayed
import time
import multiprocessing
import numpy as np
import scipy as sp
import statslib as slib
from pylab import shape
import csv
import pickle



def scale(x, mx, mn):
    a = 0.0
    b = 1.0
    return (b - a)*(x - mn) / (mx - mn) + a
    
def unscale(x, a, b):
    mx = 1.0
    mn = 0.0
    return (x-a)*(mx - mn) / (b - a) + mn    
    
def getCorrelation(t1, year, stats, tourney=False, verbose=False):
    pt_diff = TeamStat('../kraggle_data', t1).getDerivedStatsByYear(year, tourney)[:, 10]
    if verbose:
        print "Grabbing scaled stats for {}".format(t1)
    #stats = getScaledStats(t1, year, verbose)
    return np.corrcoef(stats, np.reshape(pt_diff, (1, shape(stats)[1])))[shape(stats)[0], 0:-1]

def getTeamType(stats, pt_diff, verbose):
    ttypes = pickle.load(open('./ttypes.pkl'))
    tt = np.corrcoef(stats, np.reshape(pt_diff, (1, shape(stats)[1])))[shape(stats)[0], 0:-1]
    team_type = np.zeros((len(tt),))
    devs = ttypes['mean'] - tt
    stds = ttypes['stds']
    for i in range(len(devs)):
        if devs[i] > stds[i]:
            team_type += ttypes[str(i+1)]
            if verbose:
                print str(i+1)
        elif devs[i] < -stds[i]:
            team_type += ttypes[str(-(i+1))]
            if verbose:
                print str(-(i+1))
        else:
            team_type += np.array([72.83999, 16.19815, 7.00, 37.317, 10.782, -11.429, -13.5571]) #ttypes['base']
    if verbose:
        print team_type / len(devs)
    return team_type / len(devs)
    
"""
Stats structure:
| 0 TS% Impact | 1 Opp TS% | 2 Team Average Rank | 3 Rank Differential

Also returns the number of games in the set as 'hg'.
"""

def defstatsloop(g):
    retlist = []
    oppteam_id = g[0]
    oppteam = TeamStat('../kraggle_data', int(oppteam_id))
    oppteamStats = oppteam.getDerivedStatsByYear(g[6], g[7])
    oppteamRank = oppteam.getAverageRank(g[6])
    oppRank = oppteamRank[oppteam.getGameNumber(g[6], g[1], g[7]),1]
    opp_avfgp = oppteamStats[:,18].mean()
    retlist.append(g[3] - opp_avfgp)
    retlist.append(g[4])
    retlist.append(g[2])
    retlist.append(g[5] - oppRank)
    return retlist
    
def getDefStats(t1, year, tourney=False, verbose=False):
    folder = '../kraggle_data'
    home = TeamStat(folder, t1)
    
    if verbose:
        print "Getting defensive stats...",
        
    hg = home.gamesInSeason(year, tourney)
    homeRaw = home.getStatsByYear(year, tourney)
    homeDer = home.getDerivedStatsByYear(year, tourney)
    homeRank = home.getAverageRank(year)
    passlist = []
    pt = time.time()
    for g in range(hg):
        passlist.append([homeRaw[g, 33], homeRaw[g, 2], homeRank[g, 1], homeDer[g, 19], homeDer[g, 18], homeRank[g, 1], year, tourney])
    stats = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(defstatsloop)(g) for g in passlist)
    if verbose:
        print "{:.3f} secs for loop.".format(time.time() - pt)
    stats = np.array(stats)
    return stats, hg
    
"""
Your scaled stats glossary:
| 0 Opp TS% | 1 Team TS% | 2 DR Diff. | 3 Ast Diff. | 4 Opp FT% | 5 TS% Impact
| 6 Rank Diff. |
""" 
def getScaledStats(t1, year, tourney=False, verbose=False):
    folder = '../kraggle_data'
    homeT = TeamStat(folder, t1)
    home = homeT.getDerivedStatsByYear(year, tourney)
    jeffStats = homeT.getJeffStats(year, tourney)
    #get defensive stats
    if verbose:
        print "Grabbing stats for team {} of {}".format(t1, year)
    homeStats, hg = getDefStats(t1, year, tourney, verbose)
    
    return np.array([home[:,19], home[:, 18], scale(home[:, 12], 10, -10), 
                scale(home[:, 13], 30, -30), home[:, 7], scale(homeStats[:, 0], .5, -.5), 
                scale(homeStats[:, 3], 333, -333), scale(jeffStats[:, 0], 2, 0), jeffStats[:, 2]])
                
"""
And the test_stats glossary...
|0 Team FG% | 1 Team 3FG% | 2 Team FT% | 3 Team RB% | 4 Team Ast% | 5 Opp FG% | 6 Opp 3FG%
| 7 Opp FT% | 8 Opp RB% | 9 Opp Ast% | 10 Team TS% | 11 Opp TS% | 12 Team FTR | 13 Opp FTR 
| 14 Team Poss | 15 Opp Poss | 16:28 Team stats (13 cols) | 29:41 Opp. stats (13 cols) | 42 Team Rank

Stat structure:
16 29 | 17 30 | 18 31 | 19 32 | 20 33 | 21 34 | 22 35 | 23 36 | 24 37 | 25 38 | 26 39 | 27 40 | 28 41
FGM   | FGA   | FGM3  | FGA3  |  FTM  |  FTA  |  OR   |  DR   |  Ast  |  TO   | Stl   |  Blk  |  PF
"""
def getTestStats(t1, year, tourney=False, verbose=False):
    home = TeamStat('../kraggle_data', t1)
    dstats = home.getDerivedStatsByYear(year, tourney)
    stats = home.getStatsByYear(year, tourney)
    ranks = home.getAverageRank(year)
    if tourney:
        final_rank = np.average(ranks, weights=np.exp(np.arange(len(ranks))))
        ranks = np.ones((shape(stats)[0]))*final_rank
    if len(stats) > 0:
        return np.concatenate((dstats[:, 0:10], dstats[:, 18:24], stats[:, 7:33], ranks), axis=1)
    else:
        return 0
        
def simNetworkScore(t1, t2, year, model, tourney=False, verbose=False):
    results = np.zeros((1000,))
    home = np.mean(getTestStats(t1, year, tourney, verbose), axis=0)
    away = np.mean(getTestStats(t2, year, tourney, verbose), axis=0)
    hstd = np.std(getTestStats(t1, year, tourney, verbose), axis=0)
    astd = np.std(getTestStats(t2, year, tourney, verbose), axis=0)
    avs = np.concatenate((home[0:5], away[5:10], home[10:12], away[12:14], home[14:28], away[28:]))
    stds = np.concatenate((hstd[0:5], astd[5:10], hstd[10:12], astd[12:14], hstd[14:28], astd[28:]))
    nn = Network([10])
    nn.load(model)
    for i in range(1000):
        sts = np.zeros((len(avs),))
        for n in range(len(sts)):
            sts[n] = np.random.normal(loc=avs[n], scale=stds[n])
        results[i] = nn.run(sts)
    return results
    
def genNetworkProbabilities(t1, t2, year, model, tourney=False, verbose=False):
    res = simNetworkScore(t1, t2, year, model, tourney, verbose)
    prob = (sum(res == 2) + sum(res == 3)*2.0 - sum(res == 0)) / len(res)
    if prob <= 0:
        prob = .01
    elif prob >= 1:
        prob = .99
    return prob, res
    
               
def simulateScore(hStats, aStats, pt_diff, rkg, iters = 100, verbose = False):
    results = np.zeros((iters,))
    corrs = getTeamType(hStats, pt_diff, verbose)
    gameStats = hStats
    gameStats[1, :] = hStats[1, :] + unscale(aStats[5, :].mean(), .5, -.5)
    avs = np.mean(hStats, axis=1)
    avs[0] = aStats[0, :].mean()
    avs[4] = aStats[4, :].mean()
    sigma = np.std(hStats, axis=1)
    sigma[0] = aStats[0, :].std()
    sigma[4] = aStats[4, :].std()
    avs[6] = scale(rkg, 333, -333)
    sigma[6] = 0
    for i in range(iters):
        sts = np.zeros((len(corrs),))
        for n in range(len(corrs)):
            sts[n] = np.random.normal(loc=avs[n], scale=sigma[n])
        results[i] = sum(sts*corrs)
    return results
    
def genProbabilities(t1, t2, year, tourney=False, verbose=False):
    #grab data folder and csv filenames
    folder = '../kraggle_data'
    iters = 1000
    home = TeamStat(folder, t1)
    away = TeamStat(folder, t2)
    hStats = getScaledStats(t1, year, tourney, verbose)
    aStats = getScaledStats(t2, year, tourney, verbose)
    hRank = home.getAverageRank(year)[:, 1]
    aRank = away.getAverageRank(year)[:, 1]
    rkg = np.average(hRank, weights=np.exp(np.arange(shape(hRank)[0]))) \
        - np.average(aRank, weights=np.exp(np.arange(shape(aRank)[0])))
    if verbose:
        print "Simulating {} games...".format(iters)
        print "Team {}".format(t1)
    homeScore = simulateScore(hStats, aStats, home.getDerivedStatsByYear(year, tourney)[:,10], rkg, iters, verbose)
    if verbose:
        print "Team {}".format(t2)
    awayScore = simulateScore(aStats, hStats, away.getDerivedStatsByYear(year, tourney)[:,10], -rkg, iters, verbose)
    
    hperc = sum(homeScore - awayScore > 0) / (iters + 0.0)
    
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
    
def grabLabels(t1, year, tourney=False):
    home = TeamStat('../kraggle_data', t1).getDerivedStatsByYear(year, tourney)[:, 10]
    ret = home + 0.0
    ret[home < 0] = 1
    ret[home > 0] = 2
    ret[home < -15] = 0
    ret[home > 15] = 3
    return ret
        
def getLogScore(predictions, labels):
    n = len(labels)
    return -1/n*sum(labels*np.log10(predictions[:,0]) + (1.0 - labels)*np.log10(predictions[:,1]))
