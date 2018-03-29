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
from statslib import DataLib
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
| 14 Team Poss | 15 Opp Poss | 16:22 Team stats (7 cols) | 23:29 Opp. stats (7 cols) | 30 Team Rank
| 31 Home/Away for team | 32 Team TS% Impact | 33 Rank Diff

Stat structure:
| 16 23 | 17 24 | 18 25 | 19 26 | 20 27 | 21 28 | 22 29
|  OR   |  DR   |  Ast  |  TO   | Stl   |  Blk  |  PF
"""
#def getTestStats(t1, year, tourney=False, verbose=False):
#    home = TeamStat('../kraggle_data', t1)
#    dstats = home.getDerivedStatsByYear(year, tourney)
#    stats = home.getStatsByYear(year, tourney)
#    ranks = home.getAverageRank(year)[:, 1]
#    
#    if len(stats) > 0:
#        if tourney:
#            final_rank = np.average(ranks, weights=np.exp(np.arange(len(ranks))))
#            ranks = np.ones((shape(stats)[0],1))*final_rank
#            #print ranks
#        return np.concatenate((dstats[:, 0:10], dstats[:, 18:24], stats[:, 13:20], stats[:, 26:33], np.reshape(ranks, (len(ranks), 1))), axis=1)
#    else:
#        return 0
        
def getTestStats(team, lib, verbose=False):
    dstats = team['dstats']
    stats = team['stats']
    if verbose:
        print str(team['id']) + ' ',
    defstats = np.zeros((team["gp"], 2))
        
    if len(stats) > 0:
        for oi in range(team["gp"]):
            oppteam = lib.getTeam(int(team["stats"][oi, 33]), False, team['year'])
            findgame = np.logical_and(oppteam["stats"][:, 33] == team["id"], oppteam["stats"][:, 2] == team["stats"][oi, 2])
            defstats[oi, 0] = oppteam["dstats"][:, 18].mean() - oppteam["dstats"][findgame, 18]
            defstats[oi, 1] = oppteam["ranks"][findgame, 1] - team["ranks"][oi, 1]
        ranks = team['ranks'][:, 1]
        if team['tourney']:
            final_rank = np.average(ranks, weights=np.exp(np.arange(len(ranks))))
            ranks = np.ones((shape(stats)[0],1))*final_rank
            #print ranks
        return np.concatenate((dstats[:, 0:10], dstats[:, 18:24], stats[:, 13:20], stats[:, 26:33], np.reshape(ranks, (len(ranks), 1)), np.reshape(stats[:, 5], (len(ranks), 1)), defstats), axis=1)
        #return np.concatenate((dstats[:, 0:10], dstats[:, 18:24], stats[:, 13:20], stats[:, 26:33], np.reshape(ranks, (len(ranks), 1)), np.reshape(stats[:, 5], (len(ranks), 1))), axis=1)
    else:
        return 0
        
def loadGameModel(t1, t2, lib, verbose=False):
    hstats = getTestStats(t1, lib, verbose)
    astats = getTestStats(t2, lib, verbose)
    htstats = hstats
    atstats = astats
    home = np.mean(htstats, axis=0)
    away = np.mean(atstats, axis=0)
    hstd = np.std(hstats, axis=0)
    astd = np.std(astats, axis=0)
    rkdif = away[30] - home[30]
    avs = np.concatenate((home[0:5], away[0:5], np.array([home[10] - away[32], away[10] - home[32], home[12], away[12], home[14], away[14]]), home[16:23], away[16:23], np.array([home[30], 0, home[32], rkdif])), axis=0)
    stds = np.concatenate((hstd[0:5], astd[0:5], np.array([hstd[10], astd[10], hstd[12], astd[12], hstd[14], astd[14]]), hstd[16:23], astd[16:23], np.array([0, 0, hstd[32], 0])))
    return avs, stds
        
#def loadGameModel(t1, t2, lib, verbose=False):
#    hstats = getTestStats(t1, lib, verbose)
#    astats = getTestStats(t2, lib, verbose)
#    htstats = hstats
#    atstats = astats
#    home = np.mean(htstats, axis=0)
#    away = np.mean(atstats, axis=0)
#    hstd = np.std(hstats, axis=0)
#    astd = np.std(astats, axis=0)
#    #rkdif = away[30] - home[30]
#    avs = np.concatenate((home[0:5], away[0:5], np.array([home[10], away[10], home[12], away[12], home[14], away[14]]), home[16:23], away[16:23], np.array([home[30], 0])), axis=0)
#    stds = np.concatenate((hstd[0:5], astd[0:5], np.array([hstd[10], astd[10], hstd[12], astd[12], hstd[14], astd[14]]), hstd[16:23], astd[16:23], np.array([0, 0])))
#    return avs, stds
        
def simNetworkScore(t1, t2, model, lib, verbose=False):
    results = np.zeros((1000,4))
    avs, stds = loadGameModel(t1, t2, lib, verbose)
    nn = Network([10])
    nn.load(model)
    for i in range(1000):
        sts = np.zeros((len(avs),))
        for n in range(len(sts)):
            sts[n] = np.random.normal(loc=avs[n], scale=stds[n])
        toss, results[i, :] = nn.run(sts, True)
    return results
    
def genNetworkProbabilities(t1, t2, model, lib, verbose=False):
    res = simNetworkScore(t1, t2, model, lib, verbose)
    #res2 = simNetworkScore(t2, t1, model, lib, verbose)
    tmp = np.sum(res, axis=0)# + np.flip(np.sum(res2, axis=0), 0)
    prob = (tmp[2] + tmp[3]) / sum(tmp)
    #prob = (sum(res == 2) + sum(res == 3) + 0.0) / len(res)
    if prob <= 0:
        prob = .01
    elif prob >= 1:
        prob = .99
    return prob, res
    
               
def simulateScore(t1, t2, lib, coeffs, verbose = False):
    iters = 100
    results = np.zeros((iters,))
    avs, stds = loadGameModel(t1, t2, lib, verbose)
    for i in range(iters):
        sts = np.zeros((len(avs),))
        for n in range(len(avs)):
            sts[n] = np.random.normal(loc=avs[n], scale=stds[n])
        results[i] = sum(sts*coeffs)
    return results
    
def genProbabilities(t1, t2, lib, coeffs, verbose=False):
    
    res = simulateScore(t1, t2, lib, coeffs, verbose)
    hperc = sum(res > 0) / 100.0
    return hperc, res
    
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
    
#def grabLabels(t1, year, tourney=False):
#    home = TeamStat('../kraggle_data', t1).getDerivedStatsByYear(year, tourney)[:, 10]
#    ret = home + 0.0
#    ret[home < 0] = 1
#    ret[home > 0] = 2
#    ret[home < -15] = 0
#    ret[home > 15] = 3
#    return ret
    
def grabLabels(t1, get_pts=False):
    home = t1['dstats'][:, 10]
    ret = home + 0.0
    ret[home < 0] = 1
    ret[home > 0] = 2
    ret[home < -15] = 0
    ret[home > 15] = 3
    if get_pts:
        return ret, home
    else:
        return ret
        
def getLogScore(predictions, labels):
    n = len(labels)
    return -1/n*sum(labels*np.log10(predictions[:,0]) + (1.0 - labels)*np.log10(predictions[:,1]))
