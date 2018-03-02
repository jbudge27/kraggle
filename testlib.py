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


def genProbabilities(t1, t2, year):
    #grab data folder and csv filenames
    folder = '../kraggle_data'
    data = slib.loadDataDict(folder)
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
    
    hFinal = (-homeScore[0] + homeScore[2])*(homeScore[1] + homeScore[3])/homeScore[4]
    aFinal = (-awayScore[0] + awayScore[2])*(awayScore[1] + awayScore[3])/awayScore[4]
    
    return [hFinal, aFinal], [homeScore, awayScore]
