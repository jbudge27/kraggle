# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:31 2018

@author: artemis-new
"""

from TeamStat import TeamStat
from sklearn import cluster
import statslib
from pylab import shape
import testlib as ts
#import scipy as sp
import numpy as np
import pickle
from matplotlib import pyplot as plt
#from sklearn import svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler

def build_pickles(folder, pickle_folder, tourney):
    #ranges = np.zeros((11,2))
    scsts = []
    scpts = []
    sccorrs = []
    for year in range(2003, 2018):
        print str(year),
        for i in teams:
            print '.',
            curr_team = TeamStat(folder, i)
            tmp = curr_team.getStatsByYear(year, tourney)[:, 3]
            if len(tmp) > 0:
                scpts.append(tmp)
                scalestats = ts.getScaledStats(i, year, tourney, True)
                scsts.append(scalestats)
                overcorr = ts.getCorrelation(i, year, scalestats, tourney)
                sccorrs.append(overcorr)
    if tourney:
        pickle.dump(scsts, open(pickle_folder + "/scsts_t.pkl", "w"))
        pickle.dump(scpts, open(pickle_folder + "/scpts_t.pkl", "w"))
        pickle.dump(sccorrs, open(pickle_folder + "/sccorrs_t.pkl", "w"))
    else:
        pickle.dump(scsts, open(pickle_folder + "/scsts.pkl", "w"))
        pickle.dump(scpts, open(pickle_folder + "/scpts.pkl", "w"))
        pickle.dump(sccorrs, open(pickle_folder + "/sccorrs.pkl", "w"))
    return 1

def load_pickles(folder, tourney):
    if tourney:
        ct = np.array(pickle.load(open(folder + "/sccorrs_t.pkl")))
        sts = np.concatenate(pickle.load(open(folder + "/scsts_t.pkl")), axis=1)
        spts = np.concatenate(pickle.load(open(folder + "/scpts_t.pkl"))[0:-1])
        stslist = pickle.load(open(folder + "/scsts_t.pkl"))
        sptslist = pickle.load(open(folder + "/scpts_t.pkl"))
    else:
        ct = np.array(pickle.load(open(folder + "/sccorrs.pkl")))
        sts = np.concatenate(pickle.load(open(folder + "/scsts.pkl")), axis=1)
        spts = np.concatenate(pickle.load(open(folder + "/scpts.pkl"))[0:-1])
        stslist = pickle.load(open(folder + "/scsts.pkl"))
        sptslist = pickle.load(open(folder + "/scpts.pkl"))
    return ct, sts, spts, stslist, sptslist

def build_ttypes(pickle_folder, tourney):
    ct, sts, spts, stslist, sptslist = load_pickles(pickle_folder, tourney)
    lssz = len(stslist)
    numstats = shape(sts)[0]
    ctave = np.mean(ct, axis=0)
    lssol = np.linalg.lstsq(sts.T, spts)
    sols = np.zeros((2*numstats+1, numstats))
    solkey = []
    sols[0, :] = lssol[0]
    solkey.append(0)
    ttypes = {}
    ttypes['base'] = lssol[0]
    ttypes['mean'] = ctave
    ttypes['stds'] = np.std(ct, axis=0)
    for n in range(1, numstats+1):
        deviates = []
        dev_pts = []
        undeviates = []
        undev_pts = []
        dists = np.zeros((lssz, 7))
        for i in range(lssz):
            dists[i, :] = ct[i, :] - ctave
            if dists[i, n-1] > ttypes['stds'][n-1]:
                deviates.append(stslist[i])
                dev_pts.append(sptslist[i])
            if dists[i, n-1] < -ttypes['stds'][n-1]:
                undeviates.append(stslist[i])
                undev_pts.append(sptslist[i])
        deviates = np.concatenate(deviates, axis=1)
        dev_pts = np.concatenate(dev_pts)
        undeviates = np.concatenate(undeviates, axis=1)
        undev_pts = np.concatenate(undev_pts)
        lssol = np.linalg.lstsq(deviates.T, dev_pts)
        sols[2*n-1, :] = lssol[0]
        solkey.append(n)
        ttypes[str(n)] = lssol[0]
        lssol = np.linalg.lstsq(undeviates.T, undev_pts)
        sols[2*n, :] = lssol[0]
        solkey.append(-n)
        ttypes[str(-(n))] = lssol[0]
    if tourney:
        pickle.dump(ttypes, open(pickle_folder + "/ttypes_t.pkl", "w"))
        test = pickle.load(open(pickle_folder + "/ttypes_t.pkl"))
    else:
        pickle.dump(ttypes, open(pickle_folder + "/ttypes.pkl", "w"))
        test = pickle.load(open(pickle_folder + "/ttypes.pkl"))
    return test
    
    
folder = '../kraggle_data'
pickle_folder = '.'
data = statslib.loadDataDict(folder)
teams = np.arange(1101, 1464)
#teams = [statslib.getIDFromTeam("North Carolina", data), statslib.getIDFromTeam("Gonzaga", data), statslib.getIDFromTeam("Villanova", data), statslib.getIDFromTeam("Wisconsin", data)]
#test = ts.getMatchups(2017)
#probses = np.zeros((63,2))
#for n in range(63):
#    print str(n)
#    perc, res, scores = ts.genProbabilities(test[n, 0], test[n, 1], 2017)
#    probses[n, :] = np.array(perc)
#
#finalScore = ts.getLogScore(probses, test[:, 2])
#test = ts.getCorrelation(statslib.getIDFromTeam("Virginia", data), 2017, True)
#build_ttypes(folder)
#build_pickles(folder, pickle_folder, True)
#ct, sts, spts, stslist, sptslist = load_pickles(pickle_folder, False)
perc, res, scores = ts.genProbabilities(statslib.getIDFromTeam("Villanova", data), statslib.getIDFromTeam("North Carolina", data), 2016, False, True)

