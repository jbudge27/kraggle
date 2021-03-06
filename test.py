# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:31 2018

@author: artemis-new
"""

from TeamStat import TeamStat
from sklearn import cluster
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from Network import Network
from joblib import Parallel, delayed
import multiprocessing
import statslib
from pylab import shape
import testlib as ts
import time
#import scipy as sp
import numpy as np
import pickle
from matplotlib import pyplot as plt
#from sklearn import svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler

def build_pickles(folder, pickle_folder, tourney=False):
    #ranges = np.zeros((11,2))
    scsts = []
    scpts = []
    sccorrs = []
    teams = np.arange(1101, 1464)
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
    
"""
Parallel processing loop for the stats builder.
THis grabs the labels for training the NN, along with all the stats
I've deemed worthy of inclusion to the net (in the getTestStats function).
Also grabs the true shooting percentage impact as well as the rank differential
of all the teams this particular team played that year.
"""
def faststatsloop(i, year):
    print '.',
    team = globals()["lib"].getTeam(i, False, year)
    stats = ts.getTestStats(team, globals()["lib"])
    etc, labels = ts.grabLabels(team, True)
#    defstats = np.zeros((team["gp"], 2))
#    #print team["gp"]
#    if len(labels) > 0:
#        for oi in range(len(labels)):
#            oppteam = globals()["lib"].getTeam(int(team["stats"][oi, 33]), False, year)
#            findgame = np.logical_and(oppteam["stats"][:, 33] == team["id"], oppteam["stats"][:, 2] == team["stats"][oi, 2])
#            defstats[oi, 0] = oppteam["dstats"][:, 18].mean() - oppteam["dstats"][findgame, 18]
#            defstats[oi, 1] = oppteam["ranks"][findgame, 1] - team["ranks"][oi, 1]
#        stats = np.concatenate((stats, defstats), axis=1)
    return [stats, labels]
    
def build_fast_stats(start_year=2003):
    scsts = []
    for year in range(start_year, 2018):
        print str(year),
        teams = np.arange(1101, 1464)
        tmpsts = Parallel(n_jobs=multiprocessing.cpu_count()-1)(delayed(faststatsloop)(g, year) for g in teams)
        scsts.append(tmpsts)
    return scsts

def load_pickles(folder, tourney):
    if tourney:
        ct = np.array(pickle.load(open(folder + "/sccorrs_t.pkl")))
        sts = np.concatenate(pickle.load(open(folder + "/scsts_t.pkl")), axis=1)
        spts = np.concatenate(pickle.load(open(folder + "/scpts_t.pkl")))
        stslist = pickle.load(open(folder + "/scsts_t.pkl"))
        sptslist = pickle.load(open(folder + "/scpts_t.pkl"))
    else:
        ct = np.array(pickle.load(open(folder + "/sccorrs.pkl")))
        sts = np.concatenate(pickle.load(open(folder + "/scsts.pkl")), axis=1)
        spts = np.concatenate(pickle.load(open(folder + "/scpts.pkl"))[0:-1])
        stslist = pickle.load(open(folder + "/scsts.pkl"))
        sptslist = pickle.load(open(folder + "/scpts.pkl"))
    return ct, sts, spts, stslist, sptslist

def build_ttypes(pickle_folder, tourney=False):
    ct, sts, spts, stslist, sptslist = load_pickles(pickle_folder, tourney)
    valids = (np.sum(np.isnan(ct) + (np.abs(ct) == 1), axis=1) != 0)
    lssz = len(stslist)
    numstats = shape(sts)[0]
    ctave = np.mean(ct[valids, :], axis=0)
    lssol = np.linalg.lstsq(sts.T, spts)
    #sols = np.zeros((2*numstats+1, numstats))
    #solkey = []
    #sols[0, :] = lssol[0]
    #solkey.append(0)
    ttypes = {}
    ttypes['base'] = lssol[0]
    ttypes['mean'] = ctave
    ttypes['stds'] = np.std(ct[valids, :], axis=0)
    for n in range(1, numstats+1):
        deviates = []
        dev_pts = []
        undeviates = []
        undev_pts = []
        #dists = np.zeros((lssz, 7))
        for i in range(lssz):
            if valids[i]:
                dist = ct[i, n] - ctave[n]
                if dist > ttypes['stds'][n-1]:
                    deviates.append(stslist[i])
                    dev_pts.append(sptslist[i])
                if dist < -ttypes['stds'][n-1]:
                    undeviates.append(stslist[i])
                    undev_pts.append(sptslist[i])
        if len(deviates) > 0:
            deviates = np.concatenate(deviates, axis=1)
            dev_pts = np.concatenate(dev_pts)
            lssol = np.linalg.lstsq(deviates.T, dev_pts)
            #sols[2*n-1, :] = lssol[0]
            #solkey.append(n)
            ttypes[str(n)] = lssol[0]
        else:
            ttypes[str(n)] = ttypes['base']
            print 'Not enough information for {}'.format(n)
        if len(undeviates) > 0:
            undeviates = np.concatenate(undeviates, axis=1)
            undev_pts = np.concatenate(undev_pts)
            lssol = np.linalg.lstsq(undeviates.T, undev_pts)
            #sols[2*n, :] = lssol[0]
            #solkey.append(-n)
            ttypes[str(-(n))] = lssol[0]
        else:
            ttypes[str(-(n))] = ttypes['base']
            print 'Not enough information for {}'.format(-n)
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
lib = statslib.DataLib(folder)
year = 2017

#-----------RUN TOURNEY--------------------------------------------------------
test = ts.getMatchups(year)
probses = np.zeros((63,2))
for n in range(63):
    print str(n)
    perc, res = ts.genNetworkProbabilities(lib.getTeam(test[n, 0], False, year), lib.getTeam(test[n, 1], False, year), '../kraggle_model.pkl', lib, True)
    #perc, res = ts.genProbabilities(lib.getTeam(test[n, 0], False, year), lib.getTeam(test[n, 1], False, year), lib, coeffs, True)
    probses[n, :] = np.array([perc, 1.0-perc])

finalScore = ts.getLogScore(probses, test[:, 2])
corrects = sum(np.logical_and(probses[:, 0] > probses[:, 1], test[:, 2])) / 63.0
chalk = sum(test[:, 2]) / 63.0
#------------------------------------------------------------------------------
#test = ts.getCorrelation(teams[0], 2017, ts.getScaledStats(teams[0], 2017), False, True)
#test = ts.getScaledStats(teams[0], 2017)
#build_ttypes(pickle_folder, True)
#build_pickles(folder, pickle_folder, False)
#ct, sts, spts, stslist, sptslist = load_pickles(pickle_folder, True)
#perc, res, scores = ts.genProbabilities(statslib.getIDFromTeam("North Carolina", data), statslib.getIDFromTeam("Kentucky", data), 2017, False, True)
##BUILD MODEL------------------------------------------------------------------
#scsts = build_fast_stats(2010)
#sccorrs = []
#training_data = scsts[0][1][0]
#labels = scsts[0][1][1]
#sz = len(labels)
#for i in range(0, len(scsts)):
#    for n in range(len(scsts[i])):
#        if type(scsts[i][n][0]) != int:
#            training_data = np.concatenate((training_data, scsts[i][n][0]), axis=0)
#            labels = np.concatenate((labels, scsts[i][n][1]))
#            sccorrs.append(np.corrcoef(scsts[i][n][0].T, np.reshape(scsts[i][n][1], (1, len(scsts[i][n][1]))))[32, 0:-1])
#sz = shape(training_data)[0]
#sz_cut = int(sz * 7.0/8)
#sz = len(labels)
#sz_cut = int(sz * 7.0/8)
#sccorrs = np.array(sccorrs)
#model = Network([64, 32, 16, 8, 4])
#model.train(training_data[0:sz_cut, :], labels[0:sz_cut])
#test, percs = model.run(training_data[sz_cut:, :], True)
#truths = labels[sz_cut:]
#model.save('/home/artemis-new/Documents/kraggle_model_sans_def.pkl')
##-----------------------------------------------------------------------------
#test = ts.simNetworkScore(teams[0], teams[1], 2017, '../kraggle_model.pkl', True)

        
#test, res = ts.genNetworkProbabilities(lib.getTeam(statslib.getIDFromTeam("North Carolina", data), False, year), lib.getTeam(1211, False, year), '../kraggle_model_sans_def.pkl', lib, True)
#test, res = ts.genProbabilities(lib.getTeam(statslib.getIDFromTeam("North Carolina", data), False, year), lib.getTeam(1211, False, year), lib, rep_teams, ttype_coeffs, True)

#af = cluster.AffinityPropagation(preference=-50).fit(sccorrs)

#team_types = [[], [], [], [], [], [], []]
#label_types = [[], [], [], [], [], [], []]
#for i in range(0, len(scsts)):
#    for n in range(len(scsts[i])):
#        if type(scsts[i][n][0]) != int:
#            training_data = scsts[i][n][0]
#            labels = scsts[i][n][1]
#            corr = np.corrcoef(scsts[i][n][0].T, np.reshape(scsts[i][n][1], (1, len(scsts[i][n][1]))))[32, 0:-1]
#            dists = np.zeros((7,))
#            for m in range(7):
#                dists[m] = np.linalg.norm(corr - rep_teams[m])
#            ttype = np.where(dists == dists.min())[0][0]
#            team_types[ttype].append(training_data)
#            label_types[ttype].append(labels)
#            
#ttype_coeffs = np.zeros((7, 32))
#for i in range(7):
#    curr_ttype = []
#    curr_label = []
#    for n in range(len(team_types[i])):
#        if len(curr_ttype) == 0:
#            curr_ttype = team_types[i][n]
#            curr_label = label_types[i][n]
#        else:
#            curr_ttype = np.concatenate((curr_ttype, team_types[i][n]))
#            curr_label = np.concatenate((curr_label, label_types[i][n]))
#    ttype_coeffs[i, :] = np.linalg.lstsq(curr_ttype, curr_label)[0]

