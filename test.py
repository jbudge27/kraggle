# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:31 2018

@author: artemis-new
"""

from TeamStat import TeamStat
from sklearn.cluster import KMeans
import statslib
import testlib as ts
#import scipy as sp
import numpy as np
#from sklearn import svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler
folder = '../kraggle_data'
data = statslib.loadDataDict(folder)
#teams = np.arange(1140, 1400)
#teams = [statslib.getIDFromTeam("North Carolina", data), statslib.getIDFromTeam("Gonzaga", data), statslib.getIDFromTeam("Villanova", data), statslib.getIDFromTeam("Wisconsin", data)]
test = ts.getMatchups(2017)
probses = np.zeros((63,2))
for n in range(63):
    print str(n)
    perc, res, scores = ts.genProbabilities(test[n, 0], test[n, 1], 2017)
    probses[n, :] = np.array(perc)

finalScore = ts.getLogScore(probses, test[:, 2])
#perc, res, scores = ts.genProbabilities(statslib.getIDFromTeam("Oklahoma", data), statslib.getIDFromTeam("BYU", data), 2017, True)
#team = TeamStat(folder, 1140)
#test = ts.getScaledStats(1140, 2017, True)
#lssol = np.linalg.lstsq(test.T, team.getStatsByYear(2017)[:, 3].T)

#goodones = [19, 18, 16, 12, 13, 9, 7, 5, 6]
##10 is ptdiff
#corrs = np.zeros((11,))
#ranges = np.zeros((11,2))
#scsts = []
#scpts = []
#for i in teams:
#    curr_team = TeamStat(folder, i)
#    tmp = curr_team.getStatsByYear(2017)[:, 3]
#    if len(tmp) > 0:
#        scpts.append(tmp)
#        scsts.append(ts.getScaledStats(i, 2017))
#    sts = curr_team.getDerivedStatsByYear(2017)
#    for idx, g in enumerate(goodones):
#        corrs[idx] += np.corrcoef(sts[:, g], sts[:, 10])[1, 0]
#        ranges[idx, 0] = max([ranges[idx, 0], sts[:,g].max()])
#        ranges[idx, 1] = min([ranges[idx, 1], sts[:,g].min()])
#    stsdef, hg = ts.getDefStats(i, 2017, True)
#    corrs[9] += np.corrcoef(stsdef[:, 0], sts[:, 10])[1, 0] 
#    corrs[10] += np.corrcoef(stsdef[:, 3], sts[:, 10])[1, 0]
#    ranges[9, 0] = max([ranges[9, 0], stsdef[:,0].max()])
#    ranges[9, 1] = min([ranges[9, 1], stsdef[:,3].min()])
#    ranges[10, 0] = max([ranges[10, 0], stsdef[:,0].max()])
#    ranges[10, 1] = min([ranges[10, 1], stsdef[:,3].min()])
#corrs /= len(teams)
#nmstats = 7
#fp = np.corrcoef(scsts[0], scpts[0])[nmstats, 0:-1]
#corrs = np.reshape(np.corrcoef(scsts[0], scpts[0])[nmstats, 0:-1], (nmstats,1))
#dists = np.zeros((len(scpts)-1,))
#scstats = scsts[0]
#scpoints = scpts[0]
#for i in range(1, len(scpts)):
#    tcorr = np.corrcoef(scsts[i], scpts[i])[nmstats, 0:-1]
#    corrs = np.concatenate((corrs, np.reshape(tcorr, (nmstats,1))), axis=1)
#    dists[i-1] = np.linalg.norm(fp - tcorr)
#    scstats = np.concatenate((scstats, scsts[i]), axis=1)
#    scpoints = np.concatenate((scpoints, scpts[i]))
#    
#lssol = np.linalg.lstsq(scstats.T, scpoints)
#
#est = KMeans(n_clusters = 3)
#est.fit(corrs.T)
#team_types = est.cluster_centers_
#dif_stats = {'1': [], '2': [], '3': []}
#dif_points = {'1': [], '2': [], '3': []}
#for i in range(len(scpts)):
#    cdist = np.array([np.linalg.norm(corrs[:, i] - team_types[0, :]), 
#                      np.linalg.norm(corrs[:, i] - team_types[1, :]), np.linalg.norm(corrs[:, i] - team_types[2, :])])
#    if cdist.min() == cdist[0]:
#        if len(dif_stats['1']) == 0:
#            dif_stats['1'].append(scsts[i])
#            dif_points['1'].append(scpts[i])
#        else:
#            dif_stats['1'][0] = np.concatenate((dif_stats['1'][0], scsts[i]), axis=1)
#            dif_points['1'][0] = np.concatenate((dif_points['1'][0], scpts[i]))
#    if cdist.min() == cdist[1]:
#        if len(dif_stats['2']) == 0:
#            dif_stats['2'].append(scsts[i])
#            dif_points['2'].append(scpts[i])
#        else:
#            dif_stats['2'][0] = np.concatenate((dif_stats['2'][0], scsts[i]), axis=1)
#            dif_points['2'][0] = np.concatenate((dif_points['2'][0], scpts[i]))
#    if cdist.min() == cdist[2]:
#        if len(dif_stats['3']) == 0:
#            dif_stats['3'].append(scsts[i])
#            dif_points['3'].append(scpts[i])
#        else:
#            dif_stats['3'][0] = np.concatenate((dif_stats['3'][0], scsts[i]), axis=1)
#            dif_points['3'][0] = np.concatenate((dif_points['3'][0], scpts[i]))
