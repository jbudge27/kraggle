# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:31 2018

@author: artemis-new
"""

from TeamStat import TeamStat
import statslib
import testlib as ts
#import scipy as sp
import numpy as np
#from sklearn import svm
#from sklearn.neural_network import MLPClassifier
#from sklearn.preprocessing import StandardScaler
folder = '../kraggle_data'
data = statslib.loadDataDict(folder)
#teams = [statslib.getIDFromTeam("North Carolina", data), statslib.getIDFromTeam("Gonzaga", data), statslib.getIDFromTeam("Villanova", data)]
#test = ts.getMatchups(2017)
#probses = np.zeros((63,2))
#for n in range(15):
#    print str(n)
#    perc, res, scores = ts.genProbabilities(test[n, 0], test[n, 1], 2017)
#    probses[n, :] = np.array(perc)
#
#finalScore = ts.getLogScore(probses, test[:, 2])
#perc, res, scores = ts.genProbabilities(1276, 1332, 2017, True)
team = TeamStat(folder, 1140)
test = ts.getScaledStats(1140, 2017, True)
lssol = np.linalg.lstsq(test.T, team.getStatsByYear(2017)[:, 3].T)

#goodones = [19, 18, 16, 12, 13, 9, 7, 5, 6]
##10 is ptdiff
#corrs = np.zeros((11,))
#ranges = np.zeros((11,2))
#for i in teams:
#    curr_team = TeamStat(folder, i)
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
    
    
    

