# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:31 2018

@author: artemis-new
"""

from TeamStat import TeamStat
import statslib
import scipy as sp
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

folder = '/home/artemis-new/Documents/kraggle'
data = statslib.loadDataDict(folder)
team_id = statslib.getIDFromTeam("Texas", data)
#ids = np.arange(1101, 1464)
#ratings = np.zeros((len(ids), 3))
#qts = np.zeros((23,))
#for idx, team_id in enumerate(ids):
qt = np.zeros((24,))
team = TeamStat(folder, team_id, False)
year = 2017
#ratings[idx, 0] = i
teamRawStats = team.getStatsByYear(year)
teamStats = team.getDerivedStatsByYear(year)
teamRank = team.getAverageRank(year)
av_fgp = teamStats[:, 0].mean()
impact = []
oppImpact = []
rks = []

for g in range(len(team.getGamesByYear(year))):
    print '.',
    oppteam_id = teamRawStats[g, 33]
    oppteam = TeamStat(folder, int(oppteam_id))
    oppteamStats = oppteam.getDerivedStatsByYear(year)
    #oppteamRank = oppteam.getAverageRank(year)
    #oppRank = oppteamRank[oppteam.getGameNumber(year, teamRawStats[g, 2]),1]
    opp_avfgp = oppteamStats[:,4].mean()
    impact.append(teamStats[g,9] - opp_avfgp)
    oppImpact.append(teamStats[g,4])
    #rks.append(teamRank[g, 1] - oppRank)
    
impact = np.array(impact)
oppImpact = np.array(oppImpact)
#rks = np.array(rks)


#plt.close('all')
#plt.plot(impact)
#plt.plot(oppImpact)
print str(impact.mean())
#print str(oppImpact.std())

#t = team.getJeffStats(year)
#q = np.concatenate((t, teamStats,np.reshape(impact, (len(impact),1)) ,np.reshape(oppImpact, (len(oppImpact),1)), np.reshape(rks, (len(oppImpact), 1))), axis=1)
#for n in range(24):
#    qt[n] = np.corrcoef(q[:,n], teamStats[:,10])[1,0]
#qts = qts + qt
#t = team.getGameLog(team.season_games[273])



#timeouts = []
#subs = []
##8, 10
#for i in range(435, 468):
#    print '.',
#    t = team.getGameLog(team.season_games[i])
#    for n in range(len(t)):
#        if t[n][8] == str(team_id):
#            if t[n][10] == "foul_pers":
#                subs.append(t[n][7])
#            elif t[n][10] == "sub_in":
#                timeouts.append(t[n][7])
#                
#for s in range(len(subs)):
#    subs[s] = int(subs[s])
#    
#for s in range(len(timeouts)):
#    timeouts[s] = int(timeouts[s])
#    
#subs = np.array(subs)
#timeouts = np.array(timeouts)
#
#h = np.histogram(subs[subs < 2400], 40)
#q = np.histogram(timeouts[timeouts < 2400], 40)
#
#plt.plot(h[0])
#plt.plot(q[0])
    
#q = np.concatenate((np.reshape(t[:,2], (len(impact),1)), np.reshape(impact, (len(impact),1)), teamStats[:,12:14]), axis=1)
#scaler = StandardScaler()
#scaler.fit(q[0:15, :])
#training_data = scaler.transform(q[0:15, :])
#test_data = scaler.transform(q[-15:, :])
#
#svm_model = svm.LinearSVC(multi_class='crammer_singer')
#svm_model.fit(training_data, team.season_wins[0:15])
#svm_targets = svm_model.predict(test_data)
#
#plt.plot(team.season_wins[-15:])
#plt.plot(svm_targets)