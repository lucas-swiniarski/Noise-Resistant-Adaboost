import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

def prepareDataset(dataset, Y_index): 
    dataset["Y"] = np.where(dataset[Y_index] == 0, -1, 1)
    dataset = dataset.drop(Y_index, 1)
    dataset[dataset.drop(Y_index, 1).columns.values] = scale(dataset[dataset.drop(Y_index, 1).columns.values])
    return dataset


def getWaveForm():
    # We only keep two types of waveform : 0 & 1, not 2
    waveform = pd.read_csv("data/waveform.data", header=None)
    waveform = waveform[waveform[21] < 2]
    waveform["Y"] = np.where(waveform[21] == 0, -1, 1)
    waveform = waveform.drop(21, 1)
    waveform[waveform.drop("Y", 1).columns.values] = scale(waveform[waveform.drop("Y", 1).columns.values])
    waveform.to_csv("data/waveform.data.clean.csv")

def getBanana():
    # Already scaled
    banana = pd.read_csv("data/banana_data.csv", header=None)
    banana["Y"] = banana[0]
    banana = banana.drop(0, 1)
    banana.to_csv("data/banana.data.clean.csv")

def getRingNorm():
    ringnorm = pd.read_csv("data/ringnorm.data", sep='\s+', header=None)
    ringnorm["Y"] = np.where(ringnorm[20] == 0, -1, 1)
    ringnorm = ringnorm.drop(20, 1)
    ringnorm[ringnorm.drop("Y", 1).columns.values] = scale(ringnorm[ringnorm.drop("Y", 1).columns.values])
    ringnorm.to_csv("data/ringnorm.data.clean.csv")

def getTwoNorm():
    twonorm = pd.read_csv("data/twonorm.data", sep="\s+", header=None)
    twonorm["Y"] = np.where(twonorm[20] == 0, -1, 1)
    twonorm = twonorm.drop(20, 1)
    twonorm[twonorm.drop("Y", 1).columns.values] = scale(twonorm[twonorm.drop("Y", 1).columns.values])
    twonorm.to_csv("data/twonorm.data.clean.csv")

def getSpambase():
    spambase = pd.read_csv("data/spambase.data.shuffled", header=None)
    spambase["Y"] = np.where(spambase[57] == 0, -1, 1)
    spambase = spambase.drop(57, 1)
    spambase[spambase.drop("Y", 1).columns.values] = scale(spambase[spambase.drop("Y", 1).columns.values])
    spambase.to_csv("data/spambase.data.clean.csv")

def getAbalone():
    abalone = pd.read_csv("data/abalone.data")
    abalone = abalone[abalone["Sex"] != "I"]
    abalone["Y"] = np.where(abalone["Sex"] == "M", 1, -1)
    abalone = abalone.drop("Sex", 1)
    abalone.to_csv("data/abalone.data.clean.csv")

getWaveForm()
getBanana()
getRingNorm()
getTwoNorm()
getSpambase()
