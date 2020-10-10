import numpy as np
import pandas as pd
from os.path import join

datasets = ['ogi','myst','cu','cmu']
sPhonesFile = "phones.txt"

phones = pd.read_csv('phones.txt',sep=' ',names=['PhoneSymb','PhoneIndx'])

for data in datasets:
    sDataFile = data+'.ctm'
    dfCTM = pandas.read_csv(join('ali_data',sDataFile),sep=' ',names=['fileID','Channel','StartTime','Duration','PhoneIndx'])
    dfPhone = dfCTM.groupby(by=['PhoneIndx'],as_index=False).sum()[['Duration','PhoneIndx']])
    dfPhone = dfPhone.merge(phones,how='left',on='PhoneIndx')
 
