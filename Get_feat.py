import sys
import numpy as np
import gzip
import pandas as pd
from os.path import join,basename
from os import makedirs
import time
import math
import glob
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import multiprocessing as mp


def SplitScpFile(sWavScp,nJobs):
    tmpDir=join('tmp',str(np.random.randint(100000)))
    makedirs(tmpDir,exist_ok=True)
    with open(sWavScp,'r') as f:
        lLines = f.read().splitlines()
    nFiles = len(lLines)
    nFilesPerJob = math.ceil(nFiles/nJobs)
    lChunks = [lLines[i*nFilesPerJob:(i+1)*nFilesPerJob] for i in range(nJobs)]
    for i in range(len(lChunks)):
        with open(join(tmpDir,'chunk'+str(i)),'w') as f:
            [print(line,file=f) for line in lChunks[i]]
    return tmpDir


def ExtractFeatures(sWavScp, iJobID = 0, outDir='out'):
    #sWavScp = 'ogi_100_wav.scp'
    context = (10,10)
    withDelta = True
    sFeatureType = 'logfbank'
    nSamplesPerChunck = 200000
    bSaveCompress = False
    
    featFunc = mfcc
    nFeatures = 13
    if sFeatureType == 'logfbank':
        featFunc = logfbank
        nFeatures = 26
    
    nFeatures = nFeatures*3 if withDelta else nFeatures
    nContxtFrames = np.sum(context)+1
    AcFeat_all = np.zeros((int(nSamplesPerChunck*2),nContxtFrames*nFeatures),dtype=float)
    aPrePad = np.zeros((context[0],nFeatures),dtype=float)
    aPostPad = np.zeros((context[1],nFeatures),dtype=float)
    iPos = 0
    dFilePos={} #Track the start and end samples of each file

    iChunk = 0
    
    pdWavScp = pd.read_csv(sWavScp,sep=' ',names=['UttID','FilePath'])
    for file in pdWavScp['FilePath']:
        sr,data = wav.read(file)
        AcFeat = featFunc(data,sr)
        if withDelta:
          AcFeat_delta = delta(AcFeat,2)
          AcFeat_delta2 = delta(AcFeat_delta,2)
          AcFeat = np.c_[AcFeat,AcFeat_delta,AcFeat_delta2]
        AcFeat = np.r_[aPrePad,AcFeat,aPostPad]
        aShiftVer = [np.roll(AcFeat,i,axis=0) for i in np.arange(context[1],-context[0]-1,-1)]
        AcFeat = np.concatenate(aShiftVer,axis=1)[context[0]:-context[1]]
        nSamples = AcFeat.shape[0]
        #print(iJobID,iPos,iChunk,AcFeat_all.shape,AcFeat.shape)
        AcFeat_all[iPos:iPos+nSamples] = AcFeat
        dFilePos[file] = (iPos,nSamples)
        iPos += nSamples
        if iPos > nSamplesPerChunck:
            #print(iJobID,iPos,iChunk,AcFeat_all.shape)
            AcFeat_all = AcFeat_all[:iPos]
            sOutScpFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.scp')
            with open(sOutScpFile,'w') as fh:
                for item in dFilePos:
                    print(item,*dFilePos[item],file=fh)
            if bSaveCompress:
                sOutFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.gz')
                print('Saving {0} samples in {1}'.format(iPos,sOutFile))
                with gzip.GzipFile(sOutFile,'w') as fh:
                    np.save(fh,AcFeat_all)
            else:
                sOutFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.npy')
                print('Saving {0} samples in {1}'.format(iPos,sOutFile))
                np.save(sOutFile,AcFeat_all)
           
 
            iChunk += 1
            
            #print(AcFeat_all.shape)
            dFilePos = {}
            AcFeat_all = np.zeros((int(nSamplesPerChunck*2),nContxtFrames*nFeatures),dtype=float)
            iPos = 0
    #print(iJobID,iPos,iChunk,AcFeat_all.shape)
    AcFeat_all = AcFeat_all[:iPos]
    sOutScpFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.scp')
    with open(sOutScpFile,'w') as fh:
        for item in dFilePos:
            print(item,*dFilePos[item],file=fh)

    if bSaveCompress:
        sOutFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.gz')
        print('Saving {0} samples in {1}'.format(iPos,sOutFile))
        with gzip.GzipFile(sOutFile,'w') as fh:
            np.save(fh,AcFeat_all)
    else:
        sOutFile = join(outDir,'egs.'+str(iJobID)+'.'+str(iChunk)+'.npy')
        print('Saving {0} samples in {1}'.format(iPos,sOutFile))
        np.save(sOutFile,AcFeat_all)
    print('Done job {0} with {1} number of chunks'.format(iJobID,iChunk))


def main(inFile, nJobs,outDir):
    makedirs(outDir,exist_ok=True)
    sJobsDir = SplitScpFile(inFile,nJobs)
    print('Creating {0} jobs in {1}'.format(nJobs,sJobsDir))
    pool = mp.Pool(nJobs)
    [pool.apply_async(ExtractFeatures, args=(join(sJobsDir,'chunk'+str(i)),i,outDir)) for i in range(nJobs)]
    pool.close()
    pool.join()
    l = [(i,join(sJobsDir,'chunk'+str(i))) for i in range(nJobs)]
    print(l)


if __name__=='__main__':
    nJobs = int(sys.argv[1])
    inFile = sys.argv[2]
    outDir = sys.argv[3]
    main(inFile,nJobs,outDir)
