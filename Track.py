import functools
import math
import os

import numpy
  
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib import colors

COLOR1 = 'Purples'
COLOR2 = 'blue'
COLOR3 = '#ffd966'  # = (255, 217, 102), some version of yellow

class Track:
  
  def __init__(self, position, frame, intensity):

    self.positions = [position]
    self.frames = [frame]
    self.intensities = [intensity]
    self.distances = [] # change in position / sqrt(change in frame)
    
  def addPosition(self, position, frame, intensity):

    distance = _calcAdjustedDistance(position, frame, self) # have to calculate this first
    
    self.positions.append(position)
    self.frames.append(frame)
    self.intensities.append(intensity)
    
    self.distances.append(distance)
    
  @property
  def averageIntensity(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.intensities[1:-1]))
    else:
      return numpy.average(numpy.array(self.intensities))
    
  @property
  def numberPositions(self):
    
    return len(self.positions)
    
  @property
  def deltaFrames(self):
    
    return self.frames[-1] - self.frames[0]
    
  def maxDistanceTravelled(self):
    
    distances = [_calcAdjustedDistance(self.positions[n], self.frames[n], self, 0) for n in range(1, len(self.positions))]
    
    return max(distances)

def _calcAdjustedDistance(position, frame, track, trackPositionIndex=-1):
  
  delta = position - track.positions[trackPositionIndex]
  #distance = numpy.sqrt(numpy.sum(delta*delta))
  distance = numpy.sqrt(numpy.sum(delta*delta)) / numpy.sqrt(frame - track.frames[trackPositionIndex])
  
  return distance
  
def _processPosition(finishedTracks, currentTracks, position, frame, intensity, maxJumpDistance, maxFrameGap):
  
  bestDist = None
  bestTrack = None
  
  for track in list(currentTracks):
    if frame > track.frames[-1] + maxFrameGap:
      currentTracks.remove(track)
      finishedTracks.add(track)
    elif frame > track.frames[-1]:
      distance = _calcAdjustedDistance(position, frame, track)
      if distance < maxJumpDistance and (bestDist is None or distance < bestDist):
        bestDist = distance
        bestTrack = track

  if bestTrack:
    bestTrack.addPosition(position, frame, intensity)
  else:
    track = Track(position, frame, intensity)
    currentTracks.add(track)

def determineTracks(fileName, numDimensions, maxJumpDistance, maxFrameGap, minNumPositions):

  finishedTracks = set()
  currentTracks = set()

  with open(fileName, 'rU') as fp:
    
    fp.readline()  # header

    for line in fp:
      if numDimensions == 2:
        (x, y, frame, intensity) = line.rstrip().split()[:4]
        position = numpy.array((float(x), float(y)))
        base = 0
      elif numDimensions == 3:
        #(x, y, z, frame, intensity) = line.rstrip().split()[:5]
        (frame, junk, x, y, z, junk, junk, junk, intensity, base) = line.rstrip().split(',')[:10]
        position = numpy.array((float(x), float(y), float(z)))

      frame = int(frame)
      intensity = float(intensity) - float(base)
    
      _processPosition(finishedTracks, currentTracks, position, frame, intensity, maxJumpDistance, maxFrameGap)
      
  finishedTracks.update(currentTracks)

  print('Number of tracks = %d' % len(finishedTracks))

  # filter out short tracks
  finishedTracks = [track for track in finishedTracks if track.numberPositions >= minNumPositions]

  print('Number of tracks after filtering for >= %d positions = %d' % (minNumPositions, len(finishedTracks)))

  return finishedTracks
    
def calcFramesPercentage(tracks, percentage):
  
  fraction = percentage / 100.0
  
  deltaFrames = [track.deltaFrames for track in tracks]
  deltaFrames.sort()
  
  n = int(fraction * len(deltaFrames))
  
  result = deltaFrames[n]
  
  print('Track frames length which is longer than %.1f%% of them all is %d' % (percentage, result))
  
  return result
  
def _calcNumTracksByBin(tracks, binSize):

  xBinMax = yBinMax = 0
  for track in tracks:
    xPosition, yPosition = track.positions[0]
    xBin = int(xPosition / binSize)
    yBin = int(yPosition / binSize)
    xBinMax = max(xBin, xBinMax)
    yBinMax = max(yBin, yBinMax)
    
  xSize = xBinMax + 1
  ySize = yBinMax + 1
  numTracks = numpy.zeros((ySize, xSize), dtype='int32')

  for track in tracks:
    xPosition, yPosition = track.positions[0]
    xBin = int(xPosition / binSize)
    yBin = int(yPosition / binSize)
    numTracks[yBin][xBin] += 1

  return numTracks
  
def calcMaxNumTracksInBin(tracks, binSize):
  
  numTracks = _calcNumTracksByBin(tracks, binSize)
  result = numpy.max(numTracks)
  
  print('Maximum number of tracks in any bin = %d' % result)
  
  return result
  
def _determineOutputFileName(filePrefix, name):
  
  dirName = os.path.dirname(filePrefix) + '_out'
  baseName = os.path.basename(filePrefix)
  if not os.path.exists(dirName):
    os.mkdir(dirName)
  
  fileName = '%s/%s_%s' % (dirName, baseName, name)
  
  return fileName
  
def saveNumTracksInBin(tracks, filePrefix, binSize, minValue, maxValue, plotDpi):
  
  numTracks = _calcNumTracksByBin(tracks, binSize)
  
  # TEMP HACK
  fileName = _determineOutputFileName(filePrefix, 'numTracks.csv')
  with open(fileName, 'w') as fp:
    fp.write('# xBin, yBin, count\n')
    for yBin in range(201):
      for xBin in range(101):
        if numTracks[yBin][xBin] > 0:
          fp.write('%d,%d,%d\n' % (xBin, yBin, numTracks[yBin][xBin]))
  
  cmap_name = COLOR1
  plt.xlim((0, 100)) # TEMP HACK
  plt.ylim((200, 0))  # y axis is backwards
  imgplot = plt.imshow(numTracks, cmap=cmap_name, vmin=minValue, vmax=maxValue, interpolation='nearest')
  #plt.xlim((0, len(numTracks[0]-1)))
  #plt.ylim((len(numTracks)-1, 0))  # y axis is backwards

  fileName = _determineOutputFileName(filePrefix, 'countHeat.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()
  
def savePositionsFramesIntensities(tracks, filePrefix):

  fileName = _determineOutputFileName(filePrefix, 'positionsFramesIntensity.csv')
  with open(fileName, 'w') as fp:
    fp.write('# track, numberPositions, deltaFrames, averageIntensity (missing out first and last ones if >= 3 positions)\n')
    for n, track in enumerate(tracks):
      fp.write('%d,%d,%d,%.1f\n' % (n+1, track.numberPositions, track.deltaFrames, track.averageIntensity))

def _calcFramesByBin(tracks, binSize):
  
  xBinMax = yBinMax = 0
  for track in tracks:
    xPosition, yPosition = track.positions[0]
    xBin = int(xPosition / binSize)
    yBin = int(yPosition / binSize)
    xBinMax = max(xBin, xBinMax)
    yBinMax = max(yBin, yBinMax)
  
  xSize = xBinMax + 1
  ySize = yBinMax + 1
  trackFrames = numpy.zeros((ySize, xSize), dtype='float32')
  ntrackFrames = numpy.zeros((ySize, xSize), dtype='int32')

  for track in tracks:
    xPosition, yPosition = track.positions[0]
    xBin = int(xPosition / binSize)
    yBin = int(yPosition / binSize)
    trackFrames[yBin][xBin] += track.frames[-1] - track.frames[0]
    ntrackFrames[yBin][xBin] += 1
  
    ntrackFramesOne = numpy.maximum(ntrackFrames, numpy.ones((ySize, xSize), dtype='int32'))
    trackFrames /= ntrackFramesOne
  
  return trackFrames
  
def calcFramesByBinPercentage(tracks, binSize, percentage):
  
  fraction = percentage / 100.0
  
  trackFrames = _calcFramesByBin(tracks, binSize)
  
  trackBinFrames = []
  for yBin in range(len(trackFrames)):
    for xBin in range(len(trackFrames[0])):
      if trackFrames[yBin][xBin] > 0:
        trackBinFrames.append(trackFrames[yBin][xBin])

  trackBinFrames.sort()
  n = int(fraction*len(trackBinFrames))
  n = min(n, len(trackBinFrames)-1)
  result = trackBinFrames[n]
  
  print('Binned track frames length which is longer than %.1f%% of them all is %d' % (percentage, result))
  
  return result
  
def saveTrackFramesInBin(tracks, filePrefix, binSize, cutoffValue, plotDpi):
  
  trackFrames = _calcFramesByBin(tracks, binSize)
  
  ySize, xSize = trackFrames.shape
  highLengths = numpy.zeros((ySize, xSize), dtype='float32')
  for yBin in range(ySize):
    for xBin in range(xSize):
      if trackFrames[yBin][xBin] > 0:
        if trackFrames[yBin][xBin] > cutoffValue:
          value = 1
        else:
          value = 0
      else:
        value = - numpy.inf
      highLengths[yBin][xBin] = value

  cmap = colors.ListedColormap([COLOR3, COLOR2])
  bounds=[0.0,0.5,1.0]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  imgplot = plt.imshow(highLengths, interpolation='nearest', origin='lower',
                      cmap=cmap, norm=norm, vmin=0, vmax=1)
                      
  plt.xlim((0, xSize-1))
  plt.ylim((ySize-1, 0))  # y axis is backwards

  fileName = _determineOutputFileName(filePrefix, 'framesByBin.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()

def saveTracksColoredByFrames(tracks, filePrefix, cutoffValue, plotDpi, numDimensions):
  
  dims = [(0, 1)]
  if numDimensions == 3:
    dims.append((0, 2))
    dims.append((1, 2))
    
  for xDim, yDim in dims:
    for track in tracks:
      xpositions = [position[xDim] for position in track.positions]
      ypositions = [position[yDim] for position in track.positions]
    
      if track.deltaFrames >= cutoffValue:
        color = COLOR2
      else:
        color = COLOR3
      plt.plot(xpositions, ypositions, color=color)
    
    plt.ylim(plt.ylim()[::-1])
  
    if numDimensions == 2:
      d = ''
    else:
      d = '%d%d' % (xDim+1, yDim+1)
    fileName = _determineOutputFileName(filePrefix, 'tracksByFrames%s.png' % d)
    plt.savefig(fileName, dpi=plotDpi, transparent=True)
    #plt.show()
    plt.close()
  
def saveTracksColoredByDistance(tracks, filePrefix, cutoffValue, plotDpi, numDimensions):
  
  dims = [(0, 1)]
  if numDimensions == 3:
    dims.append((0, 2))
    dims.append((1, 2))
    
  for xDim, yDim in dims:
    for track in tracks:
      xpositions = [position[xDim] for position in track.positions]
      ypositions = [position[yDim] for position in track.positions]
    
      if track.maxDistanceTravelled() >= cutoffValue:
        color = COLOR2
      else:
        color = COLOR3
      plt.plot(xpositions, ypositions, color=color)
    
    plt.ylim(plt.ylim()[::-1])
  
    if numDimensions == 2:
      d = ''
    else:
      d = '%d%d' % (xDim+1, yDim+1)
    fileName = _determineOutputFileName(filePrefix, 'tracksByDistance%s.png' % d)
    plt.savefig(fileName, dpi=plotDpi, transparent=True)
    #plt.show()
    plt.close()
  
def _getResidenceTimes(tracks):
  
  return [track.deltaFrames for track in tracks]
  
def saveResidenceTimes(tracks, filePrefix):
  
  residenceTimes = _getResidenceTimes(tracks)
  
  fileName = _determineOutputFileName(filePrefix, 'residenceTimes.csv')
  with open(fileName, 'w') as fp:
    fp.write('%s\n' % ','.join(['%d' % residenceTime for residenceTime in residenceTimes]))
  
def _getSurvivalCounts(tracks, maxSize=0):

  residenceTimes = _getResidenceTimes(tracks)
  if maxSize == 0:
    #maxSize = max(residenceTimes)
    maxSize = 1 + max(residenceTimes)
  survivalCounts = numpy.zeros(maxSize, dtype='int32')
  ones = numpy.ones(maxSize, dtype='int32')
  for residenceTime in residenceTimes:
    #survivalCounts[:residenceTime] += ones[:residenceTime]
    survivalCounts[:1+residenceTime] += ones[:1+residenceTime]
    
  return survivalCounts
  
def saveSurvivalCounts(tracks, filePrefix, maxSize=0):
  
  survivalCounts = _getSurvivalCounts(tracks, maxSize)
  
  fileName = _determineOutputFileName(filePrefix, 'survivalCounts.csv')
  with open(fileName, 'w') as fp:
    fp.write('%s\n' % ','.join(['%d' % survivalCount for survivalCount in survivalCounts]))

def _fitExponentials(fitUsingLog, xdata, *params):

  nexps = len(params) // 2
  params = list(params)

  ydata = numpy.zeros(len(xdata), dtype='float32')
  for i in range(nexps):
    ydata += params[i] * numpy.exp(-xdata*params[i+nexps])

  if fitUsingLog:
    return numpy.log(ydata)
  else:
    return ydata
    
def _initialFitSurvivalParameterEstimate(ydata):
  
  # assumes ydata[0] > 0 (in fact it is 1.0)
  a = ydata[0]
  b = 0.0
  for m in range(min(len(ydata)-1, 10), 0, -1):
    if ydata[m] > 0:
      b = math.log(ydata[0] / ydata[m]) / m
      break
      
  return (a, b)
    
def _adjustedSurvivalParams(params):
  
  # if fit is A exp(-B x) + C exp(-D x) then parameters go from
  # (A, C, B, D) to (A/(A+C), 1/B, C/(A+C), 1/D)
  
  numberExponentials = len(params) // 2
  s = sum(params[:numberExponentials])
  paramsNew = (2*numberExponentials)*[0]
  for i in range(numberExponentials):
    paramsNew[2*i] = params[i] / s
    paramsNew[2*i+1] = 1 / params[i+numberExponentials]
    
  return paramsNew
  
def _bootstrapFit(xdata, ydata, params_opt, fitFunc, fitUsingLogData=False, adjustedParamsFunc=None, ntrials=1000, fp=None):
  
  ndata = len(xdata)
  paramsList =  []
  for trial in range(ntrials):
    indices = range(ndata)
    indices = numpy.random.choice(indices, ndata)
    x = xdata[indices]
    y = numpy.log(ydata[indices]) if fitUsingLogData else ydata[indices]
    bounds0 = (len(params_opt))*[0]
    bounds1 = (len(params_opt))*[numpy.inf]
    bounds = (bounds0, bounds1)
    try:
      params, params_cov = curve_fit(fitFunc, x, y, p0=params_opt, bounds=bounds)
    except: # fit might fail
      continue
    if adjustedParamsFunc:
      params = adjustedParamsFunc(params)
    if fp:
      fp.write('%s\n' % ','.join(['%.3f' % p for p in params]))
    paramsList.append(params)
    
  paramsArray = numpy.array(paramsList)
  paramsMean = numpy.mean(paramsArray, axis=0)
  paramsStd = numpy.std(paramsArray, axis=0)
  #print('Bootstrap parameter mean = %s' % paramsMean)
  #print('Bootstrap parameter standard deviation = %s' % paramsStd)
  
  return paramsStd
    
def _writeFitSurvivalHeader(fp, maxNumberExponentials):
  
  data = ['nexp']
  for m in range(maxNumberExponentials):
    data.append('ampl%d' % (m+1))
    data.append('T%d' % (m+1))
  for m in range(maxNumberExponentials):
    data.append('amplErr%d' % (m+1))
    data.append('TErr%d' % (m+1))
  data.append('rss')
  data.append('bic')
    
  data = ','.join(data)
  
  fp.write(data + '\n')
    
def _writeFitSurvivalParams(fp, params, paramsStd, rss, maxNumberExponentials, ndata):
  
  numberExponentials = len(params) // 2
  params = _adjustedSurvivalParams(params)
  n = 2 * (maxNumberExponentials - numberExponentials)
  
  data = ['%d' % numberExponentials]
  data.extend(['%.3f' % param for param in params])
  data.extend(n*[''])
  
  data.extend(['%.3f' % param for param in paramsStd])
  data.extend(n*[''])
  
  data.append('%.3f' % rss)
  
  bic = numpy.log(ndata) * (len(params) + 1) + ndata * (numpy.log(2*numpy.pi*rss/ndata) + 1)
  data.append('%.3f' % bic)
    
  data = ','.join(data)
  
  fp.write(data + '\n')
  
def fitSurvivalCounts(tracks, filePrefix, maxNumberExponentials=1, minNumPositions=2, fitUsingLogData=False, plotDpi=600):
  
  survivalCounts = _getSurvivalCounts(tracks)
  
  survivalCounts = survivalCounts[minNumPositions-1:]
  
  ydata = survivalCounts.astype('float32')
  ydata /= ydata[0]
  xdata = numpy.arange(len(ydata))
  
  data = numpy.log(ydata) if fitUsingLogData else ydata
  fitFunc = functools.partial(_fitExponentials, fitUsingLogData)
  
  params0 = _initialFitSurvivalParameterEstimate(ydata)
  
  fileName = _determineOutputFileName(filePrefix, 'fitSurvivalCounts.csv')
  with open(fileName, 'w') as fp:
    _writeFitSurvivalHeader(fp, maxNumberExponentials)
    params_list = []
    for numberExponentials in range(1, maxNumberExponentials+1):
      #params_opt, params_cov = curve_fit(_fitExponentials, xdata, ydata, p0=params0)
      bounds0 = (len(params0))*[0]
      bounds1 = (len(params0))*[numpy.inf]
      bounds = (bounds0, bounds1)
      params_opt, params_cov = curve_fit(fitFunc, xdata, data, p0=params0, bounds=bounds)
      ss = '' if numberExponentials == 1 else 's'
      params_err = numpy.sqrt(numpy.diag(params_cov))
      params_opt = tuple(params_opt)
      yfit = _fitExponentials(fitUsingLogData, xdata, *params_opt)
      if fitUsingLogData:
        yfit = numpy.exp(yfit)
      rss = numpy.sum((yfit - ydata)**2)
      print('Fitting survival counts with %d exponential%s, parameters = %s, parameter standard deviation = %s, rss = %f' % (numberExponentials, ss, params_opt, params_err, rss))
      #fileNameBootstrap = _determineOutputFileName(filePrefix, 'bootstrapParams_%d.csv' % numberExponentials)
      #with open(fileNameBootstrap, 'w') as fpBootstrap:
      #  paramsStd = _bootstrapFit(xdata, ydata, params_opt, fp=fpBootstrap)
      paramsStd = _bootstrapFit(xdata, ydata, params_opt, fitFunc, fitUsingLogData, _adjustedSurvivalParams)
      _writeFitSurvivalParams(fp, params_opt, paramsStd, rss, maxNumberExponentials, len(xdata))
      params_list.append(params_opt)
      params0 = list(params_opt[:numberExponentials]) + [0.1] + list(params_opt[numberExponentials:]) + [0.0]
    
  colors = ['blue', 'red', 'green', 'yellow', 'black']  # assumes no more than 4 exponentials
  plt.plot(xdata, ydata, color=colors[-1])
  for n in range(maxNumberExponentials):
    yfit = _fitExponentials(fitUsingLogData, xdata, *params_list[n])
    if fitUsingLogData:
      yfit = numpy.exp(yfit)
    plt.plot(xdata, yfit, color=colors[n])
  
  fileName = _determineOutputFileName(filePrefix, 'survivalCountsFit.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()
  
  fileName = _determineOutputFileName(filePrefix, 'survivalCounts.csv')
  with open(fileName, 'w') as fp:
    fp.write('%s\n' % ','.join(['%s' % w for w in xdata]))
    fp.write('%s\n' % ','.join(['%s' % w for w in survivalCounts]))
    fp.write('%s\n' % ','.join(['%s' % w for w in ydata]))
    fp.write('%s\n' % ','.join(['%s' % w for w in yfit]))
            
if __name__ == '__main__':

  import os
  import sys
  
  if len(sys.argv) == 1:
    print('Need to specify one or more data files')
    sys.exit()
  
  fileNames = sys.argv[1:]
  for fileName in fileNames:
    print('Determining tracks for %s' % fileName)
    tracks = determineTracks(fileName, numDimensions=2, maxJumpDistance=100, maxFrameGap=2, minNumPositions=3)
    filePrefix = fileName[:-4]
    savePositionsFramesIntensities(tracks, filePrefix)
    
 