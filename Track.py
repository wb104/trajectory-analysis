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

def _calcAdjustedDistance(position, frame, track):
  
  delta = position - track.positions[-1]
  #distance = numpy.sqrt(numpy.sum(delta*delta))
  distance = numpy.sqrt(numpy.sum(delta*delta)) / numpy.sqrt(frame - track.frames[-1])
  
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
  
  cmap_name = COLOR1
  imgplot = plt.imshow(numTracks, cmap=cmap_name, vmin=minValue, vmax=maxValue, interpolation='nearest')
  plt.xlim((0, len(numTracks[0]-1)))
  plt.ylim((len(numTracks)-1, 0))  # y axis is backwards

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

def saveTracksColoredByFrames(tracks, filePrefix, cutoffValue, plotDpi):
  
  for track in tracks:
    xpositions = [position[0] for position in track.positions]
    ypositions = [position[1] for position in track.positions]
    
    if (track.frames[-1]-track.frames[0]) >= cutoffValue: # HACK
      color = COLOR2
    else:
      color = COLOR3
    plt.plot(xpositions, ypositions, color=color)
    
  plt.ylim(plt.ylim()[::-1])
  
  fileName = _determineOutputFileName(filePrefix, 'tracksByFrames.png')
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
    maxSize = max(residenceTimes)
  survivalCounts = numpy.zeros(maxSize, dtype='int32')
  ones = numpy.ones(maxSize, dtype='int32')
  for residenceTime in residenceTimes:
    survivalCounts[:residenceTime] += ones[:residenceTime]
    
  return survivalCounts
  
def saveSurvivalCounts(tracks, filePrefix, maxSize=0):
  
  survivalCounts = _getSurvivalCounts(tracks, maxSize)
  
  fileName = _determineOutputFileName(filePrefix, 'survivalCounts.csv')
  with open(fileName, 'w') as fp:
    fp.write('%s\n' % ','.join(['%d' % survivalCount for survivalCount in survivalCounts]))

def _fitSurvival(xdata, *params):

  nexps = len(params) // 2
  params = list(params)

  ydata = numpy.zeros(len(xdata), dtype='float32')
  for i in range(nexps):
    ydata += params[i] * numpy.exp(-xdata*params[i+nexps])

  return ydata

def _initialFitParameterEstimate(ydata):
  
  # assumes ydata[0] > 0 (in fact it is 1.0)
  a = ydata[0]
  b = 0.0
  for m in range(min(len(ydata)-1, 10), 0, -1):
    if ydata[m] > 0:
      b = math.log(ydata[0] / ydata[m]) / m
      break
      
  return (a, b)
    
def _adjustedParams(params):
  
  # if fit is A exp(-B x) + c exp(-D x) then parameters go from
  # (A, C, B, D) to (A/(A+C), 1/B, C/(A+C), 1/D)
  
  numberExponentials = len(params) // 2
  s = sum(params[:numberExponentials])
  paramsNew = (2*numberExponentials)*[0]
  for i in range(numberExponentials):
    paramsNew[2*i] = params[i] / s
    paramsNew[2*i+1] = 1 / params[i+numberExponentials]
    
  return paramsNew
  
def _bootstrapFit(xdata, ydata, params_opt, ntrials=1000):
  
  ndata = len(xdata)
  paramsList =  []
  for trial in range(ntrials):
    indices = range(ndata)
    indices = numpy.random.choice(indices, ndata)
    x = xdata[indices]
    y = ydata[indices]
    try:
      params, params_cov = curve_fit(_fitSurvival, x, y, p0=params_opt)
    except: # fit might fail
      pass
    params = _adjustedParams(params)
    paramsList.append(params)
    
  paramsArray = numpy.array(paramsList)
  paramsMean = numpy.mean(paramsArray, axis=0)
  paramsStd = numpy.std(paramsArray, axis=0)
  #print('Bootstrap parameter mean = %s' % paramsMean)
  #print('Bootstrap parameter standard deviation = %s' % paramsStd)
  
  return paramsStd
    
def _writeParams(fp, params, paramsStd, rss, maxNumberExponentials):
  
  numberExponentials = len(params) // 2
  params = _adjustedParams(params)
  data = ['%d' % numberExponentials]
  data.extend(['%.3f' % param for param in params])
  n = 2 * (maxNumberExponentials - numberExponentials)
  data.extend(n*[''])
  data.extend(['%.3f' % param for param in paramsStd])
  data.extend(n*[''])
  data.append('%.3f' % rss)
    
  data = ','.join(data)
  
  fp.write(data + '\n')
  
def fitSurvivalCounts(tracks, filePrefix, maxNumberExponentials=1, plotDpi=600):
  
  survivalCounts = _getSurvivalCounts(tracks)
  
  ydata = survivalCounts.astype('float32')
  ydata /= ydata[0]
  xdata = numpy.arange(len(ydata))
  
  params0 = _initialFitParameterEstimate(ydata)
  
  fileName = _determineOutputFileName(filePrefix, 'fitSurvivalCounts.csv')
  with open(fileName, 'w') as fp:
    params_list = []
    for numberExponentials in range(1, maxNumberExponentials+1):
      params_opt, params_cov = curve_fit(_fitSurvival, xdata, ydata, p0=params0)
      ss = '' if numberExponentials == 1 else 's'
      params_err = numpy.sqrt(numpy.diag(params_cov))
      params_opt = tuple(params_opt)
      yfit = _fitSurvival(xdata, *params_opt)
      rss = numpy.sum((yfit - ydata)**2)
      print('Fitting survival counts with %d exponential%s, parameters = %s, parameter standard deviation = %s, rss = %f' % (numberExponentials, ss, params_opt, params_err, rss))
      paramsStd = _bootstrapFit(xdata, ydata, params_opt)
      _writeParams(fp, params_opt, paramsStd, rss, maxNumberExponentials)
      params_list.append(params_opt)
      params0 = list(params_opt[:numberExponentials]) + [0.1] + list(params_opt[numberExponentials:]) + [0.0]
    
  colors = ['blue', 'red', 'green', 'yellow', 'black']  # assumes no more than 4 exponentials
  plt.plot(xdata, ydata, color=colors[-1])
  for n in range(maxNumberExponentials):
    yfit = _fitSurvival(xdata, *params_list[n])
    plt.plot(xdata, yfit, color=colors[n])
  
  fileName = _determineOutputFileName(filePrefix, 'survivalCountsFit.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()
      
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
    
 