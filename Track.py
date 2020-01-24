import functools
import math
import os

import numpy
  
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.stats import gmean

from matplotlib import pyplot as plt
from matplotlib import colors

COLOR1 = 'Purples'
COLOR2 = 'blue'
COLOR3 = '#ffd966'  # = (255, 217, 102), some version of yellow

class Track:
  
  def __init__(self, position, frame, intensity, signal=None, background=0.0, noise=0.0, precision=0.0):

    self.positions = [position]
    self.frames = [frame]
    self.intensities = [intensity]
    self.distances = [] # change in position / sqrt(change in frame)
    
    if signal is None:
      signal = intensity + background
    self.signals = [signal]
    self.backgrounds = [background]
    self.noises = [noise]
    self.precisions = [precision]
    
  def addPosition(self, position, frame, intensity, signal=None, background=0.0, noise=0.0, precision=0.0):

    distance = _calcAdjustedDistance(position, frame, self) # have to calculate this first
    
    self.positions.append(position)
    self.frames.append(frame)
    self.intensities.append(intensity)
    
    self.distances.append(distance)
    
    if signal is None:
      signal = intensity + background
    self.signals.append(signal)
    self.backgrounds.append(background)
    self.noises.append(noise)
    self.precisions.append(precision)

  @property
  def averagePosition(self):
    
    return numpy.average(numpy.array(self.positions), axis=0)
    
  @property
  def averageIntensity(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.intensities[1:-1]))
    else:
      return numpy.average(numpy.array(self.intensities))
    
  @property
  def geometricMeanIntensity(self):
    
    if self.numberPositions >= 3:
      return gmean(numpy.array(self.intensities[1:-1]))
    else:
      return gmean(numpy.array(self.intensities))
    
  @property
  def averageSignal(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.signals[1:-1]))
    else:
      return numpy.average(numpy.array(self.signals))
    
  @property
  def averageBackground(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.backgrounds[1:-1]))
    else:
      return numpy.average(numpy.array(self.backgrounds))
    
  @property
  def averageNoise(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.noises[1:-1]))
    else:
      return numpy.average(numpy.array(self.noises))
    
  @property
  def averagePrecision(self):
    
    if self.numberPositions >= 3:
      return numpy.average(numpy.array(self.precisions[1:-1]))
    else:
      return numpy.average(numpy.array(self.precisions))
    
  @property
  def averageSignalToNoise(self):
    
    if self.numberPositions >= 3:
      noises = numpy.array(self.noises[1:-1])
      signals = numpy.array(self.signals[1:-1])
    else:
      noises = numpy.array(self.noises)
      signals = numpy.array(self.signals)

    if len(signals) > 0:
      ratio = signals / noises
      return numpy.average(ratio)
    else:
      return 0.0
      
  @property
  def numberPositions(self):
    
    return len(self.positions)
    
  @property
  def deltaFrames(self):
    
    return self.frames[-1] - self.frames[0]
    
  def maxDistanceTravelled(self):
    
    distances = [_calcAdjustedDistance(self.positions[n], self.frames[n], self, 0) for n in range(1, len(self.positions))]
    
    return max(distances)
    
  def meanSquareDisplacements(self):
    
    msds = []
    for n in range(1, self.numberPositions):
      msd = 0
      for i in range(self.numberPositions-n):
        d = _calcDistance(self.positions[i+n], self.frames[i+n], self, i)
        msd += d*d
      msd /= self.numberPositions - n
      msds.append(msd)
      
    return msds
    
  def adjustedDistance(self, firstIndex=0, lastIndex=-1):
    
    return _calcAdjustedDistance(self.positions[firstIndex], self.frames[firstIndex], self, lastIndex)

def _calcDistance(position, frame, track, trackPositionIndex=-1):
  
  delta = position - track.positions[trackPositionIndex]
  distance = numpy.sqrt(numpy.sum(delta*delta))
  
  return distance
  
def _calcAdjustedDistance(position, frame, track, trackPositionIndex=-1):
  
  delta = position - track.positions[trackPositionIndex]
  distance = numpy.sqrt(numpy.sum(delta*delta)) / numpy.sqrt(abs(frame - track.frames[trackPositionIndex]))
  
  return distance
  
def _processPosition(finishedTracks, currentTracks, position, frame, intensity, signal, background, noise, precision, maxJumpDistance, maxFrameGap):
  
  position = numpy.array(position)
  
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
    bestTrack.addPosition(position, frame, intensity, signal, background, noise, precision)
  else:
    track = Track(position, frame, intensity, signal, background, noise, precision)
    currentTracks.add(track)

def readTrack(fileName, numDimensions):
  
  track = None
  
  with open(fileName, 'rU') as fp:
    
    fp.readline()  # header
    
    for line in fp:
      
      frame, track_id, x, y, z, mean_intensity, median_intensity = line.rstrip().split(',')[:7]
      
      if numDimensions == 1:
        position = (float(x),)
      elif numDimensions == 2:
        position = (float(x), float(y))
      elif numDimensions == 3:
        position = (float(x), float(y), float(z))
      
      frame = int(frame)
      position = numpy.array(position)  
      intensity = float(median_intensity)
      
      if track is None:
        track = Track(position, frame, intensity)
      else:
        track.addPosition(position, frame, intensity)
        
  return track

def readOldPositionFile(fileName, numDimensions):
  
  with open(fileName, 'rU') as fp:
    
    fp.readline()  # header

    n = 0
    for n, line in enumerate(fp):
      if n > 0 and n % 100000 == 0:
        print('reading line %d' % n)
      if numDimensions == 1:
        (frame, x) = line.rstrip().split()[:2]
        position = (float(x),)
        intensity = base = 0
      elif numDimensions == 2:
        (x, y, frame, intensity) = line.rstrip().split()[:4]
        position = (float(x), float(y))
        base = 0
      elif numDimensions == 3:
        #(x, y, z, frame, intensity) = line.rstrip().split()[:5]
        (frame, junk, x, y, z, junk, junk, junk, signal, background) = line.rstrip().split(',')[:10]
        position = (float(x), float(y), float(z))

      frame = int(frame)
      signal = float(signal)
      background = float(background)
      intensity = signal - background
      
      noise = precision = 0.0
      
      yield (frame, intensity, position, signal, background, noise, precision)
    #print('found %d lines' % n)

def readNewPositionFile(fileName, numDimensions):
  
  with open(fileName, 'rU') as fp:
    
    n = 0
    for n, line in enumerate(fp):
      if n > 0 and n % 100000 == 0:
        print('reading line %d' % n)
        
      if line.startswith('#') or line.startswith('"#'): # comment line
        continue
        
      fields = line.rstrip().split()
      
      frame = int(fields[0])
      intensity = float(fields[8]) - float(fields[7])
      x, y, z = fields[9:12]
      
      if numDimensions == 1:
        position = (float(x),)
      elif numDimensions == 2:
        position = (float(x), float(y))
      elif numDimensions == 3:
        position = (float(x), float(y), float(z))
      
      noise = float(fields[5])
      background = float(fields[7])
      signal = float(fields[8])
      precision = float(fields[13]) if len(fields) >= 14 else 0.0

      yield (frame, intensity, position, signal, background, noise, precision)
    #print('found %d lines' % n)

def withinExcludeRadius(position1, position2, excludeRadius):
  
  delta = numpy.array(position1) - numpy.array(position2)
  
  return numpy.sum(delta*delta) < (excludeRadius*excludeRadius)
  
def analyseSingleFrameData(singleFrameData, excludeRadius):
  
  excludeSet = set()
  n = len(singleFrameData)
  for i in range(n-1):
    if i in excludeSet:
      continue
    (frame1, intensity1, position1, signal1, background1, noise1, precision1) = singleFrameData[i]
    for j in range(i+1, n):
      if j in excludeSet:
        continue
      (frame2, intensity2, position2, signal2, background2, noise2, precision2) = singleFrameData[j]
      if withinExcludeRadius(position1[:2], position2[:2], excludeRadius):  # only look at x, y, not z
        if signal1-background1 >= signal2-background2:
          excludeSet.add(j)
        else:
          excludeSet.add(i)
          break  # no point looking at i further, so skip remaining j
          
  filteredSingleFrameData = []
  for i in range(n):
    if i not in excludeSet:
      filteredSingleFrameData.append(singleFrameData[i])
      
  return filteredSingleFrameData
      
def filterPeaksExcludeRadius(frameData, excludeRadius):
  
  currentFrame = None
  singleFrameData = []
  filteredFrameData = []
  
  for n, (frame, intensity, position, signal, background, noise, precision) in enumerate(frameData):
    if n > 0 and n % 10000 == 0:
      print('filtering frame data %d' % n)
      
    if (frame != currentFrame) and (currentFrame is not None):
      filteredSingleFrameData = analyseSingleFrameData(singleFrameData, excludeRadius)
      filteredFrameData.extend(filteredSingleFrameData)
      singleFrameData = []
      
    currentFrame = frame
    singleFrameData.append((frame, intensity, position, signal, background, noise, precision))
      
  filteredSingleFrameData = analyseSingleFrameData(singleFrameData, excludeRadius)
  filteredFrameData.extend(filteredSingleFrameData)
  
  return filteredFrameData
  
def determineTracks(fileName, numDimensions, maxJumpDistance, maxFrameGap, minNumPositions, isNewPositionFile=True, excludeRadius=0):

  frameData = []
  
  if isNewPositionFile:
    readPositionFile = readNewPositionFile
  else:
    readPositionFile = readOldPositionFile
    
  for (frame, intensity, position, signal, background, noise, precision) in readPositionFile(fileName, numDimensions):
    frameData.append((frame, intensity, position, signal, background, noise, precision))
    
  print('found %d records' % len(frameData))
  
  if excludeRadius > 0:
    frameData = filterPeaksExcludeRadius(frameData, excludeRadius)
    print('have %d records after filtering on excludeRadius' % len(frameData))
    
  finishedTracks = set()
  currentTracks = set()

  frameData.sort() # old data might not be in frame order
  for n, (frame, intensity, position, signal, background, noise, precision) in enumerate(frameData):
    if n > 0 and n % 10000 == 0:
      print('processing frame data %d (finishedTracks %d, currentTracks %d)' % (n, len(finishedTracks), len(currentTracks)))
    _processPosition(finishedTracks, currentTracks, position, frame, intensity, signal, background, noise, precision, maxJumpDistance, maxFrameGap)
      
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
  #fileName = _determineOutputFileName(filePrefix, 'numTracks.csv')
  #with open(fileName, 'w') as fp:
  #  fp.write('# xBin, yBin, count\n')
  #  for yBin in range(201):
  #    for xBin in range(101):
  #      if numTracks[yBin][xBin] > 0:
  #        fp.write('%d,%d,%d\n' % (xBin, yBin, numTracks[yBin][xBin]))
  
  cmap_name = COLOR1
  #plt.xlim((0, 100)) # TEMP HACK
  #plt.ylim((200, 0))  # y axis is backwards
  imgplot = plt.imshow(numTracks, cmap=cmap_name, vmin=minValue, vmax=maxValue, interpolation='nearest')
  #plt.xlim((0, len(numTracks[0]-1)))
  #plt.ylim((len(numTracks)-1, 0))  # y axis is backwards

  fileName = _determineOutputFileName(filePrefix, 'countHeat.png')
  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  plt.close()
  
def saveTracks(tracks, filePrefix):

  fileName = _determineOutputFileName(filePrefix, 'trackPositions.csv')
  print('Saving tracks to %s' % fileName)
  with open(fileName, 'w') as fp:
    fp.write('#track,frame,x,y,z\n')
    for n, track in enumerate(tracks):
      for i, position in enumerate(track.positions):
        frame = track.frames[i]

        if len(position) == 1:
          position0 = position[0]
          position1 = position2 = 0
        elif len(position) == 2:
          position0, position1 = position
          position2 = 0
        else:
          position0, position1, position2 = position
        fields = [n+1, frame, position0, position1, position2]

        #fields = [n+1, frame]
        #fields.extend(position)
        
        fields = ['%s' % field for field in fields]
        fp.write(','.join(fields) + '\n')
  
def savePositionsFramesIntensities(tracks, filePrefix):

  fileName = _determineOutputFileName(filePrefix, 'positionsFramesIntensity.csv')
  with open(fileName, 'w') as fp:
    fp.write('# track, numberPositions, deltaFrames, averageIntensity, averageSignal, averageBackground, averageNoise, averagePrecision, averageSignalToNoise (averages miss out first and last ones if >= 3 positions), averagePosition\n')
    for n, track in enumerate(tracks):
      averagePosition = ','.join(['%.1f' % pos for pos in track.averagePosition])
      fp.write('%d,%d,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%s\n' % (n+1, track.numberPositions, track.deltaFrames, track.averageIntensity,
          track.averageSignal, track.averageBackground, track.averageNoise, track.averagePrecision, track.averageSignalToNoise, averagePosition))
      
def savePositionFramesIntensity(tracks, filePrefix):

  fileName = _determineOutputFileName(filePrefix, 'positionFramesIntensity.csv')
  with open(fileName, 'w') as fp:
    fp.write('# averagePosition, deltaFrames, averageIntensity, averageSignal, averageBackground, averageNoise, averagePrecision, averageSignalToNoise (averages miss out first and last ones if >= 3 positions)\n')
    for n, track in enumerate(tracks):
      averagePosition = ','.join(['%.1f' % pos for pos in track.averagePosition])
      fp.write('%s,%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n' % (averagePosition, track.deltaFrames, track.averageIntensity,
          track.averageSignal, track.averageBackground, track.averageNoise, track.averagePrecision, track.averageSignalToNoise))

def saveIntensityHistogram(tracks, filePrefix):

  intensities = [track.averageIntensity for track in tracks]
  
  maxIntensity = max(intensities)
  nbins = len(intensities) // 10  # gives average of 10.0 hits per bin
  binSize = maxIntensity / nbins
  
  hist = nbins * [0]
  for intensity in intensities:
    b = int(intensity / binSize)
    b = min(nbins-1, b)
    hist[b] += 1
    
  fileName = _determineOutputFileName(filePrefix, 'intensityHistogram.csv')
  with open(fileName, 'w') as fp:
    for b in range(nbins):
      intensity = b * binSize
      fp.write('%.0f,%d\n' % (intensity, hist[b]))
  
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
  
def calcMedianIntensity(tracks, filePrefix):
  
  intensities = []
  for track in tracks:
    intensities.extend(track.intensities)
    
  print('%s: median intensity = %s' % (filePrefix, numpy.median(numpy.array(intensities))))
  
  return intensities
    
def endCalcMedianIntensity(directory, intensities):
  
  print('directory %s: median intensity = %s' % (directory, numpy.median(numpy.array(intensities))))

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
    
  #import random
  #tracks = random.sample(tracks, 300) # HACK
  
  for xDim, yDim in dims:
    #if xDim == 0 and yDim == 2:
    #  tracks = random.sample(tracks, 120) # HACK
      
    for track in tracks:
      xpositions = [position[xDim] for position in track.positions]
      ypositions = [position[yDim] for position in track.positions]
    
      if track.maxDistanceTravelled() >= cutoffValue:
        #color = 'orange' #COLOR3 #2 # HACK
        color = COLOR2
      else:
        #color = COLOR2 #3 # HACK
        color = COLOR3
      plt.plot(xpositions, ypositions, color=color)
    
    plt.ylim(plt.ylim()[::-1])
    
    #if yDim != 1:
    #  plt.ylim((18000, -18000)) # HACK
      
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
            
def calcMeanSquareDisplacements(tracks, filePrefix, secondsPerFrame=0.5, plotDpi=600):
  
  xs = []
  ys = []
  
  for track in tracks:
    msds = track.meanSquareDisplacements()[:-3]  # chop off last three because those have poor stats
    x = secondsPerFrame * numpy.array(range(1, len(msds)+1))
    y = msds
    plt.loglog(x, y, alpha=0.5)
    xs.append(x)
    ys.append(y)
    
  ##fileName = _determineOutputFileName(filePrefix, 'meanSquareDisplacements.png')
  ##plt.savefig(fileName, dpi=plotDpi, transparent=True)
  #plt.show()
  ##plt.close()
  
  return (xs, ys)

""""
def _fitDoubleFunc(xlog, *params):
  
  slope1, intercept1, slope2, intercept2 = params
  alpha1 = slope1
  k1 = numpy.power(10, intercept1)
  alpha2 = slope2
  k2 = numpy.power(10, intercept2)
  x = numpy.power(10, xlog)
  ylog = numpy.log10(k1*numpy.power(x, alpha1) + k2*numpy.power(x, alpha2))
  
  return ylog
"""
  
def endMeanSquareDisplacements(directory, xs, ys, tracks, xlim=(1.0e-1, 1.0e3), ylim=(1.0e0, 1.0e6), plotDpi=600):
  
  #plt.xlim(xlim)
  #plt.ylim(ylim)

  xs = [numpy.array(x) for x in xs]
  ys = [numpy.array(y) for y in ys]
  xall = numpy.concatenate(xs)
  yall = numpy.concatenate(ys)
  
  xyall = numpy.array(sorted(zip(xall, yall)))
  xall = xyall[:,0]
  yall = xyall[:,1]
  
  n = 0
  while n < len(xall) and xall[n] <= 100: # HACK
    n += 1
  #n = len(xall)
   
  xall = xall[:n]
  yall = yall[:n]
  
  xlog = numpy.log10(xall)
  ylog = numpy.log10(yall)
  
  """
  n = len(xlog)
  slope1, intercept1, r_value1, p_value1, std_err1 = linregress(xlog[:n//2], ylog[:n//2])
  slope2, intercept2, r_value2, p_value2, std_err2 = linregress(xlog[n//2:int(0.9*n)], ylog[n//2:int(0.9*n)])
  
  params0 = (slope1, intercept1, slope2, intercept2)
  params, params_cov = curve_fit(_fitDoubleFunc, xlog, ylog, p0=params0)
"""
  
  slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
  print('slope, intercept, r_value, p_value, std_err =', slope, intercept, r_value, p_value, std_err)
  Dapp = 0.25*numpy.power(10, intercept)
  print('alpha, Dapp =', slope, Dapp)
  x0 = numpy.min(xall)
  y0 = numpy.power(10, intercept + slope*numpy.log10(x0))
  x1 = numpy.max(xall)
  y1 = numpy.power(10, intercept + slope*numpy.log10(x1))
  #plt.xlim((x0, x1))
  plt.ylim((1.0e-2, 1.0e6)) # HACK
  plt.plot((x0, x1), (y0, y1), color='black', linewidth=2)
   
  """
  xfits = []
  x = xall[0]
  while x < xall[-1]:
    xfits.append(x)
    x *= 1.1
  xfits.append(xall[-1])
  xfits = numpy.array(xfits)
  xlogs = numpy.log10(xfits)
  params = tuple(params)
  yfits = numpy.power(10, _fitLogFunc(xlogs, *params))
  plt.plot(xfits, yfits, color='black', linewidth=2)
"""
    
  directoryOut = directory + '_out'
  if not os.path.exists(directoryOut):
    os.mkdir(directoryOut)
    
  prefix = os.path.basename(directory)
  fileName = os.path.join(directoryOut, '%s_meanSquareDisplacements.png' % prefix)

  plt.savefig(fileName, dpi=plotDpi, transparent=True)
  plt.close()
  
  statsFile = os.path.join(directoryOut, '%s_fitParams.csv' % prefix)
  with open(statsFile, 'w') as fp:
    fp.write('#n,alpha,Dapp,avgInt,geomInt,time\n')
    fp.write('all,%s,%s\n' % (slope, Dapp))
    for n, x in enumerate(xs):
      y = ys[n]
      track = tracks[n]
      xlog = numpy.log10(x)
      ylog = numpy.log10(y)
      slope, intercept, r_value, p_value, std_err = linregress(xlog, ylog)
      Dapp = 0.25*numpy.power(10, intercept)
      fields = [n+1, slope, Dapp, track.averageIntensity, track.geometricMeanIntensity, 0.5*track.deltaFrames] # TBD: 0.5 hardwired for now
      fields = ['%s' % field for field in fields]
      fp.write(','.join(fields) + '\n')
  
  filePrefix = '%s/%s' % (directory, directory)
  saveTracks(tracks, filePrefix)
  
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
    
 