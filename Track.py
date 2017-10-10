import numpy

numDimensions = 2
assert numDimensions in (2, 3)
maxJumpDist = 100  # max distance can jump between frame (adjusted distance, taking frame difference into account)
maxFrameGap = 2  # maximum change in frame for two consecutive positions on track
minTrackPositions = 3  # minimum number of positions on track to be further considered

class Track:
  
  def __init__(self, position, frame, intensity):

    self.positions = [position]
    self.frames = [frame]
    self.intensities = [intensity]
    self.distances = [] # change in position / sqrt(change in frame)
    
  def addPosition(self, position, frame, intensity):

    distance = calcAdjustedDistance(position, frame, self) # have to calculate this first
    
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

def calcAdjustedDistance(position, frame, track):
  
  delta = position - track.positions[-1]
  #distance = numpy.sqrt(numpy.sum(delta*delta))
  distance = numpy.sqrt(numpy.sum(delta*delta)) / numpy.sqrt(frame - track.frames[-1])
  
  return distance
  
def processPosition(finishedTracks, currentTracks, position, frame, intensity):
  
  bestDist = None
  bestTrack = None
  
  for track in list(currentTracks):
    if frame > track.frames[-1] + maxFrameGap:
      currentTracks.remove(track)
      finishedTracks.add(track)
    elif frame > track.frames[-1]:
      distance = calcAdjustedDistance(position, frame, track)
      if distance < maxJumpDist and (bestDist is None or distance < bestDist):
        bestDist = distance
        bestTrack = track

  if bestTrack:
    bestTrack.addPosition(position, frame, intensity)
  else:
    track = Track(position, frame, intensity)
    currentTracks.add(track)

def determineTracks(fileName):

  finishedTracks = set()
  currentTracks = set()

  with open(fileName, 'rU') as fp:
    
    fp.readline()  # header

    for line in fp:
      if numDimensions == 2:
        (x, y, frame, intensity) = line.rstrip().split()[:4]
        position = numpy.array((float(x), float(y)))
      elif numDimensions == 3:
        (x, y, z, frame, intensity) = line.rstrip().split()[:5]
        position = numpy.array((float(x), float(y), float(z)))

      frame = int(frame)
      intensity = float(intensity)
    
      processPosition(finishedTracks, currentTracks, position, frame, intensity)
      
  finishedTracks.update(currentTracks)

  print('Number of tracks = %d' % len(finishedTracks))

  # filter out short tracks
  finishedTracks = [track for track in finishedTracks if track.numberPositions >= minTrackPositions]

  print('Number of tracks after filtering for >= %d positions = %d' % (minTrackPositions, len(finishedTracks)))

  return finishedTracks
    
def savePositionsFramesIntensities(tracks, filePrefix):

  fileName = '%s_positionsFramesIntensity.csv' % filePrefix
  with open(fileName, 'w') as fp:
    fp.write('# track, numberPositions, deltaFrames, averageIntensity (missing out first and last ones if >= 3 positions)\n')
    for n, track in enumerate(tracks):
      fp.write('%d,%d,%d,%.1f\n' % (n+1, track.numberPositions, track.deltaFrames, track.averageIntensity))

if __name__ == '__main__':

  import os
  import sys
  
  if len(sys.argv) == 1:
    print('Need to specify one or more data files')
    sys.exit()
  
  fileNames = sys.argv[1:]
  for fileName in fileNames:
    print('Determining tracks for %s' % fileName)
    tracks = determineTracks(fileName)
    #shortTracks = [track for track in tracks if track.numberPositions == minTrackPositions]
    #print('Number of tracks with minimum number of track positions = %d (%.1f%%)' % (len(shortTracks), 100*len(shortTracks)/len(tracks)))
    filePrefix = fileName[:-4]
    savePositionsFramesIntensities(tracks, filePrefix)
    
 