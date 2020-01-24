import os

import Track

PROG_NAME = 'trajectoryAnalysis'
DESCRIPTION = 'Analysis of trajectories for cell image data'

    
def main():
  
  SUFFIX1D = '.txt'
  SUFFIX2D = '.txt'
  SUFFIX3D = '.csv'
  SUFFIXNEW = '.xls' # it's not really an xls file, it's a tab-separated text file
  
  from argparse import ArgumentParser
  
  epilog = 'For further help on running this program please email wb104@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
                             
  arg_parse.add_argument('directories', nargs='+',
                         help='Directories containing *csv (or *txt or *xls) files to be analysed')

  arg_parse.add_argument('-suffix', default="",
                         help='Suffix of files to be analysed (usually csv or txt or xls)')

  arg_parse.add_argument('-numDimensions', default=2, type=int,
                         help='Number of dimensions for the tracks (1, 2 or 3)')

  arg_parse.add_argument('-maxJumpDistance', default=100, type=float,
                         help='Maximum distance can jump between frames (adjusted distance, taking frame difference into account)')

  arg_parse.add_argument('-maxFrameGap', default=2, type=int,
                         help='Maximum change in frame number for two consecutive positions on track')

  arg_parse.add_argument('-minNumPositions', default=3, type=int,
                         help='Minimum number of positions on track to be further considered')

  arg_parse.add_argument('-binSize', default=312, type=int,
                         help='The bin size for binned calculations')
  
  arg_parse.add_argument('-excludeRadius', default=0, type=int,
                         help='Exclude peaks of lower intensity if within this radius of peak of higher intensity (in x, y)')

  arg_parse.add_argument('-plotDpi', default=600, type=int,
                         help='DPI value for binned plots')
  
  arg_parse.add_argument('-isNewPositionFile', default=False, action='store_true',
                         help='Whether new format position file')
  
  arg_parse.add_argument('-fileContainsTrack', default=False, action='store_true',
                         help='Whether have track instead of position file')
  
  arg_parse.add_argument('-saveTracks', default=False, action='store_true',
                         help='Save frames and positions of tracks to csv file')
  
  arg_parse.add_argument('-savePositionsFramesIntensities', default=False, action='store_true',
                         help='Save number of positions, number of frames, average intensity and average position of tracks to csv file')
  
  arg_parse.add_argument('-savePositionFramesIntensity', default=False, action='store_true',
                         help='Save average position, number of frames, and average intensity of tracks to csv file')
  
  arg_parse.add_argument('-saveIntensityHistogram', default=False, action='store_true',
                         help='Save average intensity of tracks as histogram to csv file')
  
  arg_parse.add_argument('-calcFramesPercentage', default=0, type=float,
                         help='Calculate track frames length which is >= than specified percentage over all tracks')
  
  arg_parse.add_argument('-calcMaxNumTracksInBin', default=False, action='store_true',
                         help='Calculate maximum number of tracks in any bin')
  
  arg_parse.add_argument('-calcMedianIntensity', default=False, action='store_true',
                         help='Calculate median intensity across all positions')
  
  arg_parse.add_argument('-saveNumTracksInBin', default=0, type=int,
                         help='Save number of tracks in each bin (as a plot, saved as png) with maximum number for color being specified by given value, -1 means use maximum number of tracks in any bin')
  
  arg_parse.add_argument('-minNumTracksColorInBin', default=0, type=int,
                         help='Specify minimum number number of tracks in each bin for start color')
  
  arg_parse.add_argument('-calcFramesByBinPercentage', default=0, type=float,
                         help='Calculate binned track (average) frames length which is >= than specified percentage over all tracks')
  
  arg_parse.add_argument('-saveTrackFramesInBin', default=0, type=int,
                         help='Save track (average) frames in each bin (as a plot, saved as png) with maximum number for color being specified by given value, -1 means use maximum number of (average) frames in any bin')
  
  arg_parse.add_argument('-saveTracksColoredByFrames', default=0, type=int,
                         help='Save tracks (as a plot, saved as png) where those with frames >= specified value are colored blue and others yellow')
                             
  arg_parse.add_argument('-saveTracksColoredByDistance', default=0, type=float,
                         help='Save tracks (as a plot, saved as png) where those which ever travel >= specified value from start point are colored blue and others yellow')
                             
  arg_parse.add_argument('-saveResidenceTimes', default=False, action='store_true',
                         help='Save residence times to a csv file')
                             
  arg_parse.add_argument('-saveSurvivalCounts', default=0, type=int,
                         help='Save survival counts (how many tracks last at least a given amount) with specified cutoff to a csv file')
                             
  arg_parse.add_argument('-fitSurvivalCounts', default=0, type=int,
                         help='Fit survival counts (how many tracks last at least a given amount) to specified number of exponentials, and save result to a csv file')
  
  arg_parse.add_argument('-fitUsingLogData', default=False, action='store_true',
                         help='Fit survival count using log of data (so later points count as much as early points)')
  
  arg_parse.add_argument('-calcMeanSquareDisplacements', default=False, action='store_true',
                         help='Calculate mean square displacements for one frame, two frames, three frames, etc.')
                         
  arg_parse.add_argument('-secondsPerFrame', default=0.5, type=float,
                         help='Seconds per frame (used in calcMeanSquareDisplacements)')
  
  args = arg_parse.parse_args()

  assert args.numDimensions in (1, 2, 3), 'numDimensions = %d, must be in (1, 2, 3)' % args.numDimensions

  if args.suffix:
    suffix = args.suffix
  elif args.isNewPositionFile:
    suffix = SUFFIXNEW
  elif args.numDimensions == 1:
    suffix = SUFFIX1D
  elif args.numDimensions == 2:
    suffix = SUFFIX2D
  else:
    suffix = SUFFIX3D
      
  for directory in args.directories:
    print('Processing directory %s' % directory)
    xs = []
    ys = []
    all_tracks = []
    intensities = []
    relfileNames = os.listdir(directory)
    relfileNames = [relfileName for relfileName in relfileNames if relfileName.endswith(suffix)]
    for relfileName in relfileNames:
      filePrefix = os.path.join(directory, relfileName[:-len(suffix)])
      fileName = os.path.join(directory, relfileName)
      if args.fileContainsTrack:
        print('Reading track from %s' % fileName)
        track = Track.readTrack(fileName, args.numDimensions)
        tracks = [track]
      else:
        print('Determining tracks for %s' % fileName)
        tracks = Track.determineTracks(fileName, args.numDimensions, args.maxJumpDistance, args.maxFrameGap,
                                      args.minNumPositions, args.isNewPositionFile, args.excludeRadius)
        
      if args.saveTracks:
        Track.saveTracks(tracks, filePrefix)
        
      if args.savePositionsFramesIntensities:
        Track.savePositionsFramesIntensities(tracks, filePrefix)

      if args.savePositionFramesIntensity:
        Track.savePositionFramesIntensity(tracks, filePrefix)

      if args.saveIntensityHistogram:
        Track.saveIntensityHistogram(tracks, filePrefix)

      if args.calcFramesPercentage > 0:
        Track.calcFramesPercentage(tracks, args.calcFramesPercentage)
  
      if args.calcMaxNumTracksInBin:
        Track.calcMaxNumTracksInBin(tracks, args.binSize)
        
      if args.saveNumTracksInBin:
        if args.saveNumTracksInBin == -1:
          value = Track.calcMaxNumTracksInBin(tracks, args.binSize)
        else:
          value = args.saveNumTracksInBin
        Track.saveNumTracksInBin(tracks, filePrefix, args.binSize, args.minNumTracksColorInBin, value, args.plotDpi)
        
      if args.calcFramesByBinPercentage > 0:
        Track.calcFramesByBinPercentage(tracks, args.binSize, args.calcFramesByBinPercentage)
        
      if args.calcMedianIntensity:
        intensities.extend(Track.calcMedianIntensity(tracks, filePrefix))
        
      if args.saveTrackFramesInBin:
        if args.saveTrackFramesInBin == -1:
          value = Track.calcFramesByBinPercentage(tracks, args.binSize, 100.0)
        else:
          value = args.saveTrackFramesInBin
        Track.saveTrackFramesInBin(tracks, filePrefix, args.binSize, value, args.plotDpi)
        
      if args.saveTracksColoredByFrames:
        Track.saveTracksColoredByFrames(tracks, filePrefix, args.saveTracksColoredByFrames, args.plotDpi, args.numDimensions)

      if args.saveTracksColoredByDistance:
        Track.saveTracksColoredByDistance(tracks, filePrefix, args.saveTracksColoredByDistance, args.plotDpi, args.numDimensions)

      if args.saveResidenceTimes:
        Track.saveResidenceTimes(tracks, filePrefix)

      if args.saveSurvivalCounts > 0:
        Track.saveSurvivalCounts(tracks, filePrefix, args.saveSurvivalCounts)
        
      if args.fitSurvivalCounts > 0:
        Track.fitSurvivalCounts(tracks, filePrefix, args.fitSurvivalCounts, args.minNumPositions, args.fitUsingLogData, args.plotDpi)
        
      if args.calcMeanSquareDisplacements:
        track_xs, track_ys = Track.calcMeanSquareDisplacements(tracks, filePrefix, args.secondsPerFrame, args.plotDpi)
        xs.extend(track_xs)
        ys.extend(track_ys)
        all_tracks.extend(tracks)
        
    if args.calcMeanSquareDisplacements and xs:
      Track.endMeanSquareDisplacements(directory, xs, ys, all_tracks, plotDpi=args.plotDpi)
        
    if args.calcMedianIntensity:
      Track.endCalcMedianIntensity(directory, intensities)
        
if __name__ == '__main__':
  
  main()
  
      
    