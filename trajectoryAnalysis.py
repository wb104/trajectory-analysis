import os

import Track

PROG_NAME = 'trajectoryAnalysis'
DESCRIPTION = 'Analysis of trajectories for cell image data'

    
def main():
  
  SUFFIX2D = '.txt'
  SUFFIX3D = '.csv'
  
  from argparse import ArgumentParser
  
  epilog = 'For further help on running this program please email wb104@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
                             
  arg_parse.add_argument('directories', nargs='+',
                         help='Directories containing *%s (or *%s) files to be analysed' % (SUFFIX2D, SUFFIX3D))

  arg_parse.add_argument('-numDimensions', default=2, type=int,
                         help='Number of dimensions for the tracks (2 or 3)')

  arg_parse.add_argument('-maxJumpDistance', default=100, type=float,
                         help='Maximum distance can jump between frames (adjusted distance, taking frame difference into account)')

  arg_parse.add_argument('-maxFrameGap', default=2, type=int,
                         help='Maximum change in frame number for two consecutive positions on track')

  arg_parse.add_argument('-minNumPositions', default=3, type=int,
                         help='Minimum number of positions on track to be further considered')

  arg_parse.add_argument('-binSize', default=312, type=int,
                         help='The bin size for binned calculations')
  
  arg_parse.add_argument('-plotDpi', default=600, type=int,
                         help='DPI value for binned plots')
  
  arg_parse.add_argument('-savePositionsFramesIntensities', default=False, action='store_true',
                         help='Save positions, frames and intensities of tracks to csv file')
  
  arg_parse.add_argument('-calcFramesPercentage', default=0, type=float,
                         help='Calculate track frames length which is >= than specified percentage over all tracks')
  
  arg_parse.add_argument('-calcMaxNumTracksInBin', default=False, action='store_true',
                         help='Calculate maximum number of tracks in any bin')
  
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
                             
  arg_parse.add_argument('-saveResidenceTimes', default=False, action='store_true',
                         help='Save residence times to a csv file')
                             
  arg_parse.add_argument('-saveSurvivalCounts', default=0, type=int,
                         help='Save survival counts (how many tracks last at least a given amount) with specified cutoff to a csv file')
                             
  arg_parse.add_argument('-fitSurvivalCounts', default=0, type=int,
                         help='Fit survival counts (how many tracks last at least a given amount) to specified number of exponentials, and save result to a csv file')
  
  args = arg_parse.parse_args()

  assert args.numDimensions in (2, 3), 'numDimensions = %d, must be in (2, 3)' % args.numDimensions

  suffix = SUFFIX2D if args.numDimensions == 2 else SUFFIX3D
  
  for directory in args.directories:
    print('Processing directory %s' % directory)
    relfileNames = os.listdir(directory)
    relfileNames = [relfileName for relfileName in relfileNames if relfileName.endswith(suffix)]
    for relfileName in relfileNames:
      filePrefix = os.path.join(directory, relfileName[:-len(suffix)])
      fileName = os.path.join(directory, relfileName)
      print('Determining tracks for %s' % fileName)
      tracks = Track.determineTracks(fileName, args.numDimensions, args.maxJumpDistance, args.maxFrameGap, args.minNumPositions)
        
      if args.savePositionsFramesIntensities:
        Track.savePositionsFramesIntensities(tracks, filePrefix)

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
        
      if args.saveTrackFramesInBin:
        if args.saveTrackFramesInBin == -1:
          value = Track.calcFramesByBinPercentage(tracks, args.binSize, 100.0)
        else:
          value = args.saveTrackFramesInBin
        Track.saveTrackFramesInBin(tracks, filePrefix, args.binSize, value, args.plotDpi)
        
      if args.saveTracksColoredByFrames:
        Track.saveTracksColoredByFrames(tracks, filePrefix, args.saveTracksColoredByFrames, args.plotDpi)

      if args.saveResidenceTimes:
        Track.saveResidenceTimes(tracks, filePrefix)

      if args.saveSurvivalCounts > 0:
        Track.saveSurvivalCounts(tracks, filePrefix, args.saveSurvivalCounts)
        
      if args.fitSurvivalCounts > 0:
        Track.fitSurvivalCounts(tracks, filePrefix, args.fitSurvivalCounts, args.plotDpi)
        
if __name__ == '__main__':
  
  main()
  
      
    