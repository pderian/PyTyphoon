###
import re
###
import numpy
###

### .INR MOTION/IMAGES ###
##########################

def readINR(fileName, printInfo=False):
    """
    Read an INR image file.

    Arguments:
        * fileName    the .inr filename
        * printInfo   print image info

    Note: to preserve memory continuity, image is loaded with dimensions in the following order:
    0-V (channel), 1-Z (depth), 2-Y (height), 3-X (width).
    """
    # open file
    with open(fileName, 'rb') as inrFile:
        ##### HEADER #####
        ##################
        # read the header: chunk of 256 char
        header = inrFile.read(256).decode('utf-8')
        # the keys we look for
        dimKey=['VDIM','ZDIM','YDIM', 'XDIM']
        typeKey='TYPE'
        sizeKey='PIXSIZE'
        # default values
        dataDim = []
        dataType = ''
        dataBitSize = 0
        # read dimensions
        for key in dimKey:
            match = re.search(key+'=(?P<val>[0-9]+)', header)
            if match:
                dataDim.append(int(match.group('val')))
            else:
                print(str(key)+' not found !')
        # read type
        key = typeKey
        match = re.search(key+'=(?P<val>[a-z]+)', header)
        if match:
            dataType = match.group('val')
        else:
            print(str(key)+' not found !')
        # read pixel size (in BITS)
        key = sizeKey
        match = re.search(key+'=(?P<val>[0-9]+) bits',header)
        if match:
            dataBitSize = int(match.group('val'))
        else:
            print(str(key)+' not found !')
        # print info
        if printInfo:
            print('File: '+fileName+'\nDimensions: '+str(dataDim)+'\nType: '\
                  +str(dataType)+'\nPixel Size: '+str(dataBitSize))
        ##### DATA #####
        ################
        # number of bytes to read
        nByte = dataDim[0]*dataDim[1]*dataDim[2]*dataDim[3]*(dataBitSize//8)
        # create a 1D float array
        values = numpy.frombuffer(inrFile.read(nByte), dtype=numpy.dtype('{:s}{:d}'.format(dataType, dataBitSize)))
        # resize to 4-D array
        values.shape = dataDim
        # clean shape from dimensions equal to 1 and return
        return values.squeeze()

def readMotion(fileName, printInfo=False, reverse=False):
    """
    Read a motion from an INR file.

    Arguments:

        * fileName    the .inr file
        * printInfo   print motion info.
        * reverse     reverse the vertical axis and the sign of vertical component.
    """
    # read INR file
    motion = readINR(fileName,printInfo)
    v1 = motion[0,:,:]
    v2 = motion[1,:,:]
    # reverse reference maybe
    if reverse:
        v1 = numpy.flipud(v1)
        v2 = -numpy.flipud(v2)
    return v1,v2
