"""
split/merge utilities to help processing large images; 'object masks' version
"""

import numpy as np
from gpfunctions import *

def bb_IoU(box_a, box_b):
    xmin_a, ymin_a, xmax_a, ymax_a = box_a
    xmin_b, ymin_b, xmax_b, ymax_b = box_b

    min_ymax = np.minimum(ymax_a, ymax_b)
    max_ymin = np.maximum(ymin_a, ymin_b)
    min_xmax = np.minimum(xmax_a, xmax_b)
    max_xmin = np.maximum(xmin_a, xmin_b)

    x_intersection = np.maximum(min_xmax-max_xmin, 0)
    y_intersection = np.maximum(min_ymax-max_ymin, 0)
    
    area_intersection = x_intersection*y_intersection
    area_a = (xmax_a-xmin_a)*(ymax_a-ymin_a)
    area_b = (xmax_b-xmin_b)*(ymax_b-ymin_b)
    area_union = area_a+area_b-area_intersection

    if area_union == 0:
        return 0.0

    return area_intersection/area_union

class PI2D:
    """
    PartitionOfImage 2D

    *demo:*
    ::

        from PartitionOfImage import PI2D
        import numpy as np
        from gpfunctions import *

        I = np.random.rand(128,128)
        PI2D.setup(I,64,4,'accumulate')

        nChannels = 2
        PI2D.createOutput(nChannels)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            Q = np.zeros((nChannels,P.shape[0],P.shape[1]))
            for j in range(nChannels):
                Q[j,:,:] = P
            PI2D.patchOutput(i,Q)

        J = PI2D.getValidOutput()
        J = J[0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        K = cat(1,cat(1,I,J),D)
        imshow(K)
    """

    Image = None
    SuggestedPatchSize = 128
    Margin = 14
    PC = None # patch coordinates
    NumPatches = 0
    Output = None
    NR = None
    NC = None
    Boxes = []
    Contours = []
    OutputRaw = None
    Output = None

    def setup(image,suggestedPatchSize,margin):
        """
        initialize PI2D

        *inputs:*
            image: 2D image to partition; if the image nas more than 1 channel,
            the channel dimension is assumed to be the 1st

            suggestedPatchSize: suggested size of square patch (tile);
            actual patch sizes may vary depending on image size

            margin: half the amount of overlap between adjacent patches; margin should be
            an integer greater than 0 and smaller than suggestedPatchSize/2
        """

        PI2D.Image = image
        PI2D.SuggestedPatchSize = suggestedPatchSize
        PI2D.Margin = margin

        if len(image.shape) == 2:
            nr,nc = image.shape
        elif len(image.shape) == 3: # multi-channel image
            nz,nr,nc = image.shape

        PI2D.NR = nr
        PI2D.NC = nc

        npr = int(np.ceil(nr/suggestedPatchSize)) # number of patch rows
        npc = int(np.ceil(nc/suggestedPatchSize)) # number of patch cols

        pcRows = np.linspace(0, nr, npr+1).astype(int)
        pcCols = np.linspace(0, nc, npc+1).astype(int)

        PI2D.PC = [] # patch coordinates [r0,r1,c0,c1]
        for i in range(npr):
            r0 = np.maximum(pcRows[i]-margin, 0)
            r1 = np.minimum(pcRows[i+1]+margin, nr)
            for j in range(npc):
                c0 = np.maximum(pcCols[j]-margin, 0)
                c1 = np.minimum(pcCols[j+1]+margin, nc)
                PI2D.PC.append([r0,r1,c0,c1])

        PI2D.NumPatches = len(PI2D.PC)

        PI2D.OutputRaw = 0.25*PI2D.Image
        PI2D.Output = np.copy(PI2D.OutputRaw)

    def getPatch(i):
        """
        returns the i-th patch for processing
        """

        r0,r1,c0,c1 = PI2D.PC[i]
        if len(PI2D.Image.shape) == 2:
            return PI2D.Image[r0:r1,c0:c1]
        if len(PI2D.Image.shape) == 3:
            return PI2D.Image[:,r0:r1,c0:c1]

    def patchOutput(i,bbs,cts):
        """
        adds result bounding boxes (bbs) and countours (cts)
        of i-th tile processing to the output image
        """

        r0,r1,c0,c1 = PI2D.PC[i]
        for idx in range(len(bbs)):
            xmin, ymin, xmax, ymax = bbs[idx] # x: cols; y: rows
            ct = np.array(cts[idx])
            
            for row in range(ymin,ymax+1):
                PI2D.OutputRaw[r0+row, c0+xmin] = 0.5
                PI2D.OutputRaw[r0+row, c0+xmax] = 0.5
            for col in range(xmin, xmax+1):
                PI2D.OutputRaw[r0+ymin, c0+col] = 0.5
                PI2D.OutputRaw[r0+ymax, c0+col] = 0.5
            for rc in ct:
                PI2D.OutputRaw[r0+rc[0],c0+rc[1]] = 1

            xmin += c0
            xmax += c0
            ymin += r0
            ymax += r0
            ct[:,0] += r0
            ct[:,1] += c0
            
            candidate_box = [xmin, ymin, xmax, ymax]
            candidate_contour = ct

            if PI2D.Boxes:
                did_find_redundancy = False
                for index_box in range(len(PI2D.Boxes)):
                    box = PI2D.Boxes[index_box]
                    if bb_IoU(candidate_box, box) > 0:
                        candidate_area = (xmax-xmin)*(ymax-ymin)
                        area = (box[2]-box[0])*(box[3]-box[1])
                        if candidate_area > area:
                            PI2D.Boxes[index_box] = candidate_box
                            PI2D.Contours[index_box] = candidate_contour
                        did_find_redundancy = True
                        break
                if not did_find_redundancy:
                    PI2D.Boxes.append(candidate_box)
                    PI2D.Contours.append(candidate_contour)
            else:
                PI2D.Boxes.append(candidate_box)
                PI2D.Contours.append(candidate_contour)


    def prepareOutput():
        """
        recovers output with resolved intersections in overlapping areas
        """

        boxes = PI2D.Boxes
        contours = PI2D.Contours
        for idx in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[idx] # x: cols; y: rows
            ct = contours[idx]
            
            for row in range(ymin,ymax+1):
                PI2D.Output[row, xmin] = 0.5
                PI2D.Output[row, xmax] = 0.5
            for col in range(xmin, xmax+1):
                PI2D.Output[ymin, col] = 0.5
                PI2D.Output[ymax, col] = 0.5
            for rc in ct:
                PI2D.Output[rc[0],rc[1]] = 1

    def demo():
        I = np.random.rand(128,128)
        PI2D.setup(I,64,4,'accumulate')

        nChannels = 2
        PI2D.createOutput(nChannels)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            Q = np.zeros((nChannels,P.shape[0],P.shape[1]))
            for j in range(nChannels):
                Q[j,:,:] = P
            PI2D.patchOutput(i,Q)

        J = PI2D.getValidOutput()
        J = J[0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        K = cat(1,cat(1,I,J),D)
        imshow(K)
