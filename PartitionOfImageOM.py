"""
split/merge utilities to help processing large images; 'object masks' version
"""

import numpy as np
from gpfunctions import *

class PI2D:
    """
    PartitionOfImage 2D

    *demo:*
    ::

        im_side = 300
        n_circs = 30
        rad_min = 5
        rad_max = 10
        suggested_patch_size = 100
        margin = 50

        prm = np.arange(im_side)
        x, y = np.meshgrid(prm, prm)
        M = np.zeros((im_side, im_side))

        c_rows = np.random.randint(im_side, size=n_circs)
        c_cols = np.random.randint(im_side, size=n_circs)
        rads = np.random.randint(rad_min, rad_max, size=n_circs)

        for i in range(n_circs):
            mask = np.sqrt((x-c_rows[i])**2+(y-c_cols[i])**2) < rads[i]
            M[mask] = 1

        PI2D.setup(np.double(M), suggested_patch_size, margin)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            bbs, cts = labels_to_boxes_and_contours(mask2label(P > 0))
            PI2D.patchOutput(i, bbs, cts)

        PI2D.prepareOutput()
        Output = PI2D.Output

        imshowlist([PI2D.OutputRaw, Output])
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
            image: 2D image to partition; assumed double, single channel, in range [0, 1]

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
            ct = cts[idx]#np.array(cts[idx])
            
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
                    if boxes_intersect(candidate_box, box):
                        contour = PI2D.Contours[index_box]

                        cc = np.concatenate((candidate_contour, contour), axis=0)
                        cc_min_r, cc_min_c = np.min(cc, axis=0)
                        cc_max_r, cc_max_c = np.max(cc, axis=0)

                        cc_box_a = np.zeros((cc_max_r-cc_min_r+1, cc_max_c-cc_min_c+1), dtype=bool)
                        cc_box_b = np.copy(cc_box_a)

                        for idx_c in range(candidate_contour.shape[0]):
                            cc_box_a[candidate_contour[idx_c,0]-cc_min_r,candidate_contour[idx_c,1]-cc_min_c] = True

                        for idx_c in range(contour.shape[0]):
                            cc_box_b[contour[idx_c,0]-cc_min_r,contour[idx_c,1]-cc_min_c] = True

                        cc_box_a = imfillholes(cc_box_a)
                        cc_box_b = imfillholes(cc_box_b)

                        if np.sum(cc_box_a*cc_box_b) > 0:
                            candidate_area = np.sum(cc_box_a)
                            area = np.sum(cc_box_b)
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
        computes output with resolved intersections in overlapping areas
        which is accessible at PI2D.Output; the output with unresolved
        intersections is accessible at PI2D.OutputRaw
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
        im_side = 300
        n_circs = 30
        rad_min = 5
        rad_max = 10
        suggested_patch_size = 100
        margin = 50

        prm = np.arange(im_side)
        x, y = np.meshgrid(prm, prm)
        M = np.zeros((im_side, im_side))

        c_rows = np.random.randint(im_side, size=n_circs)
        c_cols = np.random.randint(im_side, size=n_circs)
        rads = np.random.randint(rad_min, rad_max, size=n_circs)

        for i in range(n_circs):
            mask = np.sqrt((x-c_rows[i])**2+(y-c_cols[i])**2) < rads[i]
            M[mask] = 1

        PI2D.setup(np.double(M), suggested_patch_size, margin)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            bbs, cts = labels_to_boxes_and_contours(mask2label(P > 0))
            PI2D.patchOutput(i, bbs, cts)

        PI2D.prepareOutput()
        Output = PI2D.Output

        imshowlist([PI2D.OutputRaw, Output])
