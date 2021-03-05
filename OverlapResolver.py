"""
resolves overlap in box/contour predictions from multiple inference models
"""

import numpy as np
from gpfunctions import *

class OR2D:
    """
    OverlapResolver 2D

    *demo:*
    ::

        import numpy as np
        from gpfunctions import *
        from OverlapResolver import OR2D

        im_side = 300
        n_circs = 30
        rad_min = 5
        rad_max = 10

        prm = np.arange(im_side)
        x, y = np.meshgrid(prm, prm)
        M = np.zeros((im_side, im_side))

        c_rows = np.random.randint(im_side, size=n_circs)
        c_cols = np.random.randint(im_side, size=n_circs)
        rads = np.random.randint(rad_min, rad_max, size=n_circs)

        for i in range(n_circs):
            mask = np.sqrt((x-c_rows[i])**2+(y-c_cols[i])**2) < rads[i]
            M[mask] = 1


        M = imgaussfilt(np.double(M), 5)
        M1 = M > 0.3
        M2 = M > 0.2

        bbs1, cts1 = labels_to_boxes_and_contours(mask2label(M1))
        bbs2, cts2 = labels_to_boxes_and_contours(mask2label(M2))

        I1 = draw_boxes_and_contours(M, bbs1, cts1)
        I2 = draw_boxes_and_contours(M, bbs2, cts2)

        OR2D.setup(M)
        # OR2D.resolve(bbs1, cts1)
        # OR2D.resolve(bbs2, cts2)
        OR2D.resolve(bbs1+bbs2, cts1+cts2)

        OR2D.prepareOutput()
        Output = OR2D.Output

        imshowlist([I1, I2, OR2D.OutputRaw, Output])
    """

    Image = None
    Output = None
    NR = None
    NC = None
    Boxes = []
    Contours = []
    OutputRaw = None
    Output = None

    def setup(image):
        """
        initialize OR2D

        *input:*
            image: 2D image; assumed double, single channel, in range [0, 1]
        """

        OR2D.Boxes = []
        OR2D.Contours = []

        OR2D.Image = image

        assert len(image.shape) == 2
        
        nr,nc = image.shape

        OR2D.NR = nr
        OR2D.NC = nc

        OR2D.OutputRaw = 0.25*OR2D.Image
        OR2D.Output = np.copy(OR2D.OutputRaw)

    def resolve(bbs, cts):
        """
        resolves bounding boxes (bbs) and countours (cts)

        *inputs:*
            bbs: list of bounding boxes [xmin, ymin, xmax, ymax], where y = rows, x = cols

            contours: list of contours, where contours is a Nx2 array with each row being a [row, col] contour location
        """

        for idx in range(len(bbs)):
            xmin, ymin, xmax, ymax = bbs[idx] # x: cols; y: rows
            ct = cts[idx]#np.array(cts[idx])
            
            for row in range(ymin,ymax+1):
                OR2D.OutputRaw[row, xmin] = 0.5
                OR2D.OutputRaw[row, xmax] = 0.5
            for col in range(xmin, xmax+1):
                OR2D.OutputRaw[ymin, col] = 0.5
                OR2D.OutputRaw[ymax, col] = 0.5
            for rc in ct:
                OR2D.OutputRaw[rc[0],rc[1]] = 1
            
            candidate_box = [xmin, ymin, xmax, ymax]
            candidate_contour = ct

            if OR2D.Boxes:
                did_find_redundancy = False
                for index_box in range(len(OR2D.Boxes)):
                    box = OR2D.Boxes[index_box]
                    if boxes_intersect(candidate_box, box):
                        contour = OR2D.Contours[index_box]

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

                        # if np.any(cc_box_a*cc_box_b):
                        if masks_IoU(cc_box_a, cc_box_b) > 0.1: # allow for some intersection before replacement
                            candidate_area = np.sum(cc_box_a)
                            area = np.sum(cc_box_b)
                            if candidate_area > area:
                                OR2D.Boxes[index_box] = candidate_box
                                OR2D.Contours[index_box] = candidate_contour
                            did_find_redundancy = True
                            break
                if not did_find_redundancy:
                    OR2D.Boxes.append(candidate_box)
                    OR2D.Contours.append(candidate_contour)
            else:
                OR2D.Boxes.append(candidate_box)
                OR2D.Contours.append(candidate_contour)

    def prepareOutput():
        """
        computes output with resolved contour overlaps,
        which is accessible at OR2D.Output; the output with unresolved
        intersections is accessible at OR2D.OutputRaw
        """

        boxes = OR2D.Boxes
        contours = OR2D.Contours
        for idx in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[idx] # x: cols; y: rows
            ct = contours[idx]
            
            for row in range(ymin,ymax+1):
                OR2D.Output[row, xmin] = 0.5
                OR2D.Output[row, xmax] = 0.5
            for col in range(xmin, xmax+1):
                OR2D.Output[ymin, col] = 0.5
                OR2D.Output[ymax, col] = 0.5
            for rc in ct:
                OR2D.Output[rc[0],rc[1]] = 1

    def demo():
        im_side = 300
        n_circs = 30
        rad_min = 5
        rad_max = 10

        prm = np.arange(im_side)
        x, y = np.meshgrid(prm, prm)
        M = np.zeros((im_side, im_side))

        c_rows = np.random.randint(im_side, size=n_circs)
        c_cols = np.random.randint(im_side, size=n_circs)
        rads = np.random.randint(rad_min, rad_max, size=n_circs)

        for i in range(n_circs):
            mask = np.sqrt((x-c_rows[i])**2+(y-c_cols[i])**2) < rads[i]
            M[mask] = 1


        M = imgaussfilt(np.double(M), 5)
        M1 = M > 0.3
        M2 = M > 0.2

        bbs1, cts1 = labels_to_boxes_and_contours(mask2label(M1))
        bbs2, cts2 = labels_to_boxes_and_contours(mask2label(M2))

        I1 = draw_boxes_and_contours(M, bbs1, cts1)
        I2 = draw_boxes_and_contours(M, bbs2, cts2)

        OR2D.setup(M)
        # OR2D.resolve(bbs1, cts1)
        # OR2D.resolve(bbs2, cts2)
        OR2D.resolve(bbs1+bbs2, cts1+cts2)

        OR2D.prepareOutput()
        Output = OR2D.Output

        imshowlist([I1, I2, OR2D.OutputRaw, Output])
