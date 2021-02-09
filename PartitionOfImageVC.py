"""
split/merge utilities to help processing large images; 'valid convolutions' version;

this 'VC' version of PartitionOfImage is ideal for processing tasks where the ouput is smaller than the
input, as in convolutional networks where the convolution ouput is 'valid', as opposed to 'same'
"""

import numpy as np

class PI2D:
    """
    PartitionOfImageVC 2D

    *demo:*
    ::

        from PartitionOfImageVC import PI2D
        from gpfunctions import imshowlist
        import numpy as np

        imSize = 210
        patchSize = 60
        margin = 20

        I = np.zeros((imSize,imSize))
        I[margin:-margin,margin:-margin] = np.random.rand(imSize-2*margin,imSize-2*margin)

        PI2D.setup(I,patchSize,margin)

        nChannels = 2
        PI2D.createOutput(nChannels)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            Q = np.zeros((nChannels,PI2D.SubPatchSize,PI2D.SubPatchSize))
            # Q = np.zeros((PI2D.SubPatchSize,PI2D.SubPatchSize))
            for j in range(nChannels):
                Q[j,:,:] = P[margin:-margin,margin:-margin]
            # Q[:,:] = P[margin:-margin,margin:-margin]
            PI2D.patchOutput(i,Q)

        J = PI2D.Output
        J = J[0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        imshowlist([I,J,D])
    """

    Image = None
    PatchSize = 60
    Margin = 20
    SubPatchSize = PatchSize-2*Margin
    PC = None # patch coordinates
    SPC = None # subpatch coordinates
    NumPatches = 0
    Output = None
    NR = None # rows
    NC = None # cols

    def setup(image,patchSize,margin):
        """
        initialize PI2D

        *inputs:*
            image: 2D image to partition; if the image nas more than 1 channel,
            the channel dimension is assumed to be the 1st

            patchSize: size of square patch (tile)

            margin: half the amount of overlap between adjacent patches; margin should be
            an integer greater than 0 and smaller than patchSize
        """

        PI2D.Image = image
        PI2D.PatchSize = patchSize
        PI2D.Margin = margin
        subPatchSize = patchSize-2*margin
        PI2D.SubPatchSize = subPatchSize

        if len(image.shape) == 2:
            nr,nc = image.shape
        elif len(image.shape) == 3: # multi-channel image
            nw,nr,nc = image.shape

        PI2D.NR = nr
        PI2D.NC = nc

        npr = int(np.floor((nr-2*margin)/subPatchSize)) # number of patch rows
        npc = int(np.floor((nc-2*margin)/subPatchSize)) # number of patch cols

        if npr*subPatchSize+2*margin < nr:
            npr += 1
        if npc*subPatchSize+2*margin < nc:
            npc += 1

        PI2D.PC = [] # patch coordinates [r0,r1,c0,c1]
        PI2D.SPC = []  # subpatch coordinates [r0,r1,c0,c1]
        for i in range(npr):
            r0 = np.minimum(i*subPatchSize,nr-patchSize)
            r1 = r0+patchSize
            sr0 = r0 + margin
            sr1 = sr0 + subPatchSize
            for j in range(npc):
                c0 = np.minimum(j*subPatchSize,nc-patchSize)
                c1 = c0+patchSize
                sc0 = c0 + margin
                sc1 = sc0 + subPatchSize
                PI2D.PC.append([r0,r1,c0,c1])
                PI2D.SPC.append([sr0, sr1, sc0, sc1])

        PI2D.NumPatches = len(PI2D.PC)

    def getPatch(i):
        """
        returns the i-th patch for processing
        """

        r0,r1,c0,c1 = PI2D.PC[i]
        if len(PI2D.Image.shape) == 2:
            return PI2D.Image[r0:r1,c0:c1]
        if len(PI2D.Image.shape) == 3:
            return PI2D.Image[:,r0:r1,c0:c1]

    def createOutput(nChannels):
        """
        creates output image to store results of tile processing;
        the output can be accessed at PI2D.Output
        """

        if nChannels == 1:
            PI2D.Output = np.zeros((PI2D.NR,PI2D.NC))
        else:
            PI2D.Output = np.zeros((nChannels,PI2D.NR,PI2D.NC))

    def patchOutput(i,P):
        """
        adds result P of i-th tile processing to the output image
        """

        sr0,sr1,sc0,sc1 = PI2D.SPC[i]
        if len(P.shape) == 2:
            PI2D.Output[sr0:sr1,sc0:sc1] = P
        elif len(P.shape) == 3:
            PI2D.Output[:,sr0:sr1,sc0:sc1] = P

    def demo():
        imSize = 210
        patchSize = 60
        margin = 20

        I = np.zeros((imSize,imSize))
        I[margin:-margin,margin:-margin] = np.random.rand(imSize-2*margin,imSize-2*margin)

        PI2D.setup(I,patchSize,margin)

        nChannels = 2
        PI2D.createOutput(nChannels)

        for i in range(PI2D.NumPatches):
            P = PI2D.getPatch(i)
            Q = np.zeros((nChannels,PI2D.SubPatchSize,PI2D.SubPatchSize))
            # Q = np.zeros((PI2D.SubPatchSize,PI2D.SubPatchSize))
            for j in range(nChannels):
                Q[j,:,:] = P[margin:-margin,margin:-margin]
            # Q[:,:] = P[margin:-margin,margin:-margin]
            PI2D.patchOutput(i,Q)

        J = PI2D.Output
        J = J[0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        from gpfunctions import imshowlist
        imshowlist([I,J,D])


class PI3D:
    """
    PartitionOfImageVC 3D

    *demo*:
    ::

        from PartitionOfImageVC import PI3D
        from gpfunctions import imshowlist
        import numpy as np

        imSize = 210
        patchSize = 60
        margin = 20

        I = np.zeros((imSize,imSize,imSize))
        I[margin:-margin,margin:-margin,margin:-margin] = np.random.rand(imSize-2*margin,imSize-2*margin,imSize-2*margin)

        PI3D.setup(I,patchSize,margin)

        nChannels = 1
        PI3D.createOutput(nChannels)

        for i in range(PI3D.NumPatches):
            P = PI3D.getPatch(i)
            # Q = np.zeros((PI3D.SubPatchSize,nChannels,PI3D.SubPatchSize,PI3D.SubPatchSize))
            Q = np.zeros((PI3D.SubPatchSize,PI3D.SubPatchSize,PI3D.SubPatchSize))
            # for j in range(nChannels):
            #     Q[:,j,:,:] = P[margin:-margin,margin:-margin,margin:-margin]
            Q[:,:,:] = P[margin:-margin,margin:-margin,margin:-margin]
            PI3D.patchOutput(i,Q)

        J = PI3D.Output
        # J = J[:,0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        pI = I[int(imSize/2),:,:]
        pJ = J[int(imSize/2),:,:]
        pD = D[int(imSize/2),:,:]

        from gpfunctions import imshowlist
        imshowlist([pI,pJ,pD])
    """

    Image = None
    PatchSize = 60
    Margin = 20
    SubPatchSize = PatchSize-2*Margin
    PC = None # patch coordinates
    SPC = None # subpatch coordinates
    NumPatches = 0
    Output = None
    NR = None # rows
    NC = None # cols
    NZ = None # planes

    def setup(image,patchSize,margin):
        """
        initialize PI3D

        *inputs:*
            image: 3D image to partition; if the image nas more than 1 channel,
            the channel dimension is assumed to be the 2nd, i.e. dimensions are:
            planes, channels, rows, columns

            patchSize: size of cubic patch (tile)

            margin: half the amount of overlap between adjacent patches; margin should be
            an integer greater than 0 and smaller than patchSize
        """

        PI3D.Image = image
        PI3D.PatchSize = patchSize
        PI3D.Margin = margin
        subPatchSize = patchSize-2*margin
        PI3D.SubPatchSize = subPatchSize

        if len(image.shape) == 3:
            nz,nr,nc = image.shape
        elif len(image.shape) == 4: # multi-channel image
            nz,nw,nr,nc = image.shape

        PI3D.NR = nr
        PI3D.NC = nc
        PI3D.NZ = nz

        npr = int(np.floor((nr-2*margin)/subPatchSize)) # number of patch rows
        npc = int(np.floor((nc-2*margin)/subPatchSize)) # number of patch cols
        npz = int(np.floor((nz-2*margin)/subPatchSize)) # number of patch planes

        if npr*subPatchSize+2*margin < nr:
            npr += 1
        if npc*subPatchSize+2*margin < nc:
            npc += 1
        if npz*subPatchSize+2*margin < nz:
            npz += 1

        PI3D.PC = [] # patch coordinates [z0,z1,r0,r1,c0,c1]
        PI3D.SPC = []  # subpatch coordinates [z0,z1,r0,r1,c0,c1]
        for iZ in range(npz):
            z0 = np.minimum(iZ*subPatchSize,nz-patchSize)
            z1 = z0+patchSize
            sz0 = z0+margin
            sz1 = sz0+subPatchSize
            for i in range(npr):
                r0 = np.minimum(i*subPatchSize,nr-patchSize)
                r1 = r0+patchSize
                sr0 = r0 + margin
                sr1 = sr0 + subPatchSize
                for j in range(npc):
                    c0 = np.minimum(j*subPatchSize,nc-patchSize)
                    c1 = c0+patchSize
                    sc0 = c0 + margin
                    sc1 = sc0 + subPatchSize
                    PI3D.PC.append([z0,z1,r0,r1,c0,c1])
                    PI3D.SPC.append([sz0, sz1, sr0, sr1, sc0, sc1])

        PI3D.NumPatches = len(PI3D.PC)

    def getPatch(i):
        """
        returns the i-th patch for processing
        """

        z0,z1,r0,r1,c0,c1 = PI3D.PC[i]
        if len(PI3D.Image.shape) == 3:
            return PI3D.Image[z0:z1,r0:r1,c0:c1]
        if len(PI3D.Image.shape) == 4:
            return PI3D.Image[z0:z1,:,r0:r1,c0:c1]

    def createOutput(nChannels):
        """
        creates output image to store results of tile processing;
        the output can be accessed at PI3D.Output
        """

        if nChannels == 1:
            PI3D.Output = np.zeros((PI3D.NZ,PI3D.NR,PI3D.NC))
        else:
            PI3D.Output = np.zeros((PI3D.NZ,nChannels,PI3D.NR,PI3D.NC))

    def patchOutput(i,P):
        """
        adds result P of i-th tile processing to the output image
        """

        sz0,sz1,sr0,sr1,sc0,sc1 = PI3D.SPC[i]
        if len(P.shape) == 3:
            PI3D.Output[sz0:sz1,sr0:sr1,sc0:sc1] = P
        elif len(P.shape) == 4:
            PI3D.Output[sz0:sz1,:,sr0:sr1,sc0:sc1] = P

    def demo():
        imSize = 210
        patchSize = 60
        margin = 20

        I = np.zeros((imSize,imSize,imSize))
        I[margin:-margin,margin:-margin,margin:-margin] = np.random.rand(imSize-2*margin,imSize-2*margin,imSize-2*margin)

        PI3D.setup(I,patchSize,margin)

        nChannels = 1
        PI3D.createOutput(nChannels)

        for i in range(PI3D.NumPatches):
            P = PI3D.getPatch(i)
            # Q = np.zeros((PI3D.SubPatchSize,nChannels,PI3D.SubPatchSize,PI3D.SubPatchSize))
            Q = np.zeros((PI3D.SubPatchSize,PI3D.SubPatchSize,PI3D.SubPatchSize))
            # for j in range(nChannels):
            #     Q[:,j,:,:] = P[margin:-margin,margin:-margin,margin:-margin]
            Q[:,:,:] = P[margin:-margin,margin:-margin,margin:-margin]
            PI3D.patchOutput(i,Q)

        J = PI3D.Output
        # J = J[:,0,:,:]

        D = np.abs(I-J)
        print(np.max(D))

        pI = I[int(imSize/2),:,:]
        pJ = J[int(imSize/2),:,:]
        pD = D[int(imSize/2),:,:]

        from gpfunctions import imshowlist
        imshowlist([pI,pJ,pD])
