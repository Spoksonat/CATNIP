import numpy as np
from scipy import signal as sig

class UMPA:
    def __init__(self, I_refs, I_samps, dict_params):
        """
        Initializes the UMPA class for speckle tracking phase retrieval.

        Args:
            I_refs (np.ndarray): Reference speckle images stack.
            I_samps (np.ndarray): Sample speckle images stack.
            dict_params (dict): Dictionary of retrieval parameters.
        """
        self.I_refs = I_refs
        self.I_samps = I_samps

        n_rows, n_cols = I_refs.shape[1], I_refs.shape[2]

        if(n_rows > n_cols):
            delta = n_rows - n_cols
            self.I_refs = self.I_refs[:,:-delta, :]
            self.I_samps = self.I_samps[:,:-delta, :]
        elif(n_rows < n_cols):
            delta = n_cols - n_rows
            self.I_refs = self.I_refs[:,:, :-delta]
            self.I_samps = self.I_samps[:,:, :-delta]
        else:
            self.I_refs = self.I_refs
            self.I_samps = self.I_samps
       
        print(self.I_refs.shape)
        
        self.dict_params = dict_params
        self.Nw = int(float(dict_params["Nw"]))
        self.step = 1
        self.max_shift = int(float(dict_params["max_shift"]))
        self.df = dict_params["df_bool"]

        self.match_speckles(printout=False)

    def match_speckles(self, printout=True):
        
        """
        Speckle matching
        
        Author: Pierre Thibault
        Date: July 2015
        
        Compare speckle images with sample (self.I_samps) and w/o sample
        (self.I_refs) using a given window.
        max_shift can be set to the number of pixels for an "acceptable"
        speckle displacement.
    
        :param self.I_samps: A list  of measurements, with the sample aligned but speckles shifted
        :param self.I_refs: A list of empty speckle measurements with the same displacement as self.I_samps.
        :param self.Nw: 2*self.Nw + 1 is the width of the window.
        :param step: perform the analysis on every other _step_ pixels in both directions (default 1)
        :param max_shift: Do not allow shifts larger than this number of pixels (default 4)
        :param df: Compute dark field (default True)
    
        Return T, dx, dy, df, f
        """
    
        Ish = self.I_samps[0].shape
    
        # Create the window
        w = np.multiply.outer(np.hamming(2*self.Nw+1), np.hamming(2*self.Nw+1))
        w /= w.sum()
    
        NR = len(self.I_samps)
    
        S2 = sum(I**2 for I in self.I_samps)
        R2 = sum(I**2 for I in self.I_refs)
        if self.df:
            S1 = sum(I for I in self.I_samps)
            R1 = sum(I for I in self.I_refs)
            Im = R1.mean()/NR
    
        L1 = self.cc(S2, w)
        L3 = self.cc(R2, w)
        if self.df:
            L2 = Im * Im * NR
            L4 = Im * self.cc(S1, w)
            L6 = Im * self.cc(R1, w)
        # (We need a loop for L5)
    
        # 2*Ns + 1 is the width of the window explored to find the best fit.
        Ns = self.max_shift
    
        ROIx = np.arange(Ns+self.Nw, Ish[0]-Ns-self.Nw-1, self.step)
        ROIy = np.arange(Ns+self.Nw, Ish[1]-Ns-self.Nw-1, self.step)
    
        # The final images will have this size
        sh = (len(ROIy), len(ROIx))
        tx = np.zeros(sh)
        ty = np.zeros(sh)
        tr = np.zeros(sh)
        do = np.zeros(sh)
        MD = np.zeros(sh)
    
        # Loop through all positions
        for xi, i in enumerate(ROIy):
            if printout:
                print('line %d, %d/%d' % (i, xi, sh[0]))
            for xj, j in enumerate(ROIx):
                # Define local values of L1, L2, ...
                t1 = L1[i, j]
                t3 = L3[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
                if self.df:
                    t2 = L2
                    t4 = L4[i, j]
                    t6 = L6[(i-Ns):(i+Ns+1), (j-Ns):(j+Ns+1)]
                else:
                    t2 = 0.
                    t4 = 0.
                    t6 = 0.
    
                # Now we can compute t5 (local L5)
                t5 = np.zeros((2*Ns+1, 2*Ns+1))
                for k in range(NR):
                    t5 += self.cc(self.I_refs[k][(i-self.Nw-Ns):(i+self.Nw+Ns+1), (j-self.Nw-Ns):(j+self.Nw+Ns+1)],
                             w * self.I_samps[k][(i-self.Nw):(i+self.Nw+1), (j-self.Nw):(j+self.Nw+1)], mode='valid')
    
                # Compute K and beta
                if self.df:
                    K = (t2*t5 - t4*t6)/(t2*t3 - t6**2)
                    beta = (t3*t4 - t5*t6)/(t2*t3 - t6**2)
                else:
                    K = t5/t3
                    beta = 0.
    
                # Compute v and a
                a = beta + K
                v = K/a
    
                # Construct D
                D = t1 + (beta**2)*t2 + (K**2)*t3 - 2*beta*t4 - 2*K*t5 + 2*beta*K*t6
    
                # Find subpixel optimum for tx an ty
                sy, sx = self.sub_pix_min(D)
    
                # We should re-evaluate the other values with sub-pixel precision but here we just round
                # We also need to clip because "sub_pix_min" can return the position of the minimum outside of the bounds...
                isy = np.clip(int(np.round(sy)), 0, 2*Ns)
                isx = np.clip(int(np.round(sx)), 0, 2*Ns)
    
                # store everything
                ty[xi, xj] = sy - Ns
                tx[xi, xj] = sx - Ns
                tr[xi, xj] = a[isy, isx]
                do[xi, xj] = v[isy, isx]
                MD[xi, xj] = D[isy, isx]
    
        self.T = tr
        self.dx = self.wrap_phase(tx)
        self.dy = self.wrap_phase(ty)
        self.D = do
        #return {'T': tr, 'dx': tx, 'dy': ty, 'df': do, 'f': MD}
    
    def wrap_phase(self, image):
        """
        Wraps the phase of an image to the interval [-pi, pi].

        Args:
            image (np.ndarray): Input phase image.

        Returns:
            np.ndarray: Phase-wrapped image.
        """
        image_wrap = np.angle(np.exp(1j * image))
        return image_wrap 
    
    def cc(self, A, B, mode='same'):
        """
        Computes fast cross-correlation between two images using FFT convolution.

        Args:
            A (np.ndarray): Reference image.
            B (np.ndarray): Template image to match.
            mode (str, optional): Convolution mode ('same', 'full', 'valid').

        Returns:
            np.ndarray: Cross-correlation result.
        """
        return sig.fftconvolve(A, B[::-1, ::-1], mode=mode)
    
    
    def quad_fit(self, a):
        """\
        Fits a paraboloid to a 2D array and returns the optimum value, position, and Hessian.

        Args:
            a (np.ndarray): Input 2D array.

        Returns:
            tuple: Optimum value (c), position (x0), and Hessian matrix (H).
        """
    
        sh = a.shape
    
        i0, i1 = np.indices(sh)
        i0f = i0.flatten()
        i1f = i1.flatten()
        af = a.flatten()
    
        # Model = p(1) + p(2) x + p(3) y + p(4) x^2 + p(5) y^2 + p(6) xy
        #       = c + (x-x0)' h (x-x0)
        A = np.vstack([np.ones_like(i0f), i0f, i1f, i0f**2, i1f**2, i0f*i1f]).T
        r = np.linalg.lstsq(A, af)
        p = r[0]
        x0 = - (np.matrix([[2*p[3], p[5]], [p[5], 2*p[4]]]).I * np.matrix([p[1], p[2]]).T).A1
        c = p[0] + .5*(p[1]*x0[0] + p[2]*x0[1])
        h = np.matrix([[p[3], .5*p[5]], [.5*p[5], p[4]]])
        return c, x0, h
    
    
    def quad_max(self, a):
        """\
        (c, x0) = quad_max(a)
    
        Fits a parabola (or paraboloid) to A and returns the
        maximum value c of the fitted function, along with its
        position x0 (in pixel units).
        All entries are None upon failure. Failure occurs if :
        * A has a positive curvature (it then has a minimum, not a maximum).
        * A has a saddle point
        * the hessian of the fit is singular, that is A is (nearly) flat.
        """
    
        c, x0, h = self.quad_fit(a)
    
        failed = False
        if a.ndim == 1:
            if h > 0:
                print('Warning: positive curvature!')
                failed = True
        else:
            if h[0, 0] > 0:
                print('Warning: positive curvature along first axis!')
                failed = True
            elif h[1, 1] > 0:
                print('Warning: positive curvature along second axis!')
                failed = True
            elif np.linalg.det(h) < 0:
                print('Warning: the provided data fits to a saddle!')
                failed = True
    
        if failed:
            c = None
        return c, x0
    
    
    def pshift(self, a, ctr):
        """\
        Shift an array so that ctr becomes the origin using weighted contributions.

        Args:
            a (np.ndarray): Input array.
            ctr (array-like): Center coordinates to shift to origin.

        Returns:
            np.ndarray: Shifted array.
        """
        sh  = np.array(a.shape)
        out = np.zeros_like(a)
    
        ctri = np.floor(ctr).astype(int)
        ctrx = np.empty((2, a.ndim))
        ctrx[1,:] = ctr - ctri     # second weight factor
        ctrx[0,:] = 1 - ctrx[1,:]  # first  weight factor
    
        # walk through all combinations of 0 and 1 on a length of a.ndim:
        #   0 is the shift with shift index floor(ctr[d]) for a dimension d
        #   1 the one for floor(ctr[d]) + 1
        comb_num = 2**a.ndim
        for comb_i in range(comb_num):
            comb = np.asarray(tuple(("{0:0" + str(a.ndim) + "b}").format(comb_i)), dtype=int)
    
            # add the weighted contribution for the shift corresponding to this combination
            cc = ctri + comb
            out += np.roll( np.roll(a, -cc[1], axis=1), -cc[0], axis=0) * ctrx[comb,range(a.ndim)].prod()
    
        return out
    
    
    def sub_pix_min(self, a, width=1):
        """
        Find the position of the minimum in 2D array a with subpixel precision (using a paraboloid fit).

        Args:
            a (np.ndarray): Input 2D array.
            width (int, optional): Window size for the fit.

        Returns:
            np.ndarray: Subpixel position of the minimum.
        """
    
        sh = a.shape
    
        # Find the global minimum
        cmin = np.array(np.unravel_index(a.argmin(), sh))
    
        # Move away from edges
        if cmin[0] < width:
            cmin[0] = width
        elif cmin[0]+width >= sh[0]:
            cmin[0] = sh[0] - width - 1
        if cmin[1] < width:
            cmin[1] = width
        elif cmin[1]+width >= sh[1]:
            cmin[1] = sh[1] - width - 1
    
        # Sub-pixel minimum position.
        mindist, r = self.quad_max(-np.real(a[(cmin[0]-width):(cmin[0]+width+1), (cmin[1]-width):(cmin[1]+width+1)]))
        r -= (width - cmin)
    
        return r
