from functools import partial
import numpy as np

from .hungarian import Hungarian

def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())

class RMOT:
    pInit = np.array(
            [
                [1,0,1,0,0,0],
                [0,1,0,1,0,0],
                [1,0,1,0,0,0],
                [0,1,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ])
    F = np.array(
            [
                [1,0,1,0,0,0],
                [0,1,0,1,0,0],
                [0,0,1,0,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ])
    H = np.array(
            [
                [1,0,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]
            ])

    Q = 1e-4*np.identity(6)
    R = 1e-4*np.identity(4)

    Fr = np.array(
            [
                [1,0,1,0],
                [0,1,0,1],
                [0,0,1,0],
                [0,0,0,1]
            ])
    Qr = 1e-4*np.identity(4)
    Rr = 1e-4*np.identity(2)

    def __init__(self, xs):
        self.N = xs.shape[0]
        self.xPrev = xs

        self.hungarian = Hungarian()
        self.tau       = 5
        self.costMtx   = None
        self.assignMtx = None

        self.o   = np.ones(self.N, dtype=np.int)
        self.RMN = np.ones((self.N, self.N), dtype=np.int)
        self.relativeWeightMtx = np.full((self.N, self.N), 1.0/self.N)

        self.pPrev = np.zeros((self.N,6,6)) + RMOT.pInit

        r = np.tile(self.xPrev[:,:4], self.N).flatten() - np.tile(self.xPrev[:,:4], (self.N,1)).flatten()
        self.r = r.reshape(self.N,self.N,4)
        self.pr = np.zeros((self.N,self.N,4,4)) + symmetrize(1e-2*np.random.randn(4,4))

    def calcConditionalMtx(self):
        # r = np.tile(self.xPrev[:,:4], self.N).flatten() - np.tile(self.xPrev[:,:4], (self.N,1)).flatten()
        # self.r = r.reshape(self.N,self.N,4)

        xi = np.tile(self.xPrev, self.N).reshape(self.N,self.N,6)
        x = xi + np.concatenate((self.r, np.zeros((self.N,self.N,2))), axis=2)
        x = x + 0.01*np.random.randn(*x.shape)

        self.xCond = np.einsum('...ij,...j', RMOT.F, x)
        self.pCond = np.dot(np.einsum('...ij,...jk->...ik',RMOT.F,self.pPrev), RMOT.F.T) + RMOT.Q

    def likelihood(self,z1,z2):
        x, y = z1[:2]
        wz1, hz1 = z1[2:4]
        iwz1 = [x-wz1/2, x+wz1/2]
        ihz1 = [y-hz1/2, y+hz1/2]

        x, y = z2[:2]
        wz2, hz2 = z2[4:6]
        iwz2 = [x-wz2/2, x+wz2/2]
        ihz2 = [y-hz2/2, y+hz2/2]

        x_overlap = max(0.0, min(iwz1[1],iwz2[1]) - max(iwz1[0],iwz2[0]))
        y_overlap = max(0.0, min(ihz1[1],ihz2[1]) - max(ihz1[0],ihz2[0]))
        intersection = x_overlap * y_overlap
        union = wz1*hz1 + wz2*hz2 - intersection

        # d = np.dot(z1[:2]-(z2[:2]-z2[2:4]), z2[2:4])/(np.linalg.norm(z1[:2]-(z2[:2]-z2[2:4])) * np.linalg.norm(z2[2:4]))
        # d = 1.0 / (1.0 + np.exp(-5.0*(d)))

        # return 1.0/(1.0 + np.linalg.norm(z1[:2]-z2[:2]))

        # s = 0.0
        # if 1>union or intersection>union:
        #     s = 0.0
        # else:
        #     s = intersection/union
        s = intersection/union

        # return 0.2*s + 0.8*1.0/(1.0+np.linalg.norm(z1[:2]-z2[:2]))
        # return 1.0/(1.0+np.linalg.norm(z1[:2]-z2[:2]))
        return s/(1.0 + np.linalg.norm(z1[:2]-z2[:2]))

    def calcCostMtx(self,z):
        self.z = z
        self.M = z.shape[0]

        self.costMtx = np.zeros((self.N, self.M))

        for k in range(self.M):
            likelihoodMtx = np.full(self.RMN.shape, -1)
            func = partial(self.likelihood, z[k])
            likelihoodMtx[self.RMN==1] = np.apply_along_axis(func, 1, self.xCond[self.RMN==1]) * self.relativeWeightMtx[self.RMN==1]
            for i in range(self.N):
                j = np.argmax(likelihoodMtx[i,:])
                self.costMtx[i,k] = -np.log(self.likelihood(self.z[k], self.xCond[i,j]))
        self.costMtx = np.nan_to_num(self.costMtx)

    def calcAssignMtx(self):
        self.hungarian.calculate(self.costMtx)

        RC = np.max([self.N, self.M])
        self.assignMtx = np.zeros((RC, RC))
        for t in self.hungarian.get_results():
            self.assignMtx[t] = 1
        self.assignMtx = self.assignMtx[:self.N,:self.M]

    def calcObservationMtx(self):
        nonzeroPos = np.nonzero(self.assignMtx)

        obsMtx = np.full((self.N, self.M), False, dtype=np.bool_)
        obsMtx[nonzeroPos] = self.costMtx[nonzeroPos]<self.tau

        self.gammaMtx = np.zeros((self.N, self.M))
        self.gammaMtx[obsMtx==True] = 1

        self.o = np.sum(self.gammaMtx, axis=1).astype(np.int)

    def calcEventProb(self):
        RMNsum = np.sum(self.RMN, axis=1).reshape(self.N,1)

        self.P_i_k = self.gammaMtx/RMNsum
        self.P_i_j = np.ones((self.N,self.N))/RMNsum - np.sum(self.P_i_k, axis=1).reshape(self.N,1)

    def calcRelativeWeight(self):
        Pz_i_j = self.P_i_j

        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.M):
                    Pz_i_j[i,j] += self.P_i_k[i,k]*self.likelihood(self.z[k], self.xCond[i,j])

        weightProbMtx = self.relativeWeightMtx * Pz_i_j
        weightProbMtx[self.RMN==0] = 0
        self.relativeWeightMtx = weightProbMtx/np.sum(weightProbMtx)

    def calcObjectStates(self):
        for i in range(self.N):
            if self.o[i]==1:
                j = np.argmax(self.relativeWeightMtx[i,:])

                K    = np.dot(self.pCond[i], RMOT.H.T)
                HPH  = np.dot(RMOT.H, K)
                HPHR = HPH + RMOT.R
                inv  = np.linalg.inv( HPHR )
                K    = np.dot(K, inv)

                gammaZ = self.z-np.dot(RMOT.H, self.xCond[i,j])
                gammaZ = self.gammaMtx[i,:].reshape(self.M,1)*gammaZ

                self.xPrev[i] = self.xCond[i,j] + np.dot(K, np.sum(gammaZ, axis=0))

                self.pPrev[i] = self.pCond[i] -np.dot(np.dot(K, HPHR), K.T)
            else:
                # self.xPrev[i] = self.xCond[i,i]
                # self.pPrev[i] = self.pCond[i]
                pass

        self.RMN[:] = 0
        np.fill_diagonal(self.RMN,1)
        self.RMN[:, self.o==1] = 1
        # print(self.RMN)
        # print(self.xPrev)
        # print(self.pPrev)

    def calcRelativeMotion(self):
        # https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%9E%E3%83%B3%E3%83%95%E3%82%A3%E3%83%AB%E3%82%BF%E3%83%BC#.E3.82.AB.E3.83.AB.E3.83.9E.E3.83.B3.E3.83.95.E3.82.A3.E3.83.AB.E3.82.BF.E3.83.BC
        rCond  = np.einsum('...ij,...j', RMOT.Fr, self.r)
        rCond  = rCond + 1e-2*np.random.randn(*rCond.shape)
        prCond = np.einsum('...ij,...jk->...ik', np.einsum('...ij,...jk->...ik', RMOT.Fr, self.pr), RMOT.Fr.T) + RMOT.Qr

        rObs = np.tile(self.xPrev[:,:2], self.N).flatten() - np.tile(self.xPrev[:,:2], (self.N,1)).flatten()
        rObs = rObs.reshape(self.N, self.N, 2)

        e = rObs - rCond[:,:,:2]
        S = prCond[:,:,:2,:2] + RMOT.Rr
        K = np.einsum('...ij,...jk->...ik', prCond[:,:,:,:2], np.linalg.inv(S))

        self.r = rCond + np.einsum('...ij,...j', K, e)

        prTransitionMtx = np.identity(4) - np.concatenate((K, np.zeros((self.N,self.N,4,2))), axis=3)

        self.pr = np.einsum('...ij,...jk->...ik', prTransitionMtx, prCond)

    def calculation(self, zs):
        self.calcConditionalMtx()
        self.calcCostMtx(zs)
        self.calcAssignMtx()
        self.calcObservationMtx()
        self.calcEventProb()
        self.calcRelativeWeight()
        self.calcObjectStates()
        self.calcRelativeMotion()

        return self.xPrev

if __name__=="__main__":
    x1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    x2 = np.array([3.0, 3.0, 0.0, 0.0, 1.0, 1.0])
    x3 = np.array([1.5, 1.5, 0.0, 0.0, 1.0, 1.0])
    xs = np.array([x1, x2])

    z1 = np.array([0.2, 0.2, 1.0, 1.0])
    z2 = np.array([3.1, 3.1, 1.0, 1.0])
    z3 = np.array([1.5, 1.5, 1.0, 1.0])
    zs = np.array([z1, z2, z3])

    # x1 = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    # x2 = np.array([3.0, 3.0, 0.0, 0.0, 1.0, 1.0])
    # xs = np.array([x1, x2])
    #
    # z1 = np.array([0., 0., 1.0, 1.0])
    # z2 = np.array([3., 3., 1.0, 1.0])
    # zs = np.array([z1, z2])

    rmot = RMOT(xs)
    rmot.calcConditionalMtx()
    rmot.calcCostMtx(zs)
    rmot.calcAssignMtx()
    rmot.calcObservationMtx()
    rmot.calcEventProb()
    rmot.calcRelativeWeight()
    rmot.calcObjectStates()
    rmot.calcRelativeMotion()


    z1 = np.array([0.1, 0.1, 1.0, 1.0])
    z2 = np.array([3.4, 3.5, 1.0, 1.0])
    z3 = np.array([3.0, 3.0, 1.0, 1.0])
    zs = np.array([z1, z2, z3])

    rmot.calcConditionalMtx()
    rmot.calcCostMtx(zs)
    rmot.calcAssignMtx()
    rmot.calcObservationMtx()
    rmot.calcEventProb()
    rmot.calcRelativeWeight()
    rmot.calcObjectStates()
    rmot.calcRelativeMotion()
