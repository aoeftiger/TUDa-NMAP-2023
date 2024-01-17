# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 18:47:33 2012

@author: ldrosdal
"""
import math
import warnings
import numpy as np
# import matplotlib.pyplot as plt
import operator
import numpy
from numpy import matrix

# filename_Q26 = '/Users/vkain/cernbox/CERN_dfs/SPS/twisses/twiss_Q26'
# filename_Q20 = '/Users/vkain/cernbox/CERN_dfs/SPS/twisses/twiss_Q20'
# filename_Q22 = '/Users/vkain/cernbox/CERN_dfs/SPS/twisses/twiss_Q22'

filename_awakeElectron = 'electron_tt43.out'

# type_monitor = 'M'
# type_kicker = 'K'
# type_other = 'O'
# type_collimator = 'C'

FACTOR_UM = 1000000.0  # TODO: when to scale?

FIELD_NAME = 'NAME'
FIELD_S = 'S'
FIELD_X = 'X'
FIELD_PX = 'PX'
FIELD_BETX = 'BETX'
FIELD_MUX = 'MUX'
FIELD_ALPX = 'ALFX'
FIELD_Y = 'Y'
FIELD_PY = 'PY'
FIELD_BETY = 'BETY'
FIELD_MUY = 'MUY'
FIELD_ALPY = 'ALFY'
FIELD_DX = 'DX'
FIELD_DY = 'DY'

PLANES = ['H', 'V']


def getSigma(beta, emittance=3.5):  # mm
    return numpy.sqrt(emittance * beta * 0.938272 / 450)  # print 3.5*0.938272/450 #7.3 e-3

def getQuadPhases(filename, testvalues):
    quads = {}

    seq = readTwissFromMADX(filename)

    for p in [0, 1]:
        for element in seq[p].elements:
            if 'QF.' in element.name or 'QD.' in element.name or 'MQ' in element.name or 'QT' in element.name or 'QDA' in element.name or 'QFA' in element.name:
                if not element.name in quads:
                    quads[element.name] = [0, 0, 0, 0]
                quads[element.name][p] = element.main_actor_net
                diffCheck = element.main_actor_net - element.main_actor_net // 1
                diffCheck = abs(np.array(
                    [diffCheck - testvalues[p] - 0.5, diffCheck - testvalues[p], diffCheck - testvalues[p] + 0.5,
                     diffCheck - testvalues[p] + 1]))
                quads[element.name][p + 2] = min(diffCheck)

    for ik in quads.iteritems():
        print(ik)


def readTwissFromMADX(inputFile, name=''):
    if name == '':
        name = inputFile.split('/')[-1].split('.')[0]
    data = open(inputFile, 'r')
    fieldNames = []
    i_start = 0

    hSequence = TwissSequence('H', name)
    vSequence = TwissSequence('V', name)

    for i, line in enumerate(data):
        if (line.startswith('*')):
            for idx, val in enumerate(line.split()):
                fieldNames.append(val)
            i_start = i + 1

            if not FIELD_NAME in fieldNames:
                raise TwissException('MISSING FIELD', FIELD_NAME, 'IN TWISS INPUT')
            if not FIELD_S in fieldNames:
                raise TwissException('MISSING FIELD', FIELD_S, 'IN TWISS INPUT')
            if not FIELD_X in fieldNames:
                x = 0
                warnings.warn('MISSING FIELD' + FIELD_X + 'IN TWISS INPUT' + inputFile)
            if not FIELD_PX in fieldNames:
                px = 0
                warnings.warn('MISSING FIELD ' + FIELD_PX + ' IN TWISS INPUT ' + inputFile)
            if not FIELD_BETX in fieldNames:
                bx = 0
                warnings.warn('MISSING FIELD ' + FIELD_BETX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_MUX in fieldNames:
                mux = 0
                warnings.warn('MISSING FIELD ' + FIELD_MUX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_ALPX in fieldNames:
                alfx = 0
                warnings.warn('MISSING FIELD ' + FIELD_ALPX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_Y in fieldNames:
                y = 0
                warnings.warn('MISSING FIELD' + FIELD_Y + 'IN TWISS INPUT' + inputFile)
            if not FIELD_PY in fieldNames:
                py = 0
                warnings.warn('MISSING FIELD ' + FIELD_PY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_BETY in fieldNames:
                by = 0
                warnings.warn('MISSING FIELD ' + FIELD_BETY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_MUY in fieldNames:
                muy = 0
                warnings.warn('MISSING FIELD ' + FIELD_MUY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_ALPY in fieldNames:
                alfy = 0
                warnings.warn('MISSING FIELD ' + FIELD_ALPY + ' IN TWISS INPUT' + inputFile)
            if not FIELD_DX in fieldNames:
                dx = 0
                warnings.warn('MISSING FIELD ' + FIELD_DX + ' IN TWISS INPUT' + inputFile)
            if not FIELD_DY in fieldNames:
                dy = 0
                warnings.warn('MISSING FIELD ' + FIELD_DY + ' IN TWISS INPUT' + inputFile)

        elif (i > i_start and i_start > 0):
            for idx, val in enumerate(line.split()):
                if (fieldNames[idx + 1] == FIELD_NAME):
                    name = val.strip('"')
                #                    if name.startswith('DRIFT'):
                #                        break
                elif (fieldNames[idx + 1] == FIELD_X):
                    x = float(val)
                elif (fieldNames[idx + 1] == FIELD_Y):
                    y = float(val)
                elif (fieldNames[idx + 1] == FIELD_S):
                    s = float(val)
                elif (fieldNames[idx + 1] == FIELD_MUX):
                    mux = float(val)
                elif (fieldNames[idx + 1] == FIELD_MUY):
                    muy = float(val)
                elif (fieldNames[idx + 1] == FIELD_BETX):
                    bx = float(val)
                elif (fieldNames[idx + 1] == FIELD_BETY):
                    by = float(val)
                elif (fieldNames[idx + 1] == FIELD_ALPX):
                    alfx = float(val)
                elif (fieldNames[idx + 1] == FIELD_ALPY):
                    alfy = float(val)
                elif (fieldNames[idx + 1] == FIELD_PX):
                    px = float(val)
                elif (fieldNames[idx + 1] == FIELD_PY):
                    py = float(val)
                elif (fieldNames[idx + 1] == FIELD_DX):
                    dx = float(val)
                elif (fieldNames[idx + 1] == FIELD_DY):
                    dy = float(val)

            hSequence.add(TwissElement(name, s, x, px, bx, alfx, mux, dx))  # , t))
            vSequence.add(TwissElement(name, s, y, py, by, alfy, muy, dy))  # , t))

    return hSequence, vSequence


def readAWAKEelectronTwiss():
    filename_awakeElectron = 'actor_critic/utils/electron_tt43.out'
    twissH, twissV = readTwissFromMADX(filename_awakeElectron)
    return twissH, twissV


def getDispersionAsSource(filename):
    seq = readTwissFromMADX(filename)[0]

    seq.name = 'dispersion'

    for e in seq.elements:
        e.x = e.d

    return seq


class TwissSequence:
    def __init__(self, plane, name=''):
        self.plane = plane
        self.name = name
        self.elements = []
        self.elementNames = []

    def add(self, e):

        #        if e.name in self.getNames():
        #            raise Exception(e.name+' already in sequence '+str(self.getNames()))
        self.elements.append(e)
        self.elementNames.append(e.n)

    #    def clean(self): #TODO: do smarter
    #        for i, e in enumerate(self.elements):
    #            if e.type==type_monitor:
    #                if self.monitorNames[-1].endswith(e.n):
    #                    i_n=i+1
    #
    #
    ##        l=len(self.elements)
    #        self.elements=self.elements[0:i_n]
    #        self.elementNames=self.elementNames[0:i_n]
    ##        print 'Reduced elements:', l, '-->', len(self.elements), self.elements[0].name, self.elements[-1].name
    #
    #

    def remove(self, index):
        self.elements.pop(index)
        self.elementNames.pop(index)




    def __getitem__(self, i):
        return self.elements[i]

    def calculateTrajectory(self, monitorNames, kickerNames, kicks, kN,
                            allPoints=False):  ##TODO: calculate at every value?

        p = self.elements[0].x, self.elements[0].px
        x_um = [p[0] * FACTOR_UM]

        for i in range(1, len(self.elementNames)):
            kick = self.findKick(kicks, kN, self.elements[i])
            p = self.calculateTransfer(self.elements[i - 1], self.elements[i], p, kick)
            #            if 'BPMI' in self.elements[i].name:
            #                print self.elements[i].name, p
            x_um.append(p[0] * FACTOR_UM)

        return self.extractMonitorValues(x_um, monitorNames)  # TODO: need this?

    def findKick(self, kicks, kN, e):
        if not e.name.startswith('M'):  # ==type_kicker:
            return 0

        try:
            return kicks[kN.index(e.n)] / FACTOR_UM
        except ValueError:
            return 0

    def extractMonitorValues(self, x_um, names):
        monValues = []
        for m in names:
            monValues.append(x_um[self.elementNames.index(m.split('.')[-1])])
        return monValues

    def getMonitors(self, names):
        newSequence = TwissSequence(self.plane, self.name)
        for m in names:
            newSequence.add(self.elements[self.elementNames.index(m.split('.')[-1])])

        return newSequence

    def removePlaneFromMonitors(self):
        for element in self.elements:
            if 'BPMI' in element.name:
                test = element.name.split('.')
                test[0] = test[0].replace('H', '').replace('V', '')
                element.name = '.'.join(test)

    def getElementsByNames(self, names):
        newSequence = TwissSequence(self.plane, self.name)
        for m in names:
            index = self.getNames().index(m)
            newSequence.add(self.elements[index])

        return newSequence

    def getElementsByPosKeys(self, ns):
        newSequence = TwissSequence(self.plane, self.name)
        for m in ns:
            #            print m, self.elementNames
            index = self.elementNames.index(m.split('.')[-1])
            newSequence.add(self.elements[index])

        return newSequence

    # key in element
    def getElements(self, key):
        newSequence = TwissSequence(self.plane, self.name)
        for element in self.elements:
            if key in element.name:
                newSequence.add(element)
        return newSequence

    def getNames(self):
        names = []
        for e in self.elements:
            names.append(e.name)
        return names

    def getS(self):
        s = []
        for e in self.elements:
            s.append(e.s)
        return s

    def getX(self):  # returns also y in vplane
        x = []
        for e in self.elements:
            x.append(e.x * FACTOR_UM)  # *factor?
        return x

    def getMu(self):  # returns also y in vplane
        mu = []
        for e in self.elements:
            mu.append(e.main_actor_net)
        return mu

    def getAlpha(self):
        return [e.alpha for e in self.elements]

    def getBeta(self):  # returns also y in vplane
        beta = []
        for e in self.elements:
            beta.append(e.beta)
        return beta

    def getD(self):
        d = []
        for e in self.elements:
            d.append(e.d)
        return d

    def getElement(self, name):
        for e in self.elements:
            if (e.name == name):
                return e
        else:
            raise TwissException('Element not found: ' + name)

    # MDMH and MDSV, MDMV in ti8 is hkicker
    def calculateTransfer(self, e0, e1, x0, kick):
        if kick:
            if 'MDLH' in e1.name or 'MDLV' in e1.name:  # print 'not hkicker, but rbend, length=1.4'
                kick = -kick
        #            elif not 'MCIA' in e1.name:
        #                print 'kick by', e1.name

        M11, M12, M21, M22 = self.transferMatrix(e0, e1)
        return [M11 * x0[0] + M12 * x0[1], M21 * x0[0] + M22 * x0[1] + kick]

    def interpolateBetween(self, e0, x0, e1, x1, e2):
        if (e0.s < e2.s):
            a11, a12, a21, a22 = self.transferMatrix(e0, e2)
            A = matrix([[a11, a12], [a21, a22]])
        else:
            a11, a12, a21, a22 = self.transferMatrix(e2, e0)
            A = matrix([[a11, a12], [a21, a22]])
            A = A.I
        if (e1.s < e2.s):
            b11, b12, b21, b22 = self.transferMatrix(e1, e2)
            B = matrix([[b11, b12], [b21, b22]])
        else:
            b11, b12, b21, b22 = self.transferMatrix(e2, e1)
            B = matrix([[b11, b12], [b21, b22]])
            B = B.I
        # print (B)
        # print(B[0,0])
        e = (B[1, 0] * x1 - A[1, 0] * x0) / A[1, 1]
        g = B[1, 1] / A[1, 1]
        f = (B[0, 0] * x1 - A[0, 0] * x0) / A[0, 1]
        h = B[0, 1] / A[0, 1]

        px1 = (e - f) / (h - g)
        px0 = (B[0, 0] * x1 + B[0, 1] * px1 - A[0, 0] * x0) / A[0, 1]

        x_vec = np.array([x0, px0])
        # print(x_vec)
        x_out = A[0, 0] * x_vec[0] + A[0, 1] * x_vec[1]
        # print(x0,x1,x_out)
        return x_out

    def transferMatrix(self, e0, e1):

        dmu = (e1.main_actor_net - e0.main_actor_net) * 2 * math.pi
        cos_dmu = math.cos(dmu)
        sin_dmu = math.sin(dmu)
        sqrt_mult = math.sqrt(e0.beta * e1.beta)
        sqrt_div = math.sqrt(e1.beta / e0.beta)

        M11 = sqrt_div * (cos_dmu + e0.alpha * sin_dmu)
        M12 = sqrt_mult * sin_dmu
        M21 = ((e0.alpha - e1.alpha) * cos_dmu - (1 + e0.alpha * e1.alpha) * sin_dmu) / sqrt_mult
        M22 = (cos_dmu - e1.alpha * sin_dmu) / sqrt_div

        return M11, M12, M21, M22

    def subtractSource(self, twiss2):  # only x, y?
        if len(self.elements) == len(twiss2.elements):
            for i, e in enumerate(self.elements):
                e.x -= twiss2.elements[i].x
        else:
            warnings.warn('BAD LENGTH FOR SUBTRACTION')

    def getNeighbouringBPMsforElement(self, element, plane):
        indexBefore = 0
        indexAfter = 0
        index = self.elements.index(element)

        for i in range(index - 1, 0, -1):
            # print(self.elements[i].name, indexBefore)
            if (self.elements[i].name.startswith('BP' + plane)):
                elementBefore = self.elements[i]
                indexBefore = i
                break
        if (indexBefore == 0):
            for i in range(len(self.elements) - 1, index + 1, -1):
                # print(self.elements[i].name)
                if (self.elements[i].name.startswith('BP' + plane)):
                    elementBefore = self.elements[i]
                    break

        for i in range(index + 1, len(self.elements)):
            if (self.elements[i].name.startswith('BP' + plane)):
                elementAfter = self.elements[i]
                break

        # elementBefore = self.elements[indexBefore-1]
        # elementAfter = self.elements[indexAfter+1]
        return elementBefore, elementAfter


class TwissElement:

    def __init__(self, name, s, x, px, beta, alpha, mu, d):  #:, t=type_monitor):
        self.name = name
        self.s = s
        self.x = x
        self.px = px
        self.beta = beta
        self.mu = mu
        self.alpha = alpha
        #        self.type = t
        self.d = d
        self.n = name.split('.')[-1]


class TwissException(Exception):
    pass


class TwissWarning(Warning):
    pass
