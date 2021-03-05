class Scopes:
    def __init__(self, shot, currentStart=0., path=r'//LINNA/scopes/'):
        self.sh = shot
        self.pth = path
        self.start = float(currentStart)
    def getCh(self, scopeNumber, channel):
        voltName = 'scope' + str(scopeNumber)
        if(channel[0]=='E'):
            voltName += 'E'
        voltName += '_' + self.sh
        timeName = voltName + 'time'
        voltName += '_' + channel
        volts = np.genfromtxt(self.pth+voltName)[0:-1]
        time = np.genfromtxt(self.pth+timeName) - self.start
        return time, volts
    def getRogA(self):
        return self.getCh(2,'C1')
    def getRogB(self):
        return self.getCh(2,'C2')
    def getdIdt_G(self):
        return self.getCh(10,'A1')
    def getdIdt_H(self):
        return self.getCh(10,'A2')
    def getdIdt_C(self):
        return self.getCh(10,'B1')
    def getdIdt_Z(self):
        return self.getCh(10,'B2')
    def getLine_G(self):
        return self.getCh(10,'C1')
    def getLine_H(self):
        return self.getCh(10,'C2')
    def getLine_C(self):
        return self.getCh(10,'D1')
    def getLine_Z(self):
        return self.getCh(10,'D2')
    def getMarx_G(self):
        return self.getCh(11,'C1')
    def getMarx_H(self):
        return self.getCh(11,'C2')
    def getMarx_C(self):
        return self.getCh(11,'D1')
    def getMarx_Z(self):
        return self.getCh(11,'D2')
  
  class Rogowski:
    calibration = 3. #A/Vns
    attenA = 206.
    attenB = 216.
    backoff=10. #Number of ns before current start to begin intergrating
    def __init__(self, scope):
        self.scope = scope
    def getInductiveComponent(self):
        tA, rogA = self.scope.getRogA()
        tB, rogB = self.scope.getRogB() 
        inductive = (rogA*self.attenA - rogB*self.attenB)*0.5
        return tA, inductive
    def getCurrent(self, numPosts=8):
        t, V = self.getInductiveComponent()
        currentStartIndex = np.argmin( np.abs(t+self.backoff) )
        t_trimmed = t[currentStartIndex:]
        V_trimmed = V[currentStartIndex:]
        intergrated = cumtrapz(V_trimmed, x=t_trimmed)
        intergrated *= self.calibration
        intergrated *= numPosts
        return t_trimmed, intergrated