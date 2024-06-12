import numpy as np
from scipy.integrate import cumtrapz
from scipy.fft import rfft, rfftfreq, irfft
class Scopes:
    def __init__(self, shot, currentStart=1400., path=r'//LINNA/scopes/'):
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
        time = np.genfromtxt(self.pth+timeName)
        return time, volts
    def getRog1(self):
        return self.getCh(2,'C1')
    def getRog2(self):
        return self.getCh(2,'C2')
    def getRogA(self):
        return self.getCh(1,'D1')
    def getRogB(self):
        return self.getCh(1, 'D2')
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
        
class Channel:
    def __init__(self, scope_object, scope_number, channel_key, attenuation):
        self.currentStart = scope_object.start
        self.t, self.v = scope_object.getCh(scope_number, channel_key)
        self.v *= attenuation
    def low_pass(self, cuttoff_s):
        c_ns = cuttoff_s*1e-9
        v_fft = rfft(self.v)
        N = self.t.shape[0]
        dt = self.t[1]-self.t[0]
        f = rfftfreq(N, dt)
        window = f<c_ns
        self.v_filt = irfft(v_fft*window)
        
class Rogowski:
    def __init__(self, channel, calibration=3.):
        self.channel = channel
        self.calibration = calibration
    def getInductiveComponent(self, other_probe):
        t = self.channel.t - self.channel.currentStart
        v1 = self.channel.v
        v2 = other_probe.channel.v       
        inductive = (v1 - v2)*0.5
        return t, inductive
    def getcapacitiveComponent(self, other_probe):
        t = self.channel.t - self.channel.currentStart
        v1 = self.channel.v
        v2 = other_probe.channel.v       
        inductive = (v1 + v2)*0.5
        return t, inductive
    def getCurrent(self, other_probe=None, numPosts=8, backoff=10.):
        if(other_probe is None):
            t = self.channel.t - self.channel.currentStart
            V = self.channel.v
        else:
            t, V = self.getInductiveComponent(other_probe)
        currentStartIndex = np.argmin( np.abs(t+backoff) )
        t_trimmed = t[currentStartIndex:]
        V_trimmed = V[currentStartIndex:]
        intergrated = cumtrapz(V_trimmed, x=t_trimmed, initial=0.)
        intergrated *= self.calibration
        intergrated *= numPosts
        return t_trimmed, intergrated
    
        
