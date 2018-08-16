import ROOT
from scipy.interpolate import UnivariateSpline
import numpy as np

class ParametricRecoilFunction:
    def __init__(self,mu,sigma,aL,aR,offset=0,valRange=None,pdfThrForCDF=1e-5):
        """init function"""

        #configure
        self.mu=mu
        self.sigma=abs(sigma)
        self.aL=abs(aL)
        self.aR=abs(aR)
        self.offset=abs(offset)
        self.tval=None
        self.valRange=valRange
        if self.valRange:
            self.tval=[ (x-mu)/sigma for x in valRange ]
        
        #compute first moments
        self.Ot = [ self.moment(i) for i in xrange(0,3) ]
        self.norm=1./(self.Ot[0]*self.sigma)

        #transform back to x=st+u
        self.Ox = [ self.sigma*self.Ot[0],
                    (self.sigma**2)*self.Ot[1]+self.mu*self.sigma*self.Ot[0],
                    (self.sigma**3)*self.Ot[2]+2*(self.sigma**2)*self.mu*self.Ot[1]+(self.mu**2)*self.sigma*self.Ot[0] ]
        self.Ox = [ x*self.norm for x in self.Ox ]

        #mean and width of the x variable
        self.mean=self.Ox[1]
        self.width=(self.Ox[2]-self.Ox[1]**2)
        
        self.cdf_inv=None

    def moment(self,i):
        """compute the first 3 Mellin moments of the distribution"""

        mval=0
        if i==0:
            mval+=1./(self.aL*ROOT.TMath.Exp(0.5*(self.aL**2)))
            mval+=1./(self.aR*ROOT.TMath.Exp(0.5*(self.aR**2)))
            mval+=ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)*(ROOT.TMath.Erf(self.aL/ROOT.TMath.Sqrt(2))+ROOT.TMath.Erf(self.aR/ROOT.TMath.Sqrt(2)))
        if i==1:
            mval+=-((1.+self.aL**2)/(self.aL**2*ROOT.TMath.Exp(self.aL**2/2)))
            mval+=(1.+self.aR**2)/(self.aR**2*ROOT.TMath.Exp(self.aR**2/2))
            mval+=ROOT.TMath.Exp(-self.aL**2/2)-ROOT.TMath.Exp(-self.aR**2/2)
        if i==2:
            mval+=(2.+2*self.aL**2+self.aL**4)/(self.aL**3*ROOT.TMath.Exp(self.aL**2/2))
            mval+=(2.+2*self.aR**2+self.aR**4)/(self.aR**3*ROOT.TMath.Exp(self.aR**2/2))
            mval+=self.aL/ROOT.TMath.Exp(self.aL**2/2)-self.aR/ROOT.TMath.Exp(self.aR**2/2)
            mval+=-(self.aL/ROOT.TMath.Exp(self.aL**2/2))-self.aR/ROOT.TMath.Exp(self.aR**2/2)
            mval+=ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)*ROOT.TMath.Erf(self.aL/ROOT.TMath.Sqrt(2))
            mval+=ROOT.TMath.Sqrt(ROOT.TMath.Pi()/2)*ROOT.TMath.Erf(self.aR/ROOT.TMath.Sqrt(2))

        if self.tval:
            if i==0:
                mval+=-(ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL))/self.aL)
                mval+=-(ROOT.TMath.Exp(-self.aR**2/2+self.aR*(-self.tval[1]+self.aR))/self.aR)
                mval+=(-self.tval[0]+self.tval[1])*self.offset
            if i==1:
                mval+=ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL))/self.aL**2
                mval+=-(self.tval[0]*ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL)))/self.aL
                mval+=-(ROOT.TMath.Exp(-self.aR**2/2+self.aR*(-self.tval[1]+self.aR))/self.aR**2)
                mval+=-(self.tval[1]*ROOT.TMath.Exp(-self.aR**2/2+self.aR*(-self.tval[1]+self.aR)))/self.aR
                mval+=-(self.tval[0]**2*self.offset)/2+(self.tval[1]**2*self.offset)/2
            if i==2:
                mval+=(-2*ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL)))/self.aL**3
                mval+=(2*self.tval[0]*ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL)))/self.aL**2
                mval+=-(self.tval[0]**2*ROOT.TMath.Exp(-self.aL**2/2+self.aL*(self.tval[0]+self.aL)))/self.aL
                mval+=(-2*ROOT.TMath.Exp((self.aR*(-2*self.tval[1]+self.aR))/2))/self.aR**3
                mval+=-(2*self.tval[1]*ROOT.TMath.Exp((self.aR*(-2*self.tval[1]+self.aR))/2))/self.aR**2
                mval+=-(self.tval[1]**2*ROOT.TMath.Exp((self.aR*(-2*self.tval[1]+self.aR))/2))/self.aR
                mval+=-(self.tval[0]**3*self.offset)/3+(self.tval[1]**3*self.offset)/3
        return mval

    def eval(self,x):
        """evaluate at a value of x, after transforming it by the mean and sigma of the distribution"""

        t=(x-self.mu)/self.sigma
        if self.tval:
            if t<self.tval[0] or t>self.tval[1]:
                return 0.

        #evaluate depending on the branch
        ft=self.offset
        if t<-self.aL:
            ft+=ROOT.TMath.Exp(self.aL*(t+0.5*self.aL))
        elif t>self.aR:
            ft+=ROOT.TMath.Exp(-self.aR*(t-0.5*self.aR))
        else:
            ft+=ROOT.TMath.Exp(-0.5*t*t)

        #normalize
        ft *= self.norm

        return ft

        
    def getEventPDF(self,bins):
        """returns the event PDF"""
        return [ self.eval(x) for x in bins ]

    def getRandom(self,nintpts=100):
        """ generate a random number based on this PDF """

        #determine the cdf if needed
        if not self.cdf_inv:
            while True:
                try:
                    self.generateCDFInv()
                    break
                except Exception as e:
                    nintpts *= 2
                    if nintpts>1e4:
                        raise ValueError('Unable to find enough points to interpolate inverse CDF')
                    pass
        
        #generate random
        r=np.random.uniform()
        return self.cdf_inv(r)

    def generateCDFInv(self,nintpts=100):
        """generates the CDF from integrating the PDF in a grid
        then inverts it with a spline function"""
        xmin=self.valRange[0] if self.valRange else self.mean-10*self.width
        xmax=self.valRange[1] if self.valRange else self.mean+10*self.width
        dx=(xmax-xmin)/float(nintpts)
        x=np.arange(xmin,xmax,dx)
        cdf=np.cumsum(self.getEventPDF(x))
        cdf*=dx
        cdf_inv=[]
        for i in xrange(0,len(cdf)):
            if cdf[i]<1e-10 : continue
            if abs(1-cdf[i])<1e-10: continue
            if len(cdf_inv)>0 and abs(cdf[i]-cdf_inv[-1][1])<1e-10 : continue
            cdf_inv.append( (x[i],cdf[i]) )        
        cdf_inv=np.array(cdf_inv)
        self.cdf_inv = UnivariateSpline(cdf_inv[:,1],cdf_inv[:,0])
        self.cdf_inv.set_smoothing_factor(0.5)

def getParametrizedRecoilMomenta(prediction):
    """wrapper to get the mean and width from the parameterized function"""
    
    mean,width=[0,0]
    try:
        mu=prediction[0]        
        sigma=prediction[1]
        aL=prediction[2]
        aR=prediction[3]
        offset,valRange=0.,None
        if len(prediction)>4:
            offset=prediction[4]
            valRange=[-ROOT.TMath.Pi(),ROOT.TMath.Pi()]
        prf=ParametricRecoilFunction(mu,sigma,aL,aR,offset,valRange)
        mean=prf.mean
        width=prf.width
    except:
        pass

    return [mean,width]
