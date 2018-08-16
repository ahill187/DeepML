from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad
from DeepJetCore.stopwatch import stopwatch
from ParametricRecoilFunction import getParametrizedRecoilMomenta
from argparse import ArgumentParser
import numpy

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

class TrainData_Recoil(TrainData):
    '''
    Class for Recoil training
    '''
    
    def __init__(self,args):
        import numpy
        TrainData.__init__(self)

        #parse specific arguments to this class
        parser=ArgumentParser('TrainData_Recoil parser')
        parser.add_argument('--regress',
                            help='Regression targets list (CSV where first variable is the truth)', 
                            default='mu')
        parser.add_argument('--target',
                            help='Regression target',
                            default='lne1')
        parser.add_argument('--varList',
                            help='Variables to use in the regression', 
                            default='lne1,tkmet_pt,tkmet_phi,tkmet_n,tkmet_m,tkmet_sphericity,ntnpv_pt,ntnpv_sphericity,nvert,mindz,absdphi_ntnpv_tk,cosdphi_ntnpv_tk,dphi_puppi_tk')
        parser.add_argument('--isolateVars',
                            help='Variables to isolate and pass as second input',
                            default=None)
        parser.add_argument('--sel',
                            help='Apply branch flag [%default]', 
                            default=None)
        args=parser.parse_args(args.split())

        #event selection
        self.selection=args.sel
        
        #setting DeepJet specific defaults
        self.treename="data"
        self.undefTruth=[]
        
        self.referenceclass='flatten'
        self.truthclasses=['isGood'] 
        
        # register ALL branches you intend to use in the numpy recArray style
        # no need to register those that use the DeepJetCore conversion funtions
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)

        #regression target
        self.regressiontarget=args.target

        regressionBranches=args.varList.split(',')
        otherBranches=[self.regressiontarget]
        if self.selection: otherBranches.append(self.selection)
        self.registerBranches(regressionBranches+otherBranches)
        self.addBranches(regressionBranches)

        self.inputs0=range(len(regressionBranches))
        self.inputs1=None
        if args.isolateVars: 
            self.inputs1 = [ regressionBranches.index(x) for x in args.isolateVars.split(',') ]
            self.inputs0 = [ i for i in range(len(regressionBranches)) if not i in self.inputs1 ]
            print 'Will isolate the following branches in inputs:'
            print self.inputs0
            print self.inputs1

        #make these distributions uniform in the training (disabled for the moment)
        self.weightbranchX='nvert'
        self.weightbranchY='tkmet_sphericity'
        self.weight_binX = numpy.array([0,100],dtype=float)
        self.weight_binY = numpy.array([0,10000],dtype=float)

        #define the regression classes
        self.regressiontargetclasses=args.regress.split(',')

        #add a function of the regressed values to compute the mean and width on the fly
        #setattr(self, 'predictionFunctor',        getParametrizedRecoilMomenta)
        #setattr(self, 'predictionFunctorClasses', ["p_mean","p_width"])

        #print some information
        print '-'*50
        print 'Configured TrainData_Recoil'
        print 'Target classes to regress are',self.regressiontargetclasses
        print 'Truth will be set to',self.regressiontarget
        print self.selection,'events will be removed'
        print 'Variables used in the training are',regressionBranches
        print '-'*50
             
        #Call this and the end!
        self.reduceTruth(None)
        
    # this funtion defines the conversion
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        #for reacArray operations
        Tuple = self.readTreeFromRootToTuple(filename)

        #create weights and remove indices
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            if self.selection:
                print 'Removing events selected with',self.selection
                notremoves -= Tuple[self.selection].view(numpy.ndarray)
            weights=notremoves
            #print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            #print('creating weights')
            weights= weighter.getJetWeights(Tuple)
        else:
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)

        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        #that would be for labels
        alltruth=self.reduceTruth(truthtuple)
        
        reg_truth=Tuple[self.regressiontarget].view(numpy.ndarray)        
        
        #stuff all in one long vector
        x_all=MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        #print(alltruth.shape)
        if self.remove:
            #print('remove')
            weights=weights[notremoves > 0]
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            reg_truth=reg_truth[notremoves > 0]
            print len(Tuple),'->',len(x_all),'after remove'

        newnsamp=x_all.shape[0]
        #print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w=[weights]
        if self.inputs1:
            self.x=[ x_all[:,self.inputs0], x_all[:,self.inputs1] ]
        else:
            self.x=[x_all]
        self.y=[reg_truth]    
        
    
