import os
import sys
from datastructures.ParametricRecoilFunction import *

import optparse

def rangeTransform(x,a,b):
    """transform to a limited range as Minuit does (cf. https://root.cern.ch/download/minuit.pdf Sec. 1.2.1)"""
    return a+0.5*(b-a)*(ROOT.TMath.Sin(x)+1.0)


def makeIdeogramPlots(data,maxEvts,nbins=100,xmin=0,xmax=100):

    """Uses the regressed PDF to re-destribute the contribution of a single event"""

    dogPDF=hasattr(data,'sigma') or hasattr(data,'qp')

    #prepare histograms
    ideograms={}
    dx=float((xmax-xmin)/nbins)
    xcen=[xmin+(xbin+0.5)*dx for xbin in xrange(0,nbins)]
    xof = xcen[-1]+0.5*dx
    def fillIdeogram(key,val,truth,wgt=1.0):
        name=key if isinstance(key, str) else '_'.join( [str(s) for s in key] )
        if not key in ideograms:
            ideograms[key]=ROOT.TH1F('h'+name,'%s;Recoil [GeV];Events'%name,nbins,xmin,xmax)
            ideograms[key].Sumw2()
            ideograms[name+'corr']=ROOT.TH2F('hcorr'+name,'%s;True recoil [GeV]; Recoil [GeV];Events'%name,nbins,xmin,xmax,nbins,xmin,xmax)
            ideograms[name+'corr'].Sumw2()
        ideograms[key].Fill(val,wgt)
        ideograms[name+'corr'].Fill(truth,val,wgt)
    def fillIdeogramResol(key,mean,sigma,truth):
        if sigma==0: return
        name=key if isinstance(key, str) else '_'.join( [str(s) for s in key] )
        name+='chi2'
        if not name in ideograms:
            ideograms[name]=ROOT.TH1F('h'+name,'%s;Recoil [GeV];Event #chi^{2}'%name,nbins,xmin,xmax)
            ideograms[name].Sumw2()
        ideograms[name].Fill(truth,((mean-truth)/sigma)**2)


    #loop over events
    nentries=data.GetEntries() 
    if maxEvts>0: nentries=min(nentries,maxEvts)
    for i in xrange(0,nentries):
        data.GetEntry(i)
        dogdPDF=hasattr(data,'a1')

        truth=data.trueh
        fillIdeogram('true',truth,truth)
        fillIdeogram('peak',data.tkmet_pt*ROOT.TMath.Exp(data.mu),truth)

        #PDF
        bins=[ ROOT.TMath.Log(x/data.tkmet_pt) for x in xcen ]

        #gaussian core
        mu    = data.mu
        sigma = None
        a1    = 1e4
        a2    = 1e4
        if dogPDF:
            try:
                sigma=data.sigma 
            except:
                sigma=0.5*(data.qp-data.qm)
                
        if dogdPDF:
            mu    = rangeTransform( mu,      -3,   3)
            sigma = rangeTransform( sigma,   1e-3, 5)
            a1    = rangeTransform( data.a1, 1e-3, 5)
            a2    = rangeTransform( data.a2, 1e-3, 5)

        #gaussian PDF
        g_pdf,g_pdf_sum=None,None
        if dogPDF:
            g_prf=ParametricRecoilFunction(mu,sigma,1e9,1e9)
            g_pdf=g_prf.getEventPDF(bins)
            g_pdf=[ g_pdf[xbin]/xcen[xbin] for xbin in xrange(0,nbins) ]
            g_pdf_sum=sum(g_pdf)

        #gaussian double expo 
        gd_pdf,gd_pdf_sum=None,None
        if dogdPDF:
            gd_prf=ParametricRecoilFunction(mu,sigma,a1,a2)
            gd_pdf=gd_prf.getEventPDF(bins)
            gd_pdf=[ gd_pdf[xbin]/xcen[xbin] for xbin in xrange(0,nbins) ]
            gd_pdf_sum=sum(gd_pdf)

        #fill the histograms
        for xbin in xrange(0,len(bins)): 
            if g_pdf and g_pdf_sum:
                fillIdeogram('g_pdf', xcen[xbin], truth, g_pdf[xbin])
            if gd_pdf and gd_pdf_sum: 
                fillIdeogram('gd_pdf', xcen[xbin], truth, gd_pdf[xbin])

        #overflow
        if g_pdf and g_pdf_sum:
            fillIdeogram('g_pdf', xof, truth, 1-g_pdf_sum)
        if gd_pdf and gd_pdf_sum:
            fillIdeogram('gd_pdf', xof, truth, 1-gd_pdf_sum)

        #resolutions
        if g_pdf and g_pdf_sum:
            fillIdeogramResol('g_pdf',mu,sigma,truth)
        if gd_pdf and gd_pdf_sum:
            fillIdeogramResol('gd_pdf',gd_prf.mean,gd_prf.width,truth)

        if g_pdf and g_pdf_sum:
            hrand_g=data.tkmet_pt*ROOT.TMath.Exp(np.random.normal(mu,sigma))
            fillIdeogram('g_pdf_rand', hrand_g, truth)
        if gd_pdf and gd_pdf_sum:
            fillIdeogram('gd_mean',data.tkmet_pt*ROOT.TMath.Exp(gd_prf.mean),truth)
            try:
                hrand_gd=data.tkmet_pt*ROOT.TMath.Exp(gd_prf.getRandom())
                fillIdeogram('gd_pdf_rand', hrand_gd, truth)
            except Exception as e:
                print e,'will not be filled for this histogram'

    #divide by events in each bin
    #for key in ideograms:
    #    if not 'chi2' in key: continue
    #    ideograms[key].Divide(ideograms['true'])

    return ideograms


def main():
    """wrapper to be used from command line"""

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in' ,          dest='input',       help='Input directory [%default]',            default=None)    
    parser.add_option('-n', '--maxEvts' ,     dest='maxEvts',     help='Max. events to process [%default]',     default=-1, type=int)    
    parser.add_option('-o', '--out' ,         dest='output',      help='output dir [%default]',                 default='plots')
    (opt, args) = parser.parse_args()

    #start the chain of events to plot
    fList=os.path.join(opt.input,'predict/tree_association.txt')
    data=ROOT.TChain('data')
    tree=ROOT.TChain('tree')
    with open(fList,'r') as f:
        for x in f.read().split('\n'):
            try:
                dF,tF=x.split()
                data.AddFile(dF)
                tree.AddFile(tF)
            except:
                pass
    data.AddFriend(tree)

    #make the plots
    ideograms=makeIdeogramPlots(data,opt.maxEvts)

    #save to output
    outdir=os.path.dirname(opt.output)
    if len(outdir)>0: os.system('mkdir -p %s'%outdir)
    fOut=ROOT.TFile.Open('%s.root'%opt.output,'RECREATE')
    for key in ideograms: 
        ideograms[key].Write()
    fOut.Close()
    

if __name__ == "__main__":
    sys.exit(main())
