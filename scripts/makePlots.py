import os
import sys
from DeepJetCore.evaluation import makePlots_async,makeROCs_async
import optparse

COLORS=['black','red','blue','green','black,dashed','red,dashed','blue,dashed','green,dashed']

def doScalePlots(f,tag,cut,outdir):


    vnam = ['peak',             'mean',                 'tkMET*1.8',    'MC truth']
    vexp = ['tkmet_pt*exp(mu)', 'tkmet_pt*exp(p_mean)', 'tkmet_pt*1.8', 'trueh']
    makePlots_async(f,                        #input file or file list
                    vnam,                     #legend names (needs to be list)
                    vexp,                     #variables to plot
                    [cut]*len(vnam),          #cuts to apply
                    COLORS[0:len(vnam)],      #line color and style (e.g. 'red,dashed')
                    outdir+'/hpt_%s.pdf'%tag, #outputfile
                    'Recoil p_{T} [GeV]',     #xaxisname
                    'Events (a.u.)',          #yaxisname
                    False,                    #normalise
                    nbins=50,
                    xmin=0,
                    xmax=100,
                    treename="data")

    makePlots_async(f,                                                         
                    ['peak'],
                    ['tkmet_pt*exp(mu)/trueh:trueh'],
                    [cut],
                    COLORS[0:1],
                    outdir+'/hpt_profile_htruth_%s.pdf'%tag,
                    'Truth recoil p_{T} [GeV]',
                    'h / h_{truth}',           
                    False,
                    profiles=True,
                    minimum=0.,
                    maximum=1.5,
                    widthprofile=False,
                    treename="data")
            
    makePlots_async(f,                                                         
                    ['peak'],
                    ['tkmet_pt*exp(mu)/lne1:lne1'],
                    [cut], 
                    COLORS[0:1],
                    outdir+'/hpt_profile_lne1_%s.pdf'%tag,
                    'Target log(e_{1})',
                    'h / h_{truth}',           
                    False,
                    profiles=True,
                    minimum=0.,
                    maximum=1.5,
                    widthprofile=False,
                    treename="data")


def main():
    """wrapper to be used from command line"""

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--in' ,          dest='input',       help='Input directory list [%default]',  default=None)    
    parser.add_option('-o', '--out' ,         dest='output',      help='output dir [%default]',       default='plots')
    (opt, args) = parser.parse_args()

    fList=[os.path.join(d,'predict/tree_association.txt') for d in opt.input.split(',')]
    outdir=opt.output
    os.system('mkdir -p %s'%outdir)
    
    cut='isW>0'
    tags=[]
    for f in fList:
        tag='_'.join(f.split('/')[-4:-2])
        tags.append(tag)
        doScalePlots(f,tag,cut,outdir)


#some ROCs
#truthclasses=['isW>0']
#cuts=[]
#titles=[]
#formats=[]
#for c,t,f in [ ('isW>0','inclusive','auto,black'),
#               #('tkmet_sphericity<0.05','sphericity<5%','auto,gray'),
#               #('(tkmet_sphericity>0.05 && tkmet_pt<0.20)','5%<sphericity<20%','auto,green'),
#               #('tkmet_sphericity>0.20','sphericity>20%','auto,blue') 
#           ]:
#    cuts.append('%s && %s'%(truthclasses[0],c))
#    titles.append(t)
#    formats.append(f)
#
#for var,name,xtit,ytit,ymin,ymax,doProfile in [
#        ('(tkmet_pt*exp(mu)/trueh):trueh',  'hprof',  'h(true) [GeV]', 'h / h(true) [GeV]',0,2,True),
#        ('(tkmet_pt/trueh):trueh',  'tkmetprof',  'h(true) [GeV]', 'h_{tk} / h(true) [GeV]',0,2,True),
#        ('mu-lne1:lne1',                    'lne1_prof',  'log[e_{1}(true)]=log(h_{true}/h_{tk})', 'log(e_{1})-log[e_{1}(true)]',-4,6,True),
#        #('(mu-lne1):nvert',   'nvert_prof', 'Vertex multiplicity',             'log(e_{1}^{pred}/e_{1}^{true})',-4,6,True),
#        #('(mu-lne1):tkmet_ht','ht_dist',    '#sum |p_{T}| [GeV]',              'log(e_{1}^{pred}/e_{1}^{true})',-4,6,True),
#        #('(mu-lne1):tkmet_pt','pt_dist',    'h_{tk} [GeV]',                    'log(e_{1}^{pred}/e_{1}^{true})',-4,6,True)
#        ]:
#    makePlots_async(infile, #input file or file list
#                    titles,
#                    var, #variable to plot --> yaxis:xaxis
#                    cuts, #list of cuts to apply
#                    formats,
#                    outdir+'/%s.pdf'%name,
#                    xtit, #xaxisname
#                    ytit, #yaxisname
#                    False, #normalize
#                    doProfile, #make a profile plot
#                    ymin,
#                    ymax,
#                    treename="data") #override max value of y-axis range
#

if __name__ == "__main__":
    sys.exit(main())
