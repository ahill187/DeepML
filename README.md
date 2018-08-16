# DeepML

Machine learning based on DeepJetCore 

## Installation and environment
Follow the installation instructions on https://github.com/DL4Jets/DeepJetCore. Once all is compiled 
then install these scripts to run the regression studies

```
git clone ssh://git@gitlab.cern.ch:7999/psilva/DeepML.git
cd DeepML
source lxplus_env.sh
```

## Running

Use the runRecoilScaleRegression.sh found under scripts. An example below.
Running the script with -h will show all the available options.
```
sh scripts/runRecoilRegression.sh -r train -m 0 -i /eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-v2/Chunks -t WJets -c isW==0
```
If a model exists and you want to extend the predicition to other files than the ones used in the training
you can use the same script but running in predict mode
```
sh scripts/runRecoilRegression.sh -r predict -m 0 -i /eos/cms/store/cmst3/user/psilva/Wmass/RecoilTest_regress-v2/Chunks -p regress-results/train
```

## Diagnostics

Some scripts to do basic plotting are available.
The inputs can be a directory with results or a CSV list of directories with results.
To plot the loss, mse, and all the metrics which were registered use
```
python scripts/makeTrainValidationPlots.py regression_results
```
To make some basic validation plots
```
python scripts/makePlots.py -i regression_results -o plots
```
To make a combined comparison from the validation plots
(NB this one is not automatized and requires manual editing for the moment)
```
python scripts/comparePlottedResults.py
```
To make ideogram plots (using event-by-event PDF from the regression).
As it usually takes time -n tells the number of events to use (-1=all)
```
python scripts/makeIdeogramPlots.py -i regression_results -o ideograms  -n -1
```