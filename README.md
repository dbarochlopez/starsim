# starsim

## Installation
The source code for starsim can be downloaded from GitHub and installed by running:

```
git clone https://github.com/dbarochlopez/starsim.git
cd starsim
pip install .
```

Now starsim is installed, and can be imported with python from any directory with `import starsim`

## Before starting
In order to run, starsim requires Phoenix models (ref). To do so, download them from https://www.ice.csic.es/owncloud/s/7kRTAJqKHxeHB3w, and replace the models folder with the downloaded folder. If a wider grid of temperatures and/or surface gravities are needed, the high resolution model can be found in http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/ (store them in models/Phoenix) and the intensity models in http://phoenix.astro.physik.uni-goettingen.de/data/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/Z-0.0/ (store them in models/Phoenix_mu). 
IMPORTANT: if you download a new high-resolution model to models/Phoenix, the corresponding intensity model must be downloaded to models/Phoenix_mu, and vice versa.
