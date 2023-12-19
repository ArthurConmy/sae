A quick SAE implementation

# Maybe TODO for improvements: 

* Geometric median crap?
* Untie the bias in and bias out?
* Should we really be doing bias initialized to 0 in reinits? Seems like it makes a lot of non-zero fires, eek (Note: this didn't seem promising at work)
* Should we Adam reinitialize better? ccLeo
* Can't we make the neuron resampling procedure stochastic? i.e resample a neuron with some probability that's a function of how few times it fired
* Should we improve the lib and make `pip install git+` actually work from colab? Idk why it don't work
... then we sure should implement some good vizualization utils

# Help

`pip install -e .` is optimal for now. Maybe use `pip install -U torch==1.13.1` for crap hardware
