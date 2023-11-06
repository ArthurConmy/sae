Coding up an SAE for a 1L model.

<h2> Next steps: </h2>
<b> The goal is to get L0 closer to 10-20 rather than ~200 :-( !!! </b>

Currently we have a good sweep of stuff, remember that Neel got low 100s L0 so maybe 100 is fine? We also have Anthropic A/1 replication on the way, too.

Best: some fantastic loss recovered with 50-100 neurons firing; 0.0005 learning rate seems best: https://i.imgur.com/YCmoDhd.png

# TODO: 

Implement Learning Rate pruner after resampling (so it then returns properly...)

Look into bug where a lot looks to be resampled, despite firing frequencies telling a different story.

Other ideas:

* bfloat16 buffer for bigger buffer.

* Untie the bias in and bias out?

* Do this all on MLP out not hidden???

* Should we really be doing bias initialized to 0 in reinits? Seems like it makes a lot of non-zero fires, eek

* Can't we make the neuron resampling procedure stochastic? i.e resample a neuron with some probablity that's a function of how few times it fired

... then we sure should implement some good vizualization utils

# Informal results 

```
sae.load_from_my_wandb(
    run_id = "794ulknc",
    # index_from_back_override=3, # 1 did not work
    index_from_front_override=2,
)
```

Amongst first 3402 neurons:
Top (97) was not interpretable.
Second (1485) was medium interpretable (Asian, generally Indian names)
Third (1930) was interpretable (.)
Fourth (1669) was not ineterpretable
Fifth (1745) was interpretable ("With", but also "Through" as in "With" and other similar words)

For Neel's V1 model:
Top (2395) was not interpretable
Second (2970) was not interpretable
Third (517) was not interpretable
Fourth (2479) was not interpretable
Fifth (581) was interpretable (commas)

We sampled neuron 0-19 inclusive. 5.5/8 were interpretable. 12 didn't fire on more than 3 of our 500*128 tokens so were ignored. In the same process for Neel's 5/9 were interpretable.