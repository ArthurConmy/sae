Coding up an SAE for a 1L model.

<h2> Next steps: </h2>
<b> The goal is to get L0 closer to 10-20 rather than ~200 :-( !!! </b>

Currently we have a good sweep of stuff, remember that Neel got low 100s L0 so maybe 100 is fine? We also have Anthropic A/1 replication on the way, too.
Best: 137 L0 and 90% loss recovered; https://wandb.ai/arthurconmy/sae/runs/6i5td0kj

TODO: 

Implement Learning Rate pruner after resampling


Look into bug where a lot looks to be resampled, despite firing frequencies telling a different story.



Other ideas:

* bfloat16 buffer for bigger buffer.

* Untie the bias in and bias out?

* Do this all on MLP out not hidden???

* Should we really be doing bias initialized to 0 in reinits? Seems like it makes a lot of non-zero fires, eek

... then we sure should implement some good vizualization utils
