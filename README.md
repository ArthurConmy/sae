Coding up an SAE for a 1L model.

<h2> Next steps: </h2>

<b> The goal is to get L0 closer to 10-20 rather than ~200 :-( !!! </b>

Currently we have a good sweep of stuff, remember that Neel got low 100s L0 so maybe 100 is fine? We also have Anthropic A/1 replication on the way, too.
Best: 300 L0, 80% loss recovered?! Seems unlikely to be great :-(
Let's actually have a look at our + Neel's features for comparison...

Other ideas:

* bfloat16 buffer for bigger buffer.

* Untie the bias in and bias out?

* Do this all on MLP out not hidden???

... then we sure should implement some good vizualization utils
