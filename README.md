Coding up an SAE for a 1L model.

<h2> Next steps: </h2>

<b> The goal is to get L0 closer to 10-20 rather than ~200 :-( !!! </b>

Do the thing and try and fix current issues!!!

- Most importantly, scale the \lambda to try and deal with this
- ... but also experiment dropping the learning rate a little, ie two-three LRs for each \lambda

Other ideas:

Try to reduce the resample from 1e-5 --- that seems too high!

bfloat16 buffer for bigger buffer.

Buffer only replace when it's super small.

Untie the bias in and bias out?

... then we sure should implement some good vizualization utils
