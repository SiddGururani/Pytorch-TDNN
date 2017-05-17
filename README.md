# Pytorch-TDNN

This repository contains an implementation of a Time-Delay Neural Network which is commonly used in Speech Recognition and Neural Language Models. 

**Description**

The model here implements a generalized version of the TDNN based on the model descriptions given in [\[1\]](http://www.cs.toronto.edu/~fritz/absps/waibelTDNN.pdf),[\[2\]](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/4/4c/A_time_delay_neural_network_architecture_for_efficient_modeling_of_long_temporal_contexts.pdf).

In the description given by Waibel et al. a TDNN uses the full context specified. For example: if the delay specified is `N = 2`, the model uses the current frame and frames at delays of 1 and 2.

In the description given by Peddinti et al. the TDNN only looks at the farthest frames from the current frame as specified by the context parameter. The description in section 3.1 of the paper discusses the differences between their implementation and Waibel et al. 

The TDNN implemented here allows for the usage of an arbitrary context which shall be demonstrated in the usage code snippet.

**Usage**

```python
# For the model specified in the Waibel et al. paper, the first layer is as follows:
context = [0,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=True)

# For the model specified in the Peddinti et al. paper, the second layer is as follows (taking arbitrary I/O dimensions since it's not specified):
context = [-1,2]
input_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)

# You may also use any arbitrary context like this:
context = [-11,0,5,7,10]
nput_dim = 16
output_dim = 8
net = TDNN(context, input_dim, output_dim, full_context=False)

# The above will convole the kernel with the current frame, 11 frames in the past, 5, 7, and 10 frames in the future.

output = net(input) # this will run a forward pass
```
Note that no zero-padding is done at any steps. If window-centering, etc are important for your task, make sure the input is correctly zero-padded and the context are correctly specified.

**Acknowledgement**

The code spun-off from this repository:
[pytorch_TDNN](https://github.com/analvikingur/pytorch_TDNN), which implements the TDNN  or Character-Level CNN whose output is fed into the Char-LSTM Neural Language Model[\[3\]](https://arxiv.org/pdf/1508.06615.pdf).

**References**

 1. Waibel, A., Hanazawa, T., Hinton, G., Shikano, K., & Lang, K. J. (1989). Phoneme recognition using time-delay neural networks. IEEE transactions on acoustics, speech, and signal processing, 37(3), 328-339.
 2. Peddinti, V., Povey, D., & Khudanpur, S. (2015). A time delay neural network architecture for efficient modeling of long temporal contexts. In INTERSPEECH (pp. 3214-3218).
 3. Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2016). Character-aware neural language models. In Thirtieth AAAI Conference on Artificial Intelligence (pp. 2741-2749)..
