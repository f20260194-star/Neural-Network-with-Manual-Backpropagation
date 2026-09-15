Write-up: Building a Neural Network from Scratch

I had no ML background before this. Here's what I actually understood, explained simply.

What the network does

Each input is a small image, 8x8 pixels, so 64 numbers. The network takes those 64 numbers and outputs 10 numbers, one per possible digit (0-9). Whichever output number is highest is the guess.

The network has two layers. Each layer just multiplies the input by a grid of numbers (weights), adds a few more numbers (bias), and passes the result on. Between the two layers there's a small filtering step.

Two layers are needed because two multiply-and-add steps stacked directly on top of each other are mathematically identical to one step. Nothing is gained by stacking them unless something non-linear sits in between. That's what the filtering step is for.

Weights and bias

Weights start as small random numbers, not zeros. If every weight starts at zero, every neuron in a layer computes the same output as its neighbors and stays identical to them forever, since they'd all be updated the same way too. Random numbers break that symmetry so each neuron can become different over training.

The random numbers also need to be scaled down a bit, or the output grows much larger than the input as it passes through the layer. I tested this directly: unscaled weights gave an output spread about 7-8 times larger than the input, scaled weights kept it roughly the same size as the input.

Bias starts at zero, which is fine, since it doesn't have the symmetry problem weights have. Bias size also only depends on how many numbers come out of a layer, not how many go in, since it's added after the input side has already been reduced to the output.

The filtering step (ReLU)

This just turns negative numbers into zero and leaves everything else unchanged. Chosen mainly because it's the simplest to reverse later during training, since the gradient either passes straight through or gets blocked completely.

From output numbers to a loss

The 10 output numbers are converted to probabilities that add up to 1. I tested what happens with a large input value here and got garbage output (nan) from overflow. The standard fix is subtracting the largest number in the group before doing anything else, which avoids the overflow without changing the result.

Once I have probabilities, I calculate the loss by looking at the probability given to the correct answer, and turning that into a single number that's small when the model is confidently right and large when it's confidently wrong. Averaging this across a batch gives the overall loss, which is what gets minimized during training.

Going backward

The goal of the backward pass is to figure out, for every weight and bias in the network, how a small change to that number would affect the loss. That value is the gradient. Once every gradient is known, each weight and bias is nudged slightly in the direction that reduces the loss.

This is computed by starting at the loss and working backward through the network one step at a time, reusing the gradient from the step before it.

The last step (loss and probabilities combined) has a known simplified gradient formula, which I used directly rather than re-deriving the underlying calculus. I confirmed its shape matched what it should be a gradient of, which was enough to trust it.

For each multiply-and-add layer, going backward requires three things: how the weights should change, how the bias should change, and what gradient to pass to the layer before it. The weight gradient depends on the input that was originally fed into the layer, which is why the input was saved during the forward pass. The bias gradient is just the incoming gradient summed across the batch. The gradient passed backward follows the same kind of calculation, aimed in the other direction.

One rule was useful throughout: every gradient should be the same shape as the thing it's a gradient of. Checking this caught almost every mistake before running anything, since a wrong calculation usually produces an obviously incorrect shape.

Going backward through the filtering step is simple: wherever the original value was positive, the gradient passes through unchanged; wherever it was negative, the gradient is zero. I deliberately checked this using the original pre-filtered values rather than the filtered output, even though both give the same result here, since that felt like the more correct approach to rely on going forward.

The first layer's backward pass follows the exact same logic as the second layer, just applied to different numbers.
