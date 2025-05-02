# CTC_Timing_Custom_Loss_SVIPTR
This is code that I have written that replicates the SVIPTR text recognition paper along with custom pytorch CTC Losses
We trained both the regular CTC Loss and the Timing CTC Loss just 65,000 images consisting of SynthText, SVT, ICAR, and RRC training datasets along with negative images without any text for 30 epochs with warmup and a little permutations on the images.
This model only has 4.7 million parameters.
The results while weren't state of the art, showed the true power of the Timing CTC loss where after having the exact training configuartions, the Timing CTC Loss significaly outpreformed the regular CTC Loss.
The accuracy on the testing sets of SVT and ICAR 2003 where it got a 1 if it got the word exactly correct and 0 if it got it wrong.

For ICAR 2003 here where the results for accuarcy:
Timing CTC Loss: 67%
Regular CTC Loss: 46%

For SVT here where the results for accuarcy:
Timing CTC Loss: 60%
Regular CTC Loss: 32%

Again, while these aren't close to state of the art, given the limited model size and training data, this Timing CTC Loss shows promise of helping training converge faster and better.
