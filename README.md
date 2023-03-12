# Unbiased-Feature-Position-Alignment-for-Human-Pose-Estimation

Serving as a model-agnostic plug-in, DARK significantly improves the performance of a variety of state-of-the-art human pose estimation models!


## Dynamic visualization of feature postion misalignment
We illustrate the different feature position misalignments induced by two interpolation strategies. Their difference focus on the implementation of inpterplation during upsampling.

This gif dynamically visualize how the feature position misalignment generates when using corner-aligned interpolation.
![misalignment using corner-aligned interpolation](figures/mialignment_aligned_interpolation.gif)

This gif dynamically visualize how the feature position misalignment generates when using corner-unaligned interpolation.
![misalignment using corner-unaligned interpolation](figures/mialignment_unaligned_interpolation.gif)

## Dynamic visualization of unbiased feature postion alignment
This gif dynamically visualize how the proposed unbiased feature position alignment works to solve the misalignment problem when using corner-aligned interpolation.
![alignment using corner-aligned interpolation](figures/alignment_corner.gif)

This gif dynamically visualize how the proposed unbiased feature position alignment works to solve the misalignment problem when using corner-unaligned interpolation.
![alignment using corner-unaligned interpolation](figures/alignment_uncorner.gif)