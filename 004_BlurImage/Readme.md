## Gather approach

One method is that each thread block reads a block of pixels and copies it into __shared__ memory, then applies the filter. However, using this method some threads which are operating on the edge of the block can't produce an output since their neighboring pixels are not read. Therefore, the output can only be produced for a fraction of the pixels in a block. for example, imagine a block with a size of 32x32 and a filter with size of 9x9. thread 0,0 can't produce any output becuase a filter with the center on that pixels needs to know the value of the pixels above that block. the output can be generate for (32-8)x(32-8)=24x24=576 threads which is less than 60 percent of the threads.