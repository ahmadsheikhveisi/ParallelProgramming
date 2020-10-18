/*
 * HW2.h
 *
 *  Created on: Oct 18, 2020
 *      Author: ahmadsv
 */

#ifndef HW2_H_
#define HW2_H_

#include <vector_types.h>

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename);

size_t numRows();
size_t numCols();

void postProcess(const std::string& output_file, uchar4* data_ptr);

void cleanUp(void);

void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size);

#endif /* HW2_H_ */
