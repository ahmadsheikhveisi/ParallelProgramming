/*
 * HW1.h
 *
 *  Created on: Oct 13, 2020
 *      Author: ahmadsv
 */

#ifndef HW1_H_
#define HW1_H_

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename);

size_t numRows();
size_t numCols();

void postProcess(const std::string& output_file, unsigned char* data_ptr);

void cleanup();

void generateReferenceImage(std::string input_filename, std::string output_filename);

#endif /* HW1_H_ */
