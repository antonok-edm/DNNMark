// The MIT License (MIT)
// 
// Copyright (c) 2016 Northeastern University
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef CORE_INCLUDE_DATA_BINARY_H_
#define CORE_INCLUDE_DATA_BINARY_H_

#include <vector>
#include <map>
#include <fstream>
#include <iostream>

namespace dnnmark {

class BinaryLoader {
 private:
  // Constructor
  BinaryLoader() {}

  // BinaryLoader instance
  static std::unique_ptr<BinaryLoader> instance_;
 public:

  ~BinaryLoader() {}

  static BinaryLoader *GetInstance() {
    if (instance_.get())
      return instance_.get();
    instance_.reset(new BinaryLoader());
    return instance_.get();
  }
  void GenerateDataFromBinary(float *dev_ptr, int size, std::string path, DataDim output_dim_) {
    char buffer[output_dim_.n_*(output_dim_.c_*output_dim_.h_*output_dim_.w_ + 1)];	//Assumes a 1-byte label for each image
    char data[output_dim_.n_*output_dim_.c_*output_dim_.h_*output_dim_.w_];
    std::ifstream file(path, std::ios::in|std::ios::binary);
    file.read(buffer, sizeof(buffer));
    int offset = 0;
    for(int i = 0; i < sizeof(buffer)/sizeof(char); i++) {
      if(i%(output_dim_.c_*output_dim_.h_*output_dim_.w_ + 1) == 0) offset += 1;
      else data[i-offset] = buffer[i];
    }
    CUDA_CALL(cudaMemcpy(dev_ptr, data, sizeof(data)/sizeof(char), cudaMemcpyHostToDevice));
  }
};

std::unique_ptr<BinaryLoader> BinaryLoader::instance_ = nullptr;

} // namespace dnnmark

#endif // CORE_INCLUDE_DATA_PNG_H_

