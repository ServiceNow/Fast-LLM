/*
  Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of NVIDIA CORPORATION nor the names of its
     contributors may be used to endorse or promote products derived
     from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
  AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL
  THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
 Helper methods for fast index mapping builds.
 Changes for Fast-LLM: Use int16 for dataset index, add verbose argument to build_sample_idx.
*/

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;


py::array build_sample_idx(const py::array_t<int32_t>& sizes_,
			   const py::array_t<int32_t>& doc_idx_,
			   const int32_t seq_length,
			   const int32_t num_epochs,
			   const int64_t tokens_per_epoch,
			   const bool verbose) {
    /* Sample index (sample_idx) is used for gpt2 like dataset for which
       the documents are flattened and the samples are built based on this
       1-D flatten array. It is a 2D array with sizes [number-of-samples + 1, 2]
       where [..., 0] contains the index into `doc_idx` and [..., 1] is the
       starting offset in that document.*/

    // Consistency checks.
    assert(seq_length > 1);
    assert(num_epochs > 0);
    assert(tokens_per_epoch > 1);

    // Remove bound checks.
    auto sizes = sizes_.unchecked<1>();
    auto doc_idx = doc_idx_.unchecked<1>();

    // Mapping and it's length (1D).
    int64_t num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length;
    int32_t* sample_idx = new int32_t[2*(num_samples+1)];

    if (verbose) {
      cout << "    using:" << endl << std::flush;
      cout << "     number of documents:       " <<
        doc_idx_.shape(0) / num_epochs << endl << std::flush;
      cout << "     number of epochs:          " << num_epochs <<
        endl << std::flush;
      cout << "     sequence length:           " << seq_length <<
        endl << std::flush;
      cout << "     total number of samples:   " << num_samples <<
        endl << std::flush;
    }

    // Index into sample_idx.
    int64_t sample_index = 0;
    // Index into doc_idx.
    int64_t doc_idx_index = 0;
    // Beginning offset for each document.
    int32_t doc_offset = 0;
    // Start with first document and no offset.
    sample_idx[2 * sample_index] = doc_idx_index;
    sample_idx[2 * sample_index + 1] = doc_offset;
    ++sample_index;

    while (sample_index <= num_samples) {
        // Start with a fresh sequence.
      int32_t remaining_seq_length = seq_length + 1;
      while (remaining_seq_length != 0) {
            // Get the document length.
	auto doc_id = doc_idx[doc_idx_index];
	auto doc_length = sizes[doc_id] - doc_offset;
	// And add it to the current sequence.
	remaining_seq_length -= doc_length;
	// If we have more than a full sequence, adjust offset and set
	// remaining length to zero so we return from the while loop.
	// Note that -1 here is for the same reason we have -1 in
	// `_num_epochs` calculations.
	if (remaining_seq_length <= 0) {
	  doc_offset += (remaining_seq_length + doc_length - 1);
	  remaining_seq_length = 0;
	} else {
	  // Otherwise, start from the beginning of the next document.
	  ++doc_idx_index;
	  doc_offset = 0;
	}
      }
      // Record the sequence.
      sample_idx[2 * sample_index] = doc_idx_index;
      sample_idx[2 * sample_index + 1] = doc_offset;
      ++sample_index;
    }

    // Method to deallocate memory.
    py::capsule free_when_done(sample_idx, [](void *mem_) {
	int32_t *mem = reinterpret_cast<int32_t*>(mem_);
	delete[] mem;
      });

    // Return the numpy array.
    const auto byte_size = sizeof(int32_t);
    return py::array(std::vector<int64_t>{num_samples+1, 2}, // shape
                     {2*byte_size, byte_size}, // C-style contiguous strides
                     sample_idx, // the data pointer
                     free_when_done); // numpy array references

}

PYBIND11_MODULE(data, m) {
    m.def("build_sample_idx", &build_sample_idx);
}
