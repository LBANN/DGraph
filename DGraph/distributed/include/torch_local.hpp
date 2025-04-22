#pragma once
#include <torch/extension.h>

torch::Tensor local_masked_gather(torch::Tensor input,
                                  torch::Tensor indices,
                                  torch::Tensor rank_local_placement,
                                  torch::Tensor output,
                                  const int num_batches,
                                  const int num_values_rows,
                                  const int num_cols,
                                  const int num_output_rows);