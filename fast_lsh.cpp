#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" {
    void compute_lsh_c(const double* features, const double* random_vectors, 
                       int num_nodes, int feature_dim, int m, int k, int* output) {
        int total_hashes = m * k;
        for (int i = 0; i < num_nodes; ++i) {
            for (int hash_idx = 0; hash_idx < m; ++hash_idx) {
                int bucket_id = 0;
                for (int bit = 0; bit < k; ++bit) {
                    int vector_idx = hash_idx * k + bit;
                    double dot_product = 0.0;
                    for (int d = 0; d < feature_dim; ++d) {
                        dot_product += features[i * feature_dim + d] * random_vectors[d * total_hashes + vector_idx];
                    }
                    if (dot_product > 0) {
                        bucket_id |= (1 << (k - 1 - bit));
                    }
                }
                int offset = hash_idx * (1 << k);
                output[i * m + hash_idx] = bucket_id + offset;
            }
        }
    }
}