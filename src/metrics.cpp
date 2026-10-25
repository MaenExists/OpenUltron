#include "spark/metrics.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <unistd.h>
#include <sys/resource.h>

namespace spark {

MemoryUsage get_current_memory_usage() {
    MemoryUsage mem;

    // Peak resident via getrusage
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in kilobytes on Linux
        mem.peak_resident_bytes = static_cast<size_t>(usage.ru_maxrss) * 1024ULL;
    }

    // Current virtual and resident via /proc/self/statm
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        unsigned long vmsize_pages = 0;
        unsigned long rss_pages = 0;
        if (statm >> vmsize_pages >> rss_pages) {
            long page_size = sysconf(_SC_PAGESIZE);
            if (page_size > 0) {
                mem.virtual_bytes = vmsize_pages * static_cast<size_t>(page_size);
                mem.resident_bytes = rss_pages * static_cast<size_t>(page_size);
            }
        }
    }

    if (mem.peak_resident_bytes == 0) {
        mem.peak_resident_bytes = mem.resident_bytes;
    }

    return mem;
}

ActivationStats compute_activation_stats(const std::vector<float>& activations) {
    ActivationStats stats;
    if (activations.empty()) return stats;

    stats.min_val = activations[0];
    stats.max_val = activations[0];
    double sum = 0.0;
    double sq_sum = 0.0;
    size_t sparse_count = 0;

    for (float val : activations) {
        if (!std::isfinite(val)) {
            stats.non_finite_count++;
            continue;
        }
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
        sum += val;
        sq_sum += static_cast<double>(val) * val;
        if (std::abs(val) < 0.05f) {
            sparse_count++;
        }
    }

    size_t finite_count = activations.size() - stats.non_finite_count;
    if (finite_count > 0) {
        stats.mean_val = static_cast<float>(sum / finite_count);
        double var = (sq_sum / finite_count) - (static_cast<double>(stats.mean_val) * stats.mean_val);
        stats.std_val = static_cast<float>(std::sqrt(std::max(0.0, var)));
        stats.sparsity = static_cast<float>(sparse_count) / static_cast<float>(finite_count);
    }

    return stats;
}

} // namespace spark
