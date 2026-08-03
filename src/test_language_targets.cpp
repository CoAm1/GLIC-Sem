#include "language_targets.h"

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: test_language_targets ARTIFACT_ROOT FRAME_STEM\n";
        return 2;
    }
    try
    {
        const auto basis = language::loadPcaBasis(argv[1]);
        const auto frame = language::loadPcaFrameTarget(
            argv[1], argv[2], basis.dimension);
        std::cout << "dimension=" << basis.dimension
                  << " teacher_dim=" << basis.basis.size(0)
                  << " regions=" << frame.basis_dot.size(0)
                  << " image=" << frame.region_ids.size(1) << "x"
                  << frame.region_ids.size(0)
                  << " max_region=" << frame.region_ids.max().item<std::int64_t>()
                  << " confidence_min=" << frame.confidence.min().item<float>()
                  << " confidence_max=" << frame.confidence.max().item<float>()
                  << std::endl;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
