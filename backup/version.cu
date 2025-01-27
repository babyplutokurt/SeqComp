#include <thrust/version.h>
#include <cub/version.cuh>
#include <iostream>

int main() {
    // Thrust version
    std::cout << "Thrust version: "
              << THRUST_MAJOR_VERSION << "."
              << THRUST_MINOR_VERSION << "."
              << THRUST_SUBMINOR_VERSION << std::endl;

    // CUB version
    std::cout << "CUB version: "
              << CUB_MAJOR_VERSION << "."
              << CUB_MINOR_VERSION << "."
              << CUB_SUBMINOR_VERSION << std::endl;

    return 0;
}
