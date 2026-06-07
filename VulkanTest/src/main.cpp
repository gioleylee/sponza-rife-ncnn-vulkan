#include <cstdlib>
#include <exception>
#include <iostream>

#include "VulkanNcnnRenderer.h"

int main() {
    try {
        VulkanNcnnRenderer app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
