#include <cstdlib>
#include <exception>
#include <iostream>

#include "VulkanRifeRendererApp.h"

int main() {
    try {
        VulkanRifeRendererApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
