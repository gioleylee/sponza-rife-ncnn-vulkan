// Owns vertex/index/uniform buffer resources.
#include "VulkanNcnnRenderer.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstring>

void VulkanNcnnRenderer::createVertexBuffer() {
    VkDeviceSize bufferSize = sizeof(modelVertices[0]) * modelVertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, modelVertices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, 
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(vertexBuffer), "Sponza Vertex Buffer");
    
    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void VulkanNcnnRenderer::createIndexBuffer() {
    VkDeviceSize bufferSize = sizeof(modelIndices[0]) * modelIndices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, modelIndices.data(), (size_t)bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        indexBuffer, indexBufferMemory);
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(indexBuffer), "Sponza Index Buffer");

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void VulkanNcnnRenderer::createSkinnedVertexBuffer() {
    if (cesiumMan.vertices.empty()) {
        return;
    }

    VkDeviceSize bufferSize = sizeof(cesiumMan.vertices[0]) * cesiumMan.vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, cesiumMan.vertices.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, skinnedVertexBuffer, skinnedVertexBufferMemory);
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(skinnedVertexBuffer), "CesiumMan Skinned Vertex Buffer");

    copyBuffer(stagingBuffer, skinnedVertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void VulkanNcnnRenderer::createSkinnedIndexBuffer() {
    if (cesiumMan.indices.empty()) {
        return;
    }

    VkDeviceSize bufferSize = sizeof(cesiumMan.indices[0]) * cesiumMan.indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, cesiumMan.indices.data(), static_cast<size_t>(bufferSize));
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        skinnedIndexBuffer, skinnedIndexBufferMemory);
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(skinnedIndexBuffer), "CesiumMan Skinned Index Buffer");

    copyBuffer(stagingBuffer, skinnedIndexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void VulkanNcnnRenderer::createUniformBuffers() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    VkDeviceSize skinBufferSize = sizeof(SkinUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    cubeUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    cubeUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    cubeUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    cesiumUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    cesiumUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    cesiumUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    skinUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    skinUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    skinUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(uniformBuffers[i]),
            "Scene Uniform Buffer Frame " + std::to_string(i));

        vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);

        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, cubeUniformBuffers[i], cubeUniformBuffersMemory[i]);
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(cubeUniformBuffers[i]),
            "Cube Uniform Buffer Frame " + std::to_string(i));

        vkMapMemory(device, cubeUniformBuffersMemory[i], 0, bufferSize, 0, &cubeUniformBuffersMapped[i]);

        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, cesiumUniformBuffers[i], cesiumUniformBuffersMemory[i]);
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(cesiumUniformBuffers[i]),
            "CesiumMan Uniform Buffer Frame " + std::to_string(i));

        vkMapMemory(device, cesiumUniformBuffersMemory[i], 0, bufferSize, 0, &cesiumUniformBuffersMapped[i]);

        createBuffer(skinBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, skinUniformBuffers[i], skinUniformBuffersMemory[i]);
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(skinUniformBuffers[i]),
            "CesiumMan Skin Uniform Buffer Frame " + std::to_string(i));

        vkMapMemory(device, skinUniformBuffersMemory[i], 0, skinBufferSize, 0, &skinUniformBuffersMapped[i]);
    }
}

