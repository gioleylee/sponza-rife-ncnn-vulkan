// Owns Assimp model import and procedural scene geometry.
#include "VulkanRifeRendererApp.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/types.h>
#include <assimp/vector3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"

namespace {

template <typename T>
const T* accessorElement(const tinygltf::Model& model,
                         const tinygltf::Accessor& accessor,
                         size_t elementIndex) {
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    const size_t stride = static_cast<size_t>(accessor.ByteStride(view));
    const size_t offset = view.byteOffset + accessor.byteOffset + stride * elementIndex;
    return reinterpret_cast<const T*>(buffer.data.data() + offset);
}

glm::mat4 makeTransform(const glm::vec3& translation,
                        const glm::quat& rotation,
                        const glm::vec3& scale) {
    glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation);
    transform *= glm::mat4_cast(rotation);
    transform = glm::scale(transform, scale);
    return transform;
}

glm::vec4 sampleLinearVec4(const AnimationSampler& sampler, float time) {
    if (sampler.times.empty() || sampler.values.empty()) {
        return glm::vec4(0.0f);
    }
    if (time <= sampler.times.front()) {
        return sampler.values.front();
    }
    if (time >= sampler.times.back()) {
        return sampler.values.back();
    }

    auto upper = std::upper_bound(sampler.times.begin(), sampler.times.end(), time);
    size_t next = static_cast<size_t>(upper - sampler.times.begin());
    size_t prev = next - 1;
    float t0 = sampler.times[prev];
    float t1 = sampler.times[next];
    float alpha = (time - t0) / (t1 - t0);
    return glm::mix(sampler.values[prev], sampler.values[next], alpha);
}

glm::quat sampleRotation(const AnimationSampler& sampler, float time) {
    if (sampler.times.empty() || sampler.values.empty()) {
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    if (time <= sampler.times.front()) {
        const glm::vec4& v = sampler.values.front();
        return glm::normalize(glm::quat(v.w, v.x, v.y, v.z));
    }
    if (time >= sampler.times.back()) {
        const glm::vec4& v = sampler.values.back();
        return glm::normalize(glm::quat(v.w, v.x, v.y, v.z));
    }

    auto upper = std::upper_bound(sampler.times.begin(), sampler.times.end(), time);
    size_t next = static_cast<size_t>(upper - sampler.times.begin());
    size_t prev = next - 1;
    float t0 = sampler.times[prev];
    float t1 = sampler.times[next];
    float alpha = (time - t0) / (t1 - t0);
    const glm::vec4& a = sampler.values[prev];
    const glm::vec4& b = sampler.values[next];
    return glm::normalize(glm::slerp(glm::quat(a.w, a.x, a.y, a.z),
                                     glm::quat(b.w, b.x, b.y, b.z),
                                     alpha));
}

void computeGlobalTransforms(SkinnedModel& model, int nodeIndex, const glm::mat4& parentTransform) {
    model.globalTransforms[nodeIndex] = parentTransform * model.localTransforms[nodeIndex];
    for (int child : model.nodes[nodeIndex].children) {
        computeGlobalTransforms(model, child, model.globalTransforms[nodeIndex]);
    }
}

}

void VulkanRifeRendererApp::loadModel(const std::string& path) {
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
        path,
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_PreTransformVertices |
        aiProcess_ImproveCacheLocality
    );

    if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        throw std::runtime_error(std::string("Failed to load model: ") + importer.GetErrorString());
    }

    modelVertices.clear();
    modelIndices.clear();
    materials.clear();
    submeshes.clear();

    materials.resize(scene->mNumMaterials);
    std::string baseDir;
    {
        size_t slash = path.find_last_of("/\\");
        baseDir = (slash == std::string::npos) ? "" : path.substr(0, slash + 1);
    }

    for (unsigned int m = 0; m < scene->mNumMaterials; ++m) {
        aiMaterial* mat = scene->mMaterials[m];
        Material material{};

        aiString texPath;
        if (mat->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) == AI_SUCCESS) {
            material.diffusePath = baseDir + texPath.C_Str();
        }
        else {
            material.diffusePath.clear();
        }

        materials[m] = material;
    }

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh* mesh = scene->mMeshes[m];

        Submesh sm{};
        sm.indexOffset = static_cast<uint32_t>(modelIndices.size());
        sm.materialIndex = mesh->mMaterialIndex;

        size_t baseVertex = modelVertices.size();

        for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
            Vertex vertex{};
            const aiVector3D& pos = mesh->mVertices[i];

            vertex.pos = glm::vec3(pos.x, pos.y, pos.z);

            if (mesh->mTextureCoords[0]) {
                const aiVector3D& uv = mesh->mTextureCoords[0][i];
                float u = uv.x;
                float v = 1.0f - uv.y;
                vertex.texCoord = glm::vec2(u, v);
            }
            else {
                vertex.texCoord = glm::vec2(0.0f, 0.0f);
            }

            if (mesh->mNormals) {
                const aiVector3D& n = mesh->mNormals[i];
                vertex.normal = glm::normalize(glm::vec3(n.x, n.y, n.z));
            }
            else {
                vertex.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            }

            modelVertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue;

            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[0]));
            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[1]));
            modelIndices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[2]));
        }

        sm.indexCount = static_cast<uint32_t>(modelIndices.size()) - sm.indexOffset;
        if (sm.indexCount > 0) {
            submeshes.push_back(sm);
        }
    }

    if (modelVertices.empty() || modelIndices.empty() || submeshes.empty()) {
        throw std::runtime_error("Loaded model has no geometry.");
    }

}

void VulkanRifeRendererApp::loadCesiumMan(const std::string& path) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model gltfModel;
    std::string error;
    std::string warning;

    if (!loader.LoadBinaryFromFile(&gltfModel, &error, &warning, path)) {
        throw std::runtime_error("Failed to load CesiumMan.glb: " + error);
    }
    if (gltfModel.meshes.empty() || gltfModel.skins.empty() || gltfModel.animations.empty()) {
        throw std::runtime_error("CesiumMan.glb is missing mesh, skin, or animation data.");
    }

    cesiumMan = SkinnedModel{};

    const tinygltf::Primitive& primitive = gltfModel.meshes[0].primitives[0];
    auto requireAttribute = [&](const char* name) -> const tinygltf::Accessor& {
        auto it = primitive.attributes.find(name);
        if (it == primitive.attributes.end()) {
            throw std::runtime_error(std::string("CesiumMan.glb missing attribute ") + name);
        }
        return gltfModel.accessors[it->second];
    };

    const tinygltf::Accessor& positionAccessor = requireAttribute("POSITION");
    const tinygltf::Accessor& normalAccessor = requireAttribute("NORMAL");
    const tinygltf::Accessor& texCoordAccessor = requireAttribute("TEXCOORD_0");
    const tinygltf::Accessor& jointsAccessor = requireAttribute("JOINTS_0");
    const tinygltf::Accessor& weightsAccessor = requireAttribute("WEIGHTS_0");

    if (primitive.indices < 0) {
        throw std::runtime_error("CesiumMan.glb primitive has no index accessor.");
    }
    const tinygltf::Accessor& indexAccessor = gltfModel.accessors[primitive.indices];
    if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        throw std::runtime_error("CesiumMan.glb index accessor is not uint16.");
    }

    cesiumMan.vertices.resize(positionAccessor.count);
    for (size_t i = 0; i < cesiumMan.vertices.size(); ++i) {
        SkinnedVertex vertex{};
        const float* pos = accessorElement<float>(gltfModel, positionAccessor, i);
        const float* normal = accessorElement<float>(gltfModel, normalAccessor, i);
        const float* uv = accessorElement<float>(gltfModel, texCoordAccessor, i);
        const float* weights = accessorElement<float>(gltfModel, weightsAccessor, i);

        vertex.pos = glm::vec3(pos[0], pos[1], pos[2]);
        vertex.normal = glm::normalize(glm::vec3(normal[0], normal[1], normal[2]));
        vertex.texCoord = glm::vec2(uv[0], uv[1]);
        vertex.weights = glm::vec4(weights[0], weights[1], weights[2], weights[3]);

        if (jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const uint16_t* joints = accessorElement<uint16_t>(gltfModel, jointsAccessor, i);
            vertex.joints = glm::uvec4(joints[0], joints[1], joints[2], joints[3]);
        }
        else if (jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
            const uint8_t* joints = accessorElement<uint8_t>(gltfModel, jointsAccessor, i);
            vertex.joints = glm::uvec4(joints[0], joints[1], joints[2], joints[3]);
        }
        else {
            throw std::runtime_error("CesiumMan.glb JOINTS_0 uses an unsupported component type.");
        }

        cesiumMan.vertices[i] = vertex;
    }

    cesiumMan.indices.resize(indexAccessor.count);
    for (size_t i = 0; i < cesiumMan.indices.size(); ++i) {
        cesiumMan.indices[i] = *accessorElement<uint16_t>(gltfModel, indexAccessor, i);
    }

    cesiumMan.nodes.resize(gltfModel.nodes.size());
    for (size_t nodeIndex = 0; nodeIndex < gltfModel.nodes.size(); ++nodeIndex) {
        const tinygltf::Node& gltfNode = gltfModel.nodes[nodeIndex];
        SkinnedNode& node = cesiumMan.nodes[nodeIndex];
        node.children = gltfNode.children;
        for (int child : node.children) {
            cesiumMan.nodes[child].parent = static_cast<int>(nodeIndex);
        }

        if (gltfNode.translation.size() == 3) {
            node.baseTranslation = glm::vec3(
                static_cast<float>(gltfNode.translation[0]),
                static_cast<float>(gltfNode.translation[1]),
                static_cast<float>(gltfNode.translation[2]));
        }
        if (gltfNode.rotation.size() == 4) {
            node.baseRotation = glm::normalize(glm::quat(
                static_cast<float>(gltfNode.rotation[3]),
                static_cast<float>(gltfNode.rotation[0]),
                static_cast<float>(gltfNode.rotation[1]),
                static_cast<float>(gltfNode.rotation[2])));
        }
        if (gltfNode.scale.size() == 3) {
            node.baseScale = glm::vec3(
                static_cast<float>(gltfNode.scale[0]),
                static_cast<float>(gltfNode.scale[1]),
                static_cast<float>(gltfNode.scale[2]));
        }
        if (gltfNode.matrix.size() == 16) {
            float matrixValues[16]{};
            for (size_t i = 0; i < 16; ++i) {
                matrixValues[i] = static_cast<float>(gltfNode.matrix[i]);
            }
            node.baseMatrix = glm::make_mat4(matrixValues);
            node.hasMatrix = true;
        }
    }

    const tinygltf::Skin& skin = gltfModel.skins[0];
    cesiumMan.jointNodeIndices = skin.joints;
    cesiumMan.inverseBindMatrices.resize(cesiumMan.jointNodeIndices.size(), glm::mat4(1.0f));
    if (skin.inverseBindMatrices >= 0) {
        const tinygltf::Accessor& ibmAccessor = gltfModel.accessors[skin.inverseBindMatrices];
        for (size_t i = 0; i < cesiumMan.inverseBindMatrices.size(); ++i) {
            cesiumMan.inverseBindMatrices[i] = glm::make_mat4(accessorElement<float>(gltfModel, ibmAccessor, i));
        }
    }

    const tinygltf::Animation& animation = gltfModel.animations[0];
    cesiumMan.animationSamplers.resize(animation.samplers.size());
    for (size_t samplerIndex = 0; samplerIndex < animation.samplers.size(); ++samplerIndex) {
        const tinygltf::AnimationSampler& gltfSampler = animation.samplers[samplerIndex];
        AnimationSampler& sampler = cesiumMan.animationSamplers[samplerIndex];

        const tinygltf::Accessor& inputAccessor = gltfModel.accessors[gltfSampler.input];
        sampler.times.resize(inputAccessor.count);
        for (size_t i = 0; i < sampler.times.size(); ++i) {
            sampler.times[i] = *accessorElement<float>(gltfModel, inputAccessor, i);
            cesiumMan.animationDuration = std::max(cesiumMan.animationDuration, sampler.times[i]);
        }

        const tinygltf::Accessor& outputAccessor = gltfModel.accessors[gltfSampler.output];
        sampler.values.resize(outputAccessor.count);
        for (size_t i = 0; i < sampler.values.size(); ++i) {
            const float* value = accessorElement<float>(gltfModel, outputAccessor, i);
            sampler.values[i] = outputAccessor.type == TINYGLTF_TYPE_VEC4
                ? glm::vec4(value[0], value[1], value[2], value[3])
                : glm::vec4(value[0], value[1], value[2], 0.0f);
        }
    }

    for (const tinygltf::AnimationChannel& gltfChannel : animation.channels) {
        AnimationChannel channel{};
        channel.targetNode = gltfChannel.target_node;
        channel.samplerIndex = gltfChannel.sampler;
        if (gltfChannel.target_path == "translation") {
            channel.path = AnimationChannel::Path::Translation;
        }
        else if (gltfChannel.target_path == "rotation") {
            channel.path = AnimationChannel::Path::Rotation;
        }
        else if (gltfChannel.target_path == "scale") {
            channel.path = AnimationChannel::Path::Scale;
        }
        else {
            continue;
        }
        cesiumMan.animationChannels.push_back(channel);
    }

    cesiumMan.localTransforms.resize(cesiumMan.nodes.size(), glm::mat4(1.0f));
    cesiumMan.globalTransforms.resize(cesiumMan.nodes.size(), glm::mat4(1.0f));

    if (!gltfModel.images.empty() && !gltfModel.images[0].image.empty()) {
        const tinygltf::Image& image = gltfModel.images[0];
        std::vector<uint8_t> rgbaPixels(static_cast<size_t>(image.width) * image.height * 4);
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                size_t src = (static_cast<size_t>(y) * image.width + x) * image.component;
                size_t dst = (static_cast<size_t>(y) * image.width + x) * 4;
                rgbaPixels[dst + 0] = image.image[src + 0];
                rgbaPixels[dst + 1] = image.component > 1 ? image.image[src + 1] : image.image[src + 0];
                rgbaPixels[dst + 2] = image.component > 2 ? image.image[src + 2] : image.image[src + 0];
                rgbaPixels[dst + 3] = image.component > 3 ? image.image[src + 3] : 255;
            }
        }

        uint32_t mipLevels = static_cast<uint32_t>(
            std::floor(std::log2(std::max(image.width, image.height)))) + 1;
        VkDeviceSize imageSize = rgbaPixels.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        void* data = nullptr;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
        memcpy(data, rgbaPixels.data(), static_cast<size_t>(imageSize));
        vkUnmapMemory(device, stagingBufferMemory);

        createImage(static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height),
            mipLevels,
            VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            cesiumMan.material.image,
            cesiumMan.material.imageMemory);

        transitionImageLayout(cesiumMan.material.image, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
        copyBufferToImage(stagingBuffer, cesiumMan.material.image,
            static_cast<uint32_t>(image.width),
            static_cast<uint32_t>(image.height));
        generateMipmaps(cesiumMan.material.image, VK_FORMAT_R8G8B8A8_SRGB,
            image.width, image.height, mipLevels);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

        cesiumMan.material.imageView = createImageView(cesiumMan.material.image, VK_FORMAT_R8G8B8A8_SRGB, mipLevels);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = deviceFeatures.samplerAnisotropy ? VK_TRUE : VK_FALSE;
        samplerInfo.maxAnisotropy = deviceFeatures.samplerAnisotropy ? 16.0f : 1.0f;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = static_cast<float>(mipLevels);

        if (vkCreateSampler(device, &samplerInfo, nullptr, &cesiumMan.material.sampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create CesiumMan sampler!");
        }
    }

    materials.push_back(cesiumMan.material);
}

void VulkanRifeRendererApp::updateSkinAnimation(float deltaTime, uint32_t currentImage) {
    SkinUBO skinUbo{};
    for (uint32_t i = 0; i < MAX_SKIN_JOINTS; ++i) {
        skinUbo.jointMatrices[i] = glm::mat4(1.0f);
    }

    if (cesiumMan.nodes.empty() || cesiumMan.jointNodeIndices.empty()) {
        memcpy(skinUniformBuffersMapped[currentImage], &skinUbo, sizeof(skinUbo));
        return;
    }

    cesiumMan.animationTime += deltaTime;
    if (cesiumMan.animationDuration > 0.0f) {
        cesiumMan.animationTime = std::fmod(cesiumMan.animationTime, cesiumMan.animationDuration);
    }

    std::vector<glm::vec3> translations(cesiumMan.nodes.size());
    std::vector<glm::quat> rotations(cesiumMan.nodes.size());
    std::vector<glm::vec3> scales(cesiumMan.nodes.size());
    for (size_t i = 0; i < cesiumMan.nodes.size(); ++i) {
        translations[i] = cesiumMan.nodes[i].baseTranslation;
        rotations[i] = cesiumMan.nodes[i].baseRotation;
        scales[i] = cesiumMan.nodes[i].baseScale;
    }

    // Skeletal animation update:
    // 1. Sample each glTF channel at the current animation time.
    // 2. Rebuild animated local node transforms from translation, rotation, and scale.
    // 3. Walk the node hierarchy to accumulate global joint transforms.
    // 4. Convert each skin joint to mesh space with globalTransform * inverseBindMatrix.
    for (const AnimationChannel& channel : cesiumMan.animationChannels) {
        if (channel.targetNode < 0 ||
            static_cast<size_t>(channel.targetNode) >= cesiumMan.nodes.size() ||
            channel.samplerIndex < 0 ||
            static_cast<size_t>(channel.samplerIndex) >= cesiumMan.animationSamplers.size()) {
            continue;
        }

        const AnimationSampler& sampler = cesiumMan.animationSamplers[channel.samplerIndex];
        switch (channel.path) {
        case AnimationChannel::Path::Translation: {
            translations[channel.targetNode] = glm::vec3(sampleLinearVec4(sampler, cesiumMan.animationTime));
            break;
        }
        case AnimationChannel::Path::Rotation:
            rotations[channel.targetNode] = sampleRotation(sampler, cesiumMan.animationTime);
            break;
        case AnimationChannel::Path::Scale:
            scales[channel.targetNode] = glm::vec3(sampleLinearVec4(sampler, cesiumMan.animationTime));
            break;
        }
    }

    for (size_t i = 0; i < cesiumMan.nodes.size(); ++i) {
        cesiumMan.localTransforms[i] = cesiumMan.nodes[i].hasMatrix
            ? cesiumMan.nodes[i].baseMatrix
            : makeTransform(translations[i], rotations[i], scales[i]);
    }

    for (size_t i = 0; i < cesiumMan.nodes.size(); ++i) {
        if (cesiumMan.nodes[i].parent < 0) {
            computeGlobalTransforms(cesiumMan, static_cast<int>(i), glm::mat4(1.0f));
        }
    }

    const size_t jointCount = std::min<size_t>(cesiumMan.jointNodeIndices.size(), MAX_SKIN_JOINTS);
    for (size_t jointIndex = 0; jointIndex < jointCount; ++jointIndex) {
        int nodeIndex = cesiumMan.jointNodeIndices[jointIndex];
        if (nodeIndex >= 0 && static_cast<size_t>(nodeIndex) < cesiumMan.globalTransforms.size()) {
            skinUbo.jointMatrices[jointIndex] =
                cesiumMan.globalTransforms[nodeIndex] * cesiumMan.inverseBindMatrices[jointIndex];
        }
    }

    memcpy(skinUniformBuffersMapped[currentImage], &skinUbo, sizeof(skinUbo));
}

void VulkanRifeRendererApp::appendRotatingCubeGeometry() {
    rotatingCubeIndexOffset = static_cast<uint32_t>(modelIndices.size());

    const float h = 0.5f;
    const float uvTile = 4.0f;
    const glm::vec2 uv00(0.0f, 0.0f);
    const glm::vec2 uv10(uvTile, 0.0f);
    const glm::vec2 uv11(uvTile, uvTile);
    const glm::vec2 uv01(0.0f, uvTile);

    auto addFace = [&](const glm::vec3& normal,
                       const glm::vec3& a,
                       const glm::vec3& b,
                       const glm::vec3& c,
                       const glm::vec3& d) {
        uint32_t base = static_cast<uint32_t>(modelVertices.size());
        modelVertices.push_back({ a, uv00, normal });
        modelVertices.push_back({ b, uv10, normal });
        modelVertices.push_back({ c, uv11, normal });
        modelVertices.push_back({ d, uv01, normal });
        modelIndices.push_back(base + 0);
        modelIndices.push_back(base + 1);
        modelIndices.push_back(base + 2);
        modelIndices.push_back(base + 2);
        modelIndices.push_back(base + 3);
        modelIndices.push_back(base + 0);
    };

    addFace(glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(-h, -h, h), glm::vec3(h, -h, h), glm::vec3(h, h, h), glm::vec3(-h, h, h));
    addFace(glm::vec3(0.0f, 0.0f, -1.0f),
        glm::vec3(h, -h, -h), glm::vec3(-h, -h, -h), glm::vec3(-h, h, -h), glm::vec3(h, h, -h));
    addFace(glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(h, -h, h), glm::vec3(h, -h, -h), glm::vec3(h, h, -h), glm::vec3(h, h, h));
    addFace(glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(-h, -h, -h), glm::vec3(-h, -h, h), glm::vec3(-h, h, h), glm::vec3(-h, h, -h));
    addFace(glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(-h, h, h), glm::vec3(h, h, h), glm::vec3(h, h, -h), glm::vec3(-h, h, -h));
    addFace(glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(-h, -h, -h), glm::vec3(h, -h, -h), glm::vec3(h, -h, h), glm::vec3(-h, -h, h));

    rotatingCubeIndexCount = static_cast<uint32_t>(modelIndices.size()) - rotatingCubeIndexOffset;
}
