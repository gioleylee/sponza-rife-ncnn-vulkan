// Owns Assimp model import and procedural scene geometry.
#include "VulkanRifeRendererApp.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/types.h>
#include <assimp/vector3.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

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
