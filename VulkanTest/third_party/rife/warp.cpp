// rife implemented with ncnn library

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "rife_ops.h"
#include <type_traits>
#include <utility>

#if __has_include("warp.comp.hex.h") && __has_include("warp_pack4.comp.hex.h") && __has_include("warp_pack8.comp.hex.h")
#define HAS_WARP_VK_SHADER 1
#include "warp.comp.hex.h"
#include "warp_pack4.comp.hex.h"
#include "warp_pack8.comp.hex.h"
#else
#define HAS_WARP_VK_SHADER 0
#endif

using ncnn::Mat;
using ncnn::Option;
using ncnn::Pipeline;
using ncnn::VkCompute;
using ncnn::VkMat;
using ncnn::compile_spirv_module;
using ncnn::vk_constant_type;
using ncnn::vk_specialization_type;

template<typename T>
static auto isShaderPack8EnabledImpl(const T& opt, int) -> decltype((void)opt.use_shader_pack8, bool())
{
    return opt.use_shader_pack8;
}

template<typename T>
static bool isShaderPack8EnabledImpl(const T&, long)
{
    return false;
}

static bool isShaderPack8Enabled(const Option& opt)
{
    return isShaderPack8EnabledImpl(opt, 0);
}

Warp::Warp()
{
    support_vulkan = HAS_WARP_VK_SHADER;

    pipeline_warp = 0;
    pipeline_warp_pack4 = 0;
    pipeline_warp_pack8 = 0;
}

int Warp::create_pipeline(const Option& opt)
{
#if !HAS_WARP_VK_SHADER
    (void)opt;
    return 0;
#else
    if (!vkdev)
        return 0;

    std::vector<vk_specialization_type> specializations(0 + 0);

    // pack1
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_comp_data, sizeof(warp_comp_data), opt, spirv);
            }
        }

        pipeline_warp = new Pipeline(vkdev);
        pipeline_warp->set_optimal_local_size_xyz();
        pipeline_warp->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack4
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_pack4_comp_data, sizeof(warp_pack4_comp_data), opt, spirv);
            }
        }

        pipeline_warp_pack4 = new Pipeline(vkdev);
        pipeline_warp_pack4->set_optimal_local_size_xyz();
        pipeline_warp_pack4->create(spirv.data(), spirv.size() * 4, specializations);
    }

    // pack8
    if (isShaderPack8Enabled(opt))
    {
        static std::vector<uint32_t> spirv;
        static ncnn::Mutex lock;
        {
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                compile_spirv_module(warp_pack8_comp_data, sizeof(warp_pack8_comp_data), opt, spirv);
            }
        }

        pipeline_warp_pack8 = new Pipeline(vkdev);
        pipeline_warp_pack8->set_optimal_local_size_xyz();
        pipeline_warp_pack8->create(spirv.data(), spirv.size() * 4, specializations);
    }

    return 0;
#endif
}

int Warp::destroy_pipeline(const Option& opt)
{
    delete pipeline_warp;
    pipeline_warp = 0;

    delete pipeline_warp_pack4;
    pipeline_warp_pack4 = 0;

    delete pipeline_warp_pack8;
    pipeline_warp_pack8 = 0;

    return 0;
}

int Warp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& image_blob = bottom_blobs[0];
    const Mat& flow_blob = bottom_blobs[1];

    int w = image_blob.w;
    int h = image_blob.h;
    int channels = image_blob.c;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* outptr = top_blob.channel(q);

        const Mat image = image_blob.channel(q);

        const float* fxptr = flow_blob.channel(0);
        const float* fyptr = flow_blob.channel(1);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float flow_x = fxptr[0];
                float flow_y = fyptr[0];

                float sample_x = x + flow_x;
                float sample_y = y + flow_y;

                // bilinear interpolate
                float v;
                {
                    int x0 = floor(sample_x);
                    int y0 = floor(sample_y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    x0 = std::min(std::max(x0, 0), w - 1);
                    y0 = std::min(std::max(y0, 0), h - 1);
                    x1 = std::min(std::max(x1, 0), w - 1);
                    y1 = std::min(std::max(y1, 0), h - 1);

                    float alpha = sample_x - x0;
                    float beta = sample_y - y0;

                    float v0 = image.row(y0)[x0];
                    float v1 = image.row(y0)[x1];
                    float v2 = image.row(y1)[x0];
                    float v3 = image.row(y1)[x1];

                    float v4 = v0 * (1 - alpha) + v1 * alpha;
                    float v5 = v2 * (1 - alpha) + v3 * alpha;

                    v = v4 * (1 - beta) + v5 * beta;
                }

                outptr[0] = v;

                outptr += 1;

                fxptr += 1;
                fyptr += 1;
            }
        }
    }

    return 0;
}

int Warp::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
#if !HAS_WARP_VK_SHADER
    (void)bottom_blobs;
    (void)top_blobs;
    (void)cmd;
    (void)opt;
    return -1;
#else
    const VkMat& image_blob = bottom_blobs[0];
    const VkMat& flow_blob = bottom_blobs[1];

    int w = image_blob.w;
    int h = image_blob.h;
    int channels = image_blob.c;
    size_t elemsize = image_blob.elemsize;
    int elempack = image_blob.elempack;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = image_blob;
    bindings[1] = flow_blob;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(4);
    constants[0].i = top_blob.w;
    constants[1].i = top_blob.h;
    constants[2].i = top_blob.c;
    constants[3].i = top_blob.cstep;

    if (elempack == 8 && pipeline_warp_pack8)
    {
        cmd.record_pipeline(pipeline_warp_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 4)
    {
        cmd.record_pipeline(pipeline_warp_pack4, bindings, constants, top_blob);
    }
    else // if (elempack == 1)
    {
        cmd.record_pipeline(pipeline_warp, bindings, constants, top_blob);
    }

    return 0;
#endif
}
