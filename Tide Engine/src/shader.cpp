#include "shader.h"
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <sstream>

// Read a text file, trying candidate path prefixes so it works regardless of cwd.
static bool readText(const char* relPath, std::string& out, std::string& usedPath) {
    static const char* kPrefixes[] = { "../", "", "../../" };
    for (const char* pre : kPrefixes) {
        std::string p = std::string(pre) + relPath;
        std::ifstream f(p, std::ios::binary);
        if (f) {
            std::stringstream ss;
            ss << f.rdbuf();
            out = ss.str();
            usedPath = p;
            return true;
        }
    }
    return false;
}

static shaderc_shader_kind toKind(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:   return shaderc_vertex_shader;
        case VK_SHADER_STAGE_FRAGMENT_BIT: return shaderc_fragment_shader;
        case VK_SHADER_STAGE_COMPUTE_BIT:  return shaderc_compute_shader;
        default:                           return shaderc_glsl_infer_from_source;
    }
}

VkShaderModule loadShaderModule(VkDevice device, const char* relPath,
                                VkShaderStageFlagBits stage) {
    std::string source, usedPath;
    if (!readText(relPath, source, usedPath)) {
        TE_ERROR("Shader file not found: %s\n", relPath);
        std::abort();
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
#ifdef GPU_DEBUG
    options.SetGenerateDebugInfo();
#else
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

    shaderc::SpvCompilationResult result =
        compiler.CompileGlslToSpv(source, toKind(stage), relPath, options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        TE_ERROR("Shader compile failed (%s):\n%s\n", relPath,
                 result.GetErrorMessage().c_str());
        std::abort();
    }

    std::vector<uint32_t> spirv(result.cbegin(), result.cend());

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module));
    TE_INFO("Compiled shader: %s\n", usedPath.c_str());
    return module;
}
