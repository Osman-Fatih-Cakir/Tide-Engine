#include "shader.h"
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <sstream>
#include <memory>

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

// Resolves `#include "foo.glsl"` against the shaders dir (same prefix logic).
class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    shaderc_include_result* GetInclude(const char* requested, shaderc_include_type,
                                       const char* /*requesting*/, size_t) override {
        auto* data = new Payload();
        std::string used;
        if (!readText((std::string("shaders/") + requested).c_str(), data->content, used)) {
            data->content = std::string("#error include not found: ") + requested;
            used = requested;
        }
        data->name = used;
        auto* res = new shaderc_include_result{};
        res->source_name = data->name.c_str();
        res->source_name_length = data->name.size();
        res->content = data->content.c_str();
        res->content_length = data->content.size();
        res->user_data = data;
        return res;
    }
    void ReleaseInclude(shaderc_include_result* res) override {
        delete static_cast<Payload*>(res->user_data);
        delete res;
    }
private:
    struct Payload { std::string name, content; };
};

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
    options.SetIncluder(std::make_unique<ShaderIncluder>());
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
