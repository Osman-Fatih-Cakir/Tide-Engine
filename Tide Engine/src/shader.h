#pragma once
#include "pch.h"

// Compile a GLSL source file to a VkShaderModule at runtime via shaderc.
// `relPath` is resolved against a few candidate prefixes (Bin / solution dir).
VkShaderModule loadShaderModule(VkDevice device, const char* relPath,
                                VkShaderStageFlagBits stage);
