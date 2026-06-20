#pragma once
#include "mesh.h"

// Load a glTF file into CPU-side MeshData. Returns false on failure.
// Path is relative to the working directory (repo root).
bool loadGltf(const char* path, MeshData& out);
