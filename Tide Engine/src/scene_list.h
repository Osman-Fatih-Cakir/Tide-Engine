#pragma once
// Single source of truth for the selectable scenes. Each scene owns its own
// glTF path and its own persisted-settings file, so switching scenes swaps both
// the geometry and the tuning. Shared by vk_engine.cpp (load/save) and ui.cpp
// (the scene picker) — add a row here to register a new scene.
#include <fstream>
#include <string>

struct SceneDef {
    const char* name;       // shown in the UI picker + stored in the active-scene pointer
    const char* gltf;       // scene glTF path (relative to the working dir / repo root)
    const char* stateFile;  // per-scene Settings + camera blob (see saveState/loadState)
};

inline constexpr SceneDef kScenes[] = {
    { "Cigar Room",     "../Resources/Cigar Room/Room_Windowed.gltf", "cigar_room_tide_state.bin" },
    { "Christmas Room", "../Resources/Christmas Room/Room.gltf",      "christmas_tide_state.bin"  },
};
inline constexpr int kSceneCount = (int)(sizeof(kScenes) / sizeof(kScenes[0]));

// Tiny pointer file naming the active scene; read at startup so a fresh engine
// process knows which scene + state file to open after a scene-switch restart.
inline constexpr const char* kActiveSceneFile = "tide_active_scene.txt";

inline int loadActiveScene() {
    std::ifstream f(kActiveSceneFile);
    std::string name;
    if (f && std::getline(f, name))
        for (int i = 0; i < kSceneCount; ++i)
            if (name == kScenes[i].name) return i;
    return 0; // default to the first scene
}

inline void saveActiveScene(int idx) {
    if (idx < 0 || idx >= kSceneCount) return;
    std::ofstream f(kActiveSceneFile);
    if (f) f << kScenes[idx].name;
}
