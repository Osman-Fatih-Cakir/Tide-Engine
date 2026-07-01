#include "vk_engine.h"

int main() {
    // Loop so a UI scene switch can fully tear down + reinit the engine, loading
    // the new scene from its own state file. consumeRestart() is set on switch.
    do {
        VulkanEngine engine;
        engine.init();
        engine.run();
        engine.cleanup();
    } while (VulkanEngine::consumeRestart());
    return 0;
}
