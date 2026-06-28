#pragma once
#include "pch.h"

// Free-fly camera. Hold right mouse button to look; WASD to move, Q/E down/up.
class Camera {
public:
    void update(GLFWwindow* window, float dt);
    void setLookAt(glm::vec3 eye, glm::vec3 target);

    glm::mat4 view() const;
    glm::mat4 proj(float aspect) const; // Vulkan clip space (y-flipped, 0..1 depth)

    // Forward (look) direction for an arbitrary yaw/pitch (used by the path player).
    static glm::vec3 dirFromYawPitch(float yawDeg, float pitchDeg);

    glm::vec3 position = glm::vec3(0.0f, 1.5f, 4.0f);
    float yaw   = -90.0f; // degrees; -90 looks down -Z
    float pitch = 0.0f;

    float fovDeg = 60.0f;
    float nearZ  = 0.05f;
    float farZ   = 500.0f;

    float speed = 3.0f;        // units/sec
    float sensitivity = 0.1f;  // deg/pixel

private:
    glm::vec3 forward() const;
    bool   m_looking = false;
    bool   m_firstMouse = true;
    double m_lastX = 0.0, m_lastY = 0.0;
};
