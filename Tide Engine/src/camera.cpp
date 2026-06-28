#include "camera.h"

glm::vec3 Camera::dirFromYawPitch(float yawDeg, float pitchDeg) {
    float cy = std::cos(glm::radians(yawDeg));
    float sy = std::sin(glm::radians(yawDeg));
    float cp = std::cos(glm::radians(pitchDeg));
    float sp = std::sin(glm::radians(pitchDeg));
    return glm::normalize(glm::vec3(cy * cp, sp, sy * cp));
}

glm::vec3 Camera::forward() const {
    return dirFromYawPitch(yaw, pitch);
}

void Camera::setLookAt(glm::vec3 eye, glm::vec3 target) {
    position = eye;
    glm::vec3 dir = glm::normalize(target - eye);
    pitch = glm::degrees(std::asin(glm::clamp(dir.y, -1.0f, 1.0f)));
    yaw   = glm::degrees(std::atan2(dir.z, dir.x));
}

void Camera::update(GLFWwindow* window, float dt) {
    // Mouse look while right button held.
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (rmb) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        if (!m_looking) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            m_lastX = x; m_lastY = y; m_looking = true;
        }
        float dx = (float)(x - m_lastX);
        float dy = (float)(y - m_lastY);
        m_lastX = x; m_lastY = y;
        yaw   += dx * sensitivity;
        pitch -= dy * sensitivity;
        pitch = glm::clamp(pitch, -89.0f, 89.0f);
    } else if (m_looking) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        m_looking = false;
    }

    glm::vec3 fwd = forward();
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0, 1, 0)));
    glm::vec3 up = glm::vec3(0, 1, 0);

    float v = speed * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) v *= 4.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position += fwd * v;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position -= fwd * v;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position += right * v;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position -= right * v;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) position += up * v;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) position -= up * v;
}

glm::mat4 Camera::view() const {
    return glm::lookAt(position, position + forward(), glm::vec3(0, 1, 0));
}

glm::mat4 Camera::proj(float aspect) const {
    glm::mat4 p = glm::perspective(glm::radians(fovDeg), aspect, nearZ, farZ);
    p[1][1] *= -1.0f; // flip Y for Vulkan
    return p;
}
