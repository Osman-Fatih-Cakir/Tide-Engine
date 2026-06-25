#version 460

// Solid color for the fog box wireframe (bright cyan so it reads over the scene).
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.1, 0.9, 1.0, 1.0);
}
