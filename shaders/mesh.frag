#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(N, L), 0.0) * 0.8 + 0.2; // simple lambert + ambient
    outColor = vec4(vec3(diff), 1.0);
}
