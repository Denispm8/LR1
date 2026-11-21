#version 450

layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform ShadowSceneUniforms {
    mat4 shadow_view_proj;
} shadow_scene;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 emission;
    ivec4 maps;
} model_data;

void main() {
    gl_Position = shadow_scene.shadow_view_proj * model_data.model * vec4(in_position, 1.0);
}