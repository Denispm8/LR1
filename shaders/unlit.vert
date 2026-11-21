#version 450
layout(location = 0) in vec3 in_position;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_proj;
    mat4 shadow_view_proj;
    vec4 camera_position;
    vec4 directional_light_direction;
    vec4 directional_light_color;
    vec4 ambient_light;
    uint point_light_count;
    uint spot_light_count;
    int shadow_source_type;
    float _pad;
} scene;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 emission;
    ivec4 maps;
} model_data;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = model_data.albedo;
    gl_Position = scene.view_proj * model_data.model * vec4(in_position, 1.0);
}