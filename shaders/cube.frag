#version 450

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_shadow_pos;

layout(location = 0) out vec4 out_frag_color;

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

struct PointLight {
    vec4 position;
    vec4 color;
};

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(set = 0, binding = 3) uniform sampler2DShadow shadow_map;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 emission;
    ivec4 maps;
} model_data;

layout(set = 1, binding = 1) uniform sampler2D albedo_map;
layout(set = 1, binding = 2) uniform sampler2D specular_map;
layout(set = 1, binding = 3) uniform sampler2D emissive_map;

layout(push_constant) uniform TextureEffects {
    float time;
    float strength;
    float color_modulation;
    int effect;
} effects;

vec3 safeNormalize(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 1e-12));
}

vec3 sampleAlbedo(out vec2 uv) {
    uv = in_uv;
    float strength = clamp(effects.strength, 0.0, 1.0);
    if (effects.effect != 1 || strength == 0.0) return texture(albedo_map, uv).rgb;

    float time = effects.time;
    uv += strength * 0.035 * vec2(sin(uv.y * 18.0 + time * 1.3), sin(uv.x * 15.0 - time));
    vec2 offset = vec2(strength * (0.008 + 0.004 * sin(time)), 0.0);
    vec3 color = vec3(texture(albedo_map, uv + offset).r,
                      texture(albedo_map, uv).g,
                      texture(albedo_map, uv - offset).b);

    vec3 modulation = 0.8 + 0.2 * sin(time * 0.8 + vec3(0.0, 2.094395, 4.188790));
    return color * mix(vec3(1.0), modulation, strength * clamp(effects.color_modulation, 0.0, 1.0));
}

vec3 calcBlinnPhong(vec3 light_dir, vec3 light_color, vec3 normal, vec3 view_dir,
                   vec3 diff_color, vec3 spec_color, float shininess) {
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 halfway_dir = safeNormalize(light_dir + view_dir);
    float spec = diff > 0.0 ? pow(max(dot(normal, halfway_dir), 0.0), shininess) : 0.0;
    return (diff * diff_color + spec * spec_color) * light_color;
}

float calculateShadow(vec4 shadow_pos) {
    vec3 proj_coords = shadow_pos.xyz / shadow_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;

    if (proj_coords.z > 1.0 || proj_coords.z < 0.0 ||
        proj_coords.x < 0.0 || proj_coords.x > 1.0 ||
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 1.0;
    }

    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            shadow += texture(shadow_map, vec3(proj_coords.xy + vec2(x, y) * texel_size, proj_coords.z));
        }
    }
    return shadow / 9.0;
}

void main() {
    vec3 norm = safeNormalize(in_normal);
    vec3 view_dir = safeNormalize(scene.camera_position.xyz - in_world_pos);
    vec2 uv;
    vec3 base_albedo = model_data.albedo.rgb * sampleAlbedo(uv);
    vec3 spec_color = model_data.specular.rgb;
    if (model_data.maps.x != 0) spec_color *= texture(specular_map, uv).rgb;
    float shininess = max(model_data.specular.w, 1.0);

    float shadow = calculateShadow(in_shadow_pos);

    vec3 total_light = scene.ambient_light.rgb * base_albedo;

    vec3 dir_to_light = safeNormalize(-scene.directional_light_direction.xyz);
    vec3 sun_light = calcBlinnPhong(dir_to_light, scene.directional_light_color.rgb,
                                   norm, view_dir, base_albedo, spec_color, shininess);

    if (scene.shadow_source_type == 0) {
        total_light += sun_light * shadow; 
    } else {
        total_light += sun_light;
    }

    for (uint i = 0; i < scene.point_light_count; ++i) {
        vec3 light_vec = point_lights[i].position.xyz - in_world_pos;
        float dist = length(light_vec);
        if (dist > 0.0001) {
            vec3 light_dir = light_vec / dist;
            float attenuation = point_lights[i].position.w / (dist * dist + 0.01);
            vec3 light_intensity = point_lights[i].color.rgb * attenuation;
            total_light += calcBlinnPhong(light_dir, light_intensity, norm, view_dir,
                                         base_albedo, spec_color, shininess);
        }
    }

    for (uint i = 0; i < scene.spot_light_count; ++i) {
        vec3 light_vec = spot_lights[i].position.xyz - in_world_pos;
        float dist = length(light_vec);
        if (dist > 0.0001) {
            vec3 light_dir = light_vec / dist;
            vec3 spot_dir = safeNormalize(spot_lights[i].direction.xyz);
            float theta = dot(-light_dir, spot_dir);
            float inner_cos = spot_lights[i].position.w;
            float outer_cos = spot_lights[i].direction.w;
            float spot_factor = clamp((theta - outer_cos) / max(inner_cos - outer_cos, 0.0001), 0.0, 1.0);
            float attenuation = spot_lights[i].color.w / (dist * dist + 0.01);
            vec3 light_intensity = spot_lights[i].color.rgb * (attenuation * spot_factor);

            vec3 spot_contrib = calcBlinnPhong(light_dir, light_intensity, norm, view_dir,
                                               base_albedo, spec_color, shininess);
            if (scene.shadow_source_type == 1 && i == 0) {
                total_light += spot_contrib * shadow; 
            } else {
                total_light += spot_contrib;
            }
        }
    }

    if (model_data.maps.y != 0) {
        vec3 emission_sample = texture(emissive_map, uv).rgb;
        float luminance = dot(emission_sample, vec3(0.2126, 0.7152, 0.0722));
        float mask = smoothstep(0.45, 0.8, luminance);
        total_light += emission_sample * mask * model_data.emission.rgb * model_data.emission.w;
    }

    out_frag_color = vec4(total_light, 1.0);
}