import bevy_sprite_render::mesh2d::{
    vertex_output::VertexOutput,
    view_bindings::globals,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var base_color_texture: texture_2d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var base_color_sampler: sampler;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(
        base_color_texture,
        base_color_sampler,
        mesh.uv + vec2(globals.time / 10., 0.),
    );
}
