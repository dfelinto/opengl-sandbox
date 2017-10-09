#version 330
uniform sampler2D image_texture;

out vec4 fragColor;

void main()
{
	vec2 texel_size = 1.0 / vec2(textureSize(image_texture, 0));
	vec2 uvs = gl_FragCoord.xy * texel_size;
	fragColor = textureLod(image_texture, uvs-floor(vec2(uvs.x,0.0)), floor(uvs.x));
}
