#version 330
uniform sampler2D image_texture;
in vec2 texCoord_interp;
out vec4 fragColor;

void main()
{
	fragColor = texture(image_texture, texCoord_interp);
}
