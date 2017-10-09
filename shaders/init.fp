#version 330

out vec4 fragColor;

void main()
{
	vec2 phase = mod(gl_FragCoord.xy, (16*2));

	if ((phase.x > 16 && phase.y < 16) ||
		(phase.x < 16 && phase.y > 16))
	{
		fragColor = vec4(0.0);
	}
	else {
		fragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
}
