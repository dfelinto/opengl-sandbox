#version 330
#extension GL_ARB_texture_query_lod: enable
#define GPU_ATI
#define CLIP_WORKAROUND
#define BUMP_BICUBIC
#define EEVEE_ENGINE
#define MAX_PROBE 128
#define MAX_GRID 64
#define MAX_PLANAR 16
#define MAX_LIGHT 128
#define MAX_SHADOW_CUBE 42
#define MAX_SHADOW_MAP 64
#define MAX_SHADOW_CASCADE 8
#define MAX_CASCADE_NUM 4
#define IRRADIANCE_HL2
#define MESH_SHADER
#define USE_BENT_NORMAL

#define M_PI        3.14159265358979323846  /* pi */
#define M_PI_2      1.57079632679489661923  /* pi/2 */
#define M_1_PI      0.318309886183790671538  /* 1/pi */
#define M_1_2PI     0.159154943091895335768  /* 1/(2*pi) */
#define M_1_PI2     0.101321183642337771443  /* 1/(pi^2) */

#define LUT_SIZE 64

uniform mat4 ProjectionMatrix;
uniform vec4 viewvecs[2];

/* ------- Structures -------- */

struct ProbeData {
	vec4 position_type;
	vec4 attenuation_fac_type;
	mat4 influencemat;
	mat4 parallaxmat;
};

#define PROBE_PARALLAX_BOX    1.0
#define PROBE_ATTENUATION_BOX 1.0

#define p_position      position_type.xyz
#define p_parallax_type position_type.w
#define p_atten_fac     attenuation_fac_type.x
#define p_atten_type    attenuation_fac_type.y

struct PlanarData {
	vec4 plane_equation;
	vec4 clip_vec_x_fade_scale;
	vec4 clip_vec_y_fade_bias;
	vec4 clip_edges;
	vec4 facing_scale_bias;
	mat4 reflectionmat; /* transform world space into reflection texture space */
};

#define pl_plane_eq      plane_equation
#define pl_normal        plane_equation.xyz
#define pl_facing_scale  facing_scale_bias.x
#define pl_facing_bias   facing_scale_bias.y
#define pl_fade_scale    clip_vec_x_fade_scale.w
#define pl_fade_bias     clip_vec_y_fade_bias.w
#define pl_clip_pos_x    clip_vec_x_fade_scale.xyz
#define pl_clip_pos_y    clip_vec_y_fade_bias.xyz
#define pl_clip_edges    clip_edges

struct GridData {
	mat4 localmat;
	ivec4 resolution_offset;
	vec4 ws_corner_atten_scale; /* world space corner position */
	vec4 ws_increment_x_atten_bias; /* world space vector between 2 opposite cells */
	vec4 ws_increment_y;
	vec4 ws_increment_z;
};

#define g_corner        ws_corner_atten_scale.xyz
#define g_atten_scale   ws_corner_atten_scale.w
#define g_atten_bias    ws_increment_x_atten_bias.w
#define g_increment_x   ws_increment_x_atten_bias.xyz
#define g_increment_y   ws_increment_y.xyz
#define g_increment_z   ws_increment_z.xyz
#define g_resolution    resolution_offset.xyz
#define g_offset        resolution_offset.w

struct LightData {
	vec4 position_influence;      /* w : InfluenceRadius */
	vec4 color_spec;              /* w : Spec Intensity */
	vec4 spotdata_radius_shadow;  /* x : spot size, y : spot blend, z : radius, w: shadow id */
	vec4 rightvec_sizex;          /* xyz: Normalized up vector, w: area size X or spot scale X */
	vec4 upvec_sizey;             /* xyz: Normalized right vector, w: area size Y or spot scale Y */
	vec4 forwardvec_type;         /* xyz: Normalized forward vector, w: Lamp Type */
};

/* convenience aliases */
#define l_color        color_spec.rgb
#define l_spec         color_spec.a
#define l_position     position_influence.xyz
#define l_influence    position_influence.w
#define l_sizex        rightvec_sizex.w
#define l_sizey        upvec_sizey.w
#define l_right        rightvec_sizex.xyz
#define l_up           upvec_sizey.xyz
#define l_forward      forwardvec_type.xyz
#define l_type         forwardvec_type.w
#define l_spot_size    spotdata_radius_shadow.x
#define l_spot_blend   spotdata_radius_shadow.y
#define l_radius       spotdata_radius_shadow.z
#define l_shadowid     spotdata_radius_shadow.w


struct ShadowCubeData {
	vec4 near_far_bias_exp;
};

/* convenience aliases */
#define sh_cube_near   near_far_bias_exp.x
#define sh_cube_far    near_far_bias_exp.y
#define sh_cube_bias   near_far_bias_exp.z
#define sh_cube_exp    near_far_bias_exp.w


struct ShadowMapData {
	mat4 shadowmat;
	vec4 near_far_bias;
};

/* convenience aliases */
#define sh_map_near   near_far_bias.x
#define sh_map_far    near_far_bias.y
#define sh_map_bias   near_far_bias.z

#ifndef MAX_CASCADE_NUM
#define MAX_CASCADE_NUM 4
#endif

struct ShadowCascadeData {
	mat4 shadowmat[MAX_CASCADE_NUM];
	/* arrays of float are not aligned so use vec4 */
	vec4 split_distances;
	vec4 bias;
};

struct ShadingData {
	vec3 V; /* View vector */
	vec3 N; /* World Normal of the fragment */
	vec3 W; /* World Position of the fragment */
	vec3 l_vector; /* Current Light vector */
};

/* ------- Convenience functions --------- */

vec3 mul(mat3 m, vec3 v) { return m * v; }
mat3 mul(mat3 m1, mat3 m2) { return m1 * m2; }

float min_v3(vec3 v) { return min(v.x, min(v.y, v.z)); }

float saturate(float a) { return clamp(a, 0.0, 1.0); }
vec2 saturate(vec2 a) { return clamp(a, 0.0, 1.0); }
vec3 saturate(vec3 a) { return clamp(a, 0.0, 1.0); }
vec4 saturate(vec4 a) { return clamp(a, 0.0, 1.0); }

float distance_squared(vec2 a, vec2 b) { a -= b; return dot(a, a); }
float distance_squared(vec3 a, vec3 b) { a -= b; return dot(a, a); }
float len_squared(vec3 a) { return dot(a, a); }

float inverse_distance(vec3 V) { return max( 1 / length(V), 1e-8); }

/* ------- Fast Math ------- */

/* [Drobot2014a] Low Level Optimizations for GCN */
float fast_sqrt(float x)
{
	return intBitsToFloat(0x1fbd1df5 + (floatBitsToInt(x) >> 1));
}

/* [Eberly2014] GPGPU Programming for Games and Science */
float fast_acos(float x)
{
	float res = -0.156583 * abs(x) + M_PI_2;
	res *= fast_sqrt(1.0 - abs(x));
	return (x >= 0) ? res : M_PI - res;
}

float line_plane_intersect_dist(vec3 lineorigin, vec3 linedirection, vec3 planeorigin, vec3 planenormal)
{
	return dot(planenormal, planeorigin - lineorigin) / dot(planenormal, linedirection);
}

float line_plane_intersect_dist(vec3 lineorigin, vec3 linedirection, vec4 plane)
{
	vec3 plane_co = plane.xyz * (-plane.w / len_squared(plane.xyz));
	vec3 h = lineorigin - plane_co;
	return -dot(plane.xyz, h) / dot(plane.xyz, linedirection);
}

vec3 line_plane_intersect(vec3 lineorigin, vec3 linedirection, vec3 planeorigin, vec3 planenormal)
{
	float dist = line_plane_intersect_dist(lineorigin, linedirection, planeorigin, planenormal);
	return lineorigin + linedirection * dist;
}

float line_aligned_plane_intersect_dist(vec3 lineorigin, vec3 linedirection, vec3 planeorigin)
{
	/* aligned plane normal */
	vec3 L = planeorigin - lineorigin;
	float diskdist = length(L);
	vec3 planenormal = -normalize(L);
	return -diskdist / dot(planenormal, linedirection);
}

vec3 line_aligned_plane_intersect(vec3 lineorigin, vec3 linedirection, vec3 planeorigin)
{
	float dist = line_aligned_plane_intersect_dist(lineorigin, linedirection, planeorigin);
	if (dist < 0) {
		/* if intersection is behind we fake the intersection to be
		 * really far and (hopefully) not inside the radius of interest */
		dist = 1e16;
	}
	return lineorigin + linedirection * dist;
}

float line_unit_sphere_intersect_dist(vec3 lineorigin, vec3 linedirection)
{
	float a = dot(linedirection, linedirection);
	float b = dot(linedirection, lineorigin);
	float c = dot(lineorigin, lineorigin) - 1;

	float dist = 1e15;
	float determinant = b * b - a * c;
	if (determinant >= 0)
		dist = (sqrt(determinant) - b) / a;

	return dist;
}

float line_unit_box_intersect_dist(vec3 lineorigin, vec3 linedirection)
{
	/* https://seblagarde.wordpress.com/2012/09/29/image-based-lighting-approaches-and-parallax-corrected-cubemap/ */
	vec3 firstplane  = (vec3( 1.0) - lineorigin) / linedirection;
	vec3 secondplane = (vec3(-1.0) - lineorigin) / linedirection;
	vec3 furthestplane = max(firstplane, secondplane);

	return min_v3(furthestplane);
}


/* Return texture coordinates to sample Surface LUT */
vec2 lut_coords(float cosTheta, float roughness)
{
	float theta = acos(cosTheta);
	vec2 coords = vec2(roughness, theta / M_PI_2);

	/* scale and bias coordinates, for correct filtered lookup */
	return coords * (LUT_SIZE - 1.0) / LUT_SIZE + 0.5 / LUT_SIZE;
}

/* -- Tangent Space conversion -- */
vec3 tangent_to_world(vec3 vector, vec3 N, vec3 T, vec3 B)
{
	return T * vector.x + B * vector.y + N * vector.z;
}

vec3 world_to_tangent(vec3 vector, vec3 N, vec3 T, vec3 B)
{
	return vec3( dot(T, vector), dot(B, vector), dot(N, vector));
}

void make_orthonormal_basis(vec3 N, out vec3 T, out vec3 B)
{
	vec3 UpVector = abs(N.z) < 0.99999 ? vec3(0.0,0.0,1.0) : vec3(1.0,0.0,0.0);
	T = normalize( cross(UpVector, N) );
	B = cross(N, T);
}

/* ---- Opengl Depth conversion ---- */
float linear_depth(bool is_persp, float z, float zf, float zn)
{
	if (is_persp) {
		return (zn  * zf) / (z * (zn - zf) + zf);
	}
	else {
		return (z * 2.0 - 1.0) * zf;
	}
}

float buffer_depth(bool is_persp, float z, float zf, float zn)
{
	if (is_persp) {
		return (zf * (zn - z)) / (z * (zn - zf));
	}
	else {
		return (z / (zf * 2.0)) + 0.5;
	}
}

vec3 get_view_space_from_depth(vec2 uvcoords, float depth)
{
	if (ProjectionMatrix[3][3] == 0.0) {
		float d = 2.0 * depth - 1.0;
		float zview = -ProjectionMatrix[3][2] / (d + ProjectionMatrix[2][2]);
		return (viewvecs[0].xyz + vec3(uvcoords, 0.0) * viewvecs[1].xyz) * zview;
	}
	else {
		return viewvecs[0].xyz + vec3(uvcoords, depth) * viewvecs[1].xyz;
	}
}

vec3 get_specular_dominant_dir(vec3 N, vec3 R, float roughness)
{
	float smoothness = 1.0 - roughness;
	float fac = smoothness * (sqrt(smoothness) + roughness);
	return normalize(mix(N, R, fac));
}

float specular_occlusion(float NV, float AO, float roughness)
{
	return saturate(pow(NV + AO, roughness) - 1.0 + AO);
}

/* Fresnel */
vec3 F_schlick(vec3 f0, float cos_theta)
{
	float fac = pow(1.0 - cos_theta, 5);

	/* Unreal specular matching : if specular color is below 2% intensity,
	 * (using green channel for intensity) treat as shadowning */
	return saturate(50.0 * f0.g) * fac + (1.0 - fac) * f0;
}

/* Fresnel approximation for LTC area lights (not MRP) */
vec3 F_area(vec3 f0, vec2 lut)
{
	vec2 fac = normalize(lut.xy);

	/* Unreal specular matching : if specular color is below 2% intensity,
	 * (using green channel for intensity) treat as shadowning */
	return saturate(50.0 * f0.g) * fac.y + fac.x * f0;
}

/* Fresnel approximation for LTC area lights (not MRP) */
vec3 F_ibl(vec3 f0, vec2 lut)
{
	/* Unreal specular matching : if specular color is below 2% intensity,
	 * (using green channel for intensity) treat as shadowning */
	return saturate(50.0 * f0.g) * lut.y + lut.x * f0;
}

/* GGX */
float D_ggx_opti(float NH, float a2)
{
	float tmp = (NH * a2 - NH) * NH + 1.0;
	return M_PI * tmp*tmp; /* Doing RCP and mul a2 at the end */
}

float G1_Smith_GGX(float NX, float a2)
{
	/* Using Brian Karis approach and refactoring by NX/NX
	 * this way the (2*NL)*(2*NV) in G = G1(V) * G1(L) gets canceled by the brdf denominator 4*NL*NV
	 * Rcp is done on the whole G later
	 * Note that this is not convenient for the transmition formula */
	return NX + sqrt(NX * (NX - NX * a2) + a2);
	/* return 2 / (1 + sqrt(1 + a2 * (1 - NX*NX) / (NX*NX) ) ); /* Reference function */
}

float bsdf_ggx(vec3 N, vec3 L, vec3 V, float roughness)
{
	float a = roughness;
	float a2 = a * a;

	vec3 H = normalize(L + V);
	float NH = max(dot(N, H), 1e-8);
	float NL = max(dot(N, L), 1e-8);
	float NV = max(dot(N, V), 1e-8);

	float G = G1_Smith_GGX(NV, a2) * G1_Smith_GGX(NL, a2); /* Doing RCP at the end */
	float D = D_ggx_opti(NH, a2);

	/* Denominator is canceled by G1_Smith */
	/* bsdf = D * G / (4.0 * NL * NV); /* Reference function */
	return NL * a2 / (D * G); /* NL to Fit cycles Equation : line. 345 in bsdf_microfacet.h */
}

/* Based on Practical Realtime Strategies for Accurate Indirect Occlusion
 * http://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf
 * http://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pptx */

#define MAX_PHI_STEP 32
/* NOTICE : this is multiplied by 2 */
#define MAX_THETA_STEP 12

uniform sampler2D minMaxDepthTex;
uniform vec3 aoParameters;

#define aoDistance   aoParameters.x
#define aoSamples    aoParameters.y
#define aoFactor     aoParameters.z

float get_max_horizon(vec2 co, vec3 x, float h, float lod)
{
	float depth = textureLod(minMaxDepthTex, co, floor(lod)).g;

	/* Background case */
	/* this is really slow and is only a problem
	 * if the far clip plane is near enough to notice */
	// depth += step(1.0, depth) * 1e20;

	vec3 s = get_view_space_from_depth(co, depth); /* s View coordinate */
	vec3 omega_s = s - x;
	float len = length(omega_s);

	float max_h = max(h, omega_s.z / len);
	/* Blend weight after half the aoDistance to fade artifacts */
	float blend = saturate((1.0 - len / aoDistance) * 2.0);

	return mix(h, max_h, blend);
}

void search_step(
        vec2 t_phi, vec3 x, vec2 x_, float rand, vec2 pixel_ratio,
        inout float j, inout float ofs, inout float h1, inout float h2)
{
	ofs += ofs; /* Step size is doubled each iteration */

	vec2 s_ = t_phi * ofs * rand * pixel_ratio; /* s^ Screen coordinate */
	vec2 co;

	co = x_ + s_;
	h1 = get_max_horizon(co, x, h1, j);

	co = x_ - s_;
	h2 = get_max_horizon(co, x, h2, j);

	j += 0.5;
}

void search_horizon(
        vec2 t_phi, vec3 x, vec2 x_, float rand,
        float max_dist, vec2 pixel_ratio, float pixel_len,
        inout float h1, inout float h2)
{
	float ofs = 1.5 * pixel_len;
	float j = 0.0;

#if 0 /* manually unrolled bellow */
	for (int i = 0; i < MAX_THETA_STEP; i++) {
		search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
		if (ofs > max_dist)
			return;
	}
#endif
	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
	if (ofs > max_dist)	return;

	search_step(t_phi, x, x_, rand, pixel_ratio, j, ofs, h1, h2);
}

void integrate_slice(
        float iter, vec3 x, vec3 normal, vec2 x_, vec2 noise,
        float max_dist, vec2 pixel_ratio, float pixel_len,
        inout float visibility, inout vec3 bent_normal)
{
	float phi = M_PI * ((noise.r + iter) / aoSamples);

	/* Rotate with random direction to get jittered result. */
	vec2 t_phi = vec2(cos(phi), sin(phi)); /* Screen space direction */

	/* Search maximum horizon angles h1 and h2 */
	float h1 = -1.0, h2 = -1.0; /* init at cos(pi) */
	search_horizon(t_phi, x, x_, noise.g, max_dist, pixel_ratio, pixel_len, h1, h2);

	/* (Slide 54) */
	h1 = -fast_acos(h1);
	h2 = fast_acos(h2);

	/* Projecting Normal to Plane P defined by t_phi and omega_o */
	vec3 h = vec3(t_phi.y, -t_phi.x, 0.0); /* Normal vector to Integration plane */
	vec3 t = vec3(-t_phi, 0.0);
	vec3 n_proj = normal - h * dot(h, normal);
	float n_proj_len = max(1e-16, length(n_proj));

	/* Clamping thetas (slide 58) */
	float cos_n = clamp(n_proj.z / n_proj_len, -1.0, 1.0);
	float n = sign(dot(n_proj, t)) * fast_acos(cos_n); /* Angle between view vec and normal */
	h1 = n + max(h1 - n, -M_PI_2);
	h2 = n + min(h2 - n, M_PI_2);

	/* Solving inner integral */
	float sin_n = sin(n);
	float h1_2 = 2.0 * h1;
	float h2_2 = 2.0 * h2;
	float vd = (-cos(h1_2 - n) + cos_n + h1_2 * sin_n) + (-cos(h2_2 - n) + cos_n + h2_2 * sin_n);
	vd *= 0.25 * n_proj_len;
	visibility += vd;

#ifdef USE_BENT_NORMAL
	/* Finding Bent normal */
	float b_angle = (h1 + h2) / 2.0;
	/* The 0.5 factor below is here to equilibrate the accumulated vectors.
	 * (sin(b_angle) * -t_phi) will accumulate to (phi_step * result_nor.xy * 0.5).
	 * (cos(b_angle) * 0.5) will accumulate to (phi_step * result_nor.z * 0.5). */
	/* Weight sample by vd */
	bent_normal += vec3(sin(b_angle) * -t_phi, cos(b_angle) * 0.5) * vd;
#endif
}

void gtao(vec3 normal, vec3 position, vec2 noise, out float visibility
#ifdef USE_BENT_NORMAL
	, out vec3 bent_normal
#endif
	)
{
	vec2 screenres = vec2(textureSize(minMaxDepthTex, 0)) * 2.0;
	vec2 pixel_size = vec2(1.0) / screenres.xy;

	/* Renaming */
	vec2 x_ = gl_FragCoord.xy * pixel_size; /* x^ Screen coordinate */
	vec3 x = position; /* x view space coordinate */

	/* NOTE : We set up integration domain around the camera forward axis
	 * and not the view vector like in the paper.
	 * This allows us to save a lot of dot products. */
	/* omega_o = vec3(0.0, 0.0, 1.0); */

	vec2 pixel_ratio = vec2(screenres.y / screenres.x, 1.0);
	float pixel_len = length(pixel_size);
	float homcco = ProjectionMatrix[2][3] * position.z + ProjectionMatrix[3][3];
	float max_dist = aoDistance / homcco; /* Search distance */

	/* Integral over PI */
	visibility = 0.0;
#ifdef USE_BENT_NORMAL
	bent_normal = vec3(0.0);
#else
	vec3 bent_normal = vec3(0.0);
#endif
	for (float i = 0.0; i < MAX_PHI_STEP; i++) {
		if (i >= aoSamples) break;
		integrate_slice(i, x, normal, x_, noise, max_dist, pixel_ratio, pixel_len, visibility, bent_normal);
	}

	visibility = clamp(visibility / aoSamples, 1e-8, 1.0);

#ifdef USE_BENT_NORMAL
	/* The bent normal will show the facet look of the mesh. Try to minimize this. */
	bent_normal = normalize(mix(bent_normal / visibility, normal, visibility * visibility * visibility));
#endif

	/* Scale by user factor */
	visibility = pow(visibility, aoFactor);
}

/* Multibounce approximation base on surface albedo.
 * Page 78 in the .pdf version. */
float gtao_multibounce(float visibility, vec3 albedo)
{
	/* Median luminance. Because Colored multibounce looks bad. */
	float lum = albedo.x * 0.3333;
	lum += albedo.y * 0.3333;
	lum += albedo.z * 0.3333;

	float a =  2.0404 * lum - 0.3324;
	float b = -4.7951 * lum + 0.6417;
	float c =  2.7552 * lum + 0.6903;

	float x = visibility;
	return max(x, ((x * a + b) * x + c) * x);
}
vec2 mapping_octahedron(vec3 cubevec, vec2 texel_size)
{
	/* projection onto octahedron */
	cubevec /= dot( vec3(1), abs(cubevec) );

	/* out-folding of the downward faces */
	if ( cubevec.z < 0.0 ) {
		cubevec.xy = (1.0 - abs(cubevec.yx)) * sign(cubevec.xy);
	}

	/* mapping to [0;1]ˆ2 texture space */
	vec2 uvs = cubevec.xy * (0.5) + 0.5;

	/* edge filtering fix */
	uvs *= 1.0 - 2.0 * texel_size;
	uvs += texel_size;

	return uvs;
}

vec4 textureLod_octahedron(sampler2DArray tex, vec4 cubevec, float lod, float lod_max)
{
	vec2 texelSize = 1.0 / vec2(textureSize(tex, int(lod_max)));

	vec2 uvs = mapping_octahedron(cubevec.xyz, texelSize);

	return textureLod(tex, vec3(uvs, cubevec.w), lod);
}

vec4 texture_octahedron(sampler2DArray tex, vec4 cubevec)
{
	vec2 texelSize = 1.0 / vec2(textureSize(tex, 0));

	vec2 uvs = mapping_octahedron(cubevec.xyz, texelSize);

	return texture(tex, vec3(uvs, cubevec.w));
}

uniform sampler2D irradianceGrid;

#ifdef IRRADIANCE_CUBEMAP
struct IrradianceData {
	vec3 color;
};
#elif defined(IRRADIANCE_SH_L2)
struct IrradianceData {
	vec3 shcoefs[9];
};
#else /* defined(IRRADIANCE_HL2) */
struct IrradianceData {
	vec3 cubesides[3];
};
#endif

IrradianceData load_irradiance_cell(int cell, vec3 N)
{
	/* Keep in sync with diffuse_filter_probe() */

#if defined(IRRADIANCE_CUBEMAP)

	#define AMBIANT_CUBESIZE 8
	ivec2 cell_co = ivec2(AMBIANT_CUBESIZE);
	int cell_per_row = textureSize(irradianceGrid, 0).x / cell_co.x;
	cell_co.x *= cell % cell_per_row;
	cell_co.y *= cell / cell_per_row;

	vec2 texelSize = 1.0 / vec2(AMBIANT_CUBESIZE);

	vec2 uvs = mapping_octahedron(N, texelSize);
	uvs *= vec2(AMBIANT_CUBESIZE) / vec2(textureSize(irradianceGrid, 0));
	uvs += vec2(cell_co) / vec2(textureSize(irradianceGrid, 0));

	IrradianceData ir;
	ir.color = texture(irradianceGrid, uvs).rgb;

#elif defined(IRRADIANCE_SH_L2)

	ivec2 cell_co = ivec2(3, 3);
	int cell_per_row = textureSize(irradianceGrid, 0).x / cell_co.x;
	cell_co.x *= cell % cell_per_row;
	cell_co.y *= cell / cell_per_row;

	ivec3 ofs = ivec3(0, 1, 2);

	IrradianceData ir;
	ir.shcoefs[0] = texelFetch(irradianceGrid, cell_co + ofs.xx, 0).rgb;
	ir.shcoefs[1] = texelFetch(irradianceGrid, cell_co + ofs.yx, 0).rgb;
	ir.shcoefs[2] = texelFetch(irradianceGrid, cell_co + ofs.zx, 0).rgb;
	ir.shcoefs[3] = texelFetch(irradianceGrid, cell_co + ofs.xy, 0).rgb;
	ir.shcoefs[4] = texelFetch(irradianceGrid, cell_co + ofs.yy, 0).rgb;
	ir.shcoefs[5] = texelFetch(irradianceGrid, cell_co + ofs.zy, 0).rgb;
	ir.shcoefs[6] = texelFetch(irradianceGrid, cell_co + ofs.xz, 0).rgb;
	ir.shcoefs[7] = texelFetch(irradianceGrid, cell_co + ofs.yz, 0).rgb;
	ir.shcoefs[8] = texelFetch(irradianceGrid, cell_co + ofs.zz, 0).rgb;

#else /* defined(IRRADIANCE_HL2) */

	ivec2 cell_co = ivec2(3, 2);
	int cell_per_row = textureSize(irradianceGrid, 0).x / cell_co.x;
	cell_co.x *= cell % cell_per_row;
	cell_co.y *= cell / cell_per_row;

	ivec3 is_negative = ivec3(step(0.0, -N));

	IrradianceData ir;
	ir.cubesides[0] = texelFetch(irradianceGrid, cell_co + ivec2(0, is_negative.x), 0).rgb;
	ir.cubesides[1] = texelFetch(irradianceGrid, cell_co + ivec2(1, is_negative.y), 0).rgb;
	ir.cubesides[2] = texelFetch(irradianceGrid, cell_co + ivec2(2, is_negative.z), 0).rgb;

#endif

	return ir;
}

/* http://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/ */
vec3 spherical_harmonics_L1(vec3 N, vec3 shcoefs[4])
{
	vec3 sh = vec3(0.0);

	sh += 0.282095 * shcoefs[0];

	sh += -0.488603 * N.z * shcoefs[1];
	sh += 0.488603 * N.y * shcoefs[2];
	sh += -0.488603 * N.x * shcoefs[3];

	return sh;
}

vec3 spherical_harmonics_L2(vec3 N, vec3 shcoefs[9])
{
	vec3 sh = vec3(0.0);

	sh += 0.282095 * shcoefs[0];

	sh += -0.488603 * N.z * shcoefs[1];
	sh += 0.488603 * N.y * shcoefs[2];
	sh += -0.488603 * N.x * shcoefs[3];

	sh += 1.092548 * N.x * N.z * shcoefs[4];
	sh += -1.092548 * N.z * N.y * shcoefs[5];
	sh += 0.315392 * (3.0 * N.y * N.y - 1.0) * shcoefs[6];
	sh += -1.092548 * N.x * N.y * shcoefs[7];
	sh += 0.546274 * (N.x * N.x - N.z * N.z) * shcoefs[8];

	return sh;
}

vec3 hl2_basis(vec3 N, vec3 cubesides[3])
{
	vec3 irradiance = vec3(0.0);

	vec3 n_squared = N * N;

	irradiance += n_squared.x * cubesides[0];
	irradiance += n_squared.y * cubesides[1];
	irradiance += n_squared.z * cubesides[2];

	return irradiance;
}

vec3 compute_irradiance(vec3 N, IrradianceData ird)
{
#if defined(IRRADIANCE_CUBEMAP)
	return ird.color;
#elif defined(IRRADIANCE_SH_L2)
	return spherical_harmonics_L2(N, ird.shcoefs);
#else /* defined(IRRADIANCE_HL2) */
	return hl2_basis(N, ird.cubesides);
#endif
}

vec3 get_cell_color(ivec3 localpos, ivec3 gridres, int offset, vec3 ir_dir)
{
	/* Keep in sync with update_irradiance_probe */
	int cell = offset + localpos.z + localpos.y * gridres.z + localpos.x * gridres.z * gridres.y;
	IrradianceData ir_data = load_irradiance_cell(cell, ir_dir);
	return compute_irradiance(ir_dir, ir_data);
}
/* Mainly From https://eheitzresearch.wordpress.com/415-2/ */

#define USE_LTC

#ifndef UTIL_TEX
#define UTIL_TEX
uniform sampler2DArray utilTex;
#endif /* UTIL_TEX */

/* from Real-Time Area Lighting: a Journey from Research to Production
 * Stephen Hill and Eric Heitz */
float edge_integral(vec3 p1, vec3 p2)
{
#if 0
	/* more accurate replacement of acos */
	float x = dot(p1, p2);
	float y = abs(x);

	float a = 5.42031 + (3.12829 + 0.0902326 * y) * y;
	float b = 3.45068 + (4.18814 + y) * y;
	float theta_sintheta = a / b;

	if (x < 0.0) {
		theta_sintheta = (M_PI / sqrt(1.0 - x * x)) - theta_sintheta;
	}
	vec3 u = cross(p1, p2);
	return theta_sintheta * dot(u, N);
#endif
	float cos_theta = dot(p1, p2);
	cos_theta = clamp(cos_theta, -0.9999, 0.9999);

	float theta = acos(cos_theta);
	vec3 u = normalize(cross(p1, p2));
	return theta * cross(p1, p2).z / sin(theta);
}

int clip_quad_to_horizon(inout vec3 L[5])
{
	/* detect clipping config */
	int config = 0;
	if (L[0].z > 0.0) config += 1;
	if (L[1].z > 0.0) config += 2;
	if (L[2].z > 0.0) config += 4;
	if (L[3].z > 0.0) config += 8;

	/* clip */
	int n = 0;

	if (config == 0)
	{
		/* clip all */
	}
	else if (config == 1) /* V1 clip V2 V3 V4 */
	{
		n = 3;
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
		L[2] = -L[3].z * L[0] + L[0].z * L[3];
	}
	else if (config == 2) /* V2 clip V1 V3 V4 */
	{
		n = 3;
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
	}
	else if (config == 3) /* V1 V2 clip V3 V4 */
	{
		n = 4;
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
		L[3] = -L[3].z * L[0] + L[0].z * L[3];
	}
	else if (config == 4) /* V3 clip V1 V2 V4 */
	{
		n = 3;
		L[0] = -L[3].z * L[2] + L[2].z * L[3];
		L[1] = -L[1].z * L[2] + L[2].z * L[1];
	}
	else if (config == 5) /* V1 V3 clip V2 V4) impossible */
	{
		n = 0;
	}
	else if (config == 6) /* V2 V3 clip V1 V4 */
	{
		n = 4;
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
		L[3] = -L[3].z * L[2] + L[2].z * L[3];
	}
	else if (config == 7) /* V1 V2 V3 clip V4 */
	{
		n = 5;
		L[4] = -L[3].z * L[0] + L[0].z * L[3];
		L[3] = -L[3].z * L[2] + L[2].z * L[3];
	}
	else if (config == 8) /* V4 clip V1 V2 V3 */
	{
		n = 3;
		L[0] = -L[0].z * L[3] + L[3].z * L[0];
		L[1] = -L[2].z * L[3] + L[3].z * L[2];
		L[2] =  L[3];
	}
	else if (config == 9) /* V1 V4 clip V2 V3 */
	{
		n = 4;
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
		L[2] = -L[2].z * L[3] + L[3].z * L[2];
	}
	else if (config == 10) /* V2 V4 clip V1 V3) impossible */
	{
		n = 0;
	}
	else if (config == 11) /* V1 V2 V4 clip V3 */
	{
		n = 5;
		L[4] = L[3];
		L[3] = -L[2].z * L[3] + L[3].z * L[2];
		L[2] = -L[2].z * L[1] + L[1].z * L[2];
	}
	else if (config == 12) /* V3 V4 clip V1 V2 */
	{
		n = 4;
		L[1] = -L[1].z * L[2] + L[2].z * L[1];
		L[0] = -L[0].z * L[3] + L[3].z * L[0];
	}
	else if (config == 13) /* V1 V3 V4 clip V2 */
	{
		n = 5;
		L[4] = L[3];
		L[3] = L[2];
		L[2] = -L[1].z * L[2] + L[2].z * L[1];
		L[1] = -L[1].z * L[0] + L[0].z * L[1];
	}
	else if (config == 14) /* V2 V3 V4 clip V1 */
	{
		n = 5;
		L[4] = -L[0].z * L[3] + L[3].z * L[0];
		L[0] = -L[0].z * L[1] + L[1].z * L[0];
	}
	else if (config == 15) /* V1 V2 V3 V4 */
	{
		n = 4;
	}

	if (n == 3)
		L[3] = L[0];
	if (n == 4)
		L[4] = L[0];

	return n;
}

mat3 ltc_matrix(vec4 lut)
{
	/* load inverse matrix */
	mat3 Minv = mat3(
		vec3(  1,   0, lut.y),
		vec3(  0, lut.z,   0),
		vec3(lut.w,   0, lut.x)
	);

	return Minv;
}

float ltc_evaluate(vec3 N, vec3 V, mat3 Minv, vec3 corners[4])
{
	/* construct orthonormal basis around N */
	vec3 T1, T2;
	T1 = normalize(V - N*dot(V, N));
	T2 = cross(N, T1);

	/* rotate area light in (T1, T2, R) basis */
	Minv = Minv * transpose(mat3(T1, T2, N));

	/* polygon (allocate 5 vertices for clipping) */
	vec3 L[5];
	L[0] = Minv * corners[0];
	L[1] = Minv * corners[1];
	L[2] = Minv * corners[2];
	L[3] = Minv * corners[3];

	int n = clip_quad_to_horizon(L);

	if (n == 0)
		return 0.0;

	/* project onto sphere */
	L[0] = normalize(L[0]);
	L[1] = normalize(L[1]);
	L[2] = normalize(L[2]);
	L[3] = normalize(L[3]);
	L[4] = normalize(L[4]);

	/* integrate */
	float sum = 0.0;

	sum += edge_integral(L[0], L[1]);
	sum += edge_integral(L[1], L[2]);
	sum += edge_integral(L[2], L[3]);
	if (n >= 4)
		sum += edge_integral(L[3], L[4]);
	if (n == 5)
		sum += edge_integral(L[4], L[0]);

	return abs(sum);
}

/* Aproximate circle with an octogone */
#define LTC_CIRCLE_RES 8
float ltc_evaluate_circle(vec3 N, vec3 V, mat3 Minv, vec3 p[LTC_CIRCLE_RES])
{
	/* construct orthonormal basis around N */
	vec3 T1, T2;
	T1 = normalize(V - N*dot(V, N));
	T2 = cross(N, T1);

	/* rotate area light in (T1, T2, R) basis */
	Minv = Minv * transpose(mat3(T1, T2, N));

	for (int i = 0; i < LTC_CIRCLE_RES; ++i) {
		p[i] = Minv * p[i];
		/* clip to horizon */
		p[i].z = max(0.0, p[i].z);
		/* project onto sphere */
		p[i] = normalize(p[i]);
	}

	/* integrate */
	float sum = 0.0;
	for (int i = 0; i < LTC_CIRCLE_RES - 1; ++i) {
		sum += edge_integral(p[i], p[i + 1]);
	}
	sum += edge_integral(p[LTC_CIRCLE_RES - 1], p[0]);

	return max(0.0, sum);
}

/* Bsdf direct light function */
/* in other word, how materials react to scene lamps */

/* Naming convention
 * V       View vector (normalized)
 * N       World Normal (normalized)
 * L       Outgoing Light Vector (Surface to Light in World Space) (normalized)
 * Ldist   Distance from surface to the light
 * W       World Pos
 */

/* ------------ Diffuse ------------- */

float direct_diffuse_point(LightData ld, ShadingData sd)
{
	float dist = length(sd.l_vector);
	vec3 L = sd.l_vector / dist;
	float bsdf = max(0.0, dot(sd.N, L));
	bsdf /= dist * dist;
	return bsdf;
}

/* infinitly far away point source, no decay */
float direct_diffuse_sun(LightData ld, ShadingData sd)
{
	float bsdf = max(0.0, dot(sd.N, -ld.l_forward));
	bsdf *= M_1_PI; /* Normalize */
	return bsdf;
}

/* From Frostbite PBR Course
 * Analitical irradiance from a sphere with correct horizon handling
 * http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf */
float direct_diffuse_sphere(LightData ld, ShadingData sd)
{
	float dist = length(sd.l_vector);
	vec3 L = sd.l_vector / dist;
	float radius = max(ld.l_sizex, 0.0001);
	float costheta = clamp(dot(sd.N, L), -0.999, 0.999);
	float h = min(ld.l_radius / dist , 0.9999);
	float h2 = h*h;
	float costheta2 = costheta * costheta;
	float bsdf;

	if (costheta2 > h2) {
		bsdf = M_PI * h2 * clamp(costheta, 0.0, 1.0);
	}
	else {
		float sintheta = sqrt(1.0 - costheta2);
		float x = sqrt(1.0 / h2 - 1.0);
		float y = -x * (costheta / sintheta);
		float sinthetasqrty = sintheta * sqrt(1.0 - y * y);
		bsdf = (costheta * acos(y) - x * sinthetasqrty) * h2 + atan(sinthetasqrty / x);
	}

	bsdf = max(bsdf, 0.0);
	bsdf *= M_1_PI2;

	return bsdf;
}

/* From Frostbite PBR Course
 * http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf */
float direct_diffuse_rectangle(LightData ld, ShadingData sd)
{
	vec3 corners[4];
	corners[0] = sd.l_vector + ld.l_right * -ld.l_sizex + ld.l_up *  ld.l_sizey;
	corners[1] = sd.l_vector + ld.l_right * -ld.l_sizex + ld.l_up * -ld.l_sizey;
	corners[2] = sd.l_vector + ld.l_right *  ld.l_sizex + ld.l_up * -ld.l_sizey;
	corners[3] = sd.l_vector + ld.l_right *  ld.l_sizex + ld.l_up *  ld.l_sizey;

	float bsdf = ltc_evaluate(sd.N, sd.V, mat3(1.0), corners);
	bsdf *= M_1_2PI;
	return bsdf;
}


#if 0
float direct_diffuse_unit_disc(vec3 N, vec3 L)
{

}
#endif

/* ----------- GGx ------------ */
vec3 direct_ggx_point(ShadingData sd, float roughness, vec3 f0)
{
	float dist = length(sd.l_vector);
	vec3 L = sd.l_vector / dist;
	float bsdf = bsdf_ggx(sd.N, L, sd.V, roughness);
	bsdf /= dist * dist;

	/* Fresnel */
	float VH = max(dot(sd.V, normalize(sd.V + L)), 0.0);
	return F_schlick(f0, VH) * bsdf;
}

vec3 direct_ggx_sun(LightData ld, ShadingData sd, float roughness, vec3 f0)
{
	float bsdf = bsdf_ggx(sd.N, -ld.l_forward, sd.V, roughness);
	float VH = max(dot(sd.V, normalize(sd.V - ld.l_forward)), 0.0);
	return F_schlick(f0, VH) * bsdf;
}

vec3 direct_ggx_sphere(LightData ld, ShadingData sd, float roughness, vec3 f0)
{
	vec3 L = normalize(sd.l_vector);
	vec3 spec_dir = get_specular_dominant_dir(sd.N, reflect(-sd.V, sd.N), roughness);
	vec3 P = line_aligned_plane_intersect(vec3(0.0), spec_dir, sd.l_vector);

	vec3 Px = normalize(P - sd.l_vector) * ld.l_radius;
	vec3 Py = cross(Px, L);

	vec2 uv = lut_coords(dot(sd.N, sd.V), sqrt(roughness));
	vec3 brdf_lut = texture(utilTex, vec3(uv, 1.0)).rgb;
	vec4 ltc_lut = texture(utilTex, vec3(uv, 0.0)).rgba;
	mat3 ltc_mat = ltc_matrix(ltc_lut);

// #define HIGHEST_QUALITY
#ifdef HIGHEST_QUALITY
	vec3 Pxy1 = normalize( Px + Py) * ld.l_radius;
	vec3 Pxy2 = normalize(-Px + Py) * ld.l_radius;

	/* counter clockwise */
	vec3 points[8];
	points[0] = sd.l_vector + Px;
	points[1] = sd.l_vector - Pxy2;
	points[2] = sd.l_vector - Py;
	points[3] = sd.l_vector - Pxy1;
	points[4] = sd.l_vector - Px;
	points[5] = sd.l_vector + Pxy2;
	points[6] = sd.l_vector + Py;
	points[7] = sd.l_vector + Pxy1;
	float bsdf = ltc_evaluate_circle(sd.N, sd.V, ltc_mat, points);
#else
	vec3 points[4];
	points[0] = sd.l_vector + Px;
	points[1] = sd.l_vector - Py;
	points[2] = sd.l_vector - Px;
	points[3] = sd.l_vector + Py;
	float bsdf = ltc_evaluate(sd.N, sd.V, ltc_mat, points);
	/* sqrt(pi/2) difference between square and disk area */
	bsdf *= 1.25331413731;
#endif

	bsdf *= brdf_lut.b; /* Bsdf intensity */
	bsdf *= M_1_2PI * M_1_PI;

	vec3 spec = F_area(f0, brdf_lut.xy) * bsdf;

	return spec;
}

vec3 direct_ggx_rectangle(LightData ld, ShadingData sd, float roughness, vec3 f0)
{
	vec3 corners[4];
	corners[0] = sd.l_vector + ld.l_right * -ld.l_sizex + ld.l_up *  ld.l_sizey;
	corners[1] = sd.l_vector + ld.l_right * -ld.l_sizex + ld.l_up * -ld.l_sizey;
	corners[2] = sd.l_vector + ld.l_right *  ld.l_sizex + ld.l_up * -ld.l_sizey;
	corners[3] = sd.l_vector + ld.l_right *  ld.l_sizex + ld.l_up *  ld.l_sizey;

	vec2 uv = lut_coords(dot(sd.N, sd.V), sqrt(roughness));
	vec3 brdf_lut = texture(utilTex, vec3(uv, 1.0)).rgb;
	vec4 ltc_lut = texture(utilTex, vec3(uv, 0.0)).rgba;
	mat3 ltc_mat = ltc_matrix(ltc_lut);
	float bsdf = ltc_evaluate(sd.N, sd.V, ltc_mat, corners);
	bsdf *= brdf_lut.b; /* Bsdf intensity */
	bsdf *= M_1_2PI;

	vec3 spec = F_area(f0, brdf_lut.xy) * bsdf;

	return spec;
}

#if 0
float direct_ggx_disc(vec3 N, vec3 L)
{

}
#endif
uniform int light_count;
uniform int probe_count;
uniform int grid_count;
uniform int planar_count;
uniform mat4 ViewMatrix;
uniform mat4 ViewMatrixInverse;

uniform sampler2DArray probePlanars;

uniform sampler2DArray probeCubes;
uniform float lodMax;
uniform bool specToggle;

#ifndef UTIL_TEX
#define UTIL_TEX
uniform sampler2DArray utilTex;
#endif /* UTIL_TEX */

uniform sampler2DArray shadowCubes;
uniform sampler2DArrayShadow shadowCascades;

layout(std140) uniform probe_block {
	ProbeData probes_data[MAX_PROBE];
};

layout(std140) uniform grid_block {
	GridData grids_data[MAX_GRID];
};

layout(std140) uniform planar_block {
	PlanarData planars_data[MAX_PLANAR];
};

layout(std140) uniform light_block {
	LightData lights_data[MAX_LIGHT];
};

layout(std140) uniform shadow_block {
	ShadowCubeData    shadows_cube_data[MAX_SHADOW_CUBE];
	ShadowMapData     shadows_map_data[MAX_SHADOW_MAP];
	ShadowCascadeData shadows_cascade_data[MAX_SHADOW_CASCADE];
};

in vec3 worldPosition;
in vec3 viewPosition;

#ifdef USE_FLAT_NORMAL
flat in vec3 worldNormal;
flat in vec3 viewNormal;
#else
in vec3 worldNormal;
in vec3 viewNormal;
#endif

#define cameraForward   normalize(ViewMatrixInverse[2].xyz)
#define cameraPos       ViewMatrixInverse[3].xyz

/* type */
#define POINT    0.0
#define SUN      1.0
#define SPOT     2.0
#define HEMI     3.0
#define AREA     4.0

#ifdef HAIR_SHADER
vec3 light_diffuse(LightData ld, ShadingData sd, vec3 albedo)
{
       if (ld.l_type == SUN) {
               return direct_diffuse_sun(ld, sd) * albedo;
       }
       else if (ld.l_type == AREA) {
               return direct_diffuse_rectangle(ld, sd) * albedo;
       }
       else {
               return direct_diffuse_sphere(ld, sd) * albedo;
       }
}

vec3 light_specular(LightData ld, ShadingData sd, float roughness, vec3 f0)
{
       if (ld.l_type == SUN) {
               return direct_ggx_sun(ld, sd, roughness, f0);
       }
       else if (ld.l_type == AREA) {
               return direct_ggx_rectangle(ld, sd, roughness, f0);
       }
       else {
               return direct_ggx_sphere(ld, sd, roughness, f0);
       }
}

void light_shade(
        LightData ld, ShadingData sd, vec3 albedo, float roughness, vec3 f0,
        out vec3 diffuse, out vec3 specular)
{
       const float transmission = 0.3; /* Uniform internal scattering factor */
       ShadingData sd_new = sd;

       vec3 lamp_vec;

      if (ld.l_type == SUN || ld.l_type == AREA) {
               lamp_vec = ld.l_forward;
       }
       else {
               lamp_vec = -sd.l_vector;
       }

       vec3 norm_view = cross(sd.V, sd.N);
       norm_view = normalize(cross(norm_view, sd.N)); /* Normal facing view */

       vec3 norm_lamp = cross(lamp_vec, sd.N);
       norm_lamp = normalize(cross(sd.N, norm_lamp)); /* Normal facing lamp */

       /* Rotate view vector onto the cross(tangent, light) plane */
       vec3 view_vec = normalize(norm_lamp * dot(norm_view, sd.V) + sd.N * dot(sd.N, sd.V));

       float occlusion = (dot(norm_view, norm_lamp) * 0.5 + 0.5);
       float occltrans = transmission + (occlusion * (1.0 - transmission)); /* Includes transmission component */

       sd_new.N = -norm_lamp;

       diffuse = light_diffuse(ld, sd_new, albedo) * occltrans;

       sd_new.V = view_vec;

       specular = light_specular(ld, sd_new, roughness, f0) * occlusion;
}
#else
void light_shade(
        LightData ld, ShadingData sd, vec3 albedo, float roughness, vec3 f0,
        out vec3 diffuse, out vec3 specular)
{
#ifdef USE_LTC
	if (ld.l_type == SUN) {
		/* TODO disk area light */
		diffuse = direct_diffuse_sun(ld, sd) * albedo;
		specular = direct_ggx_sun(ld, sd, roughness, f0);
	}
	else if (ld.l_type == AREA) {
		diffuse =  direct_diffuse_rectangle(ld, sd) * albedo;
		specular =  direct_ggx_rectangle(ld, sd, roughness, f0);
	}
	else {
		diffuse =  direct_diffuse_sphere(ld, sd) * albedo;
		specular =  direct_ggx_sphere(ld, sd, roughness, f0);
	}
#else
	if (ld.l_type == SUN) {
		diffuse = direct_diffuse_sun(ld, sd) * albedo;
		specular = direct_ggx_sun(ld, sd, roughness, f0);
	}
	else {
		diffuse = direct_diffuse_point(ld, sd) * albedo;
		specular = direct_ggx_point(sd, roughness, f0);
	}
#endif

	specular *= float(specToggle);
}
#endif

void light_visibility(LightData ld, ShadingData sd, out float vis)
{
	vis = 1.0;

	if (ld.l_type == SPOT) {
		float z = dot(ld.l_forward, sd.l_vector);
		vec3 lL = sd.l_vector / z;
		float x = dot(ld.l_right, lL) / ld.l_sizex;
		float y = dot(ld.l_up, lL) / ld.l_sizey;

		float ellipse = 1.0 / sqrt(1.0 + x * x + y * y);

		float spotmask = smoothstep(0.0, 1.0, (ellipse - ld.l_spot_size) / ld.l_spot_blend);

		vis *= spotmask;
		vis *= step(0.0, -dot(sd.l_vector, ld.l_forward));
	}
	else if (ld.l_type == AREA) {
		vis *= step(0.0, -dot(sd.l_vector, ld.l_forward));
	}

	/* shadowing */
	if (ld.l_shadowid >= (MAX_SHADOW_MAP + MAX_SHADOW_CUBE)) {
		/* Shadow Cascade */
		float shid = ld.l_shadowid - (MAX_SHADOW_CUBE + MAX_SHADOW_MAP);
		ShadowCascadeData smd = shadows_cascade_data[int(shid)];

		/* Finding Cascade index */
		vec4 z = vec4(-dot(cameraPos - worldPosition, cameraForward));
		vec4 comp = step(z, smd.split_distances);
		float cascade = dot(comp, comp);
		mat4 shadowmat;
		float bias;

		/* Manual Unrolling of a loop for better performance.
		 * Doing fetch directly with cascade index leads to
		 * major performance impact. (0.27ms -> 10.0ms for 1 light) */
		if (cascade == 0.0) {
			shadowmat = smd.shadowmat[0];
			bias = smd.bias[0];
		}
		else if (cascade == 1.0) {
			shadowmat = smd.shadowmat[1];
			bias = smd.bias[1];
		}
		else if (cascade == 2.0) {
			shadowmat = smd.shadowmat[2];
			bias = smd.bias[2];
		}
		else {
			shadowmat = smd.shadowmat[3];
			bias = smd.bias[3];
		}

		vec4 shpos = shadowmat * vec4(sd.W, 1.0);
		shpos.z -= bias * shpos.w;
		shpos.xyz /= shpos.w;

		vis *= texture(shadowCascades, vec4(shpos.xy, shid * float(MAX_CASCADE_NUM) + cascade, shpos.z));
	}
	else if (ld.l_shadowid >= 0.0) {
		/* Shadow Cube */
		float shid = ld.l_shadowid;
		ShadowCubeData scd = shadows_cube_data[int(shid)];

		vec3 cubevec = sd.W - ld.l_position;
		float dist = length(cubevec) - scd.sh_cube_bias;

		float z = texture_octahedron(shadowCubes, vec4(cubevec, shid)).r;

		float esm_test = saturate(exp(scd.sh_cube_exp * (z - dist)));
		float sh_test = step(0, z - dist);

		vis *= esm_test;
	}
}

vec3 probe_parallax_correction(vec3 W, vec3 spec_dir, ProbeData pd, inout float roughness)
{
	vec3 localpos = (pd.parallaxmat * vec4(W, 1.0)).xyz;
	vec3 localray = (pd.parallaxmat * vec4(spec_dir, 0.0)).xyz;

	float dist;
	if (pd.p_parallax_type == PROBE_PARALLAX_BOX) {
		dist = line_unit_box_intersect_dist(localpos, localray);
	}
	else {
		dist = line_unit_sphere_intersect_dist(localpos, localray);
	}

	/* Use Distance in WS directly to recover intersection */
	vec3 intersection = W + spec_dir * dist - pd.p_position;

	/* From Frostbite PBR Course
	 * Distance based roughness
	 * http://www.frostbite.com/wp-content/uploads/2014/11/course_notes_moving_frostbite_to_pbr.pdf */
	float original_roughness = roughness;
	float linear_roughness = sqrt(roughness);
	float distance_roughness = saturate(dist * linear_roughness / length(intersection));
	linear_roughness = mix(distance_roughness, linear_roughness, linear_roughness);
	roughness = linear_roughness * linear_roughness;

	float fac = saturate(original_roughness * 2.0 - 1.0);
	return mix(intersection, spec_dir, fac * fac);
}

float probe_attenuation(vec3 W, ProbeData pd)
{
	vec3 localpos = (pd.influencemat * vec4(W, 1.0)).xyz;

	float fac;
	if (pd.p_atten_type == PROBE_ATTENUATION_BOX) {
		vec3 axes_fac = saturate(pd.p_atten_fac - pd.p_atten_fac * abs(localpos));
		fac = min_v3(axes_fac);
	}
	else {
		fac = saturate(pd.p_atten_fac - pd.p_atten_fac * length(localpos));
	}

	return fac;
}

float planar_attenuation(vec3 W, vec3 N, PlanarData pd)
{
	float fac;

	/* Normal Facing */
	fac = saturate(dot(pd.pl_normal, N) * pd.pl_facing_scale + pd.pl_facing_bias);

	/* Distance from plane */
	fac *= saturate(abs(dot(pd.pl_plane_eq, vec4(W, 1.0))) * pd.pl_fade_scale + pd.pl_fade_bias);

	/* Fancy fast clipping calculation */
	vec2 dist_to_clip;
	dist_to_clip.x = dot(pd.pl_clip_pos_x, W);
	dist_to_clip.y = dot(pd.pl_clip_pos_y, W);
	fac *= step(2.0, dot(step(pd.pl_clip_edges, dist_to_clip.xxyy), vec2(-1.0, 1.0).xyxy)); /* compare and add all tests */

	return fac;
}

float compute_occlusion(vec3 N, float micro_occlusion, vec2 randuv, out vec3 bent_normal)
{
#ifdef USE_AO /* Screen Space Occlusion */

	float macro_occlusion;
	vec3 vnor = mat3(ViewMatrix) * N;

#ifdef USE_BENT_NORMAL
	gtao(vnor, viewPosition, randuv, macro_occlusion, bent_normal);
	bent_normal = mat3(ViewMatrixInverse) * bent_normal;
#else
	gtao(vnor, viewPosition, randuv, macro_occlusion);
	bent_normal = N;
#endif
	return min(macro_occlusion, micro_occlusion);

#else /* No added Occlusion. */

	bent_normal = N;
	return micro_occlusion;

#endif
}

vec3 eevee_surface_lit(vec3 world_normal, vec3 albedo, vec3 f0, float roughness, float ao)
{
	roughness = clamp(roughness, 1e-8, 0.9999);
	float roughnessSquared = roughness * roughness;

	ShadingData sd;
	sd.N = normalize(world_normal);
	sd.V = (ProjectionMatrix[3][3] == 0.0) /* if perspective */
	            ? normalize(cameraPos - worldPosition)
	            : cameraForward;
	sd.W = worldPosition;

	vec3 radiance = vec3(0.0);

#ifdef HAIR_SHADER
       /* View facing normal */
       vec3 norm_view = cross(sd.V, sd.N);
       norm_view = normalize(cross(norm_view, sd.N)); /* Normal facing view */
#endif


	/* Analytic Lights */
	for (int i = 0; i < MAX_LIGHT && i < light_count; ++i) {
		LightData ld = lights_data[i];
		vec3 diff, spec;
		float vis = 1.0;

		sd.l_vector = ld.l_position - worldPosition;

#ifndef HAIR_SHADER
		light_visibility(ld, sd, vis);
#endif
		light_shade(ld, sd, albedo, roughnessSquared, f0, diff, spec);

		radiance += vis * (diff + spec) * ld.l_color;
	}

#ifdef HAIR_SHADER
	sd.N = -norm_view;
#endif

	vec3 bent_normal;
	vec4 rand = textureLod(utilTex, vec3(gl_FragCoord.xy / LUT_SIZE, 2.0), 0.0).rgba;
	float final_ao = compute_occlusion(sd.N, ao, rand.rg, bent_normal);

	/* Envmaps */
	vec3 R = reflect(-sd.V, sd.N);
	vec3 spec_dir = get_specular_dominant_dir(sd.N, R, roughnessSquared);
	vec2 uv = lut_coords(dot(sd.N, sd.V), roughness);
	vec2 brdf_lut = texture(utilTex, vec3(uv, 1.0)).rg;

	vec4 spec_accum = vec4(0.0);
	vec4 diff_accum = vec4(0.0);

	/* Planar Reflections */
	for (int i = 0; i < MAX_PLANAR && i < planar_count && spec_accum.a < 0.999; ++i) {
		PlanarData pd = planars_data[i];

		float influence = planar_attenuation(sd.W, sd.N, pd);

		if (influence > 0.0) {
			float influ_spec = min(influence, (1.0 - spec_accum.a));

			/* Sample reflection depth. */
			vec4 refco = pd.reflectionmat * vec4(sd.W, 1.0);
			refco.xy /= refco.w;
			float ref_depth = textureLod(probePlanars, vec3(refco.xy, i), 0.0).a;

			/* Find view vector / reflection plane intersection. (dist_to_plane is negative) */
			float dist_to_plane = line_plane_intersect_dist(cameraPos, sd.V, pd.pl_plane_eq);
			vec3 point_on_plane = cameraPos + sd.V * dist_to_plane;

			/* How far the pixel is from the plane. */
			ref_depth = ref_depth + dist_to_plane;

			/* Compute distorded reflection vector based on the distance to the reflected object.
			 * In other words find intersection between reflection vector and the sphere center
			 * around point_on_plane. */
			vec3 proj_ref = reflect(R * ref_depth, pd.pl_normal);

			/* Final point in world space. */
			vec3 ref_pos = point_on_plane + proj_ref;

			/* Reproject to find texture coords. */
			refco = pd.reflectionmat * vec4(ref_pos, 1.0);
			refco.xy /= refco.w;

			vec3 sample = textureLod(probePlanars, vec3(refco.xy, i), 0.0).rgb;

			spec_accum.rgb += sample * influ_spec;
			spec_accum.a += influ_spec;
		}
	}

	/* Specular probes */
	/* Start at 1 because 0 is world probe */
	for (int i = 1; i < MAX_PROBE && i < probe_count && spec_accum.a < 0.999; ++i) {
		ProbeData pd = probes_data[i];

		float dist_attenuation = probe_attenuation(sd.W, pd);

		if (dist_attenuation > 0.0) {
			float roughness_copy = roughness;

			vec3 sample_vec = probe_parallax_correction(sd.W, spec_dir, pd, roughness_copy);
			vec4 sample = textureLod_octahedron(probeCubes, vec4(sample_vec, i), roughness_copy * lodMax, lodMax).rgba;

			float influ_spec = min(dist_attenuation, (1.0 - spec_accum.a));

			spec_accum.rgb += sample.rgb * influ_spec;
			spec_accum.a += influ_spec;
		}
	}

	/* Start at 1 because 0 is world irradiance */
	for (int i = 1; i < MAX_GRID && i < grid_count && diff_accum.a < 0.999; ++i) {
		GridData gd = grids_data[i];

		vec3 localpos = (gd.localmat * vec4(sd.W, 1.0)).xyz;

		float fade = min(1.0, min_v3(1.0 - abs(localpos)));
		fade = saturate(fade * gd.g_atten_scale + gd.g_atten_bias);

		if (fade > 0.0) {
			localpos = localpos * 0.5 + 0.5;
			localpos = localpos * vec3(gd.g_resolution) - 0.5;

			vec3 localpos_floored = floor(localpos);
			vec3 trilinear_weight = fract(localpos);

			float weight_accum = 0.0;
			vec3 irradiance_accum = vec3(0.0);

			/* For each neighboor cells */
			for (int i = 0; i < 8; ++i) {
				ivec3 offset = ivec3(i, i >> 1, i >> 2) & ivec3(1);
				vec3 cell_cos = clamp(localpos_floored + vec3(offset), vec3(0.0), vec3(gd.g_resolution) - 1.0);

				/* We need this because we render probes in world space (so we need light vector in WS).
				 * And rendering them in local probe space is too much problem. */
				vec3 ws_cell_location = gd.g_corner +
					(gd.g_increment_x * cell_cos.x +
					 gd.g_increment_y * cell_cos.y +
					 gd.g_increment_z * cell_cos.z);
				vec3 ws_point_to_cell = ws_cell_location - sd.W;
				vec3 ws_light = normalize(ws_point_to_cell);

				vec3 trilinear = mix(1 - trilinear_weight, trilinear_weight, offset);
				float weight = trilinear.x * trilinear.y * trilinear.z;

				/* Smooth backface test */
				// weight *= sqrt(max(0.002, dot(ws_light, sd.N)));

				/* Avoid zero weight */
				weight = max(0.00001, weight);

				vec3 color = get_cell_color(ivec3(cell_cos), gd.g_resolution, gd.g_offset, bent_normal);

				weight_accum += weight;
				irradiance_accum += color * weight;
			}

			vec3 indirect_diffuse = irradiance_accum / weight_accum;

			float influ_diff = min(fade, (1.0 - diff_accum.a));

			diff_accum.rgb += indirect_diffuse * influ_diff;
			diff_accum.a += influ_diff;

			/* For Debug purpose */
			// return texture(irradianceGrid, sd.W.xy).rgb;
		}
	}

	/* World probe */
	if (diff_accum.a < 1.0 && grid_count > 0) {
		IrradianceData ir_data = load_irradiance_cell(0, bent_normal);

		vec3 diff = compute_irradiance(bent_normal, ir_data);
		diff_accum.rgb += diff * (1.0 - diff_accum.a);
	}

	if (spec_accum.a < 1.0) {
		ProbeData pd = probes_data[0];

		vec3 spec = textureLod_octahedron(probeCubes, vec4(spec_dir, 0), roughness * lodMax, lodMax).rgb;
		spec_accum.rgb += spec * (1.0 - spec_accum.a);
	}

	vec3 indirect_radiance =
	        spec_accum.rgb * F_ibl(f0, brdf_lut) * float(specToggle) * specular_occlusion(dot(sd.N, sd.V), final_ao, roughness) +
	        diff_accum.rgb * albedo * gtao_multibounce(final_ao, albedo);

	return radiance + indirect_radiance;
}

uniform mat4 ModelViewMatrix;
#ifndef EEVEE_ENGINE
uniform mat4 ProjectionMatrix;
uniform mat4 ViewMatrixInverse;
uniform mat4 ViewMatrix;
#endif
uniform mat4 ModelMatrix;
uniform mat4 ModelMatrixInverse;
uniform mat4 ModelViewMatrixInverse;
uniform mat4 ProjectionMatrixInverse;
uniform mat3 NormalMatrix;
uniform vec4 CameraTexCoFactors;

out vec4 fragColor;

/* Converters */

float convert_rgba_to_float(vec4 color)
{
#ifdef USE_NEW_SHADING
	return dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
#else
	return (color.r + color.g + color.b) * 0.333333;
#endif
}

float exp_blender(float f)
{
	return pow(2.71828182846, f);
}

float compatible_pow(float x, float y)
{
	if (y == 0.0) /* x^0 -> 1, including 0^0 */
		return 1.0;

	/* glsl pow doesn't accept negative x */
	if (x < 0.0) {
		if (mod(-y, 2.0) == 0.0)
			return pow(-x, y);
		else
			return -pow(-x, y);
	}
	else if (x == 0.0)
		return 0.0;

	return pow(x, y);
}

void rgb_to_hsv(vec4 rgb, out vec4 outcol)
{
	float cmax, cmin, h, s, v, cdelta;
	vec3 c;

	cmax = max(rgb[0], max(rgb[1], rgb[2]));
	cmin = min(rgb[0], min(rgb[1], rgb[2]));
	cdelta = cmax - cmin;

	v = cmax;
	if (cmax != 0.0)
		s = cdelta / cmax;
	else {
		s = 0.0;
		h = 0.0;
	}

	if (s == 0.0) {
		h = 0.0;
	}
	else {
		c = (vec3(cmax) - rgb.xyz) / cdelta;

		if (rgb.x == cmax) h = c[2] - c[1];
		else if (rgb.y == cmax) h = 2.0 + c[0] -  c[2];
		else h = 4.0 + c[1] - c[0];

		h /= 6.0;

		if (h < 0.0)
			h += 1.0;
	}

	outcol = vec4(h, s, v, rgb.w);
}

void hsv_to_rgb(vec4 hsv, out vec4 outcol)
{
	float i, f, p, q, t, h, s, v;
	vec3 rgb;

	h = hsv[0];
	s = hsv[1];
	v = hsv[2];

	if (s == 0.0) {
		rgb = vec3(v, v, v);
	}
	else {
		if (h == 1.0)
			h = 0.0;

		h *= 6.0;
		i = floor(h);
		f = h - i;
		rgb = vec3(f, f, f);
		p = v * (1.0 - s);
		q = v * (1.0 - (s * f));
		t = v * (1.0 - (s * (1.0 - f)));

		if (i == 0.0) rgb = vec3(v, t, p);
		else if (i == 1.0) rgb = vec3(q, v, p);
		else if (i == 2.0) rgb = vec3(p, v, t);
		else if (i == 3.0) rgb = vec3(p, q, v);
		else if (i == 4.0) rgb = vec3(t, p, v);
		else rgb = vec3(v, p, q);
	}

	outcol = vec4(rgb, hsv.w);
}

float srgb_to_linearrgb(float c)
{
	if (c < 0.04045)
		return (c < 0.0) ? 0.0 : c * (1.0 / 12.92);
	else
		return pow((c + 0.055) * (1.0 / 1.055), 2.4);
}

float linearrgb_to_srgb(float c)
{
	if (c < 0.0031308)
		return (c < 0.0) ? 0.0 : c * 12.92;
	else
		return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

void srgb_to_linearrgb(vec4 col_from, out vec4 col_to)
{
	col_to.r = srgb_to_linearrgb(col_from.r);
	col_to.g = srgb_to_linearrgb(col_from.g);
	col_to.b = srgb_to_linearrgb(col_from.b);
	col_to.a = col_from.a;
}

void linearrgb_to_srgb(vec4 col_from, out vec4 col_to)
{
	col_to.r = linearrgb_to_srgb(col_from.r);
	col_to.g = linearrgb_to_srgb(col_from.g);
	col_to.b = linearrgb_to_srgb(col_from.b);
	col_to.a = col_from.a;
}

void color_to_normal(vec3 color, out vec3 normal)
{
	normal.x =  2.0 * ((color.r) - 0.5);
	normal.y = -2.0 * ((color.g) - 0.5);
	normal.z =  2.0 * ((color.b) - 0.5);
}

void color_to_normal_new_shading(vec3 color, out vec3 normal)
{
	normal.x =  2.0 * ((color.r) - 0.5);
	normal.y =  2.0 * ((color.g) - 0.5);
	normal.z =  2.0 * ((color.b) - 0.5);
}

void color_to_blender_normal_new_shading(vec3 color, out vec3 normal)
{
	normal.x =  2.0 * ((color.r) - 0.5);
	normal.y = -2.0 * ((color.g) - 0.5);
	normal.z = -2.0 * ((color.b) - 0.5);
}
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538
#endif

/*********** SHADER NODES ***************/

void vcol_attribute(vec4 attvcol, out vec4 vcol)
{
	vcol = vec4(attvcol.xyz, 1.0);
}

void uv_attribute(vec2 attuv, out vec3 uv)
{
	uv = vec3(attuv * 2.0 - vec2(1.0, 1.0), 0.0);
}

void geom(
        vec3 co, vec3 nor, mat4 viewinvmat, vec3 attorco, vec2 attuv, vec4 attvcol,
        out vec3 global, out vec3 local, out vec3 view, out vec3 orco, out vec3 uv,
        out vec3 normal, out vec4 vcol, out float vcol_alpha, out float frontback)
{
	local = co;
	view = (ProjectionMatrix[3][3] == 0.0) ? normalize(local) : vec3(0.0, 0.0, -1.0);
	global = (viewinvmat * vec4(local, 1.0)).xyz;
	orco = attorco;
	uv_attribute(attuv, uv);
	normal = -normalize(nor);   /* blender render normal is negated */
	vcol_attribute(attvcol, vcol);
	srgb_to_linearrgb(vcol, vcol);
	vcol_alpha = attvcol.a;
	frontback = (gl_FrontFacing) ? 1.0 : 0.0;
}

void particle_info(
        vec4 sprops, vec3 loc, vec3 vel, vec3 avel,
        out float index, out float age, out float life_time, out vec3 location,
        out float size, out vec3 velocity, out vec3 angular_velocity)
{
	index = sprops.x;
	age = sprops.y;
	life_time = sprops.z;
	size = sprops.w;

	location = loc;
	velocity = vel;
	angular_velocity = avel;
}

void vect_normalize(vec3 vin, out vec3 vout)
{
	vout = normalize(vin);
}

void direction_transform_m4v3(vec3 vin, mat4 mat, out vec3 vout)
{
	vout = (mat * vec4(vin, 0.0)).xyz;
}

void point_transform_m4v3(vec3 vin, mat4 mat, out vec3 vout)
{
	vout = (mat * vec4(vin, 1.0)).xyz;
}

void point_texco_remap_square(vec3 vin, out vec3 vout)
{
	vout = vec3(vin - vec3(0.5, 0.5, 0.5)) * 2.0;
}

void point_map_to_sphere(vec3 vin, out vec3 vout)
{
	float len = length(vin);
	float v, u;
	if (len > 0.0) {
		if (vin.x == 0.0 && vin.y == 0.0)
			u = 0.0;
		else
			u = (1.0 - atan(vin.x, vin.y) / M_PI) / 2.0;

		v = 1.0 - acos(vin.z / len) / M_PI;
	}
	else
		v = u = 0.0;

	vout = vec3(u, v, 0.0);
}

void point_map_to_tube(vec3 vin, out vec3 vout)
{
	float u, v;
	v = (vin.z + 1.0) * 0.5;
	float len = sqrt(vin.x * vin.x + vin.y * vin[1]);
	if (len > 0.0)
		u = (1.0 - (atan(vin.x / len, vin.y / len) / M_PI)) * 0.5;
	else
		v = u = 0.0;

	vout = vec3(u, v, 0.0);
}

void mapping(vec3 vec, mat4 mat, vec3 minvec, vec3 maxvec, float domin, float domax, out vec3 outvec)
{
	outvec = (mat * vec4(vec, 1.0)).xyz;
	if (domin == 1.0)
		outvec = max(outvec, minvec);
	if (domax == 1.0)
		outvec = min(outvec, maxvec);
}

void camera(vec3 co, out vec3 outview, out float outdepth, out float outdist)
{
	outdepth = abs(co.z);
	outdist = length(co);
	outview = normalize(co);
}

void lamp(
        vec4 col, float energy, vec3 lv, float dist, vec3 shadow, float visifac,
        out vec4 outcol, out vec3 outlv, out float outdist, out vec4 outshadow, out float outvisifac)
{
	outcol = col * energy;
	outlv = lv;
	outdist = dist;
	outshadow = vec4(shadow, 1.0);
	outvisifac = visifac;
}

void math_add(float val1, float val2, out float outval)
{
	outval = val1 + val2;
}

void math_subtract(float val1, float val2, out float outval)
{
	outval = val1 - val2;
}

void math_multiply(float val1, float val2, out float outval)
{
	outval = val1 * val2;
}

void math_divide(float val1, float val2, out float outval)
{
	if (val2 == 0.0)
		outval = 0.0;
	else
		outval = val1 / val2;
}

void math_sine(float val, out float outval)
{
	outval = sin(val);
}

void math_cosine(float val, out float outval)
{
	outval = cos(val);
}

void math_tangent(float val, out float outval)
{
	outval = tan(val);
}

void math_asin(float val, out float outval)
{
	if (val <= 1.0 && val >= -1.0)
		outval = asin(val);
	else
		outval = 0.0;
}

void math_acos(float val, out float outval)
{
	if (val <= 1.0 && val >= -1.0)
		outval = acos(val);
	else
		outval = 0.0;
}

void math_atan(float val, out float outval)
{
	outval = atan(val);
}

void math_pow(float val1, float val2, out float outval)
{
	if (val1 >= 0.0) {
		outval = compatible_pow(val1, val2);
	}
	else {
		float val2_mod_1 = mod(abs(val2), 1.0);

		if (val2_mod_1 > 0.999 || val2_mod_1 < 0.001)
			outval = compatible_pow(val1, floor(val2 + 0.5));
		else
			outval = 0.0;
	}
}

void math_log(float val1, float val2, out float outval)
{
	if (val1 > 0.0  && val2 > 0.0)
		outval = log2(val1) / log2(val2);
	else
		outval = 0.0;
}

void math_max(float val1, float val2, out float outval)
{
	outval = max(val1, val2);
}

void math_min(float val1, float val2, out float outval)
{
	outval = min(val1, val2);
}

void math_round(float val, out float outval)
{
	outval = floor(val + 0.5);
}

void math_less_than(float val1, float val2, out float outval)
{
	if (val1 < val2)
		outval = 1.0;
	else
		outval = 0.0;
}

void math_greater_than(float val1, float val2, out float outval)
{
	if (val1 > val2)
		outval = 1.0;
	else
		outval = 0.0;
}

void math_modulo(float val1, float val2, out float outval)
{
	if (val2 == 0.0)
		outval = 0.0;
	else
		outval = mod(val1, val2);

	/* change sign to match C convention, mod in GLSL will take absolute for negative numbers,
	 * see https://www.opengl.org/sdk/docs/man/html/mod.xhtml */
	outval = (val1 > 0.0) ? outval : outval - val2;
}

void math_abs(float val1, out float outval)
{
	outval = abs(val1);
}

void squeeze(float val, float width, float center, out float outval)
{
	outval = 1.0 / (1.0 + pow(2.71828183, -((val - center) * width)));
}

void vec_math_add(vec3 v1, vec3 v2, out vec3 outvec, out float outval)
{
	outvec = v1 + v2;
	outval = (abs(outvec[0]) + abs(outvec[1]) + abs(outvec[2])) * 0.333333;
}

void vec_math_sub(vec3 v1, vec3 v2, out vec3 outvec, out float outval)
{
	outvec = v1 - v2;
	outval = (abs(outvec[0]) + abs(outvec[1]) + abs(outvec[2])) * 0.333333;
}

void vec_math_average(vec3 v1, vec3 v2, out vec3 outvec, out float outval)
{
	outvec = v1 + v2;
	outval = length(outvec);
	outvec = normalize(outvec);
}
void vec_math_mix(float strength, vec3 v1, vec3 v2, out vec3 outvec)
{
	outvec = strength * v1 + (1 - strength) * v2;
}

void vec_math_dot(vec3 v1, vec3 v2, out vec3 outvec, out float outval)
{
	outvec = vec3(0);
	outval = dot(v1, v2);
}

void vec_math_cross(vec3 v1, vec3 v2, out vec3 outvec, out float outval)
{
	outvec = cross(v1, v2);
	outval = length(outvec);
	outvec /= outval;
}

void vec_math_normalize(vec3 v, out vec3 outvec, out float outval)
{
	outval = length(v);
	outvec = normalize(v);
}

void vec_math_negate(vec3 v, out vec3 outv)
{
	outv = -v;
}

void invert_z(vec3 v, out vec3 outv)
{
	v.z = -v.z;
	outv = v;
}

void normal(vec3 dir, vec3 nor, out vec3 outnor, out float outdot)
{
	outnor = nor;
	outdot = -dot(dir, nor);
}

void normal_new_shading(vec3 dir, vec3 nor, out vec3 outnor, out float outdot)
{
	outnor = normalize(nor);
	outdot = dot(normalize(dir), nor);
}

void curves_vec(float fac, vec3 vec, sampler2D curvemap, out vec3 outvec)
{
	outvec.x = texture(curvemap, vec2((vec.x + 1.0) * 0.5, 0.0)).x;
	outvec.y = texture(curvemap, vec2((vec.y + 1.0) * 0.5, 0.0)).y;
	outvec.z = texture(curvemap, vec2((vec.z + 1.0) * 0.5, 0.0)).z;

	if (fac != 1.0)
		outvec = (outvec * fac) + (vec * (1.0 - fac));

}

void curves_rgb(float fac, vec4 col, sampler2D curvemap, out vec4 outcol)
{
	outcol.r = texture(curvemap, vec2(texture(curvemap, vec2(col.r, 0.0)).a, 0.0)).r;
	outcol.g = texture(curvemap, vec2(texture(curvemap, vec2(col.g, 0.0)).a, 0.0)).g;
	outcol.b = texture(curvemap, vec2(texture(curvemap, vec2(col.b, 0.0)).a, 0.0)).b;

	if (fac != 1.0)
		outcol = (outcol * fac) + (col * (1.0 - fac));

	outcol.a = col.a;
}

void set_value(float val, out float outval)
{
	outval = val;
}

void set_rgb(vec3 col, out vec3 outcol)
{
	outcol = col;
}

void set_rgba(vec4 col, out vec4 outcol)
{
	outcol = col;
}

void set_value_zero(out float outval)
{
	outval = 0.0;
}

void set_value_one(out float outval)
{
	outval = 1.0;
}

void set_rgb_zero(out vec3 outval)
{
	outval = vec3(0.0);
}

void set_rgb_one(out vec3 outval)
{
	outval = vec3(1.0);
}

void set_rgba_zero(out vec4 outval)
{
	outval = vec4(0.0);
}

void set_rgba_one(out vec4 outval)
{
	outval = vec4(1.0);
}

void brightness_contrast(vec4 col, float brightness, float contrast, out vec4 outcol)
{
	float a = 1.0 + contrast;
	float b = brightness - contrast * 0.5;

	outcol.r = max(a * col.r + b, 0.0);
	outcol.g = max(a * col.g + b, 0.0);
	outcol.b = max(a * col.b + b, 0.0);
	outcol.a = col.a;
}

void mix_blend(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = mix(col1, col2, fac);
	outcol.a = col1.a;
}

void mix_add(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = mix(col1, col1 + col2, fac);
	outcol.a = col1.a;
}

void mix_mult(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = mix(col1, col1 * col2, fac);
	outcol.a = col1.a;
}

void mix_screen(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = vec4(1.0) - (vec4(facm) + fac * (vec4(1.0) - col2)) * (vec4(1.0) - col1);
	outcol.a = col1.a;
}

void mix_overlay(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = col1;

	if (outcol.r < 0.5)
		outcol.r *= facm + 2.0 * fac * col2.r;
	else
		outcol.r = 1.0 - (facm + 2.0 * fac * (1.0 - col2.r)) * (1.0 - outcol.r);

	if (outcol.g < 0.5)
		outcol.g *= facm + 2.0 * fac * col2.g;
	else
		outcol.g = 1.0 - (facm + 2.0 * fac * (1.0 - col2.g)) * (1.0 - outcol.g);

	if (outcol.b < 0.5)
		outcol.b *= facm + 2.0 * fac * col2.b;
	else
		outcol.b = 1.0 - (facm + 2.0 * fac * (1.0 - col2.b)) * (1.0 - outcol.b);
}

void mix_sub(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = mix(col1, col1 - col2, fac);
	outcol.a = col1.a;
}

void mix_div(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = col1;

	if (col2.r != 0.0) outcol.r = facm * outcol.r + fac * outcol.r / col2.r;
	if (col2.g != 0.0) outcol.g = facm * outcol.g + fac * outcol.g / col2.g;
	if (col2.b != 0.0) outcol.b = facm * outcol.b + fac * outcol.b / col2.b;
}

void mix_diff(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = mix(col1, abs(col1 - col2), fac);
	outcol.a = col1.a;
}

void mix_dark(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol.rgb = min(col1.rgb, col2.rgb * fac);
	outcol.a = col1.a;
}

void mix_light(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol.rgb = max(col1.rgb, col2.rgb * fac);
	outcol.a = col1.a;
}

void mix_dodge(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	outcol = col1;

	if (outcol.r != 0.0) {
		float tmp = 1.0 - fac * col2.r;
		if (tmp <= 0.0)
			outcol.r = 1.0;
		else if ((tmp = outcol.r / tmp) > 1.0)
			outcol.r = 1.0;
		else
			outcol.r = tmp;
	}
	if (outcol.g != 0.0) {
		float tmp = 1.0 - fac * col2.g;
		if (tmp <= 0.0)
			outcol.g = 1.0;
		else if ((tmp = outcol.g / tmp) > 1.0)
			outcol.g = 1.0;
		else
			outcol.g = tmp;
	}
	if (outcol.b != 0.0) {
		float tmp = 1.0 - fac * col2.b;
		if (tmp <= 0.0)
			outcol.b = 1.0;
		else if ((tmp = outcol.b / tmp) > 1.0)
			outcol.b = 1.0;
		else
			outcol.b = tmp;
	}
}

void mix_burn(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float tmp, facm = 1.0 - fac;

	outcol = col1;

	tmp = facm + fac * col2.r;
	if (tmp <= 0.0)
		outcol.r = 0.0;
	else if ((tmp = (1.0 - (1.0 - outcol.r) / tmp)) < 0.0)
		outcol.r = 0.0;
	else if (tmp > 1.0)
		outcol.r = 1.0;
	else
		outcol.r = tmp;

	tmp = facm + fac * col2.g;
	if (tmp <= 0.0)
		outcol.g = 0.0;
	else if ((tmp = (1.0 - (1.0 - outcol.g) / tmp)) < 0.0)
		outcol.g = 0.0;
	else if (tmp > 1.0)
		outcol.g = 1.0;
	else
		outcol.g = tmp;

	tmp = facm + fac * col2.b;
	if (tmp <= 0.0)
		outcol.b = 0.0;
	else if ((tmp = (1.0 - (1.0 - outcol.b) / tmp)) < 0.0)
		outcol.b = 0.0;
	else if (tmp > 1.0)
		outcol.b = 1.0;
	else
		outcol.b = tmp;
}

void mix_hue(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = col1;

	vec4 hsv, hsv2, tmp;
	rgb_to_hsv(col2, hsv2);

	if (hsv2.y != 0.0) {
		rgb_to_hsv(outcol, hsv);
		hsv.x = hsv2.x;
		hsv_to_rgb(hsv, tmp);

		outcol = mix(outcol, tmp, fac);
		outcol.a = col1.a;
	}
}

void mix_sat(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = col1;

	vec4 hsv, hsv2;
	rgb_to_hsv(outcol, hsv);

	if (hsv.y != 0.0) {
		rgb_to_hsv(col2, hsv2);

		hsv.y = facm * hsv.y + fac * hsv2.y;
		hsv_to_rgb(hsv, outcol);
	}
}

void mix_val(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	vec4 hsv, hsv2;
	rgb_to_hsv(col1, hsv);
	rgb_to_hsv(col2, hsv2);

	hsv.z = facm * hsv.z + fac * hsv2.z;
	hsv_to_rgb(hsv, outcol);
}

void mix_color(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	outcol = col1;

	vec4 hsv, hsv2, tmp;
	rgb_to_hsv(col2, hsv2);

	if (hsv2.y != 0.0) {
		rgb_to_hsv(outcol, hsv);
		hsv.x = hsv2.x;
		hsv.y = hsv2.y;
		hsv_to_rgb(hsv, tmp);

		outcol = mix(outcol, tmp, fac);
		outcol.a = col1.a;
	}
}

void mix_soft(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);
	float facm = 1.0 - fac;

	vec4 one = vec4(1.0);
	vec4 scr = one - (one - col2) * (one - col1);
	outcol = facm * col1 + fac * ((one - col1) * col2 * col1 + col1 * scr);
}

void mix_linear(float fac, vec4 col1, vec4 col2, out vec4 outcol)
{
	fac = clamp(fac, 0.0, 1.0);

	outcol = col1 + fac * (2.0 * (col2 - vec4(0.5)));
}

void valtorgb(float fac, sampler2D colormap, out vec4 outcol, out float outalpha)
{
	outcol = texture(colormap, vec2(fac, 0.0));
	outalpha = outcol.a;
}

void rgbtobw(vec4 color, out float outval)
{
#ifdef USE_NEW_SHADING
	vec3 factors = vec3(0.2126, 0.7152, 0.0722);
#else
	vec3 factors = vec3(0.35, 0.45, 0.2); /* keep these factors in sync with texture.h:RGBTOBW */
#endif
	outval = dot(color.rgb, factors);
}

void invert(float fac, vec4 col, out vec4 outcol)
{
	outcol.xyz = mix(col.xyz, vec3(1.0) - col.xyz, fac);
	outcol.w = col.w;
}

void clamp_vec3(vec3 vec, vec3 min, vec3 max, out vec3 out_vec)
{
	out_vec = clamp(vec, min, max);
}

void clamp_val(float value, float min, float max, out float out_value)
{
	out_value = clamp(value, min, max);
}

void hue_sat(float hue, float sat, float value, float fac, vec4 col, out vec4 outcol)
{
	vec4 hsv;

	rgb_to_hsv(col, hsv);

	hsv[0] += (hue - 0.5);
	if (hsv[0] > 1.0) hsv[0] -= 1.0; else if (hsv[0] < 0.0) hsv[0] += 1.0;
	hsv[1] *= sat;
	if (hsv[1] > 1.0) hsv[1] = 1.0; else if (hsv[1] < 0.0) hsv[1] = 0.0;
	hsv[2] *= value;
	if (hsv[2] > 1.0) hsv[2] = 1.0; else if (hsv[2] < 0.0) hsv[2] = 0.0;

	hsv_to_rgb(hsv, outcol);

	outcol = mix(col, outcol, fac);
}

void separate_rgb(vec4 col, out float r, out float g, out float b)
{
	r = col.r;
	g = col.g;
	b = col.b;
}

void combine_rgb(float r, float g, float b, out vec4 col)
{
	col = vec4(r, g, b, 1.0);
}

void separate_xyz(vec3 vec, out float x, out float y, out float z)
{
	x = vec.r;
	y = vec.g;
	z = vec.b;
}

void combine_xyz(float x, float y, float z, out vec3 vec)
{
	vec = vec3(x, y, z);
}

void separate_hsv(vec4 col, out float h, out float s, out float v)
{
	vec4 hsv;

	rgb_to_hsv(col, hsv);
	h = hsv[0];
	s = hsv[1];
	v = hsv[2];
}

void combine_hsv(float h, float s, float v, out vec4 col)
{
	hsv_to_rgb(vec4(h, s, v, 1.0), col);
}

void output_node(vec4 rgb, float alpha, out vec4 outrgb)
{
	outrgb = vec4(rgb.rgb, alpha);
}

/*********** TEXTURES ***************/

void texture_flip_blend(vec3 vec, out vec3 outvec)
{
	outvec = vec.yxz;
}

void texture_blend_lin(vec3 vec, out float outval)
{
	outval = (1.0 + vec.x) * 0.5;
}

void texture_blend_quad(vec3 vec, out float outval)
{
	outval = max((1.0 + vec.x) * 0.5, 0.0);
	outval *= outval;
}

void texture_wood_sin(vec3 vec, out float value, out vec4 color, out vec3 normal)
{
	float a = sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z) * 20.0;
	float wi = 0.5 + 0.5 * sin(a);

	value = wi;
	color = vec4(wi, wi, wi, 1.0);
	normal = vec3(0.0);
}

void texture_image(vec3 vec, sampler2D ima, out float value, out vec4 color, out vec3 normal)
{
	color = texture(ima, (vec.xy + vec2(1.0)) * 0.5);
	value = color.a;

	normal.x = 2.0 * (color.r - 0.5);
	normal.y = 2.0 * (0.5 - color.g);
	normal.z = 2.0 * (color.b - 0.5);
}

/************* MTEX *****************/

void texco_orco(vec3 attorco, out vec3 orco)
{
	orco = attorco;
}

void texco_uv(vec2 attuv, out vec3 uv)
{
	/* disabled for now, works together with leaving out mtex_2d_mapping
	   uv = vec3(attuv * 2.0 - vec2(1.0), 0.0); */
	uv = vec3(attuv, 0.0);
}

void texco_norm(vec3 normal, out vec3 outnormal)
{
	/* corresponds to shi->orn, which is negated so cancels
	   out blender normal negation */
	outnormal = normalize(normal);
}

void texco_tangent(vec4 tangent, out vec3 outtangent)
{
	outtangent = normalize(tangent.xyz);
}

void texco_global(mat4 viewinvmat, vec3 co, out vec3 global)
{
	global = (viewinvmat * vec4(co, 1.0)).xyz;
}

void texco_object(mat4 viewinvmat, mat4 obinvmat, vec3 co, out vec3 object)
{
	object = (obinvmat * (viewinvmat * vec4(co, 1.0))).xyz;
}

void texco_refl(vec3 vn, vec3 view, out vec3 ref)
{
	ref = view - 2.0 * dot(vn, view) * vn;
}

void shade_norm(vec3 normal, out vec3 outnormal)
{
	/* blender render normal is negated */
	outnormal = -normalize(normal);
}

void mtex_mirror(vec3 tcol, vec4 refcol, float tin, float colmirfac, out vec4 outrefcol)
{
	outrefcol = mix(refcol, vec4(1.0, tcol), tin * colmirfac);
}

void mtex_rgb_blend(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	incol = fact * texcol + facm * outcol;
}

void mtex_rgb_mul(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	incol = (facm + fact * texcol) * outcol;
}

void mtex_rgb_screen(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	incol = vec3(1.0) - (vec3(facm) + fact * (vec3(1.0) - texcol)) * (vec3(1.0) - outcol);
}

void mtex_rgb_overlay(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	if (outcol.r < 0.5)
		incol.r = outcol.r * (facm + 2.0 * fact * texcol.r);
	else
		incol.r = 1.0 - (facm + 2.0 * fact * (1.0 - texcol.r)) * (1.0 - outcol.r);

	if (outcol.g < 0.5)
		incol.g = outcol.g * (facm + 2.0 * fact * texcol.g);
	else
		incol.g = 1.0 - (facm + 2.0 * fact * (1.0 - texcol.g)) * (1.0 - outcol.g);

	if (outcol.b < 0.5)
		incol.b = outcol.b * (facm + 2.0 * fact * texcol.b);
	else
		incol.b = 1.0 - (facm + 2.0 * fact * (1.0 - texcol.b)) * (1.0 - outcol.b);
}

void mtex_rgb_sub(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	incol = -fact * facg * texcol + outcol;
}

void mtex_rgb_add(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	incol = fact * facg * texcol + outcol;
}

void mtex_rgb_div(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	if (texcol.r != 0.0) incol.r = facm * outcol.r + fact * outcol.r / texcol.r;
	if (texcol.g != 0.0) incol.g = facm * outcol.g + fact * outcol.g / texcol.g;
	if (texcol.b != 0.0) incol.b = facm * outcol.b + fact * outcol.b / texcol.b;
}

void mtex_rgb_diff(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm;

	fact *= facg;
	facm = 1.0 - fact;

	incol = facm * outcol + fact * abs(texcol - outcol);
}

void mtex_rgb_dark(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm, col;

	fact *= facg;
	facm = 1.0 - fact;

	incol.r = min(outcol.r, texcol.r) * fact + outcol.r * facm;
	incol.g = min(outcol.g, texcol.g) * fact + outcol.g * facm;
	incol.b = min(outcol.b, texcol.b) * fact + outcol.b * facm;
}

void mtex_rgb_light(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	float facm, col;

	fact *= facg;

	col = fact * texcol.r;
	if (col > outcol.r) incol.r = col; else incol.r = outcol.r;
	col = fact * texcol.g;
	if (col > outcol.g) incol.g = col; else incol.g = outcol.g;
	col = fact * texcol.b;
	if (col > outcol.b) incol.b = col; else incol.b = outcol.b;
}

void mtex_rgb_hue(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	vec4 col;

	mix_hue(fact * facg, vec4(outcol, 1.0), vec4(texcol, 1.0), col);
	incol.rgb = col.rgb;
}

void mtex_rgb_sat(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	vec4 col;

	mix_sat(fact * facg, vec4(outcol, 1.0), vec4(texcol, 1.0), col);
	incol.rgb = col.rgb;
}

void mtex_rgb_val(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	vec4 col;

	mix_val(fact * facg, vec4(outcol, 1.0), vec4(texcol, 1.0), col);
	incol.rgb = col.rgb;
}

void mtex_rgb_color(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	vec4 col;

	mix_color(fact * facg, vec4(outcol, 1.0), vec4(texcol, 1.0), col);
	incol.rgb = col.rgb;
}

void mtex_rgb_soft(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	vec4 col;

	mix_soft(fact * facg, vec4(outcol, 1.0), vec4(texcol, 1.0), col);
	incol.rgb = col.rgb;
}

void mtex_rgb_linear(vec3 outcol, vec3 texcol, float fact, float facg, out vec3 incol)
{
	fact *= facg;

	if (texcol.r > 0.5)
		incol.r = outcol.r + fact * (2.0 * (texcol.r - 0.5));
	else
		incol.r = outcol.r + fact * (2.0 * (texcol.r) - 1.0);

	if (texcol.g > 0.5)
		incol.g = outcol.g + fact * (2.0 * (texcol.g - 0.5));
	else
		incol.g = outcol.g + fact * (2.0 * (texcol.g) - 1.0);

	if (texcol.b > 0.5)
		incol.b = outcol.b + fact * (2.0 * (texcol.b - 0.5));
	else
		incol.b = outcol.b + fact * (2.0 * (texcol.b) - 1.0);
}

void mtex_value_vars(inout float fact, float facg, out float facm)
{
	fact *= abs(facg);
	facm = 1.0 - fact;

	if (facg < 0.0) {
		float tmp = fact;
		fact = facm;
		facm = tmp;
	}
}

void mtex_value_blend(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	incol = fact * texcol + facm * outcol;
}

void mtex_value_mul(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	facm = 1.0 - facg;
	incol = (facm + fact * texcol) * outcol;
}

void mtex_value_screen(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	facm = 1.0 - facg;
	incol = 1.0 - (facm + fact * (1.0 - texcol)) * (1.0 - outcol);
}

void mtex_value_sub(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	fact = -fact;
	incol = fact * texcol + outcol;
}

void mtex_value_add(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	fact = fact;
	incol = fact * texcol + outcol;
}

void mtex_value_div(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	if (texcol != 0.0)
		incol = facm * outcol + fact * outcol / texcol;
	else
		incol = 0.0;
}

void mtex_value_diff(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	incol = facm * outcol + fact * abs(texcol - outcol);
}

void mtex_value_dark(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	incol = facm * outcol + fact * min(outcol, texcol);
}

void mtex_value_light(float outcol, float texcol, float fact, float facg, out float incol)
{
	float facm;
	mtex_value_vars(fact, facg, facm);

	float col = fact * texcol;
	if (col > outcol) incol = col; else incol = outcol;
}

void mtex_value_clamp_positive(float fac, out float outfac)
{
	outfac = max(fac, 0.0);
}

void mtex_value_clamp(float fac, out float outfac)
{
	outfac = clamp(fac, 0.0, 1.0);
}

void mtex_har_divide(float har, out float outhar)
{
	outhar = har / 128.0;
}

void mtex_har_multiply_clamp(float har, out float outhar)
{
	outhar = clamp(har * 128.0, 1.0, 511.0);
}

void mtex_alpha_from_col(vec4 col, out float alpha)
{
	alpha = col.a;
}

void mtex_alpha_to_col(vec4 col, float alpha, out vec4 outcol)
{
	outcol = vec4(col.rgb, alpha);
}

void mtex_alpha_multiply_value(vec4 col, float value, out vec4 outcol)
{
	outcol = vec4(col.rgb, col.a * value);
}

void mtex_rgbtoint(vec4 rgb, out float intensity)
{
	intensity = dot(vec3(0.35, 0.45, 0.2), rgb.rgb);
}

void mtex_value_invert(float invalue, out float outvalue)
{
	outvalue = 1.0 - invalue;
}

void mtex_rgb_invert(vec4 inrgb, out vec4 outrgb)
{
	outrgb = vec4(vec3(1.0) - inrgb.rgb, inrgb.a);
}

void mtex_value_stencil(float stencil, float intensity, out float outstencil, out float outintensity)
{
	float fact = intensity;
	outintensity = intensity * stencil;
	outstencil = stencil * fact;
}

void mtex_rgb_stencil(float stencil, vec4 rgb, out float outstencil, out vec4 outrgb)
{
	float fact = rgb.a;
	outrgb = vec4(rgb.rgb, rgb.a * stencil);
	outstencil = stencil * fact;
}

void mtex_mapping_ofs(vec3 texco, vec3 ofs, out vec3 outtexco)
{
	outtexco = texco + ofs;
}

void mtex_mapping_size(vec3 texco, vec3 size, out vec3 outtexco)
{
	outtexco = size * texco;
}

void mtex_2d_mapping(vec3 vec, out vec3 outvec)
{
	outvec = vec3(vec.xy * 0.5 + vec2(0.5), vec.z);
}

vec3 mtex_2d_mapping(vec3 vec)
{
	return vec3(vec.xy * 0.5 + vec2(0.5), vec.z);
}

void mtex_cube_map(vec3 co, samplerCube ima, out float value, out vec4 color)
{
	color = texture(ima, co);
	value = 1.0;
}

void mtex_cube_map_refl_from_refldir(
        samplerCube ima, vec3 reflecteddirection, out float value, out vec4 color)
{
        color = texture(ima, reflecteddirection);
        value = color.a;
}

void mtex_cube_map_refl(
        samplerCube ima, vec3 vp, vec3 vn, mat4 viewmatrixinverse, mat4 viewmatrix,
        out float value, out vec4 color)
{
	vec3 viewdirection = vec3(viewmatrixinverse * vec4(vp, 0.0));
	vec3 normaldirection = normalize(vec3(vec4(vn, 0.0) * viewmatrix));
	vec3 reflecteddirection = reflect(viewdirection, normaldirection);
	color = texture(ima, reflecteddirection);
	value = 1.0;
}

void mtex_image(vec3 texco, sampler2D ima, out float value, out vec4 color)
{
	color = texture(ima, texco.xy);
	value = 1.0;
}

void mtex_normal(vec3 texco, sampler2D ima, out vec3 normal)
{
	// The invert of the red channel is to make
	// the normal map compliant with the outside world.
	// It needs to be done because in Blender
	// the normal used points inward.
	// Should this ever change this negate must be removed.
	vec4 color = texture(ima, texco.xy);
	normal = 2.0 * (vec3(-color.r, color.g, color.b) - vec3(-0.5, 0.5, 0.5));
}

void mtex_bump_normals_init(vec3 vN, out vec3 vNorg, out vec3 vNacc, out float fPrevMagnitude)
{
	vNorg = vN;
	vNacc = vN;
	fPrevMagnitude = 1.0;
}

/** helper method to extract the upper left 3x3 matrix from a 4x4 matrix */
mat3 to_mat3(mat4 m4)
{
	mat3 m3;
	m3[0] = m4[0].xyz;
	m3[1] = m4[1].xyz;
	m3[2] = m4[2].xyz;
	return m3;
}

void mtex_bump_init_objspace(
        vec3 surf_pos, vec3 surf_norm,
        mat4 mView, mat4 mViewInv, mat4 mObj, mat4 mObjInv,
        float fPrevMagnitude_in, vec3 vNacc_in,
        out float fPrevMagnitude_out, out vec3 vNacc_out,
        out vec3 vR1, out vec3 vR2, out float fDet)
{
	mat3 obj2view = to_mat3(ModelViewMatrix);
	mat3 view2obj = to_mat3(ModelViewMatrixInverse);

	vec3 vSigmaS = view2obj * dFdx(surf_pos);
	vec3 vSigmaT = view2obj * dFdy(surf_pos);
	vec3 vN = normalize(surf_norm * obj2view);

	vR1 = cross(vSigmaT, vN);
	vR2 = cross(vN, vSigmaS);
	fDet = dot(vSigmaS, vR1);

	/* pretransform vNacc (in mtex_bump_apply) using the inverse transposed */
	vR1 = vR1 * view2obj;
	vR2 = vR2 * view2obj;
	vN = vN * view2obj;

	float fMagnitude = abs(fDet) * length(vN);
	vNacc_out = vNacc_in * (fMagnitude / fPrevMagnitude_in);
	fPrevMagnitude_out = fMagnitude;
}

void mtex_bump_init_texturespace(
        vec3 surf_pos, vec3 surf_norm,
        float fPrevMagnitude_in, vec3 vNacc_in,
        out float fPrevMagnitude_out, out vec3 vNacc_out,
        out vec3 vR1, out vec3 vR2, out float fDet)
{
	vec3 vSigmaS = dFdx(surf_pos);
	vec3 vSigmaT = dFdy(surf_pos);
	vec3 vN = surf_norm; /* normalized interpolated vertex normal */

	vR1 = normalize(cross(vSigmaT, vN));
	vR2 = normalize(cross(vN, vSigmaS));
	fDet = sign(dot(vSigmaS, vR1));

	float fMagnitude = abs(fDet);
	vNacc_out = vNacc_in * (fMagnitude / fPrevMagnitude_in);
	fPrevMagnitude_out = fMagnitude;
}

void mtex_bump_init_viewspace(
        vec3 surf_pos, vec3 surf_norm,
        float fPrevMagnitude_in, vec3 vNacc_in,
        out float fPrevMagnitude_out, out vec3 vNacc_out,
        out vec3 vR1, out vec3 vR2, out float fDet)
{
	vec3 vSigmaS = dFdx(surf_pos);
	vec3 vSigmaT = dFdy(surf_pos);
	vec3 vN = surf_norm; /* normalized interpolated vertex normal */

	vR1 = cross(vSigmaT, vN);
	vR2 = cross(vN, vSigmaS);
	fDet = dot(vSigmaS, vR1);

	float fMagnitude = abs(fDet);
	vNacc_out = vNacc_in * (fMagnitude / fPrevMagnitude_in);
	fPrevMagnitude_out = fMagnitude;
}

void mtex_bump_tap3(
        vec3 texco, sampler2D ima, float hScale,
        out float dBs, out float dBt)
{
	vec2 STll = texco.xy;
	vec2 STlr = texco.xy + dFdx(texco.xy);
	vec2 STul = texco.xy + dFdy(texco.xy);

	float Hll, Hlr, Hul;
	rgbtobw(texture(ima, STll), Hll);
	rgbtobw(texture(ima, STlr), Hlr);
	rgbtobw(texture(ima, STul), Hul);

	dBs = hScale * (Hlr - Hll);
	dBt = hScale * (Hul - Hll);
}

#ifdef BUMP_BICUBIC

void mtex_bump_bicubic(
        vec3 texco, sampler2D ima, float hScale,
        out float dBs, out float dBt )
{
	float Hl;
	float Hr;
	float Hd;
	float Hu;

	vec2 TexDx = dFdx(texco.xy);
	vec2 TexDy = dFdy(texco.xy);

	vec2 STl = texco.xy - 0.5 * TexDx;
	vec2 STr = texco.xy + 0.5 * TexDx;
	vec2 STd = texco.xy - 0.5 * TexDy;
	vec2 STu = texco.xy + 0.5 * TexDy;

	rgbtobw(texture(ima, STl), Hl);
	rgbtobw(texture(ima, STr), Hr);
	rgbtobw(texture(ima, STd), Hd);
	rgbtobw(texture(ima, STu), Hu);

	vec2 dHdxy = vec2(Hr - Hl, Hu - Hd);
	float fBlend = clamp(1.0 - textureQueryLOD(ima, texco.xy).x, 0.0, 1.0);
	if (fBlend != 0.0) {
		// the derivative of the bicubic sampling of level 0
		ivec2 vDim;
		vDim = textureSize(ima, 0);

		// taking the fract part of the texture coordinate is a hardcoded wrap mode.
		// this is acceptable as textures use wrap mode exclusively in 3D view elsewhere in blender.
		// this is done so that we can still get a valid texel with uvs outside the 0,1 range
		// by texelFetch below, as coordinates are clamped when using this function.
		vec2 fTexLoc = vDim * fract(texco.xy) - vec2(0.5, 0.5);
		ivec2 iTexLoc = ivec2(floor(fTexLoc));
		vec2 t = clamp(fTexLoc - iTexLoc, 0.0, 1.0);        // sat just to be pedantic

/*******************************************************************************************
 * This block will replace the one below when one channel textures are properly supported. *
 *******************************************************************************************
		vec4 vSamplesUL = textureGather(ima, (iTexLoc+ivec2(-1,-1) + vec2(0.5,0.5))/vDim);
		vec4 vSamplesUR = textureGather(ima, (iTexLoc+ivec2(1,-1) + vec2(0.5,0.5))/vDim);
		vec4 vSamplesLL = textureGather(ima, (iTexLoc+ivec2(-1,1) + vec2(0.5,0.5))/vDim);
		vec4 vSamplesLR = textureGather(ima, (iTexLoc+ivec2(1,1) + vec2(0.5,0.5))/vDim);

		mat4 H = mat4(vSamplesUL.w, vSamplesUL.x, vSamplesLL.w, vSamplesLL.x,
		            vSamplesUL.z, vSamplesUL.y, vSamplesLL.z, vSamplesLL.y,
		            vSamplesUR.w, vSamplesUR.x, vSamplesLR.w, vSamplesLR.x,
		            vSamplesUR.z, vSamplesUR.y, vSamplesLR.z, vSamplesLR.y);
 */
		ivec2 iTexLocMod = iTexLoc + ivec2(-1, -1);

		mat4 H;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				ivec2 iTexTmp = iTexLocMod + ivec2(i, j);

				// wrap texture coordinates manually for texelFetch to work on uvs oitside the 0,1 range.
				// this is guaranteed to work since we take the fractional part of the uv above.
				iTexTmp.x = (iTexTmp.x < 0) ? iTexTmp.x + vDim.x : ((iTexTmp.x >= vDim.x) ? iTexTmp.x - vDim.x : iTexTmp.x);
				iTexTmp.y = (iTexTmp.y < 0) ? iTexTmp.y + vDim.y : ((iTexTmp.y >= vDim.y) ? iTexTmp.y - vDim.y : iTexTmp.y);

				rgbtobw(texelFetch(ima, iTexTmp, 0), H[i][j]);
			}
		}

		float x = t.x, y = t.y;
		float x2 = x * x, x3 = x2 * x, y2 = y * y, y3 = y2 * y;

		vec4 X  = vec4(-0.5 * (x3 + x) + x2,    1.5 * x3 - 2.5 * x2 + 1, -1.5 * x3 + 2 * x2 + 0.5 * x, 0.5 * (x3 - x2));
		vec4 Y  = vec4(-0.5 * (y3 + y) + y2,    1.5 * y3 - 2.5 * y2 + 1, -1.5 * y3 + 2 * y2 + 0.5 * y, 0.5 * (y3 - y2));
		vec4 dX = vec4(-1.5 * x2 + 2 * x - 0.5, 4.5 * x2 - 5 * x,        -4.5 * x2 + 4 * x + 0.5,      1.5 * x2 - x);
		vec4 dY = vec4(-1.5 * y2 + 2 * y - 0.5, 4.5 * y2 - 5 * y,        -4.5 * y2 + 4 * y + 0.5,      1.5 * y2 - y);

		// complete derivative in normalized coordinates (mul by vDim)
		vec2 dHdST = vDim * vec2(dot(Y, H * dX), dot(dY, H * X));

		// transform derivative to screen-space
		vec2 dHdxy_bicubic = vec2(dHdST.x * TexDx.x + dHdST.y * TexDx.y,
		                          dHdST.x * TexDy.x + dHdST.y * TexDy.y);

		// blend between the two
		dHdxy = dHdxy * (1 - fBlend) + dHdxy_bicubic * fBlend;
	}

	dBs = hScale * dHdxy.x;
	dBt = hScale * dHdxy.y;
}

#endif

void mtex_bump_tap5(
        vec3 texco, sampler2D ima, float hScale,
        out float dBs, out float dBt)
{
	vec2 TexDx = dFdx(texco.xy);
	vec2 TexDy = dFdy(texco.xy);

	vec2 STc = texco.xy;
	vec2 STl = texco.xy - 0.5 * TexDx;
	vec2 STr = texco.xy + 0.5 * TexDx;
	vec2 STd = texco.xy - 0.5 * TexDy;
	vec2 STu = texco.xy + 0.5 * TexDy;

	float Hc, Hl, Hr, Hd, Hu;
	rgbtobw(texture(ima, STc), Hc);
	rgbtobw(texture(ima, STl), Hl);
	rgbtobw(texture(ima, STr), Hr);
	rgbtobw(texture(ima, STd), Hd);
	rgbtobw(texture(ima, STu), Hu);

	dBs = hScale * (Hr - Hl);
	dBt = hScale * (Hu - Hd);
}

void mtex_bump_deriv(
        vec3 texco, sampler2D ima, float ima_x, float ima_y, float hScale,
        out float dBs, out float dBt)
{
	float s = 1.0;      // negate this if flipped texture coordinate
	vec2 TexDx = dFdx(texco.xy);
	vec2 TexDy = dFdy(texco.xy);

	// this variant using a derivative map is described here
	// http://mmikkelsen3d.blogspot.com/2011/07/derivative-maps.html
	vec2 dim = vec2(ima_x, ima_y);
	vec2 dBduv = hScale * dim * (2.0 * texture(ima, texco.xy).xy - 1.0);

	dBs = dBduv.x * TexDx.x + s * dBduv.y * TexDx.y;
	dBt = dBduv.x * TexDy.x + s * dBduv.y * TexDy.y;
}

void mtex_bump_apply(
        float fDet, float dBs, float dBt, vec3 vR1, vec3 vR2, vec3 vNacc_in,
        out vec3 vNacc_out, out vec3 perturbed_norm)
{
	vec3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);

	vNacc_out = vNacc_in - vSurfGrad;
	perturbed_norm = normalize(vNacc_out);
}

void mtex_bump_apply_texspace(
        float fDet, float dBs, float dBt, vec3 vR1, vec3 vR2,
        sampler2D ima, vec3 texco, float ima_x, float ima_y, vec3 vNacc_in,
        out vec3 vNacc_out, out vec3 perturbed_norm)
{
	vec2 TexDx = dFdx(texco.xy);
	vec2 TexDy = dFdy(texco.xy);

	vec3 vSurfGrad = sign(fDet) * (
	        dBs / length(vec2(ima_x * TexDx.x, ima_y * TexDx.y)) * vR1 +
	        dBt / length(vec2(ima_x * TexDy.x, ima_y * TexDy.y)) * vR2);

	vNacc_out = vNacc_in - vSurfGrad;
	perturbed_norm = normalize(vNacc_out);
}

void mtex_negate_texnormal(vec3 normal, out vec3 outnormal)
{
	outnormal = vec3(-normal.x, -normal.y, normal.z);
}

void mtex_nspace_tangent(vec4 tangent, vec3 normal, vec3 texnormal, out vec3 outnormal)
{
	vec3 B = tangent.w * cross(normal, tangent.xyz);

	outnormal = texnormal.x * tangent.xyz + texnormal.y * B + texnormal.z * normal;
	outnormal = normalize(outnormal);
}

void mtex_nspace_world(mat4 viewmat, vec3 texnormal, out vec3 outnormal)
{
	outnormal = normalize((viewmat * vec4(texnormal, 0.0)).xyz);
}

void mtex_nspace_object(vec3 texnormal, out vec3 outnormal)
{
	outnormal = normalize(NormalMatrix * texnormal);
}

void mtex_blend_normal(float norfac, vec3 normal, vec3 newnormal, out vec3 outnormal)
{
	outnormal = (1.0 - norfac) * normal + norfac * newnormal;
	outnormal = normalize(outnormal);
}

/******* MATERIAL *********/

void lamp_visibility_sun_hemi(vec3 lampvec, out vec3 lv, out float dist, out float visifac)
{
	lv = lampvec;
	dist = 1.0;
	visifac = 1.0;
}

void lamp_visibility_other(vec3 co, vec3 lampco, out vec3 lv, out float dist, out float visifac)
{
	lv = co - lampco;
	dist = length(lv);
	lv = normalize(lv);
	visifac = 1.0;
}

void lamp_falloff_invlinear(float lampdist, float dist, out float visifac)
{
	visifac = lampdist / (lampdist + dist);
}

void lamp_falloff_invsquare(float lampdist, float dist, out float visifac)
{
	visifac = lampdist / (lampdist + dist * dist);
}

void lamp_falloff_sliders(float lampdist, float ld1, float ld2, float dist, out float visifac)
{
	float lampdistkw = lampdist * lampdist;

	visifac = lampdist / (lampdist + ld1 * dist);
	visifac *= lampdistkw / (lampdistkw + ld2 * dist * dist);
}

void lamp_falloff_invcoefficients(float coeff_const, float coeff_lin, float coeff_quad, float dist, out float visifac)
{
	vec3 coeff = vec3(coeff_const, coeff_lin, coeff_quad);
	vec3 d_coeff = vec3(1.0, dist, dist * dist);
	float visifac_r = dot(coeff, d_coeff);
	if (visifac_r > 0.0)
		visifac = 1.0 / visifac_r;
	else
		visifac = 0.0;
}

void lamp_falloff_curve(float lampdist, sampler2D curvemap, float dist, out float visifac)
{
	visifac = texture(curvemap, vec2(dist / lampdist, 0.0)).x;
}

void lamp_visibility_sphere(float lampdist, float dist, float visifac, out float outvisifac)
{
	float t = lampdist - dist;

	outvisifac = visifac * max(t, 0.0) / lampdist;
}

void lamp_visibility_spot_square(vec3 lampvec, mat4 lampimat, vec2 scale, vec3 lv, out float inpr)
{
	if (dot(lv, lampvec) > 0.0) {
		vec3 lvrot = (lampimat * vec4(lv, 0.0)).xyz;
		/* without clever non-uniform scale, we could do: */
		// float x = max(abs(lvrot.x / lvrot.z), abs(lvrot.y / lvrot.z));
		float x = max(abs((lvrot.x / scale.x) / lvrot.z), abs((lvrot.y / scale.y) / lvrot.z));

		inpr = 1.0 / sqrt(1.0 + x * x);
	}
	else
		inpr = 0.0;
}

void lamp_visibility_spot_circle(vec3 lampvec, mat4 lampimat, vec2 scale, vec3 lv, out float inpr)
{
	/* without clever non-uniform scale, we could do: */
	// inpr = dot(lv, lampvec);
	if (dot(lv, lampvec) > 0.0) {
		vec3 lvrot = (lampimat * vec4(lv, 0.0)).xyz;
		float x = abs(lvrot.x / lvrot.z);
		float y = abs(lvrot.y / lvrot.z);

		float ellipse = abs((x * x) / (scale.x * scale.x) + (y * y) / (scale.y * scale.y));

		inpr = 1.0 / sqrt(1.0 + ellipse);
	}
	else
		inpr = 0.0;
}

void lamp_visibility_spot(float spotsi, float spotbl, float inpr, float visifac, out float outvisifac)
{
	float t = spotsi;

	if (inpr <= t) {
		outvisifac = 0.0;
	}
	else {
		t = inpr - t;

		/* soft area */
		if (spotbl != 0.0)
			inpr *= smoothstep(0.0, 1.0, t / spotbl);

		outvisifac = visifac * inpr;
	}
}

void lamp_visibility_clamp(float visifac, out float outvisifac)
{
	outvisifac = (visifac < 0.001) ? 0.0 : visifac;
}

void world_paper_view(vec3 vec, out vec3 outvec)
{
	vec3 nvec = normalize(vec);
	outvec = (ProjectionMatrix[3][3] == 0.0) ? vec3(nvec.x, 0.0, nvec.y) : vec3(0.0, 0.0, -1.0);
}

void world_zen_mapping(vec3 view, float zenup, float zendown, out float zenfac)
{
	if (view.z >= 0.0)
		zenfac = zenup;
	else
		zenfac = zendown;
}

void world_blend_paper_real(vec3 vec, out float blend)
{
	blend = abs(vec.y);
}

void world_blend_paper(vec3 vec, out float blend)
{
	blend = (vec.y + 1.0) * 0.5;
}

void world_blend_real(vec3 vec, out float blend)
{
	blend = abs(normalize(vec).z);
}

void world_blend(vec3 vec, out float blend)
{
	blend = (normalize(vec).z + 1) * 0.5;
}

void shade_view(vec3 co, out vec3 view)
{
	/* handle perspective/orthographic */
	view = (ProjectionMatrix[3][3] == 0.0) ? normalize(co) : vec3(0.0, 0.0, -1.0);
}

void shade_tangent_v(vec3 lv, vec3 tang, out vec3 vn)
{
	vec3 c = cross(lv, tang);
	vec3 vnor = cross(c, tang);

	vn = -normalize(vnor);
}

void shade_inp(vec3 vn, vec3 lv, out float inp)
{
	inp = dot(vn, lv);
}

void shade_is_no_diffuse(out float is)
{
	is = 0.0;
}

void shade_is_hemi(float inp, out float is)
{
	is = 0.5 * inp + 0.5;
}

float area_lamp_energy(mat4 area, vec3 co, vec3 vn)
{
	vec3 vec[4], c[4];
	float rad[4], fac;

	vec[0] = normalize(co - area[0].xyz);
	vec[1] = normalize(co - area[1].xyz);
	vec[2] = normalize(co - area[2].xyz);
	vec[3] = normalize(co - area[3].xyz);

	c[0] = normalize(cross(vec[0], vec[1]));
	c[1] = normalize(cross(vec[1], vec[2]));
	c[2] = normalize(cross(vec[2], vec[3]));
	c[3] = normalize(cross(vec[3], vec[0]));

	rad[0] = acos(dot(vec[0], vec[1]));
	rad[1] = acos(dot(vec[1], vec[2]));
	rad[2] = acos(dot(vec[2], vec[3]));
	rad[3] = acos(dot(vec[3], vec[0]));

	fac =  rad[0] * dot(vn, c[0]);
	fac += rad[1] * dot(vn, c[1]);
	fac += rad[2] * dot(vn, c[2]);
	fac += rad[3] * dot(vn, c[3]);

	return max(fac, 0.0);
}

void shade_inp_area(
        vec3 position, vec3 lampco, vec3 lampvec, vec3 vn, mat4 area, float areasize, float k,
        out float inp)
{
	vec3 co = position;
	vec3 vec = co - lampco;

	if (dot(vec, lampvec) < 0.0) {
		inp = 0.0;
	}
	else {
		float intens = area_lamp_energy(area, co, vn);

		inp = pow(intens * areasize, k);
	}
}

void shade_diffuse_oren_nayer(float nl, vec3 n, vec3 l, vec3 v, float rough, out float is)
{
	vec3 h = normalize(v + l);
	float nh = max(dot(n, h), 0.0);
	float nv = max(dot(n, v), 0.0);
	float realnl = dot(n, l);

	if (realnl < 0.0) {
		is = 0.0;
	}
	else if (nl < 0.0) {
		is = 0.0;
	}
	else {
		float vh = max(dot(v, h), 0.0);
		float Lit_A = acos(realnl);
		float View_A = acos(nv);

		vec3 Lit_B = normalize(l - realnl * n);
		vec3 View_B = normalize(v - nv * n);

		float t = max(dot(Lit_B, View_B), 0.0);

		float a, b;

		if (Lit_A > View_A) {
			a = Lit_A;
			b = View_A;
		}
		else {
			a = View_A;
			b = Lit_A;
		}

		float A = 1.0 - (0.5 * ((rough * rough) / ((rough * rough) + 0.33)));
		float B = 0.45 * ((rough * rough) / ((rough * rough) + 0.09));

		b *= 0.95;
		is = nl * (A + (B * t * sin(a) * tan(b)));
	}
}

void shade_diffuse_toon(vec3 n, vec3 l, vec3 v, float size, float tsmooth, out float is)
{
	float rslt = dot(n, l);
	float ang = acos(rslt);

	if (ang < size) is = 1.0;
	else if (ang > (size + tsmooth) || tsmooth == 0.0) is = 0.0;
	else is = 1.0 - ((ang - size) / tsmooth);
}

void shade_diffuse_minnaert(float nl, vec3 n, vec3 v, float darkness, out float is)
{
	if (nl <= 0.0) {
		is = 0.0;
	}
	else {
		float nv = max(dot(n, v), 0.0);

		if (darkness <= 1.0)
			is = nl * pow(max(nv * nl, 0.1), darkness - 1.0);
		else
			is = nl * pow(1.0001 - nv, darkness - 1.0);
	}
}

float fresnel_fac(vec3 view, vec3 vn, float grad, float fac)
{
	float t1, t2;
	float ffac;

	if (fac == 0.0) {
		ffac = 1.0;
	}
	else {
		t1 = dot(view, vn);
		if (t1 > 0.0) t2 = 1.0 + t1;
		else t2 = 1.0 - t1;

		t2 = grad + (1.0 - grad) * pow(t2, fac);

		if (t2 < 0.0) ffac = 0.0;
		else if (t2 > 1.0) ffac = 1.0;
		else ffac = t2;
	}

	return ffac;
}

void shade_diffuse_fresnel(vec3 vn, vec3 lv, vec3 view, float fac_i, float fac, out float is)
{
	is = fresnel_fac(lv, vn, fac_i, fac);
}

void shade_cubic(float is, out float outis)
{
	if (is > 0.0 && is < 1.0)
		outis = smoothstep(0.0, 1.0, is);
	else
		outis = is;
}

void shade_visifac(float i, float visifac, float refl, out float outi)
{
	/*if (i > 0.0)*/
	outi = max(i * visifac * refl, 0.0);
	/*else
	    outi = i;*/
}

void shade_tangent_v_spec(vec3 tang, out vec3 vn)
{
	vn = tang;
}

void shade_add_to_diffuse(float i, vec3 lampcol, vec3 col, out vec3 outcol)
{
	if (i > 0.0)
		outcol = i * lampcol * col;
	else
		outcol = vec3(0.0);
}

void shade_hemi_spec(vec3 vn, vec3 lv, vec3 view, float spec, float hard, float visifac, out float t)
{
	lv += view;
	lv = normalize(lv);

	t = dot(vn, lv);
	t = 0.5 * t + 0.5;

	t = visifac * spec * pow(t, hard);
}

void shade_phong_spec(vec3 n, vec3 l, vec3 v, float hard, out float specfac)
{
	vec3 h = normalize(l + v);
	float rslt = max(dot(h, n), 0.0);

	specfac = pow(rslt, hard);
}

void shade_cooktorr_spec(vec3 n, vec3 l, vec3 v, float hard, out float specfac)
{
	vec3 h = normalize(v + l);
	float nh = dot(n, h);

	if (nh < 0.0) {
		specfac = 0.0;
	}
	else {
		float nv = max(dot(n, v), 0.0);
		float i = pow(nh, hard);

		i = i / (0.1 + nv);
		specfac = i;
	}
}

void shade_blinn_spec(vec3 n, vec3 l, vec3 v, float refrac, float spec_power, out float specfac)
{
	if (refrac < 1.0) {
		specfac = 0.0;
	}
	else if (spec_power == 0.0) {
		specfac = 0.0;
	}
	else {
		if (spec_power < 100.0)
			spec_power = sqrt(1.0 / spec_power);
		else
			spec_power = 10.0 / spec_power;

		vec3 h = normalize(v + l);
		float nh = dot(n, h);
		if (nh < 0.0) {
			specfac = 0.0;
		}
		else {
			float nv = max(dot(n, v), 0.01);
			float nl = dot(n, l);
			if (nl <= 0.01) {
				specfac = 0.0;
			}
			else {
				float vh = max(dot(v, h), 0.01);

				float a = 1.0;
				float b = (2.0 * nh * nv) / vh;
				float c = (2.0 * nh * nl) / vh;

				float g = 0.0;

				if (a < b && a < c) g = a;
				else if (b < a && b < c) g = b;
				else if (c < a && c < b) g = c;

				float p = sqrt(((refrac * refrac) + (vh * vh) - 1.0));
				float f = ((((p - vh) * (p - vh)) / ((p + vh) * (p + vh))) *
				           (1.0 + ((((vh * (p + vh)) - 1.0) * ((vh * (p + vh)) - 1.0)) /
				                   (((vh * (p - vh)) + 1.0) * ((vh * (p - vh)) + 1.0)))));
				float ang = acos(nh);

				specfac = max(f * g * exp_blender((-(ang * ang) / (2.0 * spec_power * spec_power))), 0.0);
			}
		}
	}
}

void shade_wardiso_spec(vec3 n, vec3 l, vec3 v, float rms, out float specfac)
{
	vec3 h = normalize(l + v);
	float nh = max(dot(n, h), 0.001);
	float nv = max(dot(n, v), 0.001);
	float nl = max(dot(n, l), 0.001);
	float angle = tan(acos(nh));
	float alpha = max(rms, 0.001);

	specfac = nl * (1.0 / (4.0 * M_PI * alpha * alpha)) * (exp_blender(-(angle * angle) / (alpha * alpha)) / (sqrt(nv * nl)));
}

void shade_toon_spec(vec3 n, vec3 l, vec3 v, float size, float tsmooth, out float specfac)
{
	vec3 h = normalize(l + v);
	float rslt = dot(h, n);
	float ang = acos(rslt);

	if (ang < size) rslt = 1.0;
	else if (ang >= (size + tsmooth) || tsmooth == 0.0) rslt = 0.0;
	else rslt = 1.0 - ((ang - size) / tsmooth);

	specfac = rslt;
}

void shade_spec_area_inp(float specfac, float inp, out float outspecfac)
{
	outspecfac = specfac * inp;
}

void shade_spec_t(float shadfac, float spec, float visifac, float specfac, out float t)
{
	t = shadfac * spec * visifac * specfac;
}

void shade_add_spec(float t, vec3 lampcol, vec3 speccol, out vec3 outcol)
{
	outcol = t * lampcol * speccol;
}

void shade_add_mirror(vec3 mir, vec4 refcol, vec3 combined, out vec3 result)
{
	result = mir * refcol.gba + (vec3(1.0) - mir * refcol.rrr) * combined;
}

void alpha_spec_correction(vec3 spec, float spectra, float alpha, out float outalpha)
{
	if (spectra > 0.0) {
		float t = clamp(max(max(spec.r, spec.g), spec.b) * spectra, 0.0, 1.0);
		outalpha = (1.0 - t) * alpha + t;
	}
	else {
		outalpha = alpha;
	}
}

void shade_add(vec4 col1, vec4 col2, out vec4 outcol)
{
	outcol = col1 + col2;
}

void shade_madd(vec4 col, vec4 col1, vec4 col2, out vec4 outcol)
{
	outcol = col + col1 * col2;
}

void shade_add_clamped(vec4 col1, vec4 col2, out vec4 outcol)
{
	outcol = col1 + max(col2, vec4(0.0));
}

void shade_madd_clamped(vec4 col, vec4 col1, vec4 col2, out vec4 outcol)
{
	outcol = col + max(col1 * col2, vec4(0.0));
}

void env_apply(vec4 col, vec3 hor, vec3 zen, vec4 f, mat4 vm, vec3 vn, out vec4 outcol)
{
	vec3 vv = normalize(vm[2].xyz);
	float skyfac = 0.5 * (1.0 + dot(vn, -vv));
	outcol = col + f * vec4(mix(hor, zen, skyfac), 0);
}

void shade_maddf(vec4 col, float f, vec4 col1, out vec4 outcol)
{
	outcol = col + f * col1;
}

void shade_mul(vec4 col1, vec4 col2, out vec4 outcol)
{
	outcol = col1 * col2;
}

void shade_mul_value(float fac, vec4 col, out vec4 outcol)
{
	outcol = col * fac;
}

void shade_mul_value_v3(float fac, vec3 col, out vec3 outcol)
{
	outcol = col * fac;
}

void shade_obcolor(vec4 col, vec4 obcol, out vec4 outcol)
{
	outcol = vec4(col.rgb * obcol.rgb, col.a);
}

void ramp_rgbtobw(vec3 color, out float outval)
{
	outval = dot(color, vec3(0.3, 0.58, 0.12));
}

void shade_only_shadow(float i, float shadfac, float energy, vec3 shadcol, out vec3 outshadrgb)
{
	outshadrgb = i * energy * (1.0 - shadfac) * (vec3(1.0) - shadcol);
}

void shade_only_shadow_diffuse(vec3 shadrgb, vec3 rgb, vec4 diff, out vec4 outdiff)
{
	outdiff = diff - vec4(rgb * shadrgb, 0.0);
}

void shade_only_shadow_specular(vec3 shadrgb, vec3 specrgb, vec4 spec, out vec4 outspec)
{
	outspec = spec - vec4(specrgb * shadrgb, 0.0);
}

void shade_clamp_positive(vec4 col, out vec4 outcol)
{
	outcol = max(col, vec4(0.0));
}

void test_shadowbuf(
        vec3 rco, sampler2DShadow shadowmap, mat4 shadowpersmat, float shadowbias, float inp,
        out float result)
{
	if (inp <= 0.0) {
		result = 0.0;
	}
	else {
		vec4 co = shadowpersmat * vec4(rco, 1.0);

		//float bias = (1.5 - inp*inp)*shadowbias;
		co.z -= shadowbias * co.w;

		if (co.w > 0.0 && co.x > 0.0 && co.x / co.w < 1.0 && co.y > 0.0 && co.y / co.w < 1.0) {
			result = textureProj(shadowmap, co);
		}
		else {
			result = 1.0;
		}
	}
}

void test_shadowbuf_vsm(
        vec3 rco, sampler2D shadowmap, mat4 shadowpersmat, float shadowbias, float bleedbias, float inp,
        out float result)
{
	if (inp <= 0.0) {
		result = 0.0;
	}
	else {
		vec4 co = shadowpersmat * vec4(rco, 1.0);
		if (co.w > 0.0 && co.x > 0.0 && co.x / co.w < 1.0 && co.y > 0.0 && co.y / co.w < 1.0) {
			vec2 moments = textureProj(shadowmap, co).rg;
			float dist = co.z / co.w;
			float p = 0.0;

			if (dist <= moments.x)
				p = 1.0;

			float variance = moments.y - (moments.x * moments.x);
			variance = max(variance, shadowbias / 10.0);

			float d = moments.x - dist;
			float p_max = variance / (variance + d * d);

			// Now reduce light-bleeding by removing the [0, x] tail and linearly rescaling (x, 1]
			p_max = clamp((p_max - bleedbias) / (1.0 - bleedbias), 0.0, 1.0);

			result = max(p, p_max);
		}
		else {
			result = 1.0;
		}
	}
}

void shadows_only(
        vec3 rco, sampler2DShadow shadowmap, mat4 shadowpersmat,
        float shadowbias, vec3 shadowcolor, float inp,
        out vec3 result)
{
	result = vec3(1.0);

	if (inp > 0.0) {
		float shadfac;

		test_shadowbuf(rco, shadowmap, shadowpersmat, shadowbias, inp, shadfac);
		result -= (1.0 - shadfac) * (vec3(1.0) - shadowcolor);
	}
}

void shadows_only_vsm(
        vec3 rco, sampler2D shadowmap, mat4 shadowpersmat,
        float shadowbias, float bleedbias, vec3 shadowcolor, float inp,
        out vec3 result)
{
	result = vec3(1.0);

	if (inp > 0.0) {
		float shadfac;

		test_shadowbuf_vsm(rco, shadowmap, shadowpersmat, shadowbias, bleedbias, inp, shadfac);
		result -= (1.0 - shadfac) * (vec3(1.0) - shadowcolor);
	}
}

void shade_light_texture(vec3 rco, sampler2D cookie, mat4 shadowpersmat, out vec4 result)
{

	vec4 co = shadowpersmat * vec4(rco, 1.0);

	result = textureProj(cookie, co);
}

void shade_exposure_correct(vec3 col, float linfac, float logfac, out vec3 outcol)
{
	outcol = linfac * (1.0 - exp(col * logfac));
}

void shade_mist_factor(
        vec3 co, float enable, float miststa, float mistdist, float misttype, float misi,
        out float outfac)
{
	if (enable == 1.0) {
		float fac, zcor;

		zcor = (ProjectionMatrix[3][3] == 0.0) ? length(co) : -co[2];

		fac = clamp((zcor - miststa) / mistdist, 0.0, 1.0);
		if (misttype == 0.0) fac *= fac;
		else if (misttype == 1.0) ;
		else fac = sqrt(fac);

		outfac = 1.0 - (1.0 - fac) * (1.0 - misi);
	}
	else {
		outfac = 0.0;
	}
}

void shade_world_mix(vec3 hor, vec4 col, out vec4 outcol)
{
	float fac = clamp(col.a, 0.0, 1.0);
	outcol = vec4(mix(hor, col.rgb, fac), col.a);
}

void shade_alpha_opaque(vec4 col, out vec4 outcol)
{
	outcol = vec4(col.rgb, 1.0);
}

void shade_alpha_obcolor(vec4 col, vec4 obcol, out vec4 outcol)
{
	outcol = vec4(col.rgb, col.a * obcol.a);
}

/*********** NEW SHADER UTILITIES **************/

float fresnel_dielectric_0(float eta)
{
	/* compute fresnel reflactance at normal incidence => cosi = 1.0 */
	float A = (eta - 1.0) / (eta + 1.0);

	return A * A;
}

float fresnel_dielectric_cos(float cosi, float eta)
{
	/* compute fresnel reflectance without explicitly computing
	 * the refracted direction */
	float c = abs(cosi);
	float g = eta * eta - 1.0 + c * c;
	float result;

	if (g > 0.0) {
		g = sqrt(g);
		float A = (g - c) / (g + c);
		float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
		result = 0.5 * A * A * (1.0 + B * B);
	}
	else {
		result = 1.0;  /* TIR (no refracted component) */
	}

	return result;
}

float fresnel_dielectric(vec3 Incoming, vec3 Normal, float eta)
{
	/* compute fresnel reflectance without explicitly computing
	 * the refracted direction */
	return fresnel_dielectric_cos(dot(Incoming, Normal), eta);
}

float hypot(float x, float y)
{
	return sqrt(x * x + y * y);
}

void generated_from_orco(vec3 orco, out vec3 generated)
{
	generated = orco;
}

int floor_to_int(float x)
{
	return int(floor(x));
}

int quick_floor(float x)
{
	return int(x) - ((x < 0) ? 1 : 0);
}

float integer_noise(int n)
{
	int nn;
	n = (n + 1013) & 0x7fffffff;
	n = (n >> 13) ^ n;
	nn = (n * (n * n * 60493 + 19990303) + 1376312589) & 0x7fffffff;
	return 0.5 * (float(nn) / 1073741824.0);
}

uint hash(uint kx, uint ky, uint kz)
{
#define rot(x, k) (((x) << (k)) | ((x) >> (32 - (k))))
#define final(a, b, c) \
{ \
	c ^= b; c -= rot(b, 14); \
	a ^= c; a -= rot(c, 11); \
	b ^= a; b -= rot(a, 25); \
	c ^= b; c -= rot(b, 16); \
	a ^= c; a -= rot(c, 4);  \
	b ^= a; b -= rot(a, 14); \
	c ^= b; c -= rot(b, 24); \
}
	// now hash the data!
	uint a, b, c, len = 3u;
	a = b = c = 0xdeadbeefu + (len << 2u) + 13u;

	c += kz;
	b += ky;
	a += kx;
	final (a, b, c);

	return c;
#undef rot
#undef final
}

uint hash(int kx, int ky, int kz)
{
	return hash(uint(kx), uint(ky), uint(kz));
}

float bits_to_01(uint bits)
{
	float x = float(bits) * (1.0 / float(0xffffffffu));
	return x;
}

float cellnoise(vec3 p)
{
	int ix = quick_floor(p.x);
	int iy = quick_floor(p.y);
	int iz = quick_floor(p.z);

	return bits_to_01(hash(uint(ix), uint(iy), uint(iz)));
}

vec3 cellnoise_color(vec3 p)
{
	float r = cellnoise(p);
	float g = cellnoise(vec3(p.y, p.x, p.z));
	float b = cellnoise(vec3(p.y, p.z, p.x));

	return vec3(r, g, b);
}

float floorfrac(float x, out int i)
{
	i = floor_to_int(x);
	return x - i;
}


/* Principled BSDF operations */

float sqr(float a)
{
	return a*a;
}

float schlick_fresnel(float u)
{
	float m = clamp(1.0 - u, 0.0, 1.0);
	float m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

float GTR1(float NdotH, float a)
{
	if (a >= 1.0) return M_1_PI;
	float a2 = a*a;
	float t = 1.0 + (a2 - 1.0) * NdotH*NdotH;
	return (a2 - 1.0) / (M_PI * log(a2) * t);
}

float GTR2(float NdotH, float a)
{
	float a2 = a*a;
	float t = 1.0 + (a2 - 1.0) * NdotH*NdotH;
	return a2 / (M_PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
	return 1.0 / (M_PI * ax*ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH*NdotH));
}

float smithG_GGX(float NdotV, float alphaG)
{
	float a = alphaG*alphaG;
	float b = NdotV*NdotV;
	return 1.0 / (NdotV + sqrt(a + b - a * b));
}

vec3 rotate_vector(vec3 p, vec3 n, float theta) {
	return (
	           p * cos(theta) + cross(n, p) *
	           sin(theta) + n * dot(p, n) *
	           (1.0 - cos(theta))
	       );
}


/*********** NEW SHADER NODES ***************/

#define NUM_LIGHTS 3

struct glLight {
	vec4 position;
	vec4 diffuse;
	vec4 specular;
	vec4 halfVector;
};

layout(std140) uniform lightSource {
	glLight glLightSource[NUM_LIGHTS];
};

/* bsdfs */

void node_bsdf_diffuse(vec4 color, float roughness, vec3 N, out vec4 result)
{
	/* ambient light */
	vec3 L = vec3(0.2);

	/* directional lights */
	for (int i = 0; i < NUM_LIGHTS; i++) {
		vec3 light_position = glLightSource[i].position.xyz;
		vec3 light_diffuse = glLightSource[i].diffuse.rgb;

		float bsdf = max(dot(N, light_position), 0.0);
		L += light_diffuse * bsdf;
	}

	result = vec4(L * color.rgb, 1.0);
}

void node_bsdf_glossy(vec4 color, float roughness, vec3 N, out vec4 result)
{
	/* ambient light */
	vec3 L = vec3(0.2);

	/* directional lights */
	for (int i = 0; i < NUM_LIGHTS; i++) {
		vec3 light_position = glLightSource[i].position.xyz;
		vec3 H = glLightSource[i].halfVector.xyz;
		vec3 light_diffuse = glLightSource[i].diffuse.rgb;
		vec3 light_specular = glLightSource[i].specular.rgb;

		/* we mix in some diffuse so low roughness still shows up */
		float bsdf = 0.5 * pow(max(dot(N, H), 0.0), 1.0 / roughness);
		bsdf += 0.5 * max(dot(N, light_position), 0.0);
		L += light_specular * bsdf;
	}

	result = vec4(L * color.rgb, 1.0);
}

void node_bsdf_anisotropic(
        vec4 color, float roughness, float anisotropy, float rotation, vec3 N, vec3 T,
        out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_bsdf_glass(vec4 color, float roughness, float ior, vec3 N, out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_bsdf_toon(vec4 color, float size, float tsmooth, vec3 N, out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_bsdf_principled(vec4 base_color, float subsurface, vec3 subsurface_radius, vec4 subsurface_color, float metallic, float specular,
	float specular_tint, float roughness, float anisotropic, float anisotropic_rotation, float sheen, float sheen_tint, float clearcoat,
	float clearcoat_gloss, float ior, float transmission, float transmission_roughness, vec3 N, vec3 CN, vec3 T, vec3 I, out vec4 result)
{
	/* ambient light */
	// TODO: set ambient light to an appropriate value
	vec3 L = vec3(mix(0.1, 0.03, metallic)) * base_color.rgb;

	float eta = (2.0 / (1.0 - sqrt(0.08 * specular))) - 1.0;

	/* set the viewing vector */
	vec3 V = -normalize(I);

	/* get the tangent */
	vec3 Tangent = T;
	if (T == vec3(0.0)) {
		// if no tangent is set, use a default tangent
		Tangent = vec3(1.0, 0.0, 0.0);
		if (N.x != 0.0 || N.y != 0.0) {
			vec3 N_xz = normalize(vec3(N.x, 0.0, N.z));

			vec3 axis = normalize(cross(vec3(0.0, 0.0, 1.0), N_xz));
			float angle = acos(dot(vec3(0.0, 0.0, 1.0), N_xz));

			Tangent = normalize(rotate_vector(vec3(1.0, 0.0, 0.0), axis, angle));
		}
	}

	/* rotate tangent */
	if (anisotropic_rotation != 0.0) {
		Tangent = rotate_vector(Tangent, N, anisotropic_rotation * 2.0 * M_PI);
	}

	/* calculate the tangent and bitangent */
	vec3 Y = normalize(cross(N, Tangent));
	vec3 X = cross(Y, N);

	/* fresnel normalization parameters */
	float F0 = fresnel_dielectric_0(eta);
	float F0_norm = 1.0 / (1.0 - F0);

	/* directional lights */
	for (int i = 0; i < NUM_LIGHTS; i++) {
		vec3 light_position_world = glLightSource[i].position.xyz;
		vec3 light_position = normalize(NormalMatrix * light_position_world);

		vec3 H = normalize(light_position + V);

		vec3 light_specular = glLightSource[i].specular.rgb;

		float NdotL = dot(N, light_position);
		float NdotV = dot(N, V);
		float LdotH = dot(light_position, H);

		vec3 diffuse_and_specular_bsdf = vec3(0.0);
		if (NdotL >= 0.0 && NdotV >= 0.0) {
			float NdotH = dot(N, H);

			float Cdlum = dot(base_color.rgb, vec3(0.3, 0.6, 0.1)); // luminance approx.

			vec3 Ctint = Cdlum > 0 ? base_color.rgb / Cdlum : vec3(1.0); // normalize lum. to isolate hue+sat
			vec3 Cspec0 = mix(specular * 0.08 * mix(vec3(1.0), Ctint, specular_tint), base_color.rgb, metallic);
			vec3 Csheen = mix(vec3(1.0), Ctint, sheen_tint);

			// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
			// and mix in diffuse retro-reflection based on roughness

			float FL = schlick_fresnel(NdotL), FV = schlick_fresnel(NdotV);
			float Fd90 = 0.5 + 2.0 * LdotH*LdotH * roughness;
			float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

			// Based on Hanrahan-Krueger brdf approximation of isotropic bssrdf
			// 1.25 scale is used to (roughly) preserve albedo
			// Fss90 used to "flatten" retroreflection based on roughness
			float Fss90 = LdotH*LdotH * roughness;
			float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
			float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);

			// specular
			float aspect = sqrt(1.0 - anisotropic * 0.9);
			float a = sqr(roughness);
			float ax = max(0.001, a / aspect);
			float ay = max(0.001, a * aspect);
			float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay); //GTR2(NdotH, a);
			float FH = (fresnel_dielectric_cos(LdotH, eta) - F0) * F0_norm;
			vec3 Fs = mix(Cspec0, vec3(1.0), FH);
			float roughg = sqr(roughness * 0.5 + 0.5);
			float Gs = smithG_GGX(NdotL, roughg) * smithG_GGX(NdotV, roughg);

			// sheen
			vec3 Fsheen = schlick_fresnel(LdotH) * sheen * Csheen;

			diffuse_and_specular_bsdf = (M_1_PI * mix(Fd, ss, subsurface) * base_color.rgb + Fsheen)
			                            * (1.0 - metallic) + Gs * Fs * Ds;
		}
		diffuse_and_specular_bsdf *= max(NdotL, 0.0);

		float CNdotL = dot(CN, light_position);
		float CNdotV = dot(CN, V);

		vec3 clearcoat_bsdf = vec3(0.0);
		if (CNdotL >= 0.0 && CNdotV >= 0.0 && clearcoat > 0.0) {
			float CNdotH = dot(CN, H);
			//float FH = schlick_fresnel(LdotH);

			// clearcoat (ior = 1.5 -> F0 = 0.04)
			float Dr = GTR1(CNdotH, mix(0.1, 0.001, clearcoat_gloss));
			float Fr = fresnel_dielectric_cos(LdotH, 1.5); //mix(0.04, 1.0, FH);
			float Gr = smithG_GGX(CNdotL, 0.25) * smithG_GGX(CNdotV, 0.25);

			clearcoat_bsdf = clearcoat * Gr * Fr * Dr * vec3(0.25);
		}
		clearcoat_bsdf *= max(CNdotL, 0.0);

		L += light_specular * (diffuse_and_specular_bsdf + clearcoat_bsdf);
	}

	result = vec4(L, 1.0);
}

void node_bsdf_translucent(vec4 color, vec3 N, out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_bsdf_transparent(vec4 color, out vec4 result)
{
	/* this isn't right */
	result.r = color.r;
	result.g = color.g;
	result.b = color.b;
	result.a = 0.0;
}

void node_bsdf_velvet(vec4 color, float sigma, vec3 N, out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_subsurface_scattering(
        vec4 color, float scale, vec3 radius, float sharpen, float texture_blur, vec3 N,
        out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_bsdf_hair(vec4 color, float offset, float roughnessu, float roughnessv, vec3 tangent, out vec4 result)
{
	result = color;
}

void node_bsdf_refraction(vec4 color, float roughness, float ior, vec3 N, out vec4 result)
{
	node_bsdf_diffuse(color, 0.0, N, result);
}

void node_ambient_occlusion(vec4 color, out vec4 result)
{
	result = color;
}

/* emission */

void node_emission(vec4 color, float strength, vec3 N, out vec4 result)
{
	result = color * strength;
}

/* background */

void background_transform_to_world(vec3 viewvec, out vec3 worldvec)
{
	vec4 v = (ProjectionMatrix[3][3] == 0.0) ? vec4(viewvec, 1.0) : vec4(0.0, 0.0, 1.0, 1.0);
	vec4 co_homogenous = (ProjectionMatrixInverse * v);

	vec4 co = vec4(co_homogenous.xyz / co_homogenous.w, 0.0);
#ifdef WORLD_BACKGROUND
	worldvec = (ViewMatrixInverse * co).xyz;
#else
	worldvec = (ModelViewMatrixInverse * co).xyz;
#endif
}

#if defined(PROBE_CAPTURE) || defined(WORLD_BACKGROUND)
void environment_default_vector(out vec3 worldvec)
{
#ifdef WORLD_BACKGROUND
	background_transform_to_world(viewPosition, worldvec);
#else
	worldvec = normalize(worldPosition);
#endif
}
#endif

void node_background(vec4 color, float strength, out vec4 result)
{
	result = color * strength;
}

/* closures */

void node_mix_shader(float fac, vec4 shader1, vec4 shader2, out vec4 shader)
{
	shader = mix(shader1, shader2, fac);
}

void node_add_shader(vec4 shader1, vec4 shader2, out vec4 shader)
{
	shader = shader1 + shader2;
}

/* fresnel */

void node_fresnel(float ior, vec3 N, vec3 I, out float result)
{
	/* handle perspective/orthographic */
	vec3 I_view = (ProjectionMatrix[3][3] == 0.0) ? normalize(I) : vec3(0.0, 0.0, -1.0);

	float eta = max(ior, 0.00001);
	result = fresnel_dielectric(I_view, N, (gl_FrontFacing) ? eta : 1.0 / eta);
}

/* layer_weight */

void node_layer_weight(float blend, vec3 N, vec3 I, out float fresnel, out float facing)
{
	/* fresnel */
	float eta = max(1.0 - blend, 0.00001);
	vec3 I_view = (ProjectionMatrix[3][3] == 0.0) ? normalize(I) : vec3(0.0, 0.0, -1.0);

	fresnel = fresnel_dielectric(I_view, N, (gl_FrontFacing) ? 1.0 / eta : eta);

	/* facing */
	facing = abs(dot(I_view, N));
	if (blend != 0.5) {
		blend = clamp(blend, 0.0, 0.99999);
		blend = (blend < 0.5) ? 2.0 * blend : 0.5 / (1.0 - blend);
		facing = pow(facing, blend);
	}
	facing = 1.0 - facing;
}

/* gamma */

void node_gamma(vec4 col, float gamma, out vec4 outcol)
{
	outcol = col;

	if (col.r > 0.0)
		outcol.r = compatible_pow(col.r, gamma);
	if (col.g > 0.0)
		outcol.g = compatible_pow(col.g, gamma);
	if (col.b > 0.0)
		outcol.b = compatible_pow(col.b, gamma);
}

/* geometry */

void node_attribute(vec3 attr, out vec4 outcol, out vec3 outvec, out float outf)
{
	outcol = vec4(attr, 1.0);
	outvec = attr;
	outf = (attr.x + attr.y + attr.z) / 3.0;
}

void node_uvmap(vec3 attr_uv, out vec3 outvec)
{
	outvec = attr_uv;
}

void tangent_orco_x(vec3 orco_in, out vec3 orco_out)
{
	orco_out = vec3(0.0, (orco_in.z - 0.5) * -0.5, (orco_in.y - 0.5) * 0.5);
}

void tangent_orco_y(vec3 orco_in, out vec3 orco_out)
{
	orco_out = vec3((orco_in.z - 0.5) * -0.5, 0.0, (orco_in.x - 0.5) * 0.5);
}

void tangent_orco_z(vec3 orco_in, out vec3 orco_out)
{
	orco_out = vec3((orco_in.y - 0.5) * -0.5, (orco_in.x - 0.5) * 0.5, 0.0);
}

void node_tangentmap(vec4 attr_tangent, mat4 toworld, out vec3 tangent)
{
	tangent = (toworld * vec4(attr_tangent.xyz, 0.0)).xyz;
}

void node_tangent(vec3 N, vec3 orco, mat4 objmat, mat4 toworld, out vec3 T)
{
	N = (toworld * vec4(N, 0.0)).xyz;
	T = (objmat * vec4(orco, 0.0)).xyz;
	T = cross(N, normalize(cross(T, N)));
}

void node_geometry(
        vec3 I, vec3 N, vec3 orco, mat4 objmat, mat4 toworld,
        out vec3 position, out vec3 normal, out vec3 tangent,
        out vec3 true_normal, out vec3 incoming, out vec3 parametric,
        out float backfacing, out float pointiness)
{
	position = (toworld * vec4(I, 1.0)).xyz;
	normal = (toworld * vec4(N, 0.0)).xyz;
	tangent_orco_z(orco, orco);
	node_tangent(N, orco, objmat, toworld, tangent);
	true_normal = normal;

	/* handle perspective/orthographic */
	vec3 I_view = (ProjectionMatrix[3][3] == 0.0) ? normalize(I) : vec3(0.0, 0.0, -1.0);
	incoming = -(toworld * vec4(I_view, 0.0)).xyz;

	parametric = vec3(0.0);
	backfacing = (gl_FrontFacing) ? 0.0 : 1.0;
	pointiness = 0.5;
}

void node_tex_coord(
        vec3 I, vec3 N, mat4 viewinvmat, mat4 obinvmat, vec4 camerafac,
        vec3 attr_orco, vec3 attr_uv,
        out vec3 generated, out vec3 normal, out vec3 uv, out vec3 object,
        out vec3 camera, out vec3 window, out vec3 reflection)
{
	generated = attr_orco;
	normal = normalize((obinvmat * (viewinvmat * vec4(N, 0.0))).xyz);
	uv = attr_uv;
	object = (obinvmat * (viewinvmat * vec4(I, 1.0))).xyz;
	camera = vec3(I.xy, -I.z);
	vec4 projvec = ProjectionMatrix * vec4(I, 1.0);
	window = vec3(mtex_2d_mapping(projvec.xyz / projvec.w).xy * camerafac.xy + camerafac.zw, 0.0);

	vec3 shade_I;
	shade_view(I, shade_I);
	vec3 view_reflection = reflect(shade_I, normalize(N));
	reflection = (viewinvmat * vec4(view_reflection, 0.0)).xyz;
}

void node_tex_coord_background(
        vec3 I, vec3 N, mat4 viewinvmat, mat4 obinvmat, vec4 camerafac,
        vec3 attr_orco, vec3 attr_uv,
        out vec3 generated, out vec3 normal, out vec3 uv, out vec3 object,
        out vec3 camera, out vec3 window, out vec3 reflection)
{
	vec4 v = (ProjectionMatrix[3][3] == 0.0) ? vec4(I, 1.0) : vec4(0.0, 0.0, 1.0, 1.0);
	vec4 co_homogenous = (ProjectionMatrixInverse * v);

	vec4 co = vec4(co_homogenous.xyz / co_homogenous.w, 0.0);

	co = normalize(co);

#ifdef PROBE_CAPTURE
	vec3 coords = normalize(worldPosition);
#elif defined(WORLD_BACKGROUND)
	vec3 coords = (ViewMatrixInverse * co).xyz;
#else
	vec3 coords = (ModelViewMatrixInverse * co).xyz;
#endif

	generated = coords;
	normal = -coords;
	uv = vec3(attr_uv.xy, 0.0);
	object = coords;

	camera = vec3(co.xy, -co.z);
	window = (ProjectionMatrix[3][3] == 0.0) ?
	         vec3(mtex_2d_mapping(I).xy * camerafac.xy + camerafac.zw, 0.0) :
	         vec3(vec2(0.5) * camerafac.xy + camerafac.zw, 0.0);

	reflection = -coords;
}

#if defined(WORLD_BACKGROUND) || (defined(PROBE_CAPTURE) && !defined(MESH_SHADER))
#define node_tex_coord node_tex_coord_background
#endif

/* textures */

float calc_gradient(vec3 p, int gradient_type)
{
	float x, y, z;
	x = p.x;
	y = p.y;
	z = p.z;
	if (gradient_type == 0) {  /* linear */
		return x;
	}
	else if (gradient_type == 1) {  /* quadratic */
		float r = max(x, 0.0);
		return r * r;
	}
	else if (gradient_type == 2) {  /* easing */
		float r = min(max(x, 0.0), 1.0);
		float t = r * r;
		return (3.0 * t - 2.0 * t * r);
	}
	else if (gradient_type == 3) {  /* diagonal */
		return (x + y) * 0.5;
	}
	else if (gradient_type == 4) {  /* radial */
		return atan(y, x) / (M_PI * 2) + 0.5;
	}
	else {
		float r = max(1.0 - sqrt(x * x + y * y + z * z), 0.0);
		if (gradient_type == 5) {  /* quadratic sphere */
			return r * r;
		}
		else if (gradient_type == 6) {  /* sphere */
			return r;
		}
	}
	return 0.0;
}

void node_tex_gradient(vec3 co, float gradient_type, out vec4 color, out float fac)
{
	float f = calc_gradient(co, int(gradient_type));
	f = clamp(f, 0.0, 1.0);

	color = vec4(f, f, f, 1.0);
	fac = f;
}

void node_tex_checker(vec3 co, vec4 color1, vec4 color2, float scale, out vec4 color, out float fac)
{
	vec3 p = co * scale;

	/* Prevent precision issues on unit coordinates. */
	p.x = (p.x + 0.000001) * 0.999999;
	p.y = (p.y + 0.000001) * 0.999999;
	p.z = (p.z + 0.000001) * 0.999999;

	int xi = int(abs(floor(p.x)));
	int yi = int(abs(floor(p.y)));
	int zi = int(abs(floor(p.z)));

	bool check = ((mod(xi, 2) == mod(yi, 2)) == bool(mod(zi, 2)));

	color = check ? color1 : color2;
	fac = check ? 1.0 : 0.0;
}

vec2 calc_brick_texture(vec3 p, float mortar_size, float mortar_smooth, float bias,
                        float brick_width, float row_height,
                        float offset_amount, int offset_frequency,
                        float squash_amount, int squash_frequency)
{
	int bricknum, rownum;
	float offset = 0.0;
	float x, y;

	rownum = floor_to_int(p.y / row_height);

	if (offset_frequency != 0 && squash_frequency != 0) {
		brick_width *= (rownum % squash_frequency != 0) ? 1.0 : squash_amount; /* squash */
		offset = (rownum % offset_frequency != 0) ? 0.0 : (brick_width * offset_amount); /* offset */
	}

	bricknum = floor_to_int((p.x + offset) / brick_width);

	x = (p.x + offset) - brick_width * bricknum;
	y = p.y - row_height * rownum;

	float tint = clamp((integer_noise((rownum << 16) + (bricknum & 0xFFFF)) + bias), 0.0, 1.0);

	float min_dist = min(min(x, y), min(brick_width - x, row_height - y));
	if (min_dist >= mortar_size) {
		return vec2(tint, 0.0);
	}
	else if (mortar_smooth == 0.0) {
		return vec2(tint, 1.0);
	}
	else {
		min_dist = 1.0 - min_dist/mortar_size;
		return vec2(tint, smoothstep(0.0, mortar_smooth, min_dist));
	}
}

void node_tex_brick(vec3 co,
                    vec4 color1, vec4 color2,
                    vec4 mortar, float scale,
                    float mortar_size, float mortar_smooth, float bias,
                    float brick_width, float row_height,
                    float offset_amount, float offset_frequency,
                    float squash_amount, float squash_frequency,
                    out vec4 color, out float fac)
{
	vec2 f2 = calc_brick_texture(co * scale,
	                             mortar_size, mortar_smooth, bias,
	                             brick_width, row_height,
	                             offset_amount, int(offset_frequency),
	                             squash_amount, int(squash_frequency));
	float tint = f2.x;
	float f = f2.y;
	if (f != 1.0) {
		float facm = 1.0 - tint;
		color1 = facm * color1 + tint * color2;
	}
	color = mix(color1, mortar, f);
	fac = f;
}

void node_tex_clouds(vec3 co, float size, out vec4 color, out float fac)
{
	color = vec4(1.0);
	fac = 1.0;
}

void node_tex_environment_equirectangular(vec3 co, sampler2D ima, out vec4 color)
{
	vec3 nco = normalize(co);
	float u = -atan(nco.y, nco.x) / (2.0 * M_PI) + 0.5;
	float v = atan(nco.z, hypot(nco.x, nco.y)) / M_PI + 0.5;

	/* Fix pole bleeding */
	float half_width = 0.5 / float(textureSize(ima, 0).x);
	v = clamp(v, half_width, 1.0 - half_width);

	/* Fix u = 0 seam */
	/* This is caused by texture filtering, since uv don't have smooth derivatives
	 * at u = 0 or 2PI, hardware filtering is using the smallest mipmap for certain
	 * texels. So we force the highest mipmap and don't do anisotropic filtering. */
	color = textureLod(ima, vec2(u, v), 0.0);
}

void node_tex_environment_mirror_ball(vec3 co, sampler2D ima, out vec4 color)
{
	vec3 nco = normalize(co);

	nco.y -= 1.0;

	float div = 2.0 * sqrt(max(-0.5 * nco.y, 0.0));
	if (div > 0.0)
		nco /= div;

	float u = 0.5 * (nco.x + 1.0);
	float v = 0.5 * (nco.z + 1.0);

	color = texture(ima, vec2(u, v));
}

void node_tex_environment_empty(vec3 co, out vec4 color)
{
	color = vec4(1.0, 0.0, 1.0, 1.0);
}

void node_tex_image(vec3 co, sampler2D ima, out vec4 color, out float alpha)
{
	color = texture(ima, co.xy);
	alpha = color.a;
}

void node_tex_image_box(vec3 texco,
                        vec3 nob,
                        sampler2D ima,
                        float blend,
                        out vec4 color,
                        out float alpha)
{
	/* project from direction vector to barycentric coordinates in triangles */
	nob = vec3(abs(nob.x), abs(nob.y), abs(nob.z));
	nob /= (nob.x + nob.y + nob.z);

	/* basic idea is to think of this as a triangle, each corner representing
	 * one of the 3 faces of the cube. in the corners we have single textures,
	 * in between we blend between two textures, and in the middle we a blend
	 * between three textures.
	 *
	 * the Nxyz values are the barycentric coordinates in an equilateral
	 * triangle, which in case of blending, in the middle has a smaller
	 * equilateral triangle where 3 textures blend. this divides things into
	 * 7 zones, with an if () test for each zone */

	vec3 weight = vec3(0.0, 0.0, 0.0);
	float limit = 0.5 * (1.0 + blend);

	/* first test for corners with single texture */
	if (nob.x > limit * (nob.x + nob.y) && nob.x > limit * (nob.x + nob.z)) {
		weight.x = 1.0;
	}
	else if (nob.y > limit * (nob.x + nob.y) && nob.y > limit * (nob.y + nob.z)) {
		weight.y = 1.0;
	}
	else if (nob.z > limit * (nob.x + nob.z) && nob.z > limit * (nob.y + nob.z)) {
		weight.z = 1.0;
	}
	else if (blend > 0.0) {
		/* in case of blending, test for mixes between two textures */
		if (nob.z < (1.0 - limit) * (nob.y + nob.x)) {
			weight.x = nob.x / (nob.x + nob.y);
			weight.x = clamp((weight.x - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
			weight.y = 1.0 - weight.x;
		}
		else if (nob.x < (1.0 - limit) * (nob.y + nob.z)) {
			weight.y = nob.y / (nob.y + nob.z);
			weight.y = clamp((weight.y - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
			weight.z = 1.0 - weight.y;
		}
		else if (nob.y < (1.0 - limit) * (nob.x + nob.z)) {
			weight.x = nob.x / (nob.x + nob.z);
			weight.x = clamp((weight.x - 0.5 * (1.0 - blend)) / blend, 0.0, 1.0);
			weight.z = 1.0 - weight.x;
		}
		else {
			/* last case, we have a mix between three */
			weight.x = ((2.0 - limit) * nob.x + (limit - 1.0)) / (2.0 * limit - 1.0);
			weight.y = ((2.0 - limit) * nob.y + (limit - 1.0)) / (2.0 * limit - 1.0);
			weight.z = ((2.0 - limit) * nob.z + (limit - 1.0)) / (2.0 * limit - 1.0);
		}
	}
	else {
		/* Desperate mode, no valid choice anyway, fallback to one side.*/
		weight.x = 1.0;
	}
	color = vec4(0);
	if (weight.x > 0.0) {
		color += weight.x * texture(ima, texco.yz);
	}
	if (weight.y > 0.0) {
		color += weight.y * texture(ima, texco.xz);
	}
	if (weight.z > 0.0) {
		color += weight.z * texture(ima, texco.yx);
	}

	alpha = color.a;
}

void node_tex_image_empty(vec3 co, out vec4 color, out float alpha)
{
	color = vec4(0.0);
	alpha = 0.0;
}

void node_tex_magic(vec3 co, float scale, float distortion, float depth, out vec4 color, out float fac)
{
	vec3 p = co * scale;
	float x = sin((p.x + p.y + p.z) * 5.0);
	float y = cos((-p.x + p.y - p.z) * 5.0);
	float z = -cos((-p.x - p.y + p.z) * 5.0);

	if (depth > 0) {
		x *= distortion;
		y *= distortion;
		z *= distortion;
		y = -cos(x - y + z);
		y *= distortion;
		if (depth > 1) {
			x = cos(x - y - z);
			x *= distortion;
			if (depth > 2) {
				z = sin(-x - y - z);
				z *= distortion;
				if (depth > 3) {
					x = -cos(-x + y - z);
					x *= distortion;
					if (depth > 4) {
						y = -sin(-x + y + z);
						y *= distortion;
						if (depth > 5) {
							y = -cos(-x + y + z);
							y *= distortion;
							if (depth > 6) {
								x = cos(x + y + z);
								x *= distortion;
								if (depth > 7) {
									z = sin(x + y - z);
									z *= distortion;
									if (depth > 8) {
										x = -cos(-x - y + z);
										x *= distortion;
										if (depth > 9) {
											y = -sin(x - y + z);
											y *= distortion;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	if (distortion != 0.0) {
		distortion *= 2.0;
		x /= distortion;
		y /= distortion;
		z /= distortion;
	}

	color = vec4(0.5 - x, 0.5 - y, 0.5 - z, 1.0);
	fac = (color.x + color.y + color.z) / 3.0;
}

float noise_fade(float t)
{
	return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float noise_scale3(float result)
{
	return 0.9820 * result;
}

float noise_nerp(float t, float a, float b)
{
	return (1.0 - t) * a + t * b;
}

float noise_grad(uint hash, float x, float y, float z)
{
	uint h = hash & 15u;
	float u = h < 8u ? x : y;
	float vt = ((h == 12u) || (h == 14u)) ? x : z;
	float v = h < 4u ? y : vt;
	return (((h & 1u) != 0u) ? -u : u) + (((h & 2u) != 0u) ? -v : v);
}

float noise_perlin(float x, float y, float z)
{
	int X; float fx = floorfrac(x, X);
	int Y; float fy = floorfrac(y, Y);
	int Z; float fz = floorfrac(z, Z);

	float u = noise_fade(fx);
	float v = noise_fade(fy);
	float w = noise_fade(fz);

	float result;

	result = noise_nerp(w, noise_nerp(v, noise_nerp(u, noise_grad(hash(X, Y, Z), fx, fy, fz),
	                                                noise_grad(hash(X + 1, Y, Z), fx - 1.0, fy, fz)),
	                                  noise_nerp(u, noise_grad(hash(X, Y + 1, Z), fx, fy - 1.0, fz),
	                                             noise_grad(hash(X + 1, Y + 1, Z), fx - 1.0, fy - 1.0, fz))),
	                    noise_nerp(v, noise_nerp(u, noise_grad(hash(X, Y, Z + 1), fx, fy, fz - 1.0),
	                                             noise_grad(hash(X + 1, Y, Z + 1), fx - 1.0, fy, fz - 1.0)),
	                               noise_nerp(u, noise_grad(hash(X, Y + 1, Z + 1), fx, fy - 1.0, fz - 1.0),
	                                          noise_grad(hash(X + 1, Y + 1, Z + 1), fx - 1.0, fy - 1.0, fz - 1.0))));
	return noise_scale3(result);
}

float noise(vec3 p)
{
	return 0.5 * noise_perlin(p.x, p.y, p.z) + 0.5;
}

float snoise(vec3 p)
{
	return noise_perlin(p.x, p.y, p.z);
}

float noise_turbulence(vec3 p, float octaves, int hard)
{
	float fscale = 1.0;
	float amp = 1.0;
	float sum = 0.0;
	octaves = clamp(octaves, 0.0, 16.0);
	int n = int(octaves);
	for (int i = 0; i <= n; i++) {
		float t = noise(fscale * p);
		if (hard != 0) {
			t = abs(2.0 * t - 1.0);
		}
		sum += t * amp;
		amp *= 0.5;
		fscale *= 2.0;
	}
	float rmd = octaves - floor(octaves);
	if (rmd != 0.0) {
		float t = noise(fscale * p);
		if (hard != 0) {
			t = abs(2.0 * t - 1.0);
		}
		float sum2 = sum + t * amp;
		sum *= (float(1 << n) / float((1 << (n + 1)) - 1));
		sum2 *= (float(1 << (n + 1)) / float((1 << (n + 2)) - 1));
		return (1.0 - rmd) * sum + rmd * sum2;
	}
	else {
		sum *= (float(1 << n) / float((1 << (n + 1)) - 1));
		return sum;
	}
}

void node_tex_noise(vec3 co, float scale, float detail, float distortion, out vec4 color, out float fac)
{
	vec3 p = co * scale;
	int hard = 0;
	if (distortion != 0.0) {
		vec3 r, offset = vec3(13.5, 13.5, 13.5);
		r.x = noise(p + offset) * distortion;
		r.y = noise(p) * distortion;
		r.z = noise(p - offset) * distortion;
		p += r;
	}

	fac = noise_turbulence(p, detail, hard);
	color = vec4(fac,
	             noise_turbulence(vec3(p.y, p.x, p.z), detail, hard),
	             noise_turbulence(vec3(p.y, p.z, p.x), detail, hard),
	             1);
}

/* Musgrave fBm
 *
 * H: fractal increment parameter
 * lacunarity: gap between successive frequencies
 * octaves: number of frequencies in the fBm
 *
 * from "Texturing and Modelling: A procedural approach"
 */

float noise_musgrave_fBm(vec3 p, float H, float lacunarity, float octaves)
{
	float rmd;
	float value = 0.0;
	float pwr = 1.0;
	float pwHL = pow(lacunarity, -H);

	for (int i = 0; i < int(octaves); i++) {
		value += snoise(p) * pwr;
		pwr *= pwHL;
		p *= lacunarity;
	}

	rmd = octaves - floor(octaves);
	if (rmd != 0.0)
		value += rmd * snoise(p) * pwr;

	return value;
}

/* Musgrave Multifractal
 *
 * H: highest fractal dimension
 * lacunarity: gap between successive frequencies
 * octaves: number of frequencies in the fBm
 */

float noise_musgrave_multi_fractal(vec3 p, float H, float lacunarity, float octaves)
{
	float rmd;
	float value = 1.0;
	float pwr = 1.0;
	float pwHL = pow(lacunarity, -H);

	for (int i = 0; i < int(octaves); i++) {
		value *= (pwr * snoise(p) + 1.0);
		pwr *= pwHL;
		p *= lacunarity;
	}

	rmd = octaves - floor(octaves);
	if (rmd != 0.0)
		value *= (rmd * pwr * snoise(p) + 1.0); /* correct? */

	return value;
}

/* Musgrave Heterogeneous Terrain
 *
 * H: fractal dimension of the roughest area
 * lacunarity: gap between successive frequencies
 * octaves: number of frequencies in the fBm
 * offset: raises the terrain from `sea level'
 */

float noise_musgrave_hetero_terrain(vec3 p, float H, float lacunarity, float octaves, float offset)
{
	float value, increment, rmd;
	float pwHL = pow(lacunarity, -H);
	float pwr = pwHL;

	/* first unscaled octave of function; later octaves are scaled */
	value = offset + snoise(p);
	p *= lacunarity;

	for (int i = 1; i < int(octaves); i++) {
		increment = (snoise(p) + offset) * pwr * value;
		value += increment;
		pwr *= pwHL;
		p *= lacunarity;
	}

	rmd = octaves - floor(octaves);
	if (rmd != 0.0) {
		increment = (snoise(p) + offset) * pwr * value;
		value += rmd * increment;
	}

	return value;
}

/* Hybrid Additive/Multiplicative Multifractal Terrain
 *
 * H: fractal dimension of the roughest area
 * lacunarity: gap between successive frequencies
 * octaves: number of frequencies in the fBm
 * offset: raises the terrain from `sea level'
 */

float noise_musgrave_hybrid_multi_fractal(vec3 p, float H, float lacunarity, float octaves, float offset, float gain)
{
	float result, signal, weight, rmd;
	float pwHL = pow(lacunarity, -H);
	float pwr = pwHL;

	result = snoise(p) + offset;
	weight = gain * result;
	p *= lacunarity;

	for (int i = 1; (weight > 0.001f) && (i < int(octaves)); i++) {
		if (weight > 1.0)
			weight = 1.0;

		signal = (snoise(p) + offset) * pwr;
		pwr *= pwHL;
		result += weight * signal;
		weight *= gain * signal;
		p *= lacunarity;
	}

	rmd = octaves - floor(octaves);
	if (rmd != 0.0)
		result += rmd * ((snoise(p) + offset) * pwr);

	return result;
}

/* Ridged Multifractal Terrain
 *
 * H: fractal dimension of the roughest area
 * lacunarity: gap between successive frequencies
 * octaves: number of frequencies in the fBm
 * offset: raises the terrain from `sea level'
 */

float noise_musgrave_ridged_multi_fractal(vec3 p, float H, float lacunarity, float octaves, float offset, float gain)
{
	float result, signal, weight;
	float pwHL = pow(lacunarity, -H);
	float pwr = pwHL;

	signal = offset - abs(snoise(p));
	signal *= signal;
	result = signal;
	weight = 1.0;

	for (int i = 1; i < int(octaves); i++) {
		p *= lacunarity;
		weight = clamp(signal * gain, 0.0, 1.0);
		signal = offset - abs(snoise(p));
		signal *= signal;
		signal *= weight;
		result += signal * pwr;
		pwr *= pwHL;
	}

	return result;
}

float svm_musgrave(int type,
                   float dimension,
                   float lacunarity,
                   float octaves,
                   float offset,
                   float intensity,
                   float gain,
                   vec3 p)
{
	if (type == 0 /* NODE_MUSGRAVE_MULTIFRACTAL */)
		return intensity * noise_musgrave_multi_fractal(p, dimension, lacunarity, octaves);
	else if (type == 1 /* NODE_MUSGRAVE_FBM */)
		return intensity * noise_musgrave_fBm(p, dimension, lacunarity, octaves);
	else if (type == 2 /* NODE_MUSGRAVE_HYBRID_MULTIFRACTAL */)
		return intensity * noise_musgrave_hybrid_multi_fractal(p, dimension, lacunarity, octaves, offset, gain);
	else if (type == 3 /* NODE_MUSGRAVE_RIDGED_MULTIFRACTAL */)
		return intensity * noise_musgrave_ridged_multi_fractal(p, dimension, lacunarity, octaves, offset, gain);
	else if (type == 4 /* NODE_MUSGRAVE_HETERO_TERRAIN */)
		return intensity * noise_musgrave_hetero_terrain(p, dimension, lacunarity, octaves, offset);
	return 0.0;
}

void node_tex_musgrave(vec3 co,
                       float scale,
                       float detail,
                       float dimension,
                       float lacunarity,
                       float offset,
                       float gain,
                       float type,
                       out vec4 color,
                       out float fac)
{
	fac = svm_musgrave(int(type),
	                   dimension,
	                   lacunarity,
	                   detail,
	                   offset,
	                   1.0,
	                   gain,
	                   co * scale);

	color = vec4(fac, fac, fac, 1.0);
}

void node_tex_sky(vec3 co, out vec4 color)
{
	color = vec4(1.0);
}

void node_tex_voronoi(vec3 co, float scale, float coloring, out vec4 color, out float fac)
{
	vec3 p = co * scale;
	int xx, yy, zz, xi, yi, zi;
	float da[4];
	vec3 pa[4];

	xi = floor_to_int(p[0]);
	yi = floor_to_int(p[1]);
	zi = floor_to_int(p[2]);

	da[0] = 1e+10;
	da[1] = 1e+10;
	da[2] = 1e+10;
	da[3] = 1e+10;

	for (xx = xi - 1; xx <= xi + 1; xx++) {
		for (yy = yi - 1; yy <= yi + 1; yy++) {
			for (zz = zi - 1; zz <= zi + 1; zz++) {
				vec3 ip = vec3(xx, yy, zz);
				vec3 vp = cellnoise_color(ip);
				vec3 pd = p - (vp + ip);
				float d = dot(pd, pd);
				vp += vec3(xx, yy, zz);
				if (d < da[0]) {
					da[3] = da[2];
					da[2] = da[1];
					da[1] = da[0];
					da[0] = d;
					pa[3] = pa[2];
					pa[2] = pa[1];
					pa[1] = pa[0];
					pa[0] = vp;
				}
				else if (d < da[1]) {
					da[3] = da[2];
					da[2] = da[1];
					da[1] = d;

					pa[3] = pa[2];
					pa[2] = pa[1];
					pa[1] = vp;
				}
				else if (d < da[2]) {
					da[3] = da[2];
					da[2] = d;

					pa[3] = pa[2];
					pa[2] = vp;
				}
				else if (d < da[3]) {
					da[3] = d;
					pa[3] = vp;
				}
			}
		}
	}

	if (coloring == 0.0) {
		fac = abs(da[0]);
		color = vec4(fac, fac, fac, 1);
	}
	else {
		color = vec4(cellnoise_color(pa[0]), 1);
		fac = (color.x + color.y + color.z) * (1.0 / 3.0);
	}
}

float calc_wave(vec3 p, float distortion, float detail, float detail_scale, int wave_type, int wave_profile)
{
	float n;

	if (wave_type == 0) /* type bands */
		n = (p.x + p.y + p.z) * 10.0;
	else /* type rings */
		n = length(p) * 20.0;

	if (distortion != 0.0)
		n += distortion * noise_turbulence(p * detail_scale, detail, 0);

	if (wave_profile == 0) { /* profile sin */
		return 0.5 + 0.5 * sin(n);
	}
	else { /* profile saw */
		n /= 2.0 * M_PI;
		n -= int(n);
		return (n < 0.0) ? n + 1.0 : n;
	}
}

void node_tex_wave(
        vec3 co, float scale, float distortion, float detail, float detail_scale, float wave_type, float wave_profile,
        out vec4 color, out float fac)
{
	float f;
	f = calc_wave(co * scale, distortion, detail, detail_scale, int(wave_type), int(wave_profile));

	color = vec4(f, f, f, 1.0);
	fac = f;
}

/* light path */

void node_light_path(
	out float is_camera_ray,
	out float is_shadow_ray,
	out float is_diffuse_ray,
	out float is_glossy_ray,
	out float is_singular_ray,
	out float is_reflection_ray,
	out float is_transmission_ray,
	out float ray_length,
	out float ray_depth,
	out float diffuse_depth,
	out float glossy_depth,
	out float transparent_depth,
	out float transmission_depth)
{
#ifndef PROBE_CAPTURE
	is_camera_ray = 1.0;
	is_glossy_ray = 0.0;
	is_diffuse_ray = 0.0;
	is_reflection_ray = 0.0;
	is_transmission_ray = 0.0;
#else
	is_camera_ray = 0.0;
	is_glossy_ray = 1.0;
	is_diffuse_ray = 1.0;
	is_reflection_ray = 1.0;
	is_transmission_ray = 1.0;
#endif
	is_shadow_ray = 0.0;
	is_singular_ray = 0.0;
	ray_length = 1.0;
	ray_depth = 1.0;
	diffuse_depth = 1.0;
	glossy_depth = 1.0;
	transparent_depth = 1.0;
	transmission_depth = 1.0;
}

void node_light_falloff(float strength, float tsmooth, out float quadratic, out float linear, out float constant)
{
	quadratic = strength;
	linear = strength;
	constant = strength;
}

void node_object_info(mat4 obmat, vec3 info, out vec3 location, out float object_index, out float material_index, out float random)
{
	location = obmat[3].xyz;
	object_index = info.x;
	material_index = info.y;
	random = info.z;
}

void node_normal_map(vec4 tangent, vec3 normal, vec3 texnormal, out vec3 outnormal)
{
	vec3 B = tangent.w * cross(normal, tangent.xyz);

	outnormal = texnormal.x * tangent.xyz + texnormal.y * B + texnormal.z * normal;
	outnormal = normalize(outnormal);
}

void node_bump(float strength, float dist, float height, vec3 N, vec3 surf_pos, float invert, out vec3 result)
{
	if (invert != 0.0) {
		dist *= -1.0;
	}
	vec3 dPdx = dFdx(surf_pos);
	vec3 dPdy = dFdy(surf_pos);

	/* Get surface tangents from normal. */
	vec3 Rx = cross(dPdy, N);
	vec3 Ry = cross(N, dPdx);

	/* Compute surface gradient and determinant. */
	float det = dot(dPdx, Rx);
	float absdet = abs(det);

	float dHdx = dFdx(height);
	float dHdy = dFdy(height);
	vec3 surfgrad = dHdx * Rx + dHdy * Ry;

	strength = max(strength, 0.0);

	result = normalize(absdet * N - dist * sign(det) * surfgrad);
	result = normalize(strength * result + (1.0 - strength) * N);
}

/* output */

void node_output_material(vec4 surface, vec4 volume, float displacement, out vec4 result)
{
	result = surface;
}

uniform float backgroundAlpha;

void node_output_world(vec4 surface, vec4 volume, out vec4 result)
{
	result = vec4(surface.rgb, backgroundAlpha);
}

void convert_metallic_to_specular(vec4 basecol, float metallic, float specular_fac, out vec4 diffuse, out vec4 f0)
{
	vec4 dielectric = vec4(0.034) * specular_fac * 2.0;
	diffuse = mix(basecol, vec4(0.0), metallic);
	f0 = mix(dielectric, basecol, metallic);
}

/* TODO : clean this ifdef mess */
/* EEVEE output */
#ifdef EEVEE_ENGINE
void world_normals_get(out vec3 N)
{
	N = gl_FrontFacing ? worldNormal : -worldNormal;
}

void node_eevee_metallic(
        vec4 basecol, float metallic, float specular, float roughness, vec4 emissive, float transp, vec3 normal,
        float clearcoat, float clearcoat_roughness, vec3 clearcoat_normal,
        float occlusion, out vec4 result)
{
	vec4 diffuse, f0;
	convert_metallic_to_specular(basecol, metallic, specular, diffuse, f0);

	result = vec4(eevee_surface_lit(normal, diffuse.rgb, f0.rgb, roughness, occlusion) + emissive.rgb, 1.0 - transp);
}

void node_eevee_specular(
        vec4 diffuse, vec4 specular, float roughness, vec4 emissive, float transp, vec3 normal,
        float clearcoat, float clearcoat_roughness, vec3 clearcoat_normal,
        float occlusion, out vec4 result)
{
	result = vec4(eevee_surface_lit(normal, diffuse.rgb, specular.rgb, roughness, occlusion) + emissive.rgb, 1.0 - transp);
}

void node_output_eevee_material(vec4 surface, out vec4 result)
{
	result = vec4(surface.rgb, length(viewPosition));
}

#endif

/* ********************** matcap style render ******************** */

void material_preview_matcap(vec4 color, sampler2D ima, vec4 N, vec4 mask, out vec4 result)
{
	vec3 normal;
	
#ifndef USE_OPENSUBDIV
	/* remap to 0.0 - 1.0 range. This is done because OpenGL 2.0 clamps colors
	 * between shader stages and we want the full range of the normal */
	normal = 2.0 * N.xyz - vec3(1.0);
	normal.z = max(normal.z, 0.0);
	normal = normalize(normal);
#else
	normal = inpt.v.normal;
	mask = vec4(1.0);
#endif

	vec2 tex = 0.49 * normal.xy + vec2(0.5);

	result = texture(ima, tex) * mask;
}
const vec4 cons3 = vec4(0.800000011921, 0.800000011921, 0.800000011921, 1.000000000000);
const float cons4 = float(0.000000000000);
const float cons5 = float(0.500000000000);
const float cons6 = float(0.200000002980);
const vec4 cons7 = vec4(0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000);
const float cons8 = float(0.000000000000);
const float cons10 = float(0.000000000000);
const float cons11 = float(0.000000000000);
const float cons13 = float(1.000000000000);

void main()
{
	vec3 tmp1;
	vec3 tmp2;
	vec4 tmp14;
	vec4 tmp16;

	world_normals_get(tmp1);
	world_normals_get(tmp2);
	node_eevee_metallic(cons3, cons4, cons5, cons6, cons7, cons8, tmp1, cons10, cons11, tmp2, cons13, tmp14);
	node_output_eevee_material(tmp14, tmp16);

	fragColor = tmp16;
}
