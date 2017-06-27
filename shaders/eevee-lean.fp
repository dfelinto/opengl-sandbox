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


/*********** NEW SHADER NODES ***************/

/* output */

void node_output_material(vec4 surface, vec4 volume, float displacement, out vec4 result)
{
	result = surface;
}

uniform float backgroundAlpha;

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

void node_output_eevee_material(vec4 surface, out vec4 result)
{
	result = vec4(surface.rgb, length(viewPosition));
}

#endif

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
