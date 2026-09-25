// =========================================
// =========================================
// crates/motionloom/src/weaver/backend/wgpu/shaders/path_trace.wgsl

// Standalone compute integrator: no preview color/depth/shadow maps are inputs.
struct Params { v: array<vec4<f32>, 26> }
struct Film {
    sum: vec4<f32>,
    stats: vec4<f32>,
    albedo: vec4<f32>,
    normal: vec4<f32>,
    // x=coverage sum, yz=motion sum in output pixels, w=reserved.
    aov: vec4<f32>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> scene: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pixels: array<u32>;
@group(0) @binding(3) var<storage, read_write> film: array<Film>;
const PI: f32 = 3.14159265359;
const INF: f32 = 1e30;
var<private> rng: u32;
fn random() -> f32 {
    rng = rng * 747796405u + 2891336453u;
    let w = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
    // Use 24 bits so conversion cannot round the RNG endpoint up to one.
    return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}
fn unit(v:vec3<f32>, fallback:vec3<f32>)->vec3<f32> {
    let len2=dot(v,v);if len2<1e-20 {return fallback;}return v*inverseSqrt(len2);
}
fn linear(c: vec3<f32>) -> vec3<f32> { return select(c/12.92, pow((c+0.055)/1.055,vec3<f32>(2.4)),c>vec3<f32>(0.04045)); }
fn mip_dim(desc:vec4<f32>, level:u32) -> vec2<i32> {
    let w=max(1,i32(desc.y)>>level); let h=max(1,i32(desc.z)>>level);
    return vec2<i32>(w,h);
}
// Levels are packed contiguously after the base level; the w component of the
// descriptor carries the level count (1 when mipmaps are off).
fn mip_offset(desc:vec4<f32>, level:u32) -> u32 {
    var offset=bitcast<u32>(desc.x); var l=0u;
    while l<level { let d=mip_dim(desc,l); offset+=u32(d.x)*u32(d.y); l++; }
    return offset;
}
fn texel(desc:vec4<f32>, level:u32, xy:vec2<i32>) -> vec4<f32> {
    let wh = mip_dim(desc,level);
    let q = ((xy % wh) + wh) % wh;
    // Texture offsets use their raw u32 bits, avoiding f32's 24-bit integer limit.
    return unpack4x8unorm(pixels[mip_offset(desc,level)+u32(q.y*wh.x+q.x)]);
}
fn sample_level(desc:vec4<f32>, uv:vec2<f32>, srgb:bool, level:u32) -> vec4<f32> {
    let wh = vec2<f32>(mip_dim(desc,level));
    let q = uv*wh-0.5; let i = vec2<i32>(floor(q)); let f = fract(q);
    var a=texel(desc,level,i); var b=texel(desc,level,i+vec2<i32>(1,0)); var c=texel(desc,level,i+vec2<i32>(0,1)); var d=texel(desc,level,i+vec2<i32>(1,1));
    if srgb { a=vec4<f32>(linear(a.rgb),a.a); b=vec4<f32>(linear(b.rgb),b.a); c=vec4<f32>(linear(c.rgb),c.a); d=vec4<f32>(linear(d.rgb),d.a); }
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}
fn texture_lod(desc:vec4<f32>, uv:vec2<f32>, srgb:bool, lod:f32) -> vec4<f32> {
    if desc.y < 1.0 { return vec4<f32>(1.0); }
    let levels=max(1u,u32(desc.w+0.5));
    let clamped=clamp(lod,0.0,f32(levels-1u));
    let l0=u32(floor(clamped)); let l1=min(l0+1u,levels-1u); let f=fract(clamped);
    return mix(sample_level(desc,uv,srgb,l0),sample_level(desc,uv,srgb,l1),f);
}
// Automatic LOD from the ray footprint; single-level textures ignore the bias.
fn texture_auto(desc:vec4<f32>, uv:vec2<f32>, srgb:bool, lod_bias:f32) -> vec4<f32> {
    return texture_lod(desc,uv,srgb,lod_bias+log2(max(max(desc.y,desc.z),1.0)));
}
fn texture(desc:vec4<f32>, uv:vec2<f32>, srgb:bool) -> vec4<f32> {
    return texture_lod(desc,uv,srgb,0.0);
}
fn material_channel(sample:vec4<f32>, encoded:u32)->f32 {
    let channel=encoded&7u; var value=sample.r;
    if channel==1u { value=sample.g; }
    else if channel==2u { value=sample.b; }
    else if channel==3u { value=sample.a; }
    else if channel==4u { value=dot(sample.rgb,vec3<f32>(0.2126,0.7152,0.0722)); }
    return select(value,1.0-value,(encoded&8u)!=0u);
}
fn material_channels(sample:vec4<f32>, packed_value:f32)->vec3<f32> {
    let packed=u32(packed_value+0.5);
    return vec3<f32>(material_channel(sample,packed&15u),material_channel(sample,(packed>>4u)&15u),material_channel(sample,(packed>>8u)&15u));
}
struct Hit { t: f32, tri: u32, uv: vec2<f32> }
fn box_hit(o:vec3<f32>, d:vec3<f32>, lo:vec3<f32>, hi:vec3<f32>, limit:f32) -> bool {
    var near=0.0; var far=limit;
    for(var a=0u;a<3u;a++) {
        if abs(d[a])<1e-12 { if o[a]<lo[a] || o[a]>hi[a] { return false; } }
        else { let x=(lo[a]-o[a])/d[a]; let y=(hi[a]-o[a])/d[a]; near=max(near,min(x,y)); far=min(far,max(x,y)); }
    }
    return far>=near;
}
fn intersect(o:vec3<f32>, d:vec3<f32>, limit:f32, primary_camera_ray:bool) -> Hit {
    var hit=Hit(limit,0xffffffffu,vec2<f32>(0.0));
    var stack:array<u32,64>; stack[0]=0u; var size=1u;
    loop {
        if size==0u { break; } size--; let n=stack[size]*3u;
        if !box_hit(o,d,scene[n].xyz,scene[n+1u].xyz,hit.t) { continue; }
        let start=u32(scene[n+2u].x); let count=u32(scene[n+2u].y);
        if count==0u { stack[size]=u32(scene[n].w); stack[size+1u]=u32(scene[n+1u].w); size+=2u; continue; }
        for(var j=0u;j<count;j++) {
            let t=bitcast<u32>(p.v[1].x)+(start+j)*16u;
            let material=bitcast<u32>(p.v[1].y)+u32(scene[t+15u].x)*10u;
            if primary_camera_ray && scene[material+9u].y<0.5 { continue; }
            let a=scene[t].xyz; let e1=scene[t+5u].xyz-a; let e2=scene[t+10u].xyz-a;
            let h=cross(d,e2); let det=dot(e1,h); if abs(det)<1e-9 { continue; }
            let s=o-a; let u=dot(s,h)/det; if u<0.0 || u>1.0 { continue; }
            let q=cross(s,e1); let v=dot(d,q)/det; if v<0.0 || u+v>1.0 { continue; }
            let dist=dot(e2,q)/det;
            if dist>1e-5 && dist<hit.t { hit=Hit(dist,t,vec2<f32>(u,v)); }
        }
    }
    return hit;
}
fn interp(t:u32, field:u32, uv:vec2<f32>) -> vec4<f32> { return scene[t+field]*(1.0-uv.x-uv.y)+scene[t+5u+field]*uv.x+scene[t+10u+field]*uv.y; }
struct Surface { n:vec3<f32>, gn:vec3<f32>, base:vec3<f32>, alpha:f32, metal:f32, rough:f32, f0:vec3<f32>, ao:f32, emission:vec3<f32>, receive_caustics:f32 }
fn surface(hit:Hit, d:vec3<f32>) -> Surface {
    let m=bitcast<u32>(p.v[1].y)+u32(scene[hit.tri+15u].x)*10u;
    let uv=interp(hit.tri,3u,hit.uv).xy;
    // Screen-space UV footprint: world size per pixel divided by world size per
    // UV unit, both taken from this triangle. Single-level textures ignore it.
    let uv0=scene[hit.tri+3u].xy; let uv1=scene[hit.tri+8u].xy; let uv2=scene[hit.tri+13u].xy;
    let v0=scene[hit.tri].xyz;
    let e1=scene[hit.tri+5u].xyz-v0; let e2=scene[hit.tri+10u].xyz-v0;
    let duv1=uv1-uv0; let duv2=uv2-uv0;
    let uv_per_world=0.5*(length(duv1)/max(length(e1),1e-6)+length(duv2)/max(length(e2),1e-6));
    let pixel_angle=2.0*p.v[3].w/max(p.v[12].w,1.0);
    let lod_bias=log2(max(hit.t*pixel_angle*uv_per_world,1e-8));
    let color=interp(hit.tri,4u,hit.uv)*texture_auto(scene[m+4u],uv,true,lod_bias);
    let mr=texture_auto(scene[m+6u],uv,false,lod_bias);
    let remapped=material_channels(mr,scene[m+9u].x);
    var gn=unit(cross(scene[hit.tri+5u].xyz-scene[hit.tri].xyz,scene[hit.tri+10u].xyz-scene[hit.tri].xyz),-d);
    var n=unit(interp(hit.tri,1u,hit.uv).xyz,gn);
    if dot(gn,d)>0.0 { gn=-gn; n=-n; }
    let tangent=interp(hit.tri,2u,hit.uv); let t=unit(tangent.xyz-n*dot(n,tangent.xyz),vec3<f32>(0));
    if dot(t,t)>0.5 {
        var map=texture_auto(scene[m+5u],uv,false,lod_bias).xyz*2.0-1.0; map=vec3<f32>(map.xy*scene[m].z,map.z);
        n=unit(t*map.x+cross(n,t)*tangent.w*map.y+n*map.z,gn);
    }
    if dot(n,-d)<=0.0 { n=gn; }
    let metal=clamp(scene[m].x*remapped.x,0.0,1.0); let rough=clamp(scene[m].y*remapped.y,0.03,1.0);
    var alpha=1.0;
    if scene[m].w==1.0 { alpha=select(0.0,1.0,color.a>=scene[m+3u].x); }
    if scene[m].w==2.0 { alpha=color.a; }
    let ao_tex=material_channels(texture_auto(scene[m+8u],uv,false,lod_bias),scene[m+9u].x).z;
    // Normal-based hemisphere occlusion mirrors the preview AO strength.
    let ao_normal=clamp(1.0-p.v[15].y*(1.0-max(n.y,0.0))*0.35,0.15,1.0);
    let ao=clamp(ao_tex*ao_normal,0.0,1.0);
    let authored_ior=clamp(scene[m+3u].y,1.0,3.0);
    let ior_ratio=(authored_ior-1.0)/(authored_ior+1.0);
    let dielectric_f0=ior_ratio*ior_ratio*scene[m+2u].rgb*scene[m+2u].w;
    return Surface(n,gn,color.rgb,alpha,metal,rough,mix(dielectric_f0,color.rgb,metal),ao,scene[m+1u].rgb*scene[m+1u].w*texture_auto(scene[m+7u],uv,true,lod_bias).rgb,scene[m+9u].z);
}
fn basis(n:vec3<f32>, q:vec3<f32>) -> vec3<f32> {
    let a=select(vec3<f32>(0,1,0),vec3<f32>(1,0,0),abs(n.y)>0.95);
    let t=normalize(cross(a,n)); return t*q.x+cross(n,t)*q.y+n*q.z;
}
fn cosine(n:vec3<f32>) -> vec3<f32> { let r=sqrt(random()); let a=2.0*PI*random(); return basis(n,vec3<f32>(r*cos(a),r*sin(a),sqrt(max(0.0,1.0-r*r)))); }
fn g1(c:f32,a2:f32)->f32 { return 2.0*c/max(c+sqrt(a2+(1.0-a2)*c*c),1e-8); }
fn probability(s:Surface)->f32 { return clamp(0.25+0.5*s.metal,0.1,0.9); }
fn bsdf(s:Surface, v:vec3<f32>, l:vec3<f32>)->vec4<f32> {
    let nl=dot(s.n,l); let nv=dot(s.n,v); if nl<=0.0 || nv<=0.0 || dot(s.gn,l)<=0.0 { return vec4<f32>(0); }
    let h=unit(v+l,s.n); let nh=clamp(dot(s.n,h),0.0,1.0); let vh=clamp(dot(v,h),1e-8,1.0);
    let a2=pow(s.rough,4.0); let den=nh*nh*(a2-1.0)+1.0; let dist=a2/(PI*den*den);
    // Schlick's cosine is bounded: pow on a negative roundoff is undefined.
    let one_minus=1.0-vh;let fresnel=s.f0+(1.0-s.f0)*one_minus*one_minus*one_minus*one_minus*one_minus;
    let spec=dist*g1(nv,a2)*g1(nl,a2)*fresnel/max(4.0*nv*nl,1e-8);
    let diffuse=(1.0-s.metal)*s.base*(1.0-fresnel)/PI;
    let pdf=mix(nl/PI,dist*nh/(4.0*vh),probability(s));
    return vec4<f32>(diffuse+spec,pdf);
}
fn env_uv(d:vec3<f32>)->vec2<f32> {
    // At a sampled pole, sin(theta) can round to zero: atan2(0,0) is undefined.
    var longitude=0.0;
    if dot(d.xz,d.xz)>1e-20 {longitude=atan2(d.z,d.x);}
    return vec2<f32>(fract(longitude/(2.0*PI)+0.5+p.v[10].w/(2.0*PI)),acos(clamp(d.y,-1.0,1.0))/PI);
}
fn env_texel(q:vec2<i32>)->vec3<f32> {
    let wh=vec2<i32>(p.v[10].yz);let x=((q.x%wh.x)+wh.x)%wh.x;let y=clamp(q.y,0,wh.y-1);
    return scene[bitcast<u32>(p.v[10].x)+u32(y*wh.x+x)].rgb;
}
fn env_radiance(d:vec3<f32>)->vec3<f32> {
    if p.v[10].y<1.0 {return vec3<f32>(0);}
    let q=env_uv(d)*p.v[10].yz-0.5;let i=vec2<i32>(floor(q));let f=fract(q);
    return mix(mix(env_texel(i),env_texel(i+vec2<i32>(1,0)),f.x),mix(env_texel(i+vec2<i32>(0,1)),env_texel(i+vec2<i32>(1,1)),f.x),f.y);
}
fn env(d:vec3<f32>)->vec3<f32> {return env_radiance(d)*p.v[11].x;}
// Diffuse sky fill honours ambientIntensity/ambientColor and diffuseIntensity;
// specular reflections honour specularIntensity. Defaults keep prior output.
// `env()` already applies the base intensity, so these exclude p.v[11].x.
fn env_diffuse_scale()->vec3<f32> {
    return vec3<f32>(p.v[11].z)*p.v[19].x*p.v[19].yzw;
}
fn env_specular_scale()->vec3<f32> {
    return vec3<f32>(p.v[11].w);
}
fn env_pdf(d:vec3<f32>)->f32 {
    if p.v[13].y<1.0 {return 1.0/(4.0*PI);}
    let wh=vec2<u32>(p.v[13].yz);let cell=min(vec2<u32>(env_uv(d)*vec2<f32>(wh)),wh-vec2<u32>(1u));
    return scene[bitcast<u32>(p.v[13].x)+cell.y*wh.x+cell.x].y;
}
fn env_sample()->vec3<f32> {
    if p.v[13].y<1.0 {let z=1.0-2.0*random();let a=2.0*PI*random();let r=sqrt(max(0.0,1.0-z*z));return vec3<f32>(r*cos(a),z,r*sin(a));}
    let wh=vec2<u32>(p.v[13].yz);var lo=0u;var hi=wh.x*wh.y-1u;let u=random();
    loop {if lo>=hi {break;}let mid=(lo+hi)/2u;if scene[bitcast<u32>(p.v[13].x)+mid].x<u {lo=mid+1u;}else{hi=mid;}}
    let x=lo%wh.x;let y=lo/wh.x;
    let z=mix(cos(PI*f32(y)/f32(wh.y)),cos(PI*f32(y+1u)/f32(wh.y)),random());
    let phi=2.0*PI*((f32(x)+random())/f32(wh.x)-0.5)-p.v[10].w;
    let r=sqrt(max(0.0,1.0-z*z));return vec3<f32>(r*cos(phi),z,r*sin(phi));
}
fn medium_segment(o:vec3<f32>,d:vec3<f32>,distance:f32)->vec2<f32> {
    if p.v[16].w<=0.0 {return vec2<f32>(0);}
    var lo=0.0;var hi=distance;
    for(var a=0u;a<3u;a++) {
        if abs(d[a])<1e-12 {if o[a]<p.v[16][a] || o[a]>p.v[17][a] {return vec2<f32>(0);}}
        else {let x=(p.v[16][a]-o[a])/d[a];let y=(p.v[17][a]-o[a])/d[a];lo=max(lo,min(x,y));hi=min(hi,max(x,y));}
    }
    return vec2<f32>(lo,max(hi-lo,0.0));
}
fn medium_density(point:vec3<f32>)->f32 {
    let height=exp(-max(point.y-p.v[9].x,0.0)*p.v[9].y);
    let edge3=min(point-p.v[16].xyz,p.v[17].xyz-point);
    let edge=min(edge3.x,min(edge3.y,edge3.z));
    let feather=select(1.0,smoothstep(0.0,p.v[9].z,edge),p.v[9].z>0.000001);
    return p.v[16].w*height*feather;
}
fn caustic_pattern(point:vec3<f32>)->f32 {
    let q=point.xz/max(p.v[20].y,0.0001);
    let t=p.v[20].z;
    let a=sin(q.x*1.31+q.y*0.77+t);
    let b=sin(q.x*-0.63+q.y*1.67-t*1.23);
    let c=sin(q.x*1.91-q.y*0.41+t*0.71);
    return pow(clamp((a+b+c)*0.1667+0.5,0.0,1.0),5.0);
}
fn caustic_radiance(point:vec3<f32>)->vec3<f32> {
    let attenuation=exp(-max(0.0,p.v[17].y-point.y)*p.v[20].w);
    return p.v[21].rgb*caustic_pattern(point)*p.v[20].x*attenuation;
}
fn phase(c:f32)->f32 {let g=p.v[17].w;return (1.0-g*g)/(4.0*PI*pow(max(1.0+g*g-2.0*g*c,1e-8),1.5));}
fn phase_sample(d:vec3<f32>)->vec3<f32> {
    let g=p.v[17].w;let u=random();var c=1.0-2.0*u;
    if abs(g)>0.001 {let s=(1.0-g*g)/(1.0-g+2.0*g*u);c=(1.0+g*g-s*s)/(2.0*g);}
    c=clamp(c,-1.0,1.0);let angle=2.0*PI*random();let r=sqrt(max(0.0,1.0-c*c));return basis(d,vec3<f32>(r*cos(angle),r*sin(angle),c));
}
fn visible(o:vec3<f32>, d:vec3<f32>, distance:f32)->f32 {
    var origin=o; var remain=distance;
    let atmosphere_distance=select(0.0,distance,distance<INF*0.5 || p.v[15].z>0.5);
    let segment=medium_segment(o,d,atmosphere_distance);
    let midpoint=o+d*(segment.x+segment.y*0.5);
    var trans=exp(-medium_density(midpoint)*segment.y);
    for(var i=0u;i<u32(p.v[9].w);i++) {
        let h=intersect(origin,d,remain,false); if h.tri==0xffffffffu { return trans; }
        let s=surface(h,d); trans*=1.0-s.alpha; if trans<1e-4 { return 0.0; }
        origin+=d*(h.t+0.0002); remain-=h.t+0.0002;
    }
    return 0.0;
}
// Sample authored rectangular emitters in solid angle. The authored direction
// is the emitting face normal; width and height are physical scene units.
struct RectLightSample { direction:vec3<f32>, distance:f32, weight:f32 }
fn rect_light(light:vec4<f32>, settings:vec4<f32>, shape:vec4<f32>, dimensions:vec4<f32>, point:vec3<f32>)->RectLightSample {
    let normal=unit(settings.xyz,vec3<f32>(0,-1,0));
    let width=max(shape.w,1e-5);let height=max(dimensions.x,1e-5);
    let local=vec3<f32>((random()-0.5)*width,(random()-0.5)*height,0.0);
    let sample_point=light.xyz+basis(normal,local);let delta=sample_point-point;let distance=length(delta);
    let direction=delta/max(distance,1e-5);let facing=max(dot(normal,-direction),0.0);
    return RectLightSample(direction,distance,width*height*facing/max(distance*distance,1e-8));
}
fn power(a:f32,b:f32)->f32 {
    // Normalize before squaring to keep grazing-angle light PDFs from overflowing.
    let scale=max(max(a,b),1e-20);let x=a/scale;let y=b/scale;
    return x*x/max(x*x+y*y,1e-20);
}
fn finite3(v:vec3<f32>)->bool {return !any((bitcast<vec3<u32>>(v)&vec3<u32>(0x7f800000u))==vec3<u32>(0x7f800000u));}
struct Sample { color:vec3<f32>, albedo:vec3<f32>, normal:vec3<f32>, position:vec3<f32>, depth:f32 }
fn trace(origin:vec3<f32>, direction:vec3<f32>)->Sample {
    var o=origin; var d=direction; var throughput=vec3<f32>(1); var radiance=vec3<f32>(0);
    var first=Sample(vec3<f32>(0),vec3<f32>(0),vec3<f32>(0),vec3<f32>(0),0.0);
    var prev_pdf=0.0; var diffuse=0u; var glossy=0u; var transparent=0u; var bounce=0u;var volume_bounces=0u;var prev_emitter_nee=false;
    var prev_diffuse=true; var prev_ao=1.0;
    loop {
        if bounce>=u32(p.v[8].x) { break; }
        let h=intersect(o,d,INF,bounce==0u);
        let atmosphere_distance=select(0.0,h.t,h.tri!=0xffffffffu || p.v[15].z>0.5);
        let segment=medium_segment(o,d,atmosphere_distance);
        if segment.y>0.0 {
            let segment_density=medium_density(o+d*(segment.x+segment.y*0.5));
            let free_path=-log(max(1.0-random(),1e-7))/max(segment_density,1e-8);
            if free_path<segment.y {
                volume_bounces++;if volume_bounces>u32(p.v[18].w) {break;}
                let point=o+d*(segment.x+free_path);throughput*=p.v[18].rgb;
                for(var li=0u;li<u32(p.v[1].w);li++) {
                    let at=bitcast<u32>(p.v[1].z)+li*4u;let light=scene[at];let settings=scene[at+1u];
                    var l=normalize(-settings.xyz);var distance=INF;var energy=settings.w;
                    if p.v[6].w>0.5 && abs(f32(li+1u)-p.v[6].w)<0.5 {
                        if p.v[6].z>0.0 && length(point-origin)>p.v[6].z {continue;}
                        energy*=p.v[15].w;
                    }
                    if light.w==0.0 && p.v[15].x>0.0 {
                        let c=mix(1.0,cos(p.v[15].x),random());let a=2.0*PI*random();let r=sqrt(max(0.0,1.0-c*c));
                        l=basis(l,vec3<f32>(r*cos(a),r*sin(a),c));let pdf=1.0/max(2.0*PI*(1.0-cos(p.v[15].x)),1e-9);energy*=power(pdf,phase(dot(d,l)));
                    }
                    if light.w==1.0 {let delta=light.xyz-point;distance=length(delta);l=delta/max(distance,1e-5);energy/=max(distance*distance,1e-5);}
                    if light.w==3.0 {let sample=rect_light(light,settings,scene[at+2u],scene[at+3u],point);l=sample.direction;distance=sample.distance;energy*=sample.weight;}
                    radiance+=throughput*scene[at+2u].rgb*energy*phase(dot(d,l))*visible(point,l,distance-0.0004);
                }
                if (u32(p.v[21].w)&1u)!=0u {radiance+=throughput*caustic_radiance(point);}
                let l=env_sample();let pdf=env_pdf(l);let f=phase(dot(d,l));
                radiance+=throughput*env(l)*env_diffuse_scale()*f*visible(point,l,INF)*power(pdf,f)/pdf;
                let next=phase_sample(d);prev_pdf=phase(dot(d,next));prev_emitter_nee=false;prev_diffuse=true;prev_ao=1.0;o=point;d=next;bounce++;
                if bounce>=u32(p.v[8].w) {let survive=clamp(max(throughput.r,max(throughput.g,throughput.b)),0.05,0.95);if random()>survive {break;}throughput/=survive;}
                continue;
            }
        }
        if h.tri==0xffffffffu {
            if bounce==0u {
                radiance+=throughput*env_radiance(d)*p.v[11].y;
            } else {
                let weight=power(prev_pdf,env_pdf(d));
                let scale=select(env_specular_scale(),env_diffuse_scale()*prev_ao,prev_diffuse);
                radiance+=throughput*env_radiance(d)*weight*p.v[11].x*scale;
            }
            // Finite directional emitters also appear in glossy reflections.
            if p.v[15].x>0.0 {
                let cos_radius=cos(p.v[15].x);let pdf=1.0/max(2.0*PI*(1.0-cos_radius),1e-9);
                for(var li=0u;li<u32(p.v[1].w);li++) {
                    let at=bitcast<u32>(p.v[1].z)+li*4u;
                    if scene[at].w==0.0 && dot(d,normalize(-scene[at+1u].xyz))>=cos_radius {
                        radiance+=throughput*scene[at+2u].rgb*scene[at+1u].w*pdf*select(power(prev_pdf,pdf),1.0,bounce==0u);
                    }
                }
            }
            break;
        }
        let s=surface(h,d); let point=o+d*h.t;
        if !finite3(s.n) || !finite3(s.gn) {return Sample(vec3<f32>(bitcast<f32>(0x7fc00000u)),vec3<f32>(10),s.n,point,f32(h.tri));}
        if random()>s.alpha {
            transparent++; if transparent>=u32(p.v[9].w) { break; }
            o=point+d*0.0002; continue;
        }
        if bounce==0u { first.albedo=s.base; first.normal=s.n; first.position=point; first.depth=h.t; }
        var emission_weight=1.0;
        if bounce>0u && prev_emitter_nee && p.v[14].z>0.0 {
            let emitter_pdf=h.t*h.t/max(abs(dot(s.gn,-d))*p.v[14].z,1e-10);
            emission_weight=power(prev_pdf,emitter_pdf);
        }
        radiance+=throughput*s.emission*emission_weight;
        if u32(p.v[21].w)>=2u && s.receive_caustics>0.5 {
            radiance+=throughput*s.base*caustic_radiance(point)*max(s.n.y,0.0);
        }
        if !finite3(radiance) {return Sample(radiance,vec3<f32>(20),throughput,point,f32(h.tri));}
        if p.v[14].y>0.0 {
            var lo=0u;var hi=u32(p.v[14].y)-1u;let choice=random();
            loop {if lo>=hi {break;}let mid=(lo+hi)/2u;if scene[bitcast<u32>(p.v[14].x)+mid].x<choice {lo=mid+1u;}else{hi=mid;}}
            let tri=bitcast<u32>(scene[bitcast<u32>(p.v[14].x)+lo].y);let u=sqrt(random());let v=random();let uv=vec2<f32>(u*(1.0-v),u*v);
            let emitter_point=interp(tri,0u,uv).xyz;let delta=emitter_point-point;let distance=length(delta);let l=delta/max(distance,1e-10);
            let emitter=surface(Hit(distance,tri,uv),l);
            let pdf=distance*distance/max(abs(dot(emitter.gn,-l))*p.v[14].z,1e-10);
            let f=bsdf(s,-d,l);
            if f.a>0.0 {radiance+=throughput*f.rgb*max(dot(s.n,l),0.0)*emitter.emission*emitter.alpha*visible(point+s.gn*0.0002,l,distance-0.0006)*power(pdf,f.a)/pdf;}
            if !finite3(radiance) {return Sample(radiance,vec3<f32>(30),vec3<f32>(pdf,f.a,distance),point,f32(tri));}
        }
        // Delta lights have no competing BSDF sampling strategy.
        for(var li=0u;li<u32(p.v[1].w);li++) {
            let at=bitcast<u32>(p.v[1].z)+li*4u; let light=scene[at]; let settings=scene[at+1u];
            var l=normalize(-settings.xyz); var distance=INF; var energy=settings.w;
            var sun_pdf=0.0;
            if light.w==0.0 && p.v[15].x>0.0 {
                let c=mix(1.0,cos(p.v[15].x),random());let angle=2.0*PI*random();let radius=sqrt(max(0.0,1.0-c*c));
                l=basis(l,vec3<f32>(radius*cos(angle),radius*sin(angle),c));sun_pdf=1.0/max(2.0*PI*(1.0-cos(p.v[15].x)),1e-9);
            }
            if light.w==1.0 { let delta=light.xyz-point; distance=length(delta); l=delta/max(distance,1e-5); energy/=max(distance*distance,1e-5); }
            if light.w==3.0 { let sample=rect_light(light,settings,scene[at+2u],scene[at+3u],point); l=sample.direction; distance=sample.distance; energy*=sample.weight; }
            let f=bsdf(s,-d,l);
            if sun_pdf>0.0 {energy*=power(sun_pdf,f.a);}
            if f.a>0.0 { radiance+=throughput*f.rgb*max(dot(s.n,l),0.0)*scene[at+2u].rgb*energy*visible(point+s.gn*0.0002,l,distance-0.0004); }
            if !finite3(radiance) {return Sample(radiance,vec3<f32>(40),f.rgb,point,f32(li));}
        }
        // Environment NEE and BSDF sampling use complementary MIS weights.
        let light_dir=env_sample(); let light_pdf=env_pdf(light_dir);
        let ef=bsdf(s,-d,light_dir);
        let env_scale=mix(env_diffuse_scale()*s.ao,env_specular_scale(),probability(s));
        if ef.a>0.0 { radiance+=throughput*ef.rgb*max(dot(s.n,light_dir),0.0)*env(light_dir)*env_scale*visible(point+s.gn*0.0002,light_dir,INF)*power(light_pdf,ef.a)/light_pdf; }
        if !finite3(radiance) {return Sample(radiance,vec3<f32>(50),vec3<f32>(light_pdf,ef.a,f32(bounce)),point,f32(h.tri));}
        var next=vec3<f32>(0);
        if random()<probability(s) {
            glossy++; if glossy>u32(p.v[8].z) { break; }
            let u=random(); let a2=pow(s.rough,4.0); let c=sqrt((1.0-u)/(1.0+(a2-1.0)*u)); let phi=2.0*PI*random();
            let hh=basis(s.n,vec3<f32>(sqrt(max(0.0,1.0-c*c))*cos(phi),sqrt(max(0.0,1.0-c*c))*sin(phi),c));
            next=reflect(d,hh); prev_diffuse=false;
        } else { diffuse++; if diffuse>u32(p.v[8].y) { break; } next=cosine(s.n); prev_diffuse=true; }
        let f=bsdf(s,-d,next); if f.a<1e-10 { break; }
        throughput*=f.rgb*max(dot(s.n,next),0.0)/f.a; prev_pdf=f.a;prev_emitter_nee=true;prev_ao=s.ao;
        if !finite3(throughput) {return Sample(throughput,vec3<f32>(60),f.rgb,point,f.a);}
        bounce++;
        if bounce>=u32(p.v[8].w) {
            let survive=clamp(max(throughput.r,max(throughput.g,throughput.b)),0.05,0.95);
            if random()>survive { break; } throughput/=survive;
        }
        o=point+s.gn*0.0002; d=next;
    }
    first.color=radiance; return first;
}
fn camera_motion(position:vec3<f32>, pixel:vec2<u32>)->vec2<f32> {
    if p.v[25].w<0.5 { return vec2<f32>(0); }
    let relative=position-p.v[22].xyz;
    let depth=dot(relative,p.v[23].xyz);
    if depth<=1e-5 { return vec2<f32>(0); }
    let tangent=max(p.v[23].w,1e-6);
    let aspect=max(p.v[24].w,1e-6);
    let ndc=vec2<f32>(
        dot(relative,p.v[24].xyz)/(depth*tangent*aspect),
        -dot(relative,p.v[25].xyz)/(depth*tangent)
    );
    let previous=(ndc*0.5+vec2<f32>(0.5))*p.v[12].zw;
    return vec2<f32>(pixel)+vec2<f32>(0.5)-previous;
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let wh=vec2<u32>(p.v[0].xy); if id.x>=wh.x || id.y>=wh.y { return; }
    let index=id.y*wh.x+id.x; var f=film[index];
    for(var batch=0u;batch<u32(p.v[0].z);batch++) {
        let count=f.sum.w; if count>=p.v[7].y { break; }
        if count>=p.v[7].x && p.v[7].z>0.0 {
            let error=sqrt(max(f.stats.y/(count-1.0)/count,0.0));
            if error<=p.v[7].z*max(abs(f.stats.x),0.01) { break; }
        }
        let pixel=id.xy+vec2<u32>(p.v[12].xy);
        let global_index=pixel.y*u32(p.v[12].z)+pixel.x;
        rng=(global_index*1973u)^((u32(count)+1u)*9277u)^bitcast<u32>(p.v[0].w);
        let q=(vec2<f32>(pixel)+vec2<f32>(random(),random()))/p.v[12].zw;
        var direction=normalize(p.v[3].xyz+p.v[4].xyz*((2.0*q.x-1.0)*p.v[3].w*p.v[4].w)+p.v[5].xyz*((1.0-2.0*q.y)*p.v[3].w));
        var origin=p.v[2].xyz;
        if p.v[5].w>0.0 {
            let angle=2.0*PI*random(); let radius=sqrt(random()); var disk=vec2<f32>(cos(angle),sin(angle))*radius;
            let blades=p.v[6].y;
            if blades>=3.0 {
                let sector=2.0*PI/blades; let k=floor(angle/sector); let u=fract(angle/sector);
                disk=radius*mix(vec2<f32>(cos(k*sector),sin(k*sector)),vec2<f32>(cos((k+1.0)*sector),sin((k+1.0)*sector)),u);
            }
            let focus=origin+direction*p.v[6].x/dot(direction,p.v[3].xyz);
            origin+=(p.v[4].xyz*disk.x+p.v[5].xyz*disk.y)*p.v[5].w; direction=normalize(focus-origin);
        }
        let sample=trace(origin,direction);
        // Non-finite samples are a render error, recorded for the host to reject.
        if !finite3(sample.color) || any(abs(sample.color)>vec3<f32>(1e25)) { f.stats.w+=1.0;f.stats.z=sample.albedo.x;f.albedo=vec4<f32>(sample.color,0.0);f.normal=vec4<f32>(sample.normal,sample.depth); break; }
        f.sum+=vec4<f32>(sample.color,1.0);
        let lum=dot(sample.color,vec3<f32>(0.2126,0.7152,0.0722)); let delta=lum-f.stats.x;
        f.stats.x+=delta/f.sum.w; f.stats.y+=delta*(lum-f.stats.x);
        f.albedo+=vec4<f32>(sample.albedo,0.0); f.normal+=vec4<f32>(sample.normal,sample.depth);
        let motion=select(vec2<f32>(0),camera_motion(sample.position,pixel),sample.depth>0.0);
        f.aov+=vec4<f32>(select(0.0,1.0,sample.depth>0.0),motion,0.0);
    }
    film[index]=f;
}
