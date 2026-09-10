// =========================================
// =========================================
// crates/motionloom/src/process/kernels/stylize_look/procedural_surface.wgsl

struct SurfaceParams {
    frame: vec4<f32>,
    field: vec4<f32>,
    material: vec4<f32>,
    light_view: vec4<f32>,
    view: vec4<f32>,
    base: vec4<f32>,
    gold: vec4<f32>,
    reserved: vec4<f32>,
};
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<uniform> s: SurfaceParams;

// Polynomial hashing avoids backend-dependent trigonometric hash amplification.
fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + vec3<f32>(33.33));
    return fract((p3.x + p3.y) * p3.z);
}
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2<f32>(1,0)),u.x),mix(hash(i+vec2<f32>(0,1)),hash(i+vec2<f32>(1,1)),u.x),u.y);
}
fn fbm(p0: vec2<f32>) -> f32 {
    var p=p0; var v=0.0; var a=0.5;
    for(var i=0;i<4;i++){v+=a*noise(p);p=mat2x2<f32>(0.8,0.6,-0.6,0.8)*p*2.02+vec2<f32>(7.1,13.7);a*=0.5;}
    return v;
}
// A single continuous field supplies both metal boundaries and folded substrate.
fn ml_surface(p: vec2<f32>) -> vec2<f32> {
    // Opt-in analytic transport keeps seeking stateless and preserves legacy relief.
    if(s.view.z>0.0){
        let t=s.field.y;
        var uv=p-vec2<f32>(t*0.65,t*0.08);
        // Coherent shear stretches nested contours along a common current.
        uv.y+=0.32*sin(uv.x*1.4+t*0.2);
        let q=vec2<f32>(fbm(uv*0.65),fbm(uv*0.65+vec2<f32>(5.2,1.3)))-0.45;
        let bend=uv+q*s.field.z*2.2;
        let v=fbm(bend*vec2<f32>(0.55,1.15));
        let phase=bend.y*7.0+v*38.0;
        let ribbons=sin(phase*5.0)+0.3*sin(phase*9.0+v*3.0);
        let height=v*0.18+ribbons*0.009*s.field.w;
        if(s.view.z>=1.0){return vec2<f32>(height,v);}
        let legacy=ml_relief(p);
        return mix(legacy,vec2<f32>(height,v),s.view.z);
    }
    return ml_relief(p);
}
fn ml_relief(p: vec2<f32>) -> vec2<f32> {
    let t=s.field.y;
    let q=vec2<f32>(fbm(p+vec2<f32>(t*0.12,0)),fbm(p+vec2<f32>(5.2,1.3+t*0.1)));
    let r=vec2<f32>(fbm(p+q*4.0*s.field.z+vec2<f32>(1.7,9.2)),fbm(p+q*4.0*s.field.z+vec2<f32>(8.3,2.8)));
    let v=fbm(p+4.0*r*s.field.z);
    let wrinkle=sin(v*85.0+fbm(p*3.0+r*5.0)*12.0);
    let height=v*0.3+wrinkle*0.035*s.field.w;
    return vec2<f32>(height,v);
}
@vertex fn vs_main(@builtin(vertex_index) i:u32) -> @builtin(position) vec4<f32> {
    let x=f32((i<<1u)&2u); let y=f32(i&2u);
    return vec4<f32>(x*2.0-1.0,1.0-y*2.0,0.0,1.0);
}
@fragment fn fs_main(@builtin(position) pos:vec4<f32>) -> @location(0) vec4<f32> {
    let original=textureLoad(source,vec2<i32>(pos.xy),0);
    if(s.frame.z<=0.0 || original.a<=0.0){return original;}
    var p=(pos.xy/s.frame.xy-0.5)*vec2<f32>(s.frame.x/s.frame.y,1.0);
    let c=cos(s.view.y);let sn=sin(s.view.y);
    p=mat2x2<f32>(c,sn,-sn,c)*p*s.field.x/s.view.x+s.light_view.zw;
    p+=vec2<f32>(s.frame.w*0.013,s.frame.w*0.019);
    let epsilon=max(s.field.x/s.view.x/s.frame.y,0.0005);
    let h=ml_surface(p);
    let hx=ml_surface(p+vec2<f32>(epsilon,0));let hy=ml_surface(p+vec2<f32>(0,epsilon));
    let grad=vec2<f32>(hx.x-h.x,hy.x-h.x)/epsilon;
    let normal=normalize(vec3<f32>(-grad*s.material.z,1.0));
    let light=vec3<f32>(cos(s.light_view.x)*cos(s.light_view.y),sin(s.light_view.x)*cos(s.light_view.y),sin(s.light_view.y));
    let halfv=normalize(light+vec3<f32>(0,0,1));
    let diffuse=max(dot(normal,light),0.0);
    let spec=pow(max(dot(normal,halfv),0.0),mix(130.0,6.0,s.material.w));
    // Derivative-sized edges suppress subpixel flicker as the view pulls away.
    let edge=max(length(vec2<f32>(hx.y-h.y,hy.y-h.y)),0.001);
    let width=s.material.x*(0.35+0.65*noise(p*2.0));
    let vein=1.0-smoothstep(width,width+edge,abs(h.y-0.48));
    let pool=smoothstep(0.7-s.material.y*0.22,0.74-s.material.y*0.22,h.y);
    let metal=max(vein,pool);
    let base=pow(s.base.rgb,vec3<f32>(2.2));let gold=pow(s.gold.rgb,vec3<f32>(2.2));
    let body=base*(0.6+diffuse*1.5)+vec3<f32>(0.18,0.22,0.34)*spec*0.65;
    let liquid=gold*(0.35+diffuse*1.15)+mix(gold,vec3<f32>(1),0.6)*spec*0.9;
    let linear=mix(body,liquid,metal);
    let display=pow(max(linear,vec3<f32>(0)),vec3<f32>(1.0/2.2));
    return vec4<f32>(mix(original.rgb,display*original.a,s.frame.z),original.a);
}
