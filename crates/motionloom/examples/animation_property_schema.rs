// =========================================
// =========================================
// crates/motionloom/examples/animation_property_schema.rs

fn main() {
    // The runtime-owned registry prevents generated editor metadata from
    // drifting away from parser and renderer capabilities.
    println!("{}", motionloom::api::animation_property_schema_json());
}
