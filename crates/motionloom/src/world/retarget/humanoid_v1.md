# humanoid_v1 Retarget Standard

This file is the stable Motionloom reference for `humanoid_v1`. Do not change
the semantic meanings below to fix one model. Per-model differences must be
handled with `Retarget` and `BoneAxisMap`.

## Coordinate System

- World up: `+Y`
- Character right: `+X`
- Character forward: `-Z`
- Canonical neutral pose: `arms_down`

## Canonical Bones

The canonical bone ids are:

```text
hips
spine
chest
upper_chest
neck
head
shoulder_l
upper_arm_l
forearm_l
hand_l
shoulder_r
upper_arm_r
forearm_r
hand_r
upper_leg_l
lower_leg_l
foot_l
toe_l
upper_leg_r
lower_leg_r
foot_r
toe_r
```

These 22 body bones are required for a new full-conformance
`humanoid_v1` profile. Existing profiles remain parseable and playable; tools
report a missing required bone so the profile can be upgraded without silently
changing old content.

The following 30 finger bones are canonical but optional:

```text
thumb_1_l thumb_2_l thumb_3_l
index_1_l index_2_l index_3_l
middle_1_l middle_2_l middle_3_l
ring_1_l ring_2_l ring_3_l
pinky_1_l pinky_2_l pinky_3_l
thumb_1_r thumb_2_r thumb_3_r
index_1_r index_2_r index_3_r
middle_1_r middle_2_r middle_3_r
ring_1_r ring_2_r ring_3_r
pinky_1_r pinky_2_r pinky_3_r
```

Optional means that an unrelated body Action can still play on a model without
finger joints. If an Action actually keys one of these bones, compatibility
validation must report the missing mapping as degraded playback. Twist/helper
bones remain target-rig implementation details and are not authored directly
by portable Actions.

## Semantic Roles

These meanings are fixed:

- `turn`: rotate around the vertical body turn direction.
- `bend`: flex the joint, for example elbow bend or knee bend.
- `forward`: raise or swing a limb toward character forward.
- `side`: abduct a limb away from the body. For arms, positive `side` moves
  from arms down toward A-pose and T-pose.
- `twist`: roll around the limb's long axis.

`BoneAxisMap` maps these semantic roles to a specific GLB bone's local rotation
axis. The chosen local axis and sign may differ per model and per side. The
semantic role itself must not change.

Positive `bend` always means natural joint flexion in canonical Action data.
For a knee, that means bringing the heel towards the back of the thigh. A
source animation and a target model may use opposite local-axis signs for that
same motion. Source conversion handles the source sign; the target
`BoneAxisMap` handles the target sign. Do not place source-specific
`rotationX` compensation inside a reusable Action.

`BoneAxisMap` also owns model rest calibration. Use `restForward`, `restSide`,
`restTwist`, `restBend`, and `restTurn` on the same `<Axis>` node. These values
are semantic offsets and are converted through that axis mapping.

## Pose Types

- `Original`: raw GLB bind/rest pose. No action.
- `Raw Axis Test`: small debug actions such as `side +20`. No
  rest offsets; used only to verify `BoneAxisMap`.
- `Action Preview`: canonical actions such as `arms_down`, `a_pose`, and
  `t_pose`. `BoneAxisMap` rest offsets are applied.

## Canonical Actions

- `arms_down`: no additional action after rest calibration.
- `a_pose`: `upper_arm_l.side = 35`, `upper_arm_r.side = 35`.
- `t_pose`: `upper_arm_l.side = 90`, `upper_arm_r.side = 90`.
- `walk`: reusable humanoid walk preview, 1 second loop, using semantic
  `upper_leg.forward` and `lower_leg.bend`.
- `jump`: reusable humanoid jump preview, 1.6 second loop, matching
  `examples/motionloom/world/actions/humanoid_jump.motionloom`.
- `wave_hand`: reusable humanoid right-hand wave preview, 2 second loop, using
  semantic `upper_arm_r.side`, `forearm_r.bend`, and arm-chain `twist`.
- `side_plus_20`: raw axis debug only, `upper_arm_l.side = 20`,
  `upper_arm_r.side = 20`.
- `side_minus_20`: raw axis debug only, `upper_arm_l.side = -20`,
  `upper_arm_r.side = -20`.
- `forward_plus_20`: raw axis debug only, arms and upper legs `forward = 20`.
- `bend_plus_20`: raw axis debug only, forearms and lower legs `bend = 20`.
- `twist_plus_20`: raw axis debug only, arm chain `twist = 20`.

## Rest Calibration

Rest calibration converts a model's source bind pose to canonical `arms_down`.
This correction is per model and lives in `BoneAxisMap` as `rest*` attributes.
There is no separate `RestPoseCorrection` tag.

Example:

```xml
<BoneAxisMap>
  <Axis bone="upper_arm_l"
        forward="rotationZ:1"
        side="rotationX:1"
        twist="rotationY:1"
        restSide="-90" />
</BoneAxisMap>
```

Initial source rest presets:

- source `arms_down`: no arm side rest offset.
- source `a_pose`: upper arms `restSide = -35`.
- source `t_pose`: upper arms `restSide = -90`.

These preset values are a starting point only. If a model's bind pose is not a
clean T-pose, A-pose, or arms-down pose, the model needs custom per-bone
correction.
