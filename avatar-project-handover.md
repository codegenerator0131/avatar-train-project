# Project Handover: Custom Talking-Head Avatar Pipeline

**Status date:** June 2026
**Reference product:** Tavus (Phoenix model family)
**Current phase:** Stage 1 (capture) delivered, Stage 2 (tracking) next

---

## 1. What we are building

We are building our own foundation technology for photorealistic talking-head avatars, in-house and from first principles, rather than assembling open source repos. A user records a short video of themselves (2 to 4 minutes, single locked camera). From that video we construct a fully 3D, animatable digital head of that person. Once built, the avatar can be driven by audio or text: given speech, it produces a video of the person saying those words with correct lip sync, head motion, and expression.

The benchmark we measure ourselves against is Tavus's Phoenix model. Phoenix went through the same evolution we are planning: Phoenix-1 used NeRFs (a 3D neural representation) driven by TTS, Phoenix-2 moved to 3D Gaussian Splatting to reach real-time speed, and the current Phoenix-4 is a hybrid where a diffusion model generates motion that drives a Gaussian splat renderer. Our minimum bar is Phoenix-1 capability; our roadmap points toward the later generations.

A core decision: the system is 3D internally even though the output is 2D video. Pure 2D approaches (networks that paint new mouth pixels onto frames) suffer from flickering teeth, identity drift, and the inability to move the head. A 3D representation guarantees the face stays consistent from every angle, and Gaussian splatting renders fast enough for eventual real-time use. This is the same conclusion the entire research field and Tavus reached.

We deliberately prototype on the academic FLAME head model (research license) and will later replace it with our own artist-authored parametric head rig so the full stack is commercially ours. Our team's character art and rigging background is a genuine advantage here, most ML teams cannot author rigs.

## 2. How the system works, in one paragraph

We track a deformable 3D head mesh (a rigged parametric head with expression blendshapes and jaw/neck/eye joints) against every frame of the user's video, solving for head pose, expression weights, and camera. We then attach tens of thousands of 3D Gaussian "splats" (soft colored ellipsoids, think of a dense particle groom bound to the skin) to the mesh surface and optimize their positions, scales, colors, and opacities with differentiable rendering until renders of the splats match the video frames. The user's natural head turns during recording provide multi-view coverage, which is what makes true 3D reconstruction from a single camera possible. The result is a per-user avatar asset. Separately, a shared audio-to-motion network (trained once, used for everyone) converts speech audio into the same blendshape curves the tracker produces, so the avatar can be driven by TTS instead of video.

## 3. Pipeline stages

| Stage | Name | Input | Output | Status |
|---|---|---|---|---|
| 1 | Capture | Raw video file | Fixed-crop frames, audio wavs, meta.json | Code delivered (capture.py) |
| 2 | Tracking | Frames + meta | Per-frame FLAME params (pose, expression, jaw, eyes) + camera | Next up |
| 3 | Splat training | Frames + tracking | Trained avatar (splats rigged to mesh) | Designed |
| 4 | Audio to motion | Speech audio | Blendshape curves (same format as stage 2 output) | Designed, trained once for all users |
| 5 | Inference | Text or audio + avatar | Talking-head video | Glue layer, depends on 2 to 4 |

Two details matter for understanding the architecture. First, stage 4 outputs the identical parameter format that stage 2 extracts from video. The avatar does not know or care whether its motion came from tracking a real video or from audio, which is what makes the whole system composable. Second, stages 2 and 3 run per user (every new person gets their own tracked sequence and trained splats), while stage 4 is a shared model.

## 4. Key engineering decisions made so far

The capture stage computes one fixed square crop for the entire video rather than a per-frame moving crop. A moving crop would change the effective camera intrinsics every frame and silently corrupt the 3D math downstream. Locked camera, locked crop, constant intrinsics.

Recording protocol is part of the engineering. The capture spec requires a locked camera with locked exposure, even frontal lighting, roughly 2 minutes of natural talking, 30 seconds of slow head rotation (this provides the multi-view signal), and 30 seconds of exaggerated mouth movement (open jaw, grin, "EE AH OH LA-LA THE-THE"). The mouth section exists because the mouth interior is the hardest region to reconstruct and natural speech alone gives it too little training signal.

Teeth and tongue are the known weak point of every published method, and our plan attacks it with rigging rather than hoping optimization discovers anatomy: upper teeth rigidly bound to the skull, lower teeth rigidly to the jaw bone, and tongue animation driven procedurally from phonemes (linguistics tells us tongue position even when the camera cannot see it). This is roadmap work that builds on our character art strength.

For synthetic or historical subjects (e.g. a museum installation), the preferred path is sculpting a likeness on our own head topology from AI-generated reference images, then rendering that rig from many angles to produce perfectly multi-view-consistent synthetic training data. Raw AI-generated video is a weak training source because generated frames are not 3D-consistent with each other.

## 5. Infrastructure and current assets

Training machine: a native Linux PC with an NVIDIA RTX 4080 Laptop GPU (12GB VRAM). This is sufficient for the entire per-user pipeline (avatar training fits in 12GB at our planned settings of up to ~300k gaussians and 512 to 800px training crops). Cloud GPUs (e.g. Vast.ai) become relevant later when training the shared audio-to-motion model or a future avatar prior model on large datasets.

Delivered files so far: setup_linux.sh (full environment install: CUDA toolkit 12.4, PyTorch cu124 wheels, pipeline dependencies), verify_env.py (six environment checks including a CUDA compile check, all must pass before proceeding), and capture/capture.py (stage 1: face-box scan, fixed crop computation, frame extraction via ffmpeg, audio split into full quality and 16 kHz mono).

Required external asset: FLAME 2023 model files from flame.is.tue.mpg.de (free registration, research license, account must belong to us). These go in ~/avatar/data/flame/. Note the license restriction is exactly why the custom rig is on the roadmap.

## 6. Roadmap after the MVP

Once the per-user pipeline produces a working avatar, the agreed direction has three pillars. First, the custom head rig with proper teeth, tongue, and mouth interior, replacing FLAME and clearing the commercial license path. Second, client-side rendering: ship the splat avatar to the browser once and stream only motion coefficients (a few KB/s instead of video), rendered in our existing WebGPU splat viewer. This eliminates server render cost and most latency, and is a genuine differentiator versus server-rendered competitors. Third, an avatar prior model (in the spirit of the ELITE paper, CVPR 2026): a network trained across many identities that predicts a good avatar initialization instantly, cutting per-user training from hours toward minutes. Further out: relightable avatars and personalized motion style ("it moves like me").

## 7. Cost and timeline expectations

MVP (stages 1 to 3 working on our own faces, offline rendering): on the order of 2 to 4 weeks for the current milestone (one trained avatar of one team member), and 4 to 9 months to a complete text-to-video MVP depending on staffing. Per-user avatar training compute is small (single GPU, under an hour to a few hours). The shared audio-to-motion model is the first item needing a real dataset (50 to 200 hours of talking-head video; public research datasets exist but licenses must be checked for commercial use). Real-time conversational capability is a separate, larger effort after the MVP.

## 8. Ethics and safety requirements (non-negotiable)

This technology can clone a person's likeness. From day one the product must include verified consent from the person being avatared (e.g. a liveness/consent statement in the training video itself), watermarking or provenance signals on generated video, and a policy refusing third-party likenesses without documented consent. For historical figures, check post-mortem publicity rights per jurisdiction. These constraints are also commercially protective: consent-based is what keeps the product defensible.

## 9. Immediate next actions

The team's first checkpoint is simple: run setup_linux.sh on the training PC, run verify_env.py and confirm all six checks pass, record the training video per the capture spec, run capture.py on it, and visually confirm the full head stays inside the crop in every frame. With that done, stage 2 (FLAME fitting and tracking) development begins, which is the mathematical core of the project: landmark detection plus an optimizer that solves head pose, expression, and camera per frame.

## Glossary

**Gaussian splatting:** a 3D scene representation made of many soft, oriented, colored ellipsoids, rendered by projecting them to screen. Very fast to render and trainable via gradient descent.
**FLAME:** an academic parametric 3D head model: a mesh with identity and expression blendshapes plus jaw, neck, and eye joints. Our temporary stand-in for a custom rig.
**Differentiable rendering:** rendering implemented so that pixel errors can be backpropagated to the 3D parameters, allowing an optimizer to adjust 3D content until renders match photos.
**Tracking / fitting:** solving the rig parameters (pose, expression) that best explain each video frame.
**Viseme:** the visual mouth shape corresponding to a phoneme; the bridge between audio and facial animation.
**Multi-view consistency:** the property that all frames depict the same 3D object from different angles; the reason head turns during capture enable 3D reconstruction, and the reason AI-generated video makes poor training data.
