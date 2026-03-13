import spaces  # must be first!
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline import CausalInferencePipeline
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
import time

REPO_ROOT = Path(__file__).resolve().parent
GRADIO_TMP = REPO_ROOT / ".gradio_cache"
GRADIO_TMP.mkdir(parents=True, exist_ok=True)

os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP)
print(f"Gradio temp/cache dir: {GRADIO_TMP}")

import gradio as gr

pipeline, device, config = None, None, None

import threading
from functools import lru_cache

_init_lock = threading.Lock()

@lru_cache(maxsize=1)
def get_pipeline():
    global pipeline, device, config
    print(f'[MAIN] Loading pipeline...')

    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(0)

    torch.set_grad_enabled(False)

    config = OmegaConf.load("configs/self_forcing_dmd.yaml")
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    pipeline = CausalInferencePipeline(config, device=device)

    state_dict = torch.load("checkpoints/self_forcing_dmd.pt", map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator_ema'])

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    print("\n[MAIN] Pipeline set up.")
    print("\n[MAIN] INIT ONLY ONCE.")

    return pipeline, config, device

@spaces.GPU
def fn(prompt, duration):
    print("\n[MAIN] Prompt:", prompt + '\n')
    with _init_lock:
        pipeline, config, device = get_pipeline()

        sampled_noise = torch.randn(
            [1, config['num_frame_per_block'], 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        num_output_frames = (duration * 16 + 3) / 4 + 3
        num_output_frames = int(num_output_frames / 3) * 3  # make it divisible by 3
        print(f"\n[MAIN] Target video length:\n\t{num_output_frames} latent frames; \
                \n\t{num_output_frames*4 - 3} frames; \
                \n\t{int((num_output_frames*4 - 3)/16.0)} seconds (FPS=16)\n")

        # repeat noise to cover all output frames
        num_repeats = num_output_frames // 3
        sampled_noise_processed = sampled_noise.repeat(1, num_repeats, 1, 1, 1)

        video = pipeline.inference(
            noise=sampled_noise_processed,
            text_prompts=[prompt],
            return_latents=False,
            initial_latent=None,
            low_memory=False,
        )
        all_video = []
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        video = 255.0 * torch.cat(all_video, dim=1)
        
        pipeline.vae.model.clear_cache()
        output_path = f"{GRADIO_TMP}/{prompt[:100].replace(' ', '_')}-{time.time()}.mp4"
        write_video(output_path, video[0], fps=16)
        
        return output_path


if __name__ == "__main__":
    inputs = [
        gr.Textbox(label="Text Prompt", lines=10, placeholder="e.g., a chicken is playing basketball"),
        gr.Slider(5, 30, value=15, step=1, label="Duration (s)"),
    ]
    outputs = [
        gr.Video(label="Output Video", autoplay=True),
    ]

    demo = gr.Interface(
        fn=fn,
        title="Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion",
        description="""
            <strong>Please consider starring <span style="color: orange">&#9733;</span> our <a href="https://github.com/haodong2000/RollingSink" target="_blank" rel="noopener noreferrer">GitHub Repo</a> if you find this demo useful!</strong>
        """,
        inputs=inputs,
        outputs=outputs,
        examples=[
            [r"A cinematic high-energy action shot of a young rider on a dark bay horse, wearing a tan coat, black helmet, and a distinctive scarf fluttering behind. The horse gallops along a long, flat shoreline, hooves kicking up repeated bursts of wet sand. Waves roll in parallel lines, creating a stable layered background of surf, sea, and horizon, with distant sea stacks and a faint pier silhouette. The camera tracks from slightly behind and to the side at low height, keeping the rider and horse centered as the shoreline repeats endlessly.", 15],
            [r"A vibrant anime illustration in a dynamic, thick-line painting style of a young girl blowing a kiss to the camera. She has long flowing hair that cascades down her back, framed by soft bangs that partially cover her eyes. The girl wears a colorful floral dress with ruffled sleeves and a delicate belt. She has bright, sparkling eyes and a sweet, joyful smile. Her lips are parted, and she blows a kiss towards the camera with a playful and innocent expression. The background is a blurred outdoor setting with a gentle sunset, highlighting warm hues of orange and pink. A close-up shot from a slightly tilted angle, capturing the moment of her kiss.", 15],
            [r"A high-energy road race photograph capturing a cyclist powering up a steep hill. The cyclist is a middle-aged man with a determined expression, sweat glistening on his brow. He is dressed in a sleek, aerodynamic racing jersey and cycling shorts, with号码 clearly visible on his back. His helmet is snugly fastened, and he grips the handlebars tightly. The background shows a winding road leading upwards, with blurred trees and bushes rushing past. The sky is a mix of dark clouds and bright sunlight, creating dramatic contrast. The scene is captured from a low-angle shot, emphasizing the cyclist's struggle and determination.", 15],
            [r"A romantic wildlife photograph in a soft naturalistic style, capturing a pair of lovebirds preening each other's feathers. The birds have vibrant plumage, with the male sporting a striking red breast and the female a beautiful green hue. They sit closely together, their heads tilted towards each other, beaks gently touching as they preen. Their eyes are filled with affection, and their wings are spread slightly, creating a cozy, intimate moment. The background is a blurred forest setting, with dappled sunlight filtering through the leaves, adding a warm, serene atmosphere. A medium shot from a low angle, capturing the tender interaction between the two birds.", 15],
            [r"A close-up shot of a bright blue parrot's shimmering feathers, capturing the unique and vibrant colors in the light. The parrot's feathers glisten with a metallic sheen, showcasing a mix of deep indigos, vivid greens, and rich blues. Its eyes sparkle with curiosity, and it appears lively and alert, perched on a branch. The background is blurred, highlighting the parrot against a soft, warm environment. The photo has a naturalistic and lifelike quality, emphasizing the bird's detailed plumage and natural movements.", 15],
            [r"A high-energy action shot of a speed skater wearing a sleek navy skinsuit and a reflective silver visor, skating powerfully on an outdoor ice oval. The skater leans into repeated left-hand curves, blades carving thin white lines on the ice, with a steady cadence and consistent crossovers. Stadium lights and banners line the rink perimeter, and snowbanks frame the track with a distant treeline beyond. The camera follows from slightly behind at hip height, keeping the skater centered as the oval’s repetitive geometry supports long-duration continuity.", 15],
            [r"A stunning Santorini landscape photo captured during the blue hour, featuring a red panda and a toucan strolling hand-in-hand through the picturesque village. The red panda, with its distinctive reddish-brown fur and large round eyes, carries a small backpack, while the toucan, with its vibrant orange and black feathers and a large curved beak, holds a colorful flower. They walk along a winding cobblestone path, passing by whitewashed buildings with blue doors and windows. The setting sun casts a soft golden glow, creating a warm and serene atmosphere. The sky is painted with shades of blue and purple, with a few twinkling stars beginning to appear. A wide-angle shot from a slightly elevated angle, capturing the intimate moment between these two unlikely friends.", 15],
            [r"A close-up shot of a majestic waterfall, capturing the dynamic movement of the water as it crashes down in a cascade of frothy white waves. The water splashes and swirls, creating a sense of motion and energy. The background features a lush green forest, with sunlight filtering through the leaves, casting dappled shadows. The camera angle emphasizes the force and beauty of the water, with droplets flying and mist rising into the air. The overall scene has a crisp, vivid quality, highlighting the natural movement and power of the waterfall.", 15],
            [r"A realistic photograph of a princess riding a horse across a river. The princess, with fair skin and delicate features, wears a flowing white gown with intricate lace detailing and a long veil. She sits gracefully on a sturdy, brown horse, her hands firmly gripping the reins. The horse's mane flows freely in the breeze, and its hooves kick up small splashes of water as it gallops across the river. The riverbank is lined with tall grasses and wildflowers, with a few trees providing shade. The background shows a misty landscape, with distant hills and a hint of blue sky peeking through the clouds. The photo captures a moment of natural movement, with the princess and horse seeming almost weightless as they cross the river. A medium shot from a slightly elevated angle, emphasizing the princess's determined expression and the horse's powerful stride.", 15],
            [r"A slow-motion video in the style of a scientific documentary, depicting the gradual injection of ink into a tank of water. The camera captures the intricate and beautiful patterns formed as the ink spreads and mixes, creating dynamic and fluid shapes. The water surface is still and clear until the moment the ink is introduced, causing ripples and waves that highlight the patterns. The lighting is soft and diffused, emphasizing the beauty of the process. The camera angle is from above, providing a clear view of the entire tank, with slow-motion playback enhancing the visual appeal.", 15],
            [r"A dynamic action shot in the style of a high-energy sports magazine spread, featuring a golden retriever sprinting with all its might after a red sports car speeding down the road. The dog's fur glistens in the sunlight, and its eyes are filled with determination and excitement. It leaps forward, its tail wagging wildly, while the car speeds away in the background, leaving a trail of dust. The background shows a busy city street with blurred cars and pedestrians, adding to the sense of urgency. The photo has a crisp, vibrant color palette and a high-resolution quality. A medium-long shot capturing the dog's full run.", 15],
            [r"A dynamic photograph capturing a marathon runner in the final moments of a grueling race, crossing the finish line. The runner, a young man with a determined expression, is sprinting with arms pumping and legs striding forcefully. His face is flushed, and he is breathing heavily, sweat glistening on his forehead and body. He is wearing a white sports jersey with 'Marathon' printed on the back, and black running shorts with sponsor logos. The background is blurred, revealing a crowd cheering and a banner reading 'Finish Line.' The finish line itself is marked by a colorful tape, and the runner's shadow stretches out behind him, emphasizing his momentum. The photo has a vibrant and energetic feel, capturing the intense moment of victory. A medium shot from a slightly elevated angle, focusing on the runner's determined expression and the blur of the crowd.", 15],
            [r"A stylish woman confidently catwalks down a bustling Tokyo street as if the street were a runway, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. Cinematography: Front-facing full-body runway-style tracking shot. The camera leads her and moves backward smoothly in front of her with gimbal-stabilized motion, maintaining a constant distance and constant eye-level camera height. Fixed focal length (35mm), no zoom, no push-in, no pull-back, no reframing. Locked framing: she stays centered (or left-third) and remains full-body in frame at all times; her scale and screen position remain constant throughout (her body height occupies roughly 55–65% of the frame height consistently). Pose and gaze: shoulders square to the camera, chin slightly up, head mostly forward; she maintains strong eye contact with the lens for most of the video (about 85–90%), with only brief natural side glances that quickly return to the camera—no prolonged looking away.", 15],
            [r"A close-up 3D animated scene of a short, fluffy monster kneeling beside a melting red candle. The monster has large, wide eyes and an open mouth, gazing at the flame with a look of wonder and curiosity. Its soft, fluffy fur contrasts with the warm, dramatic lighting that highlights every detail of its gentle, innocent expression. The pose conveys a sense of playfulness and exploration, as if the creature is discovering the world for the first time. The background features a cozy, warmly lit room with subtle hints of a fireplace and soft furnishings, enhancing the overall atmosphere. The use of warm colors and dramatic lighting creates a captivating and inviting scene.", 15],
            [r"A tranquil pond scene in the style of a watercolor painting, featuring a roe deer leaping gracefully from lily pad to lily pad. The deer has soft brown fur, large expressive eyes, and delicate antlers. It moves with agility and grace, each leap capturing a moment of mid-air motion. The lily pads are lush and green, with delicate pink flowers blooming. The background features a serene landscape with gently flowing water, patches of sunlight breaking through the trees, and a soft mist hovering over the pond. A dynamic close-up shot from a slightly elevated angle, emphasizing the deer's natural movements and the vibrant greenery.", 15],
            [r"A dynamic shot from behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by towering redwood trees on a rugged mountain slope. Dust kicks up from its tires, and the sunlight shines on the SUV, casting a warm glow over the scene. The dirt road extend forward into the distance, with no other vehicles in sight. The trees on either side are dense redwoods, with patches of greenery scattered throughout. The dirt road is framed by steep hills and mountains, with a clear blue sky above and wispy clouds drifting by. The camera captures the vehicle from the rear, emphasizing its powerful and adventurous journey. The SUV drives straight forward the entire time: no left turn, no right turn, no swerving, no drifting, no lane change, minimal yaw rotation. The dirt road continues straight ahead with a fixed central vanishing point: no curves, no bends, no winding, the road does not deviate left or right at any moment. The camera remains centered directly behind the SUV, aligned with the road axis, with constant distance, constant height, fixed focal length, no zoom, no orbit, no pan.", 15],
            [r"A 3D animation of a small, round, fluffy creature with big, expressive eyes exploring a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring The scene is rendered in a detailed, fantasy style, with a soft, ethereal lighting that enhances the enchantment. The camera follows the creature as it slowly moves, capturing its playful interactions and the magical ambiance of the forest. The creature is always facing to the camera.", 15],
            [r"A dynamic action shot of a surfer accelerating on a powerful wave, carving through the water with grace and agility. The surfer, with a tanned complexion and muscular build, rides the wave with one hand gripping the board while the other extends outwards for balance. The water splashes behind, creating a foamy trail, and the sun casts a golden glow over the scene. The background features a clear blue ocean and distant white-capped waves, with a few seagulls flying overhead. The surfer's expression is one of exhilaration and focus. A mid-shot from a low-angle perspective capturing the surfer's motion and the wave's power.", 15],
            [r"A high-definition racing scene in the style of a professional racing game, showcasing a sleek, red race car accelerating through a chicane on a winding race track. The car is filled with intense speed and power, its tires smoking as it navigates the tight turn. The driver, a muscular man with focused determination, leans slightly forward, gripping the steering wheel tightly. His helmet glints under the bright lights, reflecting the excitement of the moment. The background features blurred but recognizable elements of the track, with other cars and the stands of spectators in the distance. The camera angle is from behind the car, capturing both the action and the tension of the race. A dynamic and fast-paced medium shot.", 15],
            [r"A dynamic snowboarding scene in the style of a high-energy action shot, featuring a young snowboarder accelerating down a powdery slope. The snowboarder, with a determined expression, weaves expertly between tall pine trees, their trunks partially obscured by the swirling snow. The snow is pristine and fluffy, with the sun casting soft shadows and highlighting the snowboarder's movements. The background showcases a breathtaking mountain vista, with peaks shrouded in mist and a few distant ski lifts visible. The camera angle captures the snowboarder from a slightly behind-the-action perspective, emphasizing their speed and agility.", 15],
            [r"A high-energy choreographed fight performance on a minimalist stage: two performers in playful animal helmets and padded suits exchange fast, clean boxing sequences. The white-suited fighter with a blue spiky helmet presses forward with repeated jabs and feints; the blue-suited fighter with a red helmet pivots back and fires counters, both wearing oversized red gloves and chunky boots. The lighting is theatrical—bright key light from above with a soft rim light, creating strong silhouettes against a black background. The camera tracks laterally from left to right, keeping both fighters centered as they circle and reset.", 15],
            [r"A romantic wedding photo in a classic film noir style, capturing a bride and groom sharing a tender first dance. The bride wears a stunning white silk gown with intricate lace detailing and a flowing veil, while the groom stands confidently in a tuxedo with a crisp white shirt and a black bow tie. They hold each other closely, swaying gently to the music, with soft smiles on their faces. The background features a blurred, elegant ballroom with antique chandeliers and ornate decorations, casting a warm, golden glow. The scene is filled with emotion and love, with the couple’s reflections visible in a nearby mirror. A medium shot from a slightly elevated angle, emphasizing their intimate connection.", 15],
            [r"A dynamic hip-hop dance scene in a vibrant urban style, featuring an Asian girl in a bright yellow T-shirt and white pants. She is mid-dance move, arms stretched out and feet rhythmically stepping, exuding energy and confidence. Her hair is tied up in a ponytail, and she has a mischievous smile on her face. The background shows a bustling city street with blurred reflections of tall buildings and passing cars. The scene captures the lively and energetic atmosphere of a hip-hop performance, with a slightly grainy texture. A medium shot from a low-angle perspective.", 15],
            [r"A vibrant concert photo in the style of a live performance shot, featuring a young singer belting out a high note on stage. The singer, with flowing wavy brown hair and expressive green eyes, stands confidently in a black sequined dress adorned with glitter. She is mid-singing, her mouth wide open and throat muscles tensed, conveying raw emotion and power. The background is a blurred mix of colorful lights and audience members, with some fans waving their hands excitedly. The stage is illuminated by spotlights, casting dramatic shadows. A dynamic medium shot from a slightly elevated angle, capturing the singer's intense performance.", 15],
            [r"A detailed realist photograph captures a middle-aged man methodically wiping down a kitchen counter with a clean, white cloth. His focused expression conveys determination as he ensures every surface is spotlessly clean. He stands upright, leaning slightly forward, with one hand gripping the edge of the counter and the other holding the cloth. The background features modern kitchen appliances and cabinets, with subtle reflections in the glass surfaces. Shadows cast by the overhead lights add depth to the scene. The photo has a crisp, clear texture. A medium shot from a slightly elevated angle, highlighting the man's dedication and the pristine cleanliness of the kitchen.", 15],
            [r"A dramatic underwater photograph captures a man performing an intense drumming session. He is submerged in clear blue water, with his face partially obscured by bubbles. His arms move rhythmically, striking the drums with powerful strokes. The drums, made of durable material, are suspended above him, reflecting the vibrant underwater environment. The background features a colorful coral reef with fish swimming around, adding to the vividness of the scene. The water has a soft, ethereal quality, creating a mesmerizing effect. A dynamic low-angle shot from below the surface, emphasizing the man's energetic movements and the aquatic surroundings.", 15],
        ],
        examples_per_page=30
    )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=1324,
    )
