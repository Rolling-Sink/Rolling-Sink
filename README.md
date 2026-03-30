<h2>
  <img src="assets/icon.png" style="height:1.6em;"/>&nbsp;&nbsp;<img src="assets/title.svg" style="height:1.8em;"/>
  <br/>
  Bridging <ins>Limited-Horizon</ins> Training and <ins>Open-Ended</ins> Testing in Autoregressive Video Diffusion
</h2>

[![Page](https://img.shields.io/badge/Project-Page-pink?logo=googlechrome&logoColor=white)](https://rolling-sink.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.07775)
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20-yellow)](https://huggingface.co/spaces/haodongli/RollingSink)
[![Gallery](https://img.shields.io/badge/Gallery-5~30min%20Videos-red?logo=youtube)](https://www.youtube.com/watch?v=U1eAGF_jcxI&list=PLIfXMX0d4BSJO_S3Wte7SAZpcdSFmZEvD)
[![Slides](https://img.shields.io/badge/Slides-PPT-blue?logo=slideshare&logoColor=white)](https://drive.google.com/file/d/1blgZtE0LpC8uDCtZBekasjLMXA1oNpRa/view)
[![BibTeX](https://img.shields.io/badge/BibTeX-grey?logo=googlescholar&logoColor=white)](https://rolling-sink.github.io/bibtex.txt)

[Haodong Li](https://haodong2000.github.io/)<sup>1</sup>
[Shaoteng Liu](https://www.shaotengliu.com/)<sup>2</sup>
[Zhe Lin](https://sites.google.com/site/zhelin625/)<sup>2</sup>
[Manmohan Chandraker](https://cseweb.ucsd.edu/~mkchandraker/)<sup>1&#9993;</sup>

<span class="author-block"><sup>1</sup>UC San Diego</span>
<span class="author-block"><sup>2</sup>Adobe Research</span>
<span class="author-block">
    <sup>&#9993;</sup>Corresponding author
</span>

Please click the figure below to watch the teaser video on [YouTube](https://www.youtube.com/watch?v=oNFHLi6vrHI).
[![Demo video](https://img.youtube.com/vi/oNFHLi6vrHI/maxresdefault.jpg)](https://www.youtube.com/watch?v=oNFHLi6vrHI)

## 📢 News
- 2026-03-14 Add [Causal Forcing](https://thu-ml.github.io/CausalForcing.github.io/) as a new base model option!
- 2026-02-22 Demo ([local](https://github.com/Rolling-Sink/Rolling-Sink?tab=readme-ov-file#-gradio-demo) & [online](https://huggingface.co/spaces/haodongli/RollingSink)) released!
- 2026-02-11 Code released!
- 2026-02-10 [Paper](https://arxiv.org/abs/2602.07775) released on arXiv!

## 🛠️ Setup
> This installation was tested on: Ubuntu 20.04, CUDA 12.4, NVIDIA A40.

1. Clone the repository:
```
git clone https://github.com/Rolling-Sink/Rolling-Sink.git
cd Rolling-Sink
```
2. Install dependencies using conda:
```
conda create -n RS python=3.10 -y
conda activate RS
pip install -r requirements.txt
```
3. Download checkpoints:
```
sh shell_scripts/download_ckpt.sh
```

## 🤗 Gradio Demo
1. Run the demo online: [HF Space](https://huggingface.co/spaces/haodongli/RollingSink)
2. Run it locally:
```
python app.py
```

## 🕹️ Inference
1. Prepare prompts under `prompts/`
> We've pre-uploaded some examples under `prompts/example/`
2. Run the inference command:
```
sh shell_scripts/cuda_i.sh
```
> `i` $\in$ {`0`,`1`,`2`,`3`}
> <br>
> The default video length is **5-minute**, which requires GPUs with $\geqslant$ **48GB** memory.

## 🎓 Citation
If you find our work useful in your research, please consider citing our paper🌹:
```bibtex
@article{li2026rolling,
  title={Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion},
  author={Li, Haodong and Liu, Shaoteng and Lin, Zhe and Chandraker, Manmohan},
  journal={arXiv preprint arXiv:2602.07775},
  year={2026}
}
```

## 🤝 Acknowledgement
This implementation is impossible without the awesome open-cource contributions of:
- [Self Forcing](https://self-forcing.github.io/)
- [LongLive](https://nvlabs.github.io/LongLive/)
- [CausVid](https://causvid.github.io/)
- [Wan](https://wan.video/)
- [Gradio](https://github.com/gradio-app/gradio)
- [HuggingFace Hub](https://github.com/huggingface/huggingface_hub)
- [PyTorch](https://pytorch.org/)
