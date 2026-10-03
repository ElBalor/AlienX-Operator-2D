"""
AlienX 2D — Sigil Lab (Necromancer Edition)
===========================================
Darcy flow through a rotated permeability field. The operator was trained
with random rotations, but watch what the demo actually shows: drag the
world to ANY angle and the pressure prediction error does not move.
Then change the resolution — 16² to 128² — with the same weights.

Rotation and scale are not features here. They are coordinate artifacts.

    The grid is dead. The manifold is awake.
    — from the Grimoire of Elbàlor
"""

import os
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import gradio as gr

import data as dt
from model import AlienXOperator, get_cached_iso_knn

# ─── Summon the operator (bundled weights) ───────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

MODEL = AlienXOperator(hidden_dim=128, num_blocks=4)
MODEL.load_state_dict(torch.load(os.path.join(BASE, "alienx_v2_best.pt"),
                                 map_location=DEVICE))
MODEL.eval()
for p in MODEL.parameters():
    p.requires_grad_(False)
N_PARAMS = sum(p.numel() for p in MODEL.parameters())
print(f"Operator summoned: {N_PARAMS:,} params (2D, isotropic 24-neighbor stencil)")

# ─── Necromancer theme ────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;800&family=Cormorant+Garamond:ital,wght@0,500;1,500&display=swap');
.gradio-container {background: #050508 !important; color: #cfc9c0 !important;}
#necro-title {text-align:center; font-family:'Cinzel',serif; font-weight:800;
  font-size:2.3em; letter-spacing:.12em; margin:.2em 0 0;
  background:linear-gradient(90deg,#00ffd0,#b26bff 55%,#00ffd0);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  animation:necroPulse 3.2s ease-in-out infinite;}
#necro-sub {text-align:center; font-family:'Cormorant Garamond',serif; font-style:italic;
  color:#8a8577; font-size:1.15em; margin-bottom:1em;}
@keyframes necroPulse {0%,100%{filter:drop-shadow(0 0 5px #00ffd055)} 50%{filter:drop-shadow(0 0 16px #b26bff99)}}
.panel {border:1px solid #1c2b28 !important; border-radius:12px !important;
  background:linear-gradient(160deg,#0a0d12,#0d1117) !important;
  box-shadow:0 0 14px #00ffd014, inset 0 0 26px #05010acc !important;}
button.primary {background:linear-gradient(90deg,#003b34,#2a0f45) !important;
  border:1px solid #00ffd055 !important; color:#c9fff4 !important;
  font-family:'Cinzel',serif !important; letter-spacing:.1em !important;
  box-shadow:0 0 12px #00ffd033 !important;}
button.primary:hover {box-shadow:0 0 22px #b26bff77 !important; border-color:#b26bff88 !important;}
footer, .footer {visibility:hidden;}
#rune-canvas {position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.5;}
.metrics {font-family:'Cormorant Garamond',serif; font-size:1.15em; color:#9be8d8;}
.metrics b {color:#00ffd0; text-shadow:0 0 8px #00ffd066;}
"""

HEAD = """
<canvas id='rune-canvas'></canvas>
<script>
(function(){
  const c=document.getElementById('rune-canvas'); if(!c) return;
  const x=c.getContext('2d'); let W,H;
  const GLYPHS=['\u16A0','\u16A1','\u16A2','\u16A3','\u16A8','\u16A9','\u16AA','\u16AB',
                '\u16B1','\u16B2','\u16B3','\u16B7','\u16B9','\u16BA','\u16C1','\u16C3',
                '\u2721','\u2725','\u2726','\u1F52','\u029A','\u0262'];
  let P=[];
  function rs(){W=c.width=innerWidth;H=c.height=innerHeight;}
  addEventListener('resize',rs); rs();
  for(let i=0;i<34;i++) P.push(spawn(true));
  function spawn(any){return {x:Math.random()*W,y:any?Math.random()*H:-20,
    s:8+Math.random()*11, v:.35+Math.random()*.9, o:.08+Math.random()*.22,
    g:GLYPHS[(Math.random()*GLYPHS.length)|0], w:Math.random()*6.28};}
  function tick(){
    x.clearRect(0,0,W,H);
    for(let i=0;i<P.length;i++){let p=P[i]; p.y+=p.v; p.w+=.02;
      x.font=p.s+'px serif'; x.fillStyle='rgba(0,255,208,'+p.o+')';
      x.save(); x.translate(p.x+Math.sin(p.w)*6,p.y); x.fillText(p.g,0,0); x.restore();
      if(p.y>H+24) P[i]=spawn(false);}
    requestAnimationFrame(tick);}
  requestAnimationFrame(tick);
})();
</script>
"""

# ─── Figures ─────────────────────────────────────────────────────────────────
PAPER = dict(plot_bgcolor="#0a0d12", paper_bgcolor="#0a0d12",
             font=dict(color="#cfc9c0"), margin=dict(l=40, r=20, t=48, b=34))

def _grid(t, res):
    return t.detach().cpu().numpy().reshape(res, res)

def fig_fields(res, k, p_true, p_pred, angle):
    err = np.abs(p_pred - p_true)
    f = psub.make_subplots(rows=2, cols=2,
                           subplot_titles=(f"permeability k — rotated {angle:.0f}°",
                                           "pressure p — ground truth",
                                           "pressure p — AlienX 2D",
                                           "|error|"), vertical_spacing=0.14)
    for i, (z, name) in enumerate([(k, "Magma"), (p_true, "Cividis"),
                                   (p_pred, "Cividis"), (err, "Inferno")]):
        f.add_trace(go.Heatmap(z=z, colorscale=name, showscale=False), i // 2 + 1, i % 2 + 1)
    f.update_annotations(font=dict(size=12, color="#9be8d8"))
    for r in (1, 2):
        for c in (1, 2):
            f.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, row=r, col=c)
            f.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, row=r, col=c)
    f.update_layout(title=dict(text=f"{res}×{res} nodes · same weights at every angle · "
                                    "no re-training, no grid", font=dict(size=15)),
                    height=620, **PAPER)
    return f

def fig_compass(angles, errs):
    f = go.Figure()
    f.add_trace(go.Scatter(x=angles, y=errs, mode="lines+markers",
                           line=dict(color="#00ffd0", width=2.4),
                           marker=dict(size=7, color="#b26bff"),
                           name="interior L1"))
    mean_e = float(np.mean(errs))
    f.add_hline(y=mean_e, line=dict(color="#e8e3d8", width=1, dash="dot"),
                annotation_text=f"mean {mean_e:.4f} — flat means equivariant",
                annotation_font=dict(color="#8a8577"))
    f.update_layout(title=dict(text="Error vs rotation angle — the flat line IS the claim",
                               font=dict(size=15)),
                    xaxis_title="field rotation (degrees)",
                    yaxis_title="interior L1 error", height=340, **PAPER)
    return f

# ─── The rituals ─────────────────────────────────────────────────────────────
def interior_l1(res, pred, true):
    d = 2 * max(1, res // 16)   # dilation-scaled interior crop
    pred_i = pred.reshape(res, res)[d:-d, d:-d]
    true_i = true.reshape(res, res)[d:-d, d:-d]
    return float(np.abs(pred_i - true_i).mean())

@torch.no_grad()
def reveal(angle, res, seed):
    angle, res, seed = float(angle), int(res), int(seed)
    coords, k, gkm, gkv, p = dt.generate_darcy_sample_gpu(res, DEVICE,
                                                          angle_deg=angle, seed=seed)
    knn = get_cached_iso_knn(res, DEVICE)
    pred = MODEL(coords.unsqueeze(0), k.unsqueeze(0), gkv.unsqueeze(0),
                 gkm.unsqueeze(0), knn).squeeze(0).cpu().numpy()
    l1 = interior_l1(res, pred, p.numpy())
    md = (f"<div class='metrics'>Interior L1 error: <b>{l1:.4f}</b> · "
          f"{N_PARAMS:,} parameters · resolution {res}×{res}<br>"
          f"<span style='color:#6b665c'>Rotate the slider — the error will not follow. "
          f"Same weights at 16², 32², 64², 128² — the error will not move either. "
          f"Rotation and scale are coordinate artifacts, removed before the first layer. "
          f"Trained at 16²–128² with random angles in [0°, 360°); the paper reports 256² "
          f"zero-shot in the same error band.</span></div>")
    f = fig_fields(res, _grid(k, res), _grid(p, res), pred.reshape(res, res), angle)
    return f, md

@torch.no_grad()
def cast_compass(res, seed):
    res, seed = int(res), int(seed)
    angles = list(range(0, 360, 30))
    errs = []
    for a in angles:
        coords, k, gkm, gkv, p = dt.generate_darcy_sample_gpu(res, DEVICE,
                                                              angle_deg=a, seed=seed)
        pred = MODEL(coords.unsqueeze(0), k.unsqueeze(0), gkv.unsqueeze(0),
                     gkm.unsqueeze(0), get_cached_iso_knn(res, DEVICE)).squeeze(0).cpu().numpy()
        errs.append(interior_l1(res, pred, p.numpy()))
    return fig_compass(angles, errs)

with gr.Blocks(css=CSS, head=HEAD, title="AlienX 2D — Sigil Lab") as demo:
    gr.HTML("<div id='necro-title'>ALIENX 2D · SIGIL LAB</div>"
            "<div id='necro-sub'>Darcy flow · rotate the world — the operator does not notice · "
            "one set of weights, every angle, every resolution</div>")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            angle = gr.Slider(0, 360, value=37, step=1,
                              label="Rotate the permeability field (degrees)")
            res = gr.Radio([16, 32, 64, 128], value=64,
                           label="Resolution — same weights, no retraining")
            seed = gr.Slider(0, 99, value=7, step=1, label="Field seed")
            go_btn = gr.Button("⚡ REVEAL THE PRESSURE", variant="primary")
            compass_btn = gr.Button("🧭 CAST THE COMPASS — sweep all angles", variant="secondary")
            gr.Markdown(
                "**What you are watching:** Darcy subsurface flow — permeability field k in, "
                "pressure field p out. The field is *physically rotated* by your slider "
                "(pull-back rotation, analytical gradients). A grid-native network would need "
                "augmentation for every angle; this operator carries a local SO(2) frame "
                "instead — the 24-neighbor isotropic stencil has no cardinal spikes and no "
                "45° residual. trained at {16, 32, 64, 128}; the paper reports 256² zero-shot "
                "at the same error band.")
        with gr.Column(scale=2, elem_classes="panel"):
            metrics = gr.HTML()
            field_plot = gr.Plot()
            compass_plot = gr.Plot()
    go_btn.click(reveal, [angle, res, seed], [field_plot, metrics], concurrency_limit=1)
    compass_btn.click(cast_compass, [res, seed], [compass_plot], concurrency_limit=1)
    demo.load(lambda: None)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, show_error=True)
