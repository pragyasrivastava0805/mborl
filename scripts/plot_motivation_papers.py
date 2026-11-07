import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

fig, ax = plt.subplots(figsize=(17.5, 6.6), dpi=220)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 17.5)
ax.set_ylim(0, 6.6)
ax.axis('off')

def draw_box(x, y, w, h, title, subtitle, bullet1, bullet2, fc, ec, title_color):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.16",
        facecolor=fc, edgecolor=ec, linewidth=2.0
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.24, title, ha='center', va='top', fontsize=9.8, fontweight='bold', color=title_color)
    ax.text(x + w/2, y + h - 0.56, subtitle, ha='center', va='top', fontsize=8.0, fontweight='bold', color='#334155')
    ax.text(x + 0.14, y + h - 0.94, "• " + bullet1, ha='left', va='top', fontsize=7.6, color='#1E293B')
    ax.text(x + 0.14, y + h - 1.32, "• " + bullet2, ha='left', va='top', fontsize=7.6, color='#1E293B')

# Top Title
ax.text(8.75, 6.36, "Theoretical & Methodological Lineage: Prior Literature Motivating Each Clean Stitching Mechanism",
        ha='center', va='top', fontsize=13.0, fontweight='bold', color='#0F172A')

# Column Headers
ax.text(2.85, 5.86, "1. Foundational Literature Pillars", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#1E293B')
ax.text(8.75, 5.86, "2. Why Each Pillar Fails Alone on Maze-Stitch", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#991B1B')
ax.text(14.65, 5.86, "3. Resulting Single-Principle Clean Ideas", ha='center', va='center', fontsize=11.0, fontweight='bold', color='#1D4ED8')

# Row 1: ROMI + IQL/AWR -> Idea 1 (AT-ROMI)
draw_box(0.25, 3.92, 5.2, 1.62,
         "Pillar A: Reverse Imagination + KL-Tilted BC",
         "ROMI (Wang et al., 2021) & IQL / AWR (Kostrikov et al., 2022)",
         "ROMI: Reverse model f_rev + uniform reverse BC stays near target states.",
         "AWR / IQL: Closed-form KL projection pi* propto mu * exp(beta * Adv).",
         "#F8FAFC", "#64748B", "#0F172A")

draw_box(6.15, 3.92, 5.2, 1.62,
         "Gap A: Hovering Copy vs. Zero Spatial Reach",
         "Standard ROMI: 0–15% Success | Model-Free IQL: 0% Success",
         "Standard ROMI (beta=0) copies 87% waypoint-hovering steps (Delta V <= 0).",
         "Model-free IQL (H=0) cannot cross spatial gaps between disjoint segments.",
         "#FEF2F2", "#DC2626", "#991B1B")

draw_box(12.05, 3.92, 5.2, 1.62,
         "Idea 1: Advantage-Tilted ROMI (AT-ROMI)",
         "1-Line exp(beta * Delta V) Tilt (100% / 100% / 100% / 85% / 30%)",
         "Tilts ROMI reverse BC by exp(beta*(V(s_{t+1}) - V(s_t))), zero CEM.",
         "Strictly in-support (3.0x lower drift than CEM) + H-step value bridge.",
         "#EFF6FF", "#2563EB", "#1D4ED8")

# Row 2: MBTS / SoRB -> Idea 2 (1-Step Real-to-Real Stitch)
draw_box(0.25, 2.02, 5.2, 1.62,
         "Pillar B: Real-to-Real Graph & Inverse Stitching",
         "MBTS (Hepburn & Montana, 2022) & SoRB (Eysenbach et al., 2019)",
         "Connects pairs of real replay states (s_i, s_j) via local inverse dynamics.",
         "Eliminates multi-step open-loop rollout compounding error (d_real = 0).",
         "#F8FAFC", "#64748B", "#0F172A")

draw_box(6.15, 2.02, 5.2, 1.62,
         "Gap B: Unverified Inverse Actions Hallucinate",
         "Naive Nearest-Neighbor Stitch Teleports Across Walls / Contacts",
         "Euclidean state proximity ignores dynamics feasibility & value direction.",
         "Requires forward-ensemble consistency + positive value gain verification.",
         "#FEF2F2", "#DC2626", "#991B1B")

draw_box(12.05, 2.02, 5.2, 1.62,
         "Idea 2: 1-Step Real-to-Real Model Stitch",
         "Zero Rollouts, 100% Real States (90%–100% on Point & Ant Mazes)",
         "Pairs real states (s_i in D_A, s_j in D_B) with V(s_j) > V(s_i) + delta.",
         "Accepts a_ij = f_inv(s_i, s_j) iff 1-step forward ensemble matches s_j.",
         "#ECFDF5", "#059669", "#047857")

# Row 3: STEVE/MVE + GCSL/OGBench -> Target vs. Buffer Synthesis
draw_box(0.25, 0.12, 5.2, 1.62,
         "Pillar C: Value Expansion & Goal Relabeling",
         "MVE (Feinberg et al., 2018), STEVE (2018) & OGBench (Park et al., 2025)",
         "STEVE / TD(lambda)-MVE: Uncertainty-weighted H-step Bellman targets.",
         "GCSL / RvS / HIQL: Hindsight goal-conditioned policy pi(a | s, g).",
         "#F8FAFC", "#64748B", "#0F172A")

draw_box(6.15, 0.12, 5.2, 1.62,
         "Gap C: Target-Only Expansion Starves the Actor",
         "TD(lambda)-MVE: 100% on PointMaze, 0–5% on Ant / Humanoid",
         "Updating y_t at s_t in D_A leaves zero replay tuples inside the bridge gap.",
         "High-DoF locomotion actors need physical bridge transitions in D_bridge.",
         "#FEF2F2", "#DC2626", "#991B1B")

draw_box(12.05, 0.12, 5.2, 1.62,
         "Idea 3 & Buffer-Level Bridge Synthesis",
         "Hindsight Bridge Policy pi(a | s, s*) + Buffer-Level Stitching",
         "Inserts verified bridge transitions directly into replay buffer D_bridge.",
         "Unlocks 85–100% success on 29D AntMaze and 30–40% on 69D HumanoidMaze.",
         "#FFFBEB", "#D97706", "#B45309")

# Arrows between columns
for y_mid in [4.73, 2.83, 0.93]:
    ax.annotate("", xy=(6.02, y_mid), xytext=(5.58, y_mid),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.25", color="#64748B", lw=2.2))
    ax.annotate("", xy=(11.92, y_mid), xytext=(11.48, y_mid),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.25", color="#2563EB", lw=2.2))

plt.tight_layout()
out_path = '/usr/local/google/home/rchitale/.gemini/jetski/brain/c46126e7-d7e9-47cd-a805-110b6330628f/scratch/clean_motivation_papers_lineage.png'
plt.savefig(out_path, dpi=220)
print("Saved:", out_path)
