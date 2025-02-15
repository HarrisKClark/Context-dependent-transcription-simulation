import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches



def simulate_plasmid(
    num_segments,
    alpha,
    steps_per_frame,
    genes,
    # Negative supercoiling optimum parameters:
    c_opt=-0.3,
    sigma_opt=0.2,
    # Rate-limiting parameters for mRNA:
    A=1.0,
    B=0.02,
    # Supercoil injection parameters:
    Rmax=0.1,
    sigma=1.0,
    # Stochastic parameters for supercoiling and mRNA:
    stochastic_c=False,
    noise_std_c=0.01,
    stochastic_m=False,
    noise_std_m=0.01,
    rng_seed=42
):

    :param num_segments: discrete ring size
    :param alpha: diffusion constant * dt/dx^2 (<=0.5 stable in forward Euler)
    :param steps_per_frame: how many sub-steps each yield
    :param genes: list of gene dicts:
        {
          "name": str,
          "gene_index": int,
          "orientation": +1 or -1,
          "promoter_offset": int
        }
    :param c_opt, sigma_opt: center and width for negative supercoiling optimum
    :param A, B: define the mRNA eqn
    :param Rmax, sigma: define supercoil injection rate
    :param stochastic_c, noise_std_c: optional noise in supercoiling
    :param stochastic_m, noise_std_m: optional noise in mRNA
    :param rng_seed: reproducible random seed
    :yield: (c, M)
        c: supercoiling array
        M: array of length len(genes) for mRNA
 
            "promoter_idx": promoter_idx
        })

    idx = np.arange(num_segments)
    left_idx  = (idx - 1) % num_segments
    right_idx = (idx + 1) % num_segments

    while True:
        for _ in range(steps_per_frame):
            # 1) Diffusion
            c_new = c.copy()
            for i in range(num_segments):
                c_new[i] = c[i] + alpha*(c[left_idx[i]] - 2*c[i] + c[right_idx[i]])

            if stochastic_c:
                c_new += rng.normal(0, noise_std_c, size=num_segments)

            for i, g_inf in enumerate(gene_info):
                g_idx = g_inf["gene_idx"]
                # The local c_g might define rate
                c_g = c[g_idx]
                rate = Rmax * np.exp(-(c_g**2)/ (sigma**2))

                three_prime = (g_idx + 1) % num_segments  # + supercoils
                five_prime  = (g_idx - 1) % num_segments  # - supercoils

                c_new[three_prime] += rate
                c_new[five_prime]  -= rate

            c = c_new

            for i, g_inf in enumerate(gene_info):
                p_idx = g_inf["promoter_idx"]
                c_prom = c[p_idx]
                # dm/dt = A * exp( -(c_prom - c_opt)^2 / sigma_opt^2 ) - B*m
                dM = A * np.exp(-((c_prom - c_opt)**2)/(sigma_opt)) - B*M[i]
                M[i] += dM

                # optional noise in mRNA
                if stochastic_m:
                    M[i] += rng.normal(0, noise_std_m)

        yield c, M




def animate_plasmid(
    num_segments,
    genes,
    simulation_generator,
    total_frames=100,
    scale=0.3,
    gif_filename=None
):

    angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
    base_x = np.cos(angles)
    base_y = np.sin(angles)

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gene_colors = [color_cycle[i % len(color_cycle)] for i in range(len(genes))]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12,6))
    fig.suptitle("Supercoiling Influenced Transcription")

    ax_left.set_aspect('equal')
    ax_left.axis('off')
    x_circle_closed = np.append(base_x, base_x[0])
    y_circle_closed = np.append(base_y, base_y[0])
    ax_left.plot(x_circle_closed, y_circle_closed, 'k--', linewidth=1)

    # line for supercoiling radius
    supercoil_line, = ax_left.plot([], [], 'b-', linewidth=2, label='Supercoiling')

    # Genes + promoters
    import matplotlib.patches as mpatches
    from matplotlib.patches import RegularPolygon

    patch_size = 0.05
    gene_markers = []  # list of (gene_marker, promoter_marker)
    for i, g in enumerate(genes):
        color = gene_colors[i]
        # Gene circle
        gm, = ax_left.plot([], [], marker='o', color=color, markersize=8,
                           label=g.get("name", f"Gene{i+1}"))
        # Promoter triangle
        tri = RegularPolygon((0,0), 3, radius=patch_size, orientation=0.0,
                             color=color, ec='none')
        ax_left.add_patch(tri)
        gene_markers.append((gm, tri))

    # Add a small arc from 0°->90°, labeling 3' and 5'
    arc_radius = 0.3
    arc_patch = mpatches.Arc((0, 0), 2*arc_radius, 2*arc_radius, angle=0, theta1=0, theta2=90,
                             color='black', linewidth=1.5)
    ax_left.add_patch(arc_patch)
    ax_left.text(arc_radius*1.2, 0, "3'", ha='center', va='center', fontsize=9)
    ax_left.text(0, arc_radius*1.2, "5'", ha='center', va='center', fontsize=9)

    # Legend out of the way
    ax_left.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)


    ax_right.set_xlim(0, total_frames)
    ax_right.set_ylim(0, 1)
    ax_right.set_xlabel("Time (frames)")
    ax_right.set_ylabel("mRNA Level")
    ax_right.set_title("mRNA Profiles")

    m_lines = []
    for i, g in enumerate(genes):
        color = gene_colors[i]
        ln, = ax_right.plot([], [], color=color, label=g.get("name", f"G{i+1}"))
        m_lines.append(ln)

    ax_right.legend(loc="upper left")

    # We'll store (time, M for each gene)
    t_vals = []
    M_vals = [[] for _ in genes]

    def init():
        supercoil_line.set_data([], [])
        for (gm, tri) in gene_markers:
            gm.set_data([], [])
            tri.xy = (0,0)
            tri.orientation = 0
        for ln in m_lines:
            ln.set_data([], [])
        return [supercoil_line] + [gmk for pair in gene_markers for gmk in pair] + m_lines

    def update(frame):
        c, M = next(simulation_generator)

        # supercoiling
        radial = 1.0 + scale*c
        x_sc = radial * base_x
        y_sc = radial * base_y
        x_sc_closed = np.append(x_sc, x_sc[0])
        y_sc_closed = np.append(y_sc, y_sc[0])
        supercoil_line.set_data(x_sc_closed, y_sc_closed)

        # gene + promoter
        for i, g in enumerate(genes):
            g_idx = g["gene_index"]
            orient = g["orientation"]
            offset = g["promoter_offset"]
            promoter_idx = (g_idx + orient*offset) % num_segments

            gx = radial[g_idx]*base_x[g_idx]
            gy = radial[g_idx]*base_y[g_idx]
            px = radial[promoter_idx]*base_x[promoter_idx]
            py = radial[promoter_idx]*base_y[promoter_idx]

            gm, tri = gene_markers[i]
            gm.set_data(gx, gy)

            tri.xy = (px, py)
            # orientation: from promoter -> gene
            dx, dy = gx - px, gy - py
            angle = np.arctan2(dy, dx) - np.pi/2
            tri.orientation = angle

        # mRNA lines
        t_vals.append(frame)
        for i in range(len(genes)):
            M_vals[i].append(M[i])
            m_lines[i].set_data(t_vals, M_vals[i])

        # dynamic y-limits
        max_m = max(1.0, max(max(vals) for vals in M_vals))
        ax_right.set_ylim(0, max_m * 1.1)

        # return artists
        return [supercoil_line] + [gmk for pair in gene_markers for gmk in pair] + m_lines

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        init_func=init,
        blit=False,
        interval=150
    )

    if gif_filename is not None:
        ani.save(gif_filename, writer="pillow", fps=5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define a plasmid and genes
    # orientation no longer affects supercoiling injection,
    # but it does shift the promoter index.
    N = 40
    genes = [
        {"name":"GeneA", "gene_index":5,  "orientation":+1, "promoter_offset":2},
        {"name":"GeneB", "gene_index":15, "orientation":-1, "promoter_offset":1},
        {"name":"GeneC", "gene_index":25, "orientation":+1, "promoter_offset":3},
    ]

    # Create the simulation generator with negative supercoiling optimum
    gen = simulate_plasmid(
        num_segments=N,
        alpha=0.3,
        steps_per_frame=2,
        genes=genes,
        c_opt=-0.3,         # negative supercoiling optimum
        sigma_opt=0.2,      # RBF width
        A=1.0, B=0.02,
        Rmax=0.1, sigma=1.0,
        stochastic_c=True, noise_std_c=0.01,
        stochastic_m=True, noise_std_m=0.01,
        rng_seed=123
    )

    # Animate
    animate_plasmid(
        num_segments=N,
        genes=genes,
        simulation_generator=gen,
        total_frames=100,
        scale=0.4,
        gif_filename=None  # or "my_output.gif"
    )
