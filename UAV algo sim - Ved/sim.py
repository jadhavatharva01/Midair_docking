import cv2
import numpy as np
import math
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

FPS    = 24
DT     = 1.0 / FPS
N_FR   = int(18.0 * FPS)

# mothership helix params
RADIUS = 1.2
OMEGA  = 0.45
Z_AMP  = 0.55
Z_FREQ = 0.30

# controller gains
KP_XY, KD_XY = 2.8, 0.35
KP_Z,  KD_Z  = 1.6, 0.20
V_MAX    = 2.0
VZ_MAX   = 0.55
ALIGN_THR = 0.025

# disturbances — motor imbalance + Dryden wind
np.random.seed(7)
MOTOR_BIAS_X = 0.050
WIND_SIGMA   = 0.095
WIND_TAU     = 3.5

def _init_wind():
    a = DT / WIND_TAU
    wx = wy = 0.0
    xs, ys = [], []
    for _ in range(N_FR):
        wx = (1-a)*wx + a*np.random.normal(0, WIND_SIGMA)
        wy = (1-a)*wy + a*np.random.normal(0, WIND_SIGMA)
        xs.append(wx)
        ys.append(wy)
    return np.array(xs), np.array(ys)

WIND_X, WIND_Y = _init_wind()


def mship_pos(t):
    a = OMEGA * t
    z = Z_AMP * math.sin(2*math.pi*Z_FREQ*t) + Z_AMP
    return np.array([RADIUS*math.cos(a), RADIUS*math.sin(a), z])


def mship_vel(t):
    a = OMEGA * t
    dz = Z_AMP * 2*math.pi*Z_FREQ * math.cos(2*math.pi*Z_FREQ*t)
    return np.array([-RADIUS*OMEGA*math.sin(a), RADIUS*OMEGA*math.cos(a), dz])


def pd_step(sp, sv, mp, mv, docked, wx, wy, prev):
    r = sp - mp
    v = sv - mv
    lat = math.sqrt(r[0]**2 + r[1]**2)

    ax = -KP_XY*r[0] - KD_XY*v[0] + mv[0] + MOTOR_BIAS_X + wx*0.4
    ay = -KP_XY*r[1] - KD_XY*v[1] + mv[1] + MOTOR_BIAS_X*0.6 + wy*0.4

    # one-frame actuation lag
    ax = 0.70*ax + 0.30*prev[0]
    ay = 0.70*ay + 0.30*prev[1]

    dz  = sp[2] - mp[2]
    azt = KP_Z*dz + KD_Z*(sv[2]-mv[2])
    aligned = lat < ALIGN_THR and not docked

    Vx = np.clip(ax, mv[0]-V_MAX, mv[0]+V_MAX)
    Vy = np.clip(ay, mv[1]-V_MAX, mv[1]+V_MAX)

    if aligned and dz > 0.02:
        vz_cmd = -min(KP_Z*dz + KD_Z*max(sv[2]-mv[2], 0), VZ_MAX)
        Vz = max(mv[2]+vz_cmd, mv[2]-VZ_MAX)
    elif aligned:
        Vz = mv[2]
    else:
        Vz = mv[2] - azt*0.5

    return np.array([Vx, Vy, Vz]), lat, aligned, r, v


def simulate():
    sp   = np.array([RADIUS, 0.0, 2.5])
    sv   = mship_vel(0).copy()
    prev = sv.copy()
    docked = False
    td_t = td_err = None
    dock_offset = np.zeros(3)

    traj_s, traj_m, rows = [], [], []

    for fi in range(N_FR):
        t  = fi * DT
        mp = mship_pos(t)
        mv = mship_vel(t)

        sv_new, lat, aligned, r_rel, v_rel = pd_step(
            sp, sv, mp, mv, docked, WIND_X[fi], WIND_Y[fi], prev)

        if not docked and lat < ALIGN_THR and abs(sp[2]-mp[2]) < 0.03:
            docked = True
            td_t   = t
            td_err = lat
            dock_offset = sp - mp
            print(f"  TOUCHDOWN t={t:.1f}s  lat_err={lat*100:.2f} cm  z={sp[2]:.3f} m")

        if docked:
            sp   = mp + dock_offset
            sv   = mv.copy()
            prev = sv.copy()
        else:
            prev = sv_new.copy()
            sv   = sv_new
            sp   = sp + sv*DT
            sp[2] = max(sp[2], mp[2]-0.01)

        traj_s.append(sp.copy())
        traj_m.append(mp.copy())

        rows.append({
            't': t, 'sp': sp.copy(), 'mp': mp.copy(),
            'sv': sv.copy(), 'mv': mv.copy(),
            'r_rel': r_rel.copy(), 'v_rel': v_rel.copy(),
            'lat_err': lat, 'tz': float(np.linalg.norm(sp-mp)),
            'aligned': aligned, 'docked': docked,
            'td_t': td_t, 'td_err': td_err,
            'wind_x': WIND_X[fi], 'wind_y': WIND_Y[fi]
        })

    return rows, np.array(traj_s), np.array(traj_m)


def write_log(rows):
    path = os.path.join(OUT, "aeronest_log.csv")
    fields = ['t','scout_x','scout_y','scout_z','mship_x','mship_y','mship_z',
              'sv_x','sv_y','sv_z','mv_x','mv_y','mv_z',
              'r_rel_x','r_rel_y','r_rel_z','v_rel_x','v_rel_y','v_rel_z',
              'lat_err_m','range_to_pad_m','aligned','docked','phase']

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            phase = ('DOCKED' if row['docked'] else
                     'DESCENDING' if row['aligned'] else 'LATERAL_CORRECTION')
            w.writerow({
                't':             f"{row['t']:.4f}",
                'scout_x':       f"{row['sp'][0]:.5f}",
                'scout_y':       f"{row['sp'][1]:.5f}",
                'scout_z':       f"{row['sp'][2]:.5f}",
                'mship_x':       f"{row['mp'][0]:.5f}",
                'mship_y':       f"{row['mp'][1]:.5f}",
                'mship_z':       f"{row['mp'][2]:.5f}",
                'sv_x':          f"{row['sv'][0]:.5f}",
                'sv_y':          f"{row['sv'][1]:.5f}",
                'sv_z':          f"{row['sv'][2]:.5f}",
                'mv_x':          f"{row['mv'][0]:.5f}",
                'mv_y':          f"{row['mv'][1]:.5f}",
                'mv_z':          f"{row['mv'][2]:.5f}",
                'r_rel_x':       f"{row['r_rel'][0]:.5f}",
                'r_rel_y':       f"{row['r_rel'][1]:.5f}",
                'r_rel_z':       f"{row['r_rel'][2]:.5f}",
                'v_rel_x':       f"{row['v_rel'][0]:.5f}",
                'v_rel_y':       f"{row['v_rel'][1]:.5f}",
                'v_rel_z':       f"{row['v_rel'][2]:.5f}",
                'lat_err_m':     f"{row['lat_err']:.6f}",
                'range_to_pad_m':f"{row['tz']:.6f}",
                'aligned':       '1' if row['aligned'] else '0',
                'docked':        '1' if row['docked'] else '0',
                'phase':         phase,
            })
            if row['docked']:
                break
    print(f"  Log -> {path}")


def draw_quad(ax, pos, col, arm=0.12):
    x, y, z = pos
    for dx, dy in [(arm,0),(-arm,0),(0,arm),(0,-arm)]:
        ax.plot([x,x+dx],[y,y+dy],[z,z], color=col, lw=2.5, zorder=10)
        ax.plot([x+dx],[y+dy],[z], 'o', color=col, ms=5, zorder=11)
    ax.plot([x],[y],[z], 'o', color=col, ms=7, mec='white', mew=0.8, zorder=12)


def draw_pad(ax, pos, size=0.18, col='#2ecc71'):
    x, y, z = pos
    s = size
    verts = [[(x-s,y-s,z),(x+s,y-s,z),(x+s,y+s,z),(x-s,y+s,z)]]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.85, facecolor=col,
                                          edgecolor='darkgreen', lw=1.5))
    ax.plot([x-s,x+s],[y,y],[z,z], color='white', lw=1.5, zorder=9)
    ax.plot([x,x],[y-s,y+s],[z,z], color='white', lw=1.5, zorder=9)


def render_frame(row, traj_s, traj_m, fig_size=(16,9)):
    fi  = int(row['t'] / DT)
    t   = row['t']
    sp  = row['sp']
    mp  = row['mp']
    lat = row['lat_err']
    tz  = row['tz']
    ali = row['aligned']
    doc = row['docked']
    td_e = row['td_err']
    r_rel = row['r_rel']

    fig = plt.figure(figsize=fig_size, dpi=80, facecolor='white')
    ax3 = fig.add_axes([0.0,0.0,0.63,1.0], projection='3d')
    ax3.set_facecolor('white')
    for p in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        p.fill = False
        p.set_edgecolor('#cccccc')
    ax3.grid(True, color='#dddddd', linewidth=0.5)
    ax3.tick_params(colors='#555', labelsize=8)
    ax3.set_xlim(-1.8,1.8); ax3.set_ylim(-1.8,1.8); ax3.set_zlim(-0.3,3.5)
    ax3.set_xlabel('x (m)', color='#444', fontsize=9, labelpad=6)
    ax3.set_ylabel('y (m)', color='#444', fontsize=9, labelpad=6)
    ax3.set_zlabel('z (m)', color='#444', fontsize=9, labelpad=6)
    ax3.view_init(elev=22, azim=35+t*3.0)

    theta = np.linspace(0, 2*np.pi, 240)
    zhelix = Z_AMP*np.sin(2*math.pi*Z_FREQ*(theta/OMEGA)) + Z_AMP
    ax3.plot(RADIUS*np.cos(theta), RADIUS*np.sin(theta), zhelix,
             '--', color='#88cc88', lw=1.0, alpha=0.6)

    ts = max(0, fi-60)
    if len(traj_s[ts:fi+1]) > 1:
        ax3.plot(traj_s[ts:fi+1,0], traj_s[ts:fi+1,1], traj_s[ts:fi+1,2],
                 color='#3399ff', lw=1.5, alpha=0.55)
    if len(traj_m[ts:fi+1]) > 1:
        ax3.plot(traj_m[ts:fi+1,0], traj_m[ts:fi+1,1], traj_m[ts:fi+1,2],
                 color='#66bb66', lw=1.2, alpha=0.45)

    ax3.plot([sp[0],sp[0]],[sp[1],sp[1]],[sp[2],mp[2]],
             '--', color='#ffaa00', lw=1.0, alpha=0.6)
    ax3.plot([sp[0],mp[0]],[sp[1],mp[1]],[sp[2],mp[2]],
             color='#cc7700', lw=1.3, alpha=0.8)

    draw_pad(ax3, mp)
    draw_quad(ax3, sp, col='#1a9a1a' if (ali or doc) else '#2ecc71')

    if doc:
        ps, pc = f"DOCKED  err={td_e*100:.1f} cm", '#1a7a1a'
    elif ali:
        ps, pc = "ALIGNED -> DESCENDING", '#1a7a1a'
    else:
        ps, pc = "LATERAL CORRECTION", '#cc6600'

    ax3.text2D(0.5,0.97, f"AeroNest docking sim    t = {t:.2f} s",
               transform=ax3.transAxes, ha='center', va='top',
               fontsize=11, fontweight='bold', color='#222222')
    ax3.text2D(0.5,0.92, ps, transform=ax3.transAxes, ha='center', va='top',
               fontsize=10, fontweight='bold', color=pc)

    ax_r = fig.add_axes([0.66,0.0,0.34,1.0])
    ax_r.axis('off')
    ax_bar = fig.add_axes([0.68,0.60,0.28,0.32])
    pa = max(0.0, 1.0 - lat/0.50)
    pd_ = max(0.0, min(1.0, 1.0 - abs(tz-0.10)/1.5))
    bars = ax_bar.bar(['alignment', 'depth'], [pa, pd_],
                      color=['#3399ff','#2ecc71'], width=0.5)
    ax_bar.set_ylim(0,1.2); ax_bar.set_yticks([0,0.5,1.0])
    ax_bar.set_title('performance', fontsize=9, pad=4)
    ax_bar.spines[['top','right']].set_visible(False)
    for bar, val in zip(bars, [pa, pd_]):
        ax_bar.text(bar.get_x()+bar.get_width()/2, val+0.03, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    lines = [
        ("range to pad", f"{tz:.3f} m"),
        ("scout z",      f"{sp[2]:.3f} m"),
        ("mship z",      f"{mp[2]:.3f} m"),
        ("r_rel x",      f"{r_rel[0]:+.3f} m"),
        ("r_rel y",      f"{r_rel[1]:+.3f} m"),
        ("aligned",      "YES" if (ali or doc) else "NO"),
    ]
    y0 = 0.56
    for lbl, val in lines:
        vcol = '#1a7a1a' if lbl == "aligned" and (ali or doc) else 'black'
        ax_r.text(0.04, y0, lbl, transform=ax_r.transAxes,
                  fontsize=9, color='#333333', fontfamily='monospace')
        ax_r.text(0.56, y0, val, transform=ax_r.transAxes,
                  fontsize=9, color=vcol, fontfamily='monospace',
                  fontweight='bold' if lbl == "aligned" else 'normal')
        y0 -= 0.042

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    W_f, H_f = fig.canvas.get_width_height()
    frame = cv2.cvtColor(buf.reshape(H_f,W_f,4)[:,:,1:], cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return frame


# apriltag layout on mothership pad
# ID0: 4x4 cm central | ID1-4: 1x1 cm at top/left/right/bottom | ID5: 1x1 cm inside ID0
TAG_DETECT_RANGE = 1.8
LARGE_HALF = 0.020   # 4cm tag
SMALL_HALF = 0.005   # 1cm tag
# gap between central tag edge and outer tag centre = 2cm
# so offset = LARGE_HALF + 0.020 + SMALL_HALF = 0.045m
OUTER_OFF  = 0.045

PAD_TAGS = [
    (0,  0.0,       0.0,      LARGE_HALF, [[1,1,0,0],[0,0,1,0],[1,0,0,1],[0,1,1,0]]),
    (1,  0.0,       OUTER_OFF, SMALL_HALF, [[1,0,0,1],[0,1,1,0],[1,1,0,0],[0,0,1,1]]),
    (2, -OUTER_OFF, 0.0,      SMALL_HALF, [[1,1,1,1],[0,0,0,0],[1,0,1,0],[0,1,0,1]]),
    (3,  OUTER_OFF, 0.0,      SMALL_HALF, [[0,1,1,0],[1,0,0,1],[0,0,1,1],[1,1,0,0]]),
    (4,  0.0,      -OUTER_OFF, SMALL_HALF, [[0,1,0,1],[1,1,0,0],[0,0,1,1],[1,0,1,0]]),
    (5,  0.0,       0.0,      SMALL_HALF, [[1,1,1,0],[0,0,0,1],[1,0,1,1],[0,1,0,0]]),
]

TAG_COLORS = [
    (0,255,0), (0,220,255), (255,140,0),
    (180,0,255), (255,60,60), (255,255,0)
]

# perception model constants (from CV report §4)
VIB_BASE       = 0.80
BLUR_SCALE     = 8.0
WIND_BLUR_SC   = 1.2
SHADOW_ALPHA   = 0.72
T_BUDGET       = 33.3
T_DROP_THR     = 30.5
T_QM           = 1.0
T_TAG_MIN      = 8.0
T_TAG_MAX      = 15.0
T_PNP          = 1.5
T_ESKF         = 0.8
T_RECOVERY     = 3.5
T_ENHANCE      = 5.0
T_ROS2         = 8.0
T_MAVLINK      = 2.0


def blur_params(vrx, vry, wx, wy):
    sx = vrx + wx*WIND_BLUR_SC
    sy = vry + wy*WIND_BLUR_SC
    spd = math.sqrt(sx**2+sy**2) + VIB_BASE
    k = max(7, min(1 + 2*int(spd*BLUR_SCALE), 45))
    angle = math.degrees(math.atan2(-sy, sx))
    score = 1.0 / (1.0 + 0.008*k**2)
    return k, angle, score


def proc_time(bk, n_prev, enhance, t):
    frac = (bk-7)/38.0
    tt = T_QM + T_PNP + T_ESKF + T_ROS2 + T_MAVLINK
    tt += T_TAG_MIN + frac*(T_TAG_MAX-T_TAG_MIN)
    tt += T_ENHANCE*frac if enhance else 0.0
    tt += T_RECOVERY if n_prev < 4 else 0.0
    tt += 9.0 if 0.45 < (t % 1.0) < 0.55 else 0.0
    return tt


def dir_blur(img, k, angle):
    if k <= 1:
        return img
    kern = np.zeros((k,k), dtype=np.float32)
    cx = k//2
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for i in range(k):
        off = i - cx
        x = int(round(cx + off*ca))
        y = int(round(cx + off*sa))
        if 0 <= x < k and 0 <= y < k:
            kern[y,x] = 1.0
    s = kern.sum()
    kern = kern/s if s > 0 else (kern.__setitem__((cx,cx),1.0) or kern)
    out = cv2.filter2D(img, -1, kern)
    vk = max(3, k//4)
    vk += 1 if vk%2==0 else 0
    return cv2.GaussianBlur(out, (vk,vk), 0)


def draw_tag(img, cx, cy, hp, tag_id, bits):
    s = max(int(hp), 4)
    cx, cy = int(cx), int(cy)
    cv2.rectangle(img, (cx-s,cy-s), (cx+s,cy+s), (0,0,0), -1)
    cell = 2.0*s/6.0
    ins = max(int(s-cell), 2)
    cv2.rectangle(img, (cx-ins,cy-ins), (cx+ins,cy+ins), (255,255,255), -1)
    ds = max(int(ins-cell), 1)
    cd = 2.0*ds/4.0
    if cd < 1.0:
        return
    for r in range(4):
        for c in range(4):
            x0 = int(cx-ds+c*cd); y0 = int(cy-ds+r*cd)
            cv2.rectangle(img, (x0,y0), (int(x0+cd),int(y0+cd)),
                          (0,0,0) if bits[r][c] else (255,255,255), -1)


def draw_pad_border(img, cx, cy, half):
    s = int(half); cx, cy = int(cx), int(cy)
    cv2.rectangle(img, (cx-s,cy-s), (cx+s,cy+s), (50,52,55), -1)
    r2 = int(s*0.90)
    cv2.rectangle(img, (cx-r2,cy-r2), (cx+r2,cy+r2), (72,74,78), -1)
    dash = max(5,s//10); gap = max(3,s//16); step = dash+gap
    for d in range(-s, s, step):
        d2 = min(d+dash, s)
        cv2.line(img, (cx+d,cy), (cx+d2,cy), (0,210,230), 1)
        cv2.line(img, (cx,cy+d), (cx,cy+d2), (0,210,230), 1)


def draw_shadow(img, px, py, h, max_h=2.5):
    W = img.shape[1]
    f = W/2.0
    prox = max(0.0, min(1.0, 1.0-(h/max_h)))
    if prox < 0.08:
        return
    wing_px = min(max(4, int(f*0.15/max(h,0.05))), int(W*0.18))
    body_px = min(max(2, int(f*0.06/max(h,0.05))), int(W*0.07))
    alpha = SHADOW_ALPHA * prox
    ug = int(180 - prox*160)
    sigma = max(1, int((1.0-prox)*18))
    layer = np.full_like(img, 255)
    cv2.ellipse(layer, (px,py), (wing_px,body_px), 0, 0, 360, (ug,ug,ug), -1)
    fl = int(wing_px*0.8)
    cv2.line(layer, (px,py-fl), (px,py+fl), (ug,ug,ug), max(2,body_px))
    k = sigma*2+1
    layer = cv2.GaussianBlur(layer, (k,k), sigma)
    img[:] = np.clip(img.astype(np.float32)*(1.0-alpha) +
                     layer.astype(np.float32)*alpha, 0, 255).astype(np.uint8)


def render_tag_frame(row, W=1280, H=720, prev=None, prev_n=0):
    sp  = row['sp'];  mp  = row['mp']
    sv  = row['sv'];  mv  = row['mv']
    t   = row['t'];   doc = row['docked']
    ali = row['aligned']
    td_t = row['td_t']; td_e = row['td_err']
    wx = row['wind_x']; wy = row['wind_y']

    vrx, vry = sv[0]-mv[0], sv[1]-mv[1]
    wm = math.sqrt(wx**2+wy**2)
    bk, bang, bscore = blur_params(vrx, vry, wx, wy)
    enhance = bscore < 0.55
    pms = proc_time(bk, prev_n, enhance, t)

    if pms > T_DROP_THR and prev is not None:
        drop = prev.copy()
        cv2.rectangle(drop, (W-100,6), (W-6,30), (210,220,255), -1)
        cv2.rectangle(drop, (W-100,6), (W-6,30), (0,60,200), 2)
        cv2.putText(drop, f"DROP {pms:.0f}ms", (W-96,23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,40,180), 1)
        return drop, prev_n, pms, True

    img = np.full((H,W,3), (255,255,255), dtype=np.uint8)
    for g in range(0, max(W,H), 60):
        if g < W: cv2.line(img,(g,0),(g,H),(235,235,235),1)
        if g < H: cv2.line(img,(0,g),(W,g),(235,235,235),1)

    dx  = mp[0]-sp[0]; dy = mp[1]-sp[1]; dz = sp[2]-mp[2]
    f_  = W/2.0
    h   = max(dz, 0.05)
    d3  = float(np.linalg.norm(sp-mp))
    ppx = int(W/2+(dx/h)*f_)
    ppy = int(H/2-(dy/h)*f_)

    draw_shadow(img, ppx, ppy, h)

    chalf = f_*(OUTER_OFF+LARGE_HALF+0.025)/h
    if chalf > 4:
        draw_pad_border(img, ppx, ppy, chalf)

    det = []
    spos = {}
    for tid, tdx, tdy, phm, bits in PAD_TAGS:
        sx_ = int(W/2+((dx+tdx)/h)*f_)
        sy_ = int(H/2-((dy+tdy)/h)*f_)
        spos[tid] = (sx_, sy_)
        hp = f_*phm/h
        if hp >= 3:
            draw_tag(img, sx_, sy_, hp, tid, bits)
        if hp >= 5 and d3 < TAG_DETECT_RANGE and bscore > 0.15:
            det.append(tid)
    n_det = len(det)

    for tid in det:
        sx_, sy_ = spos[tid]
        phm = LARGE_HALF if tid == 0 else SMALL_HALF
        hp  = f_*phm/h
        bsz = int(hp*1.45)
        bc  = TAG_COLORS[tid]
        tk  = max(5, bsz//4)
        for cx2,cy2,sx2,sy2 in [(sx_-bsz,sy_-bsz,1,1),(sx_+bsz,sy_-bsz,-1,1),
                                  (sx_-bsz,sy_+bsz,1,-1),(sx_+bsz,sy_+bsz,-1,-1)]:
            cv2.line(img,(cx2,cy2),(cx2+sx2*tk,cy2), bc, 2)
            cv2.line(img,(cx2,cy2),(cx2,cy2+sy2*tk), bc, 2)
        if bsz > 12:
            cv2.putText(img, f"ID:{tid}", (sx_-bsz,sy_-bsz-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bc, 1, cv2.LINE_AA)

    img = dir_blur(img, bk, bang)

    if h <= 0.25:
        df = max(0.0, (h-0.02)/(0.25-0.02))
        img = (img.astype(np.float32)*df).astype(np.uint8)

    # detection banner
    det_col = (0,150,0) if n_det > 0 else (0,0,180)
    ov = img.copy()
    cv2.rectangle(ov,(10,10),(350,60),(240,255,240),-1)
    cv2.addWeighted(ov,0.7,img,0.3,0,img)
    cv2.rectangle(img,(10,10),(350,60),det_col,2)
    cv2.putText(img, f"TAGS DETECTED: {n_det}/6", (25,48),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, det_col, 2, cv2.LINE_AA)

    # blur info
    cv2.putText(img,
        f"BLUR k={bk}  dir={int(bang)}deg  wind={wm:.2f}m/s"
        f"  [{'ENHANCE ON' if enhance else 'OFF'}]",
        (15,85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50,50,50), 1, cv2.LINE_AA)

    # per-tag status panel
    py0 = 100
    ov = img.copy()
    cv2.rectangle(ov,(10,py0),(220,py0+150),(255,255,255),-1)
    cv2.addWeighted(ov,0.6,img,0.4,0,img)
    cv2.rectangle(img,(10,py0),(220,py0+150),(180,180,180),1)
    for i,(tid,_,_,phm,_) in enumerate(PAD_TAGS):
        d = tid in det
        col = TAG_COLORS[tid] if d else (180,180,180)
        cv2.circle(img,(25,py0+20+i*22),6,col,-1)
        sz = f"{phm*200:.1f}x{phm*200:.1f}cm"
        cv2.putText(img, f"ID:{tid}  {sz}  [{'DET' if d else ' - '}]",
                    (40,py0+25+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # LED
    lc = (0,220,0) if n_det > 0 else (0,0,220)
    cv2.circle(img,(W-50,40),25,lc,-1)
    cv2.putText(img,f"{n_det}/6",(W-75,100),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,180,0),2,cv2.LINE_AA)

    # CPU meter
    mx,my,mw,mh_ = W-100, 130, 25, 150
    cv2.rectangle(img,(mx,my),(mx+mw,my+mh_),(50,50,50),1)
    cv2.putText(img,"CPU",(mx-5,my-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(150,150,0),1)
    fh = int((pms/40.0)*mh_)
    cv2.rectangle(img,(mx+1,my+mh_-fh),(mx+mw-1,my+mh_-1),(0,180,220),-1)
    by_ = my+mh_-int((33.3/40.0)*mh_)
    cv2.line(img,(mx-5,by_),(mx+mw+5,by_),(0,150,150),1)
    cv2.putText(img,"33ms",(mx-35,by_+5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(150,150,0),1)
    cv2.putText(img,f"{pms:.1f}ms",(mx-15,my+mh_+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(50,50,50),1)

    # telemetry box
    tw,th_ = 230, 240
    tx,ty_ = W-tw-10, H-th_-10
    ov = img.copy()
    cv2.rectangle(ov,(tx,ty_),(tx+tw,ty_+th_),(255,255,255),-1)
    cv2.addWeighted(ov,0.7,img,0.3,0,img)
    cv2.rectangle(img,(tx,ty_),(tx+tw,ty_+th_),(180,180,180),1)
    phase = 'DOCKED' if doc else ('DESCEND' if ali else 'ALIGN')
    tlines = [
        f"t     = {t:.2f} s",
        f"dist  = {d3:.3f} m",
        f"h_agl = {h:.3f} m",
        f"lat   = {row['lat_err']*100:.2f} cm",
        f"vrel  = {math.sqrt(vrx**2+vry**2):.3f} m/s",
        f"wind  = {wm:.3f} m/s",
        f"blur  = k{bk} / {bscore:.2f}",
        f"proc  = {pms:.1f} ms",
        f"tags  = {n_det}/6",
        f"phase = {phase}",
    ]
    for i,line in enumerate(tlines):
        cv2.putText(img,line,(tx+15,ty_+30+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(40,40,40),1,cv2.LINE_AA)

    if doc and td_t is not None:
        cv2.rectangle(img,(0,H-25),(W,H),(0,120,0),-1)
        cv2.putText(img,f"DOCKED  lat={td_e*100:.2f}cm  t={td_t:.1f}s",
                    (20,H-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,100),1)

    return img, n_det, pms, False


if __name__ == "__main__":
    print("=" * 60)
    print("  AeroNest v2 — Flying Mothership + AprilTag + CSV Log")
    print("=" * 60)

    print("\n[1/4] Simulating trajectory...")
    rows, traj_s, traj_m = simulate()

    print("\n[2/4] Writing CSV log (until touchdown)...")
    write_log(rows)

    FIG_W, FIG_H = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_3d  = os.path.join(OUT, "aeronest_3d_circle.mp4")
    out_tag = os.path.join(OUT, "aeronest_apriltag.mp4")
    vw_3d   = cv2.VideoWriter(out_3d,  fourcc, FPS, (FIG_W, FIG_H))
    vw_tag  = cv2.VideoWriter(out_tag, fourcc, FPS, (FIG_W, FIG_H))

    print(f"\n[3/4] Rendering {len(rows)} frames for 3D video...")
    for i, row in enumerate(rows):
        f3 = render_frame(row, traj_s, traj_m, fig_size=(FIG_W/80, FIG_H/80))
        vw_3d.write(cv2.resize(f3, (FIG_W, FIG_H)))
        if i % (FPS*3) == 0:
            s = 'DOCKED' if row['docked'] else ('aligned' if row['aligned'] else 'correcting')
            print(f"  t={row['t']:.0f}s  lat={row['lat_err']*100:.1f}cm  "
                  f"scout_z={row['sp'][2]:.2f}m  mship_z={row['mp'][2]:.2f}m  {s}")
    vw_3d.release()

    print(f"\n[4/4] Rendering {len(rows)} frames for AprilTag video...")
    prev_frame, prev_n, n_drops = None, 0, 0
    for i, row in enumerate(rows):
        frame, n_det, pms, dropped = render_tag_frame(
            row, W=FIG_W, H=FIG_H, prev=prev_frame, prev_n=prev_n)
        vw_tag.write(frame)
        if not dropped:
            prev_frame, prev_n = frame, n_det
        else:
            n_drops += 1
        if i % (FPS*3) == 0:
            d3 = float(np.linalg.norm(row['sp']-row['mp']))
            bk,_,_ = blur_params(row['sv'][0]-row['mv'][0], row['sv'][1]-row['mv'][1],
                                  row['wind_x'], row['wind_y'])
            print(f"  t={row['t']:.0f}s  dist={d3:.2f}m  blur=k{bk}  "
                  f"proc={pms:.1f}ms  tags={n_det}/6  {'DROP' if dropped else 'ok'}")
    vw_tag.release()
    print(f"  Total frame drops: {n_drops} / {len(rows)}")

    kb3  = os.path.getsize(out_3d)  // 1024
    kbt  = os.path.getsize(out_tag) // 1024
    print(f"\n  3D video  -> {out_3d}  ({kb3} KB)")
    print(f"  AprilTag  -> {out_tag}  ({kbt} KB)")
    print(f"  CSV log   -> {os.path.join(OUT, 'aeronest_log.csv')}")
    print("=" * 60)