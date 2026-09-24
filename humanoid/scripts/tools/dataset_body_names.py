"""
生成 / 校验 AMP 数据集 txt 的 body 顺序。

python3 humanoid/scripts/tools/dataset_body_names.py \
    'humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/*.txt' \
    --usd humanoid/assets/z1/usd/20dof_nohand/z1_v1_20dof.usd

python3 humanoid/scripts/tools/dataset_body_names.py \
    'humanoid/envs/t2pro_23dof_amp/datasets/23dof/hugingface/hands/*.txt' \
    --usd humanoid/assets/z1/usd/23dof/z1_v1_23dof.usd

python3 humanoid/scripts/tools/dataset_body_names.py \
    'humanoid/envs/t2pro_23dof_amp/datasets/23dof/hugingface/hands/*.txt' \
    --expect "base_link,leg_left_hip_pitch_link,..."

python3 humanoid/scripts/tools/dataset_body_names.py '.../txt/*.txt'

读 USD 需要 pxr（Isaac Sim 的 python 自带，或 pip install usd-core）；
没有 pxr 时用 --expect 走 --torso-prefix / --torso-chain 模板。

/home/user/miniforge3/envs/g2h/bin/python humanoid/scripts/tools/dataset_body_names.py \
    'humanoid/envs/z1_29dof_amp_sym_flat/datasets/20dof/txt/*.txt' \
    --usd humanoid/assets/z1/usd/20dof_nohand/z1_v1_20dof.usd
"""

import argparse
import glob
import json
import re
import sys
from collections import deque

import numpy as np

POS_TOL = 2e-3


def quat_conj(q):
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_apply_inverse(q, v):
    qv = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(quat_conj(q), qv), q)[..., 1:]


def load_frames(path):
    with open(path) as f:
        return np.asarray(json.load(f)["Frames"], dtype=np.float64)


def detect_layout(width):
    return [((width - 13 * nb) // 4, nb) for nb in range(1, 60)
            if width - 13 * nb > 0 and (width - 13 * nb) % 4 == 0]


def split_block(fm, nj, nb, swizzle=True):
    o = 2 * nj
    pos = fm[:, o:o + 3 * nb].reshape(-1, nb, 3)
    o += 3 * nb
    quat = fm[:, o:o + 4 * nb].reshape(-1, nb, 4)
    return pos, (quat[..., [3, 0, 1, 2]] if swizzle else quat)


def pair_stats(files, nj, nb):
    out = []
    for p in files:
        pos = split_block(load_frames(p), nj, nb)[0]
        d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)
        out.append(np.percentile(d, 75, axis=0) - np.percentile(d, 25, axis=0))
    return np.median(np.stack(out), axis=0)


def kruskal(nb, edges):
    par = list(range(nb))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    tree = []
    for (i, j), _ in sorted(edges.items(), key=lambda kv: kv[1]):
        ri, rj = find(i), find(j)
        if ri != rj:
            par[ri] = rj
            tree.append((i, j))
    return tree


def components(nb, tree):
    adj = {i: set() for i in range(nb)}
    for i, j in tree:
        adj[i].add(j)
        adj[j].add(i)
    seen, comps = set(), []
    for s in range(nb):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def resolve_layout(files, width):
    scored = []
    for nj, nb in detect_layout(width):
        if nb < 4:
            continue
        if not all(np.allclose(np.linalg.norm(split_block(load_frames(p), nj, nb)[1], axis=-1), 1.0, atol=2e-3)
                   for p in files[:3]):
            continue
        iqr = pair_stats(files[:3], nj, nb)
        edges = {(i, j): float(iqr[i, j]) for i in range(nb) for j in range(i + 1, nb)
                 if iqr[i, j] < POS_TOL}
        n_comp = len(components(nb, kruskal(nb, edges)))
        scored.append({"nj": nj, "nb": nb, "n_edge": len(edges), "n_comp": n_comp})
        print(f"  布局候选 nj={nj:3d} nb={nb:3d}：恒定距离边 {len(edges):3d}，连通块 {n_comp:2d}"
              f"{'  ← 通过' if n_comp == 1 else ''}")
    pool = [s for s in scored if s["n_comp"] == 1]
    if not pool:
        sys.exit(f"无法定位布局（帧宽 {width}）：所有候选的恒定距离图都不连通")
    s = max(pool, key=lambda x: x["nb"])
    print(f"→ 采用 num_joints={s['nj']}, num_bodies={s['nb']}\n")
    return s["nj"], s["nb"]


def edge_metrics(files, nj, nb, i, j, swizzle):
    offs, drifts, rngs = [], [], []
    for p in files:
        pos, quat = split_block(load_frames(p), nj, nb, swizzle)
        rel = quat_apply_inverse(quat[:, i], pos[:, j] - pos[:, i])
        offs.append(np.linalg.norm(rel, axis=-1).mean())
        drifts.append(np.linalg.norm(rel - rel.mean(axis=0), axis=-1).max())
        qr = quat_mul(quat_conj(quat[:, i]), quat[:, j])
        ang = np.rad2deg(2.0 * np.arccos(np.clip(np.abs(qr[:, 0]), -1.0, 1.0)))
        rngs.append(ang.max() - ang.min())
    md = lambda v: float(np.median(v))  # noqa: E731
    return md(offs), md(drifts), md(rngs)


def pick_swizzle(files, nj, nb, k=40):
    pos = split_block(load_frames(files[0]), nj, nb)[0].mean(axis=0)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    cand = sorted((d[i, j], i, j) for i in range(nb) for j in range(i + 1, nb))
    pairs = [(i, j) for _, i, j in cand[:k]]
    return min((True, False),
               key=lambda sw: sum(edge_metrics(files[:3], nj, nb, i, j, sw)[1] for i, j in pairs))


def print_fingerprint(files, nj, nb, args):
    sw = pick_swizzle(files, nj, nb)
    print(f"\n未给 --usd / --expect：打印偏移指纹（swizzle={sw}，用前 3 个文件算）")
    print(f"{'对':>8} {'|偏移|m':>9} {'漂移mm':>8} {'转角°':>7}")
    rows = []
    for i in range(nb):
        for j in range(i + 1, nb):
            off, dr, rng = edge_metrics(files[:3], nj, nb, i, j, sw)
            if dr < args.max_drift:
                rows.append((off, dr, rng, i, j))
    for off, dr, rng, i, j in sorted(rows):
        note = "共心" if off < 1e-4 else ("不动" if rng < args.min_angle else "")
        print(f"{i:3d}->{j:<3d} {off:9.4f} {dr * 1e3:8.4f} {rng:7.2f}   {note}")
    print(f"\n共 {len(rows)} 对相对平移恒定，真父子只有 {nb - 1} 对")


def chain_edges(names, args):
    slots = [(args.left_prefix_leg, "leg"), (args.right_prefix_leg, "leg"),
             (args.left_prefix_arm, "arm"), (args.right_prefix_arm, "arm")]
    chains, tor, root, notes = {}, {}, None, []
    for idx, nm in enumerate(names):
        if nm == args.root:
            root = idx
            continue
        if args.torso_prefix and nm.startswith(args.torso_prefix):
            for rank, part in enumerate(args.torso_chain.split(",")):
                if part in nm:
                    tor[rank] = idx
                    break
            continue
        hit = None
        for pre, kind in slots:
            if not nm.startswith(pre):
                continue
            parts = args.leg_chain.split(",") if kind == "leg" else args.arm_chain.split(",")
            for rank, part in enumerate(parts):
                if part in nm:
                    hit = (pre, rank)
                    break
            break
        if hit is None:
            notes.append(nm)
            continue
        chains.setdefault(hit[0], {})[hit[1]] = idx

    edges, sym_pairs, attach = [], [], root
    for rank in sorted(tor):
        if attach is None:
            break
        edges.append((attach, tor[rank]))
        attach = tor[rank]
    for pre, members in chains.items():
        ranks = sorted(members)
        for a, b in zip(ranks, ranks[1:]):
            edges.append((members[a], members[b]))
        if attach is not None:
            edges.append((attach, members[ranks[0]]))
    for kind, pl, pr in (("leg", args.left_prefix_leg, args.right_prefix_leg),
                         ("arm", args.left_prefix_arm, args.right_prefix_arm)):
        cl, cr = chains.get(pl, {}), chains.get(pr, {})
        for a, b in zip(sorted(cl), sorted(cl)[1:]):
            if a in cr and b in cr:
                sym_pairs.append((f"{kind} {a}->{b}", (cl[a], cl[b]), (cr[a], cr[b])))
        if cl and cr and attach is not None:
            sym_pairs.append((f"{kind} {attach}->{min(cl)}", (attach, cl[min(cl)]),
                              (attach, cr[min(cr)])))
    return root, edges, notes, sym_pairs


def usd_body_order(path, args):
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        sys.exit("读 USD 需要 pxr：Isaac Sim 的 python 自带，或 pip install usd-core")

    stage = Usd.Stage.Open(path)
    if stage is None:
        sys.exit(f"打不开 USD：{path}")

    def walk():
        return iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))

    links = {str(p.GetPath()): p.GetName() for p in walk() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if not links:
        sys.exit(f"{path} 里没有带 PhysicsRigidBodyAPI 的 link")
    print(f"  USD 读到 {len(links)} 个 link")

    fixed_cls = getattr(UsdPhysics, "FixedJoint", None)
    keep = re.compile(args.keep_link)
    raw, merged, children = [], {}, {}
    for prim in walk():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        if not (b0 and b1):
            continue
        p, c = str(b0[0]), str(b1[0])
        if p not in links or c not in links:
            continue
        if fixed_cls is not None and prim.IsA(fixed_cls) and not keep.search(links[c]):
            merged[c] = p
        else:
            raw.append((p, c))

    def resolve(node):
        while node in merged:
            node = merged[node]
        return node

    edges = []
    for p, c in raw:
        rp, rc = resolve(p), resolve(c)
        if rp != rc:
            edges.append((rp, rc))
    kept = {resolve(p) for p in links}

    def build(pairs):
        ch, indeg = {}, {p: 0 for p in kept}
        for p, c in pairs:
            ch.setdefault(p, []).append(c)
            if c in indeg:
                indeg[c] += 1
        return ch, [p for p, n in indeg.items() if n == 0]

    children, roots = build(edges)
    if len(roots) != 1:
        children, roots = build([(c, p) for p, c in edges])
    if len(roots) != 1:
        roots = [r for r in roots if links.get(r) == args.root] or roots
    if len(roots) != 1:
        sys.exit(f"没能从 USD 唯一确定根 link：{[links.get(r, r) for r in roots]}")

    order, queue = [], deque([roots[0]])
    while queue:
        node = queue.popleft()
        if node in order:
            continue
        order.append(node)
        queue.extend(sorted(children.get(node, []), key=lambda p: links.get(p, p)))

    names = [links[p] for p in order]
    rest = sorted(n for p, n in links.items() if p not in order)
    if rest:
        print(f"  [WARN] {len(rest)} 个 link 没进树：{rest}")
        names += rest

    parent_of = {c: pp for pp, cs in children.items() for c in cs}
    pos = {p: k for k, p in enumerate(order)}
    parents = [pos.get(parent_of.get(p)) for p in order] + [None] * len(rest)
    return names, parents


def verify_table(files, nj, nb, names, args, parents=None):
    if parents is not None:
        edges = [(p, i) for i, p in enumerate(parents) if p is not None]
        root = next((i for i in range(nb) if parents[i] is None), None)
        notes, sym_pairs, src = [], None, "USD 真树"
    else:
        root, edges, notes, sym_pairs = chain_edges(names, args)
        src = "名字模板"
    if root is None:
        sys.exit("名字表里找不到 root 名字（--root 可改）")
    if notes:
        print(f"  [WARN] 这些名字没归类：{notes}")
    print(f"\n校验（边来自{src}）：{nb} 个 body / {len(edges)} 条边，root={root} {names[root]}")

    def total(sw):
        return sum(edge_metrics(files[:3], nj, nb, i, j, sw)[1] for i, j in edges)

    swizzle = min((True, False), key=total)
    M = {e: edge_metrics(files, nj, nb, e[0], e[1], swizzle) for e in edges}
    print(f"  四元数：{'需要 xyzw→wxyz' if swizzle else '已是 wxyz'}\n")

    print(f"{'边':>6}  {'父 -> 子':32s} {'|偏移|m':>8} {'漂移mm':>8} {'转角°':>6}  判定")
    bad, weak, fails = 0, 0, []
    for i, j in edges:
        off, dr, rng = M[(i, j)]
        if dr >= args.max_drift:
            verdict, bad = "FAIL", bad + 1
            fails.append(f"  {i}->{j} {names[i]} / {names[j]}：漂移 {dr * 1e3:.3f} mm"
                         f"（阈 {args.max_drift * 1e3:.3f}）→ 不是父子")
        elif off < 1e-4:
            verdict, weak = "ok 共心", weak + 1
        elif rng < args.min_angle:
            verdict, weak = "ok 不动", weak + 1
        else:
            verdict = "ok"
        pair = f"{names[i][:15]}>{names[j][:15]}"
        print(f"{i:3d}->{j:<2d}  {pair:32s} {off:8.4f} {dr * 1e3:8.4f} {rng:6.2f}  {verdict}")

    sym_bad = 0
    if not args.skip_sym_check:
        if sym_pairs is None:
            idx = {n: i for i, n in enumerate(names)}

            def mirror(nm):
                if "left" in nm:
                    return nm.replace("left", "right")
                if "right" in nm:
                    return nm.replace("right", "left")
                return None

            sym_pairs = []
            for p, c in edges:
                if "left" not in names[c]:
                    continue
                mc, mp = mirror(names[c]), mirror(names[p])
                if mc in idx and mp in idx and (idx[mp], idx[mc]) in M:
                    sym_pairs.append((names[c][:16], (p, c), (idx[mp], idx[mc])))
        print("\n左右对称自检：")
        for tag, eL, eR in sym_pairs:
            oL, oR = M[eL][0], M[eR][0]
            rel = abs(oL - oR) / max(oL, oR, 1e-9)
            good = rel < args.sym_tol
            sym_bad += 0 if good else 1
            print(f"  {tag:16s} 左 {oL:7.4f}  右 {oR:7.4f}  相对差 {rel * 100:5.2f}%  "
                  f"{'ok' if good else 'FAIL 左右不对称'}")

    for f in fails:
        print(f)
    print(f"\n结论：{len(edges) - bad}/{len(edges)} 条边通过；{weak} 条无法判定顺序；"
          f"对称自检失败 {sym_bad} 处")
    print("[PASS] 表与数据集自洽" if not bad and not sym_bad else "[FAIL] 表与数据集不一致")
    sys.exit(1 if (bad or sym_bad) else 0)


def main():
    ap = argparse.ArgumentParser(description="生成 / 校验 AMP 数据集的 body 顺序")
    ap.add_argument("files", nargs="+", help="数据集 txt（支持通配符）")
    ap.add_argument("--usd", default=None, help="机器人资产 USD（需要 pxr）")
    ap.add_argument("--expect", default=None, help="逗号分隔的名字表（没有 pxr 时用）")
    ap.add_argument("--root", default="base_link")
    ap.add_argument("--keep-link", default="hand", help="固定关节叶子中保留为 body 的名字正则")
    ap.add_argument("--torso-prefix", default="torso_", help="躯干链名字前缀（没躯干设空串）")
    ap.add_argument("--torso-chain", default="yaw,pitch,roll", help="躯干链从根往下的名字关键字")
    ap.add_argument("--leg-chain", default="hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll")
    ap.add_argument("--arm-chain", default="shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,hand")
    ap.add_argument("--left-prefix-leg", default="leg_left_")
    ap.add_argument("--right-prefix-leg", default="leg_right_")
    ap.add_argument("--left-prefix-arm", default="arm_left_")
    ap.add_argument("--right-prefix-arm", default="arm_right_")
    ap.add_argument("--max-drift", type=float, default=1e-4, help="允许的父系平移漂移(米)")
    ap.add_argument("--min-angle", type=float, default=1.0, help="判定「关节不动」的转角下限(度)")
    ap.add_argument("--sym-tol", type=float, default=0.01, help="左右对称自检相对容差")
    ap.add_argument("--skip-sym-check", action="store_true")
    args = ap.parse_args()

    files = sorted(p for pat in args.files for p in glob.glob(pat)) or args.files
    files = [f for f in files if f.endswith((".txt", ".json"))]
    if not files:
        sys.exit("没有匹配到数据集文件")
    print(f"读入 {len(files)} 个文件：{files[0]}")

    width = load_frames(files[0]).shape[1]
    nj, nb = resolve_layout(files, width)

    if args.usd:
        names, parents = usd_body_order(args.usd, args)
        print(f"\n由 {args.usd} 推出 {len(names)} 个 body：\n")
        print("DATASET_BODY_NAMES = [")
        for n in names:
            print(f'    "{n}",')
        print("]")
        if len(names) != nb:
            sys.exit(f"[FAIL] USD 推出 {len(names)} 个 body，数据集是 {nb} 个")
        verify_table(files, nj, nb, names, args, parents)

    if args.expect:
        names = [s.strip() for s in args.expect.split(",") if s.strip()]
        if len(names) != nb:
            sys.exit(f"--expect 给了 {len(names)} 个名字，数据集有 {nb} 个 body")
        verify_table(files, nj, nb, names, args)

    print_fingerprint(files, nj, nb, args)


if __name__ == "__main__":
    main()
