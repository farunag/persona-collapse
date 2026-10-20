"""Stage 2 analysis: drift per language, reply language, token budget, charts and summary.

Inputs (from replay.py / project.py / langid.py; one or more files each):
  --proj    *.proj.jsonl    per-turn projections and normalised scores
  --langid  *.langid.jsonl  per-reply detected language
  --replay  *.jsonl         replay transcripts (token counts, context stops), optional

Method:
  - condition = (model, capped).
  - Comparable turns: per condition and domain, the turn positions reached by >= --min-convs
    conversations in EVERY language present for that domain, AND by >= --min-retention of that
    language's conversations (relative to turn 1). The window is the unbroken run of such turns
    from turn 1, so a language whose conversations hit a context limit early (e.g. Yoruba on
    Gemma's 8,192 tokens) cannot leave only its shortest conversations in the later turns.
    All statistics use only these turns.
  - Per model x capped x lang x domain x turn: mean normalised score, 95% percentile bootstrap
    CI over conversations.
  - Per-language baseline: that language's coding-domain level = mean over coding conversations
    of each conversation's mean score over the comparable coding turns.
    drift = score - baseline. Drift CIs resample the coding conversations and the target
    conversations independently (for coding itself this slightly overstates the CI).
  - Reply language: share of replies whose detected language equals the user's.

Statistics are pure Python (no numpy), so they are testable anywhere; matplotlib is imported only
when drawing charts.

Usage:
  python scripts/analyze.py --proj <dir>/qwen_uncapped.proj.jsonl <dir>/qwen_capped.proj.jsonl \
      --langid <dir>/qwen_uncapped.langid.jsonl ... --replay <dir>/qwen_uncapped.jsonl ... --out results
"""

import argparse
import math
import csv
import random
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from stage2_config import DOMAINS, LANGS, RESULTS, read_jsonl

COLORS = {"en": "#2a78d6", "ha": "#eb6834", "yo": "#1baf7a"}  # categorical slots 1-3, validated all-pairs
LANG_NAMES = {"en": "English", "ha": "Hausa", "yo": "Yoruba"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def cond_of(r):
    return (r["model"].split("/")[-1], bool(r["capped"]))


def cond_name(c):
    return f"{c[0]} {'capped' if c[1] else 'uncapped'}"


def ordered(keys, order):
    return [k for k in order if k in keys] + sorted(k for k in keys if k not in order)


def rank(x, order):
    return order.index(x) if x in order else len(order)


def percentile(sorted_vals, q):
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def bootstrap_ci(stat, samples, n_boot, rng):
    """95% percentile CI of stat(*resampled samples); each sample is a list resampled with replacement."""
    if any(len(s) < 2 for s in samples):
        return None, None
    boots = sorted(stat(*(rng.choices(s, k=len(s)) for s in samples)) for _ in range(n_boot))
    return percentile(boots, 0.025), percentile(boots, 0.975)


def comparable_turns(rows, min_convs, min_retention=0.0):
    """{(cond, domain): turns 1..K reached, in every language, by >= min_convs conversations and by
    >= min_retention of that language's turn-1 conversations (unbroken from turn 1 when
    min_retention > 0)."""
    convs = defaultdict(set)  # (cond, domain, lang, turn) -> conv_ids
    langs = defaultdict(set)
    for r in rows:
        c = cond_of(r)
        convs[(c, r["domain"], r["lang"], r["turn"])].add(r["conv_id"])
        langs[(c, r["domain"])].add(r["lang"])
    out = {}
    for (c, d), ls in langs.items():
        turns = sorted({t for (c2, d2, _, t) in convs if (c2, d2) == (c, d)})
        n1 = {l: len(convs[(c, d, l, turns[0])]) if turns else 0 for l in ls}
        ok = [t for t in turns if all(len(convs[(c, d, l, t)]) >= max(min_convs, math.ceil(min_retention * n1[l]))
                                      for l in ls)]
        if min_retention > 0:  # unbroken window from the first turn
            win = []
            for t in turns:
                if t not in ok:
                    break
                win.append(t)
            ok = win
        out[(c, d)] = ok
    return out


def per_conv_means(rows, comp):
    """{(cond, lang, domain): {conv_id: mean score over comparable turns}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = cond_of(r)
        if r["turn"] in comp.get((c, r["domain"]), ()):
            acc[(c, r["lang"], r["domain"])][r["conv_id"]].append(r["score"])
    return {k: {cid: st.mean(v) for cid, v in d.items()} for k, d in acc.items()}


def baselines(conv_means):
    """{(cond, lang): list of per-conversation coding means} (the per-language baseline sample)."""
    return {(c, l): list(v.values()) for (c, l, d), v in conv_means.items() if d == "coding"}


def turn_table(rows, comp, base, n_boot, rng):
    cells = defaultdict(list)
    for r in rows:
        c = cond_of(r)
        if r["turn"] in comp.get((c, r["domain"]), ()):
            cells[(c, r["lang"], r["domain"], r["turn"])].append(r["score"])
    out = []
    for (c, l, d, t), vals in sorted(cells.items(), key=lambda kv: (kv[0][0], rank(kv[0][1], LANGS),
                                                                    rank(kv[0][2], DOMAINS), kv[0][3])):
        lo, hi = bootstrap_ci(st.mean, [vals], n_boot, rng)
        row = {"model": c[0], "capped": c[1], "lang": l, "domain": d, "turn": t, "n": len(vals),
               "mean_score": st.mean(vals), "ci_lo": lo, "ci_hi": hi,
               "baseline": None, "drift": None, "drift_lo": None, "drift_hi": None}
        b = base.get((c, l))
        if b:
            row["baseline"] = st.mean(b)
            row["drift"] = row["mean_score"] - row["baseline"]
            row["drift_lo"], row["drift_hi"] = bootstrap_ci(lambda x, y: st.mean(x) - st.mean(y), [vals, b], n_boot, rng)
        out.append(row)
    return out


def domain_table(conv_means, base, comp, n_boot, rng):
    out = []
    for (c, l, d), cm in conv_means.items():
        vals, b = list(cm.values()), base.get((c, l))
        row = {"model": c[0], "capped": c[1], "lang": l, "domain": d, "n_convs": len(vals),
               "turns": f"{min(comp[(c, d)])}-{max(comp[(c, d)])}" if comp.get((c, d)) else "",
               "mean_score": st.mean(vals), "baseline": st.mean(b) if b else None,
               "drift": st.mean(vals) - st.mean(b) if b else None, "drift_lo": None, "drift_hi": None}
        if b:
            row["drift_lo"], row["drift_hi"] = bootstrap_ci(lambda x, y: st.mean(x) - st.mean(y), [vals, b], n_boot, rng)
        out.append(row)
    return sorted(out, key=lambda r: (r["model"], r["capped"], rank(r["domain"], DOMAINS), rank(r["lang"], LANGS)))


def gap_table(conv_means, base, n_boot, rng):
    """Drift of each non-English language minus English drift, per condition and domain."""
    out = []
    for (c, l, d), cm in conv_means.items():
        en = conv_means.get((c, "en", d))
        if l == "en" or not en or not base.get((c, l)) or not base.get((c, "en")):
            continue
        samples = [list(cm.values()), base[(c, l)], list(en.values()), base[(c, "en")]]
        gap = lambda a, b, e, f: (st.mean(a) - st.mean(b)) - (st.mean(e) - st.mean(f))
        lo, hi = bootstrap_ci(gap, samples, n_boot, rng)
        out.append({"model": c[0], "capped": c[1], "domain": d, "lang": l, "gap_vs_en": gap(*samples),
                    "gap_lo": lo, "gap_hi": hi})
    return sorted(out, key=lambda r: (r["model"], r["capped"], r["domain"], r["lang"]))


def reply_language_tables(lrows, comp):
    by_turn, overall = defaultdict(list), defaultdict(list)
    for r in lrows:
        c = cond_of(r)
        overall[(c, r["lang"], r["domain"])].append(r)
        if not comp or r["turn"] in comp.get((c, r["domain"]), ()):
            by_turn[(c, r["lang"], r["domain"], r["turn"])].append(r["matches_user"])
    turn_rows = [{"model": c[0], "capped": c[1], "lang": l, "domain": d, "turn": t, "n": len(v),
                  "share_user_lang": sum(v) / len(v)} for (c, l, d, t), v in sorted(by_turn.items())]
    overall_rows = []
    for (c, l, d), rs in sorted(overall.items()):
        det = Counter(r["detected"] for r in rs)
        overall_rows.append({"model": c[0], "capped": c[1], "lang": l, "domain": d, "n": len(rs),
                             "share_user_lang": sum(r["matches_user"] for r in rs) / len(rs),
                             **{f"detected_{k}": det.get(k, 0) for k in ("en", "ha", "yo", "other", "empty")},
                             "code_only": sum(r["code_only"] for r in rs)})
    return turn_rows, overall_rows


def token_table(replays):
    acc = defaultdict(list)
    for r in replays:
        acc[(cond_of(r), r["lang"], r["domain"])].append(r)
    out = []
    for (c, l, d), rs in sorted(acc.items()):
        reply = [n for r in rs for n in r["reply_tokens"]]
        fins = [f for r in rs for f in r["finish_reasons"]]
        first_prompt = [r["prompt_tokens"][0] for r in rs if r["prompt_tokens"]]
        last_prompt = [r["prompt_tokens"][-1] for r in rs if r["prompt_tokens"]]
        out.append({"model": c[0], "capped": c[1], "lang": l, "domain": d, "n_convs": len(rs),
                    "mean_reply_tokens": st.mean(reply) if reply else None,
                    "share_hit_max_new_tokens": sum(f == "length" for f in fins) / len(fins) if fins else None,
                    "mean_prompt_tokens_turn1": st.mean(first_prompt) if first_prompt else None,
                    "max_prompt_tokens": max(last_prompt) if last_prompt else None,
                    "context_stops": sum(r["stopped_at_turn"] is not None for r in rs)})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=2):
    return "–" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def ci(lo, hi):
    return "" if lo is None else f" [{lo:+.2f}, {hi:+.2f}]"


# ---------------------------------------------------------------- charts (matplotlib, lazy)

def _style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "legend.frameon": False})


def _panels(plt, domains):
    fig, axes = plt.subplots(1, len(domains), figsize=(4.6 * len(domains), 4.2), sharey=True, squeeze=False)
    return fig, axes[0]


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)


def plot_drift(trows, cond, path, plt):
    rows = [r for r in trows if (r["model"], r["capped"]) == cond and r["drift"] is not None]
    domains = ordered({r["domain"] for r in rows}, DOMAINS)
    fig, axes = _panels(plt, domains)
    for ax, d in zip(axes, domains):
        for l in ordered({r["lang"] for r in rows if r["domain"] == d}, LANGS):
            rs = sorted((r for r in rows if r["domain"] == d and r["lang"] == l), key=lambda r: r["turn"])
            x = [r["turn"] for r in rs]
            ax.fill_between(x, [r["drift_lo"] if r["drift_lo"] is not None else r["drift"] for r in rs],
                            [r["drift_hi"] if r["drift_hi"] is not None else r["drift"] for r in rs],
                            color=COLORS.get(l), alpha=0.15, linewidth=0)
            ax.plot(x, [r["drift"] for r in rs], color=COLORS.get(l), linewidth=2, marker="o", markersize=3,
                    label=LANG_NAMES.get(l, l))
            _end_label(ax, x[-1], rs[-1]["drift"], l, COLORS.get(l))
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_title(d, color=INK, loc="left")
        ax.set_xlabel("Reply turn")
    axes[0].set_ylabel("Drift from own coding baseline\n(normalised score; 1 = default Assistant)")
    axes[-1].legend(loc="best")
    fig.suptitle(f"{cond_name(cond)}: drift by language (95% bootstrap CI over conversations)", x=0.01,
                 ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_capping(trows, model, path, plt):
    rows = [r for r in trows if r["model"] == model]
    domains = ordered({r["domain"] for r in rows}, DOMAINS)
    fig, axes = _panels(plt, domains)
    for ax, d in zip(axes, domains):
        for capped, ls in ((False, "-"), (True, "--")):
            for l in ordered({r["lang"] for r in rows if r["domain"] == d}, LANGS):
                rs = sorted((r for r in rows if r["domain"] == d and r["lang"] == l and r["capped"] == capped),
                            key=lambda r: r["turn"])
                if rs:
                    ax.plot([r["turn"] for r in rs], [r["mean_score"] for r in rs], color=COLORS.get(l),
                            linestyle=ls, linewidth=2, label=f"{LANG_NAMES.get(l, l)}, {'capped' if capped else 'uncapped'}")
        ax.set_title(d, color=INK, loc="left")
        ax.set_xlabel("Reply turn")
    axes[0].set_ylabel("Normalised score (1 = default Assistant, 0 = average role)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"{model}: capped (dashed) vs uncapped (solid)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reply_language(lturn, cond, path, plt):
    rows = [r for r in lturn if (r["model"], r["capped"]) == cond]
    domains = ordered({r["domain"] for r in rows}, DOMAINS)
    fig, axes = _panels(plt, domains)
    for ax, d in zip(axes, domains):
        for l in ordered({r["lang"] for r in rows if r["domain"] == d}, LANGS):
            rs = sorted((r for r in rows if r["domain"] == d and r["lang"] == l), key=lambda r: r["turn"])
            ax.plot([r["turn"] for r in rs], [r["share_user_lang"] for r in rs], color=COLORS.get(l),
                    linewidth=2, marker="o", markersize=3, label=LANG_NAMES.get(l, l))
            _end_label(ax, rs[-1]["turn"], rs[-1]["share_user_lang"], l, COLORS.get(l))
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(d, color=INK, loc="left")
        ax.set_xlabel("Reply turn")
    axes[0].set_ylabel("Share of replies in the user's language")
    axes[-1].legend(loc="lower left")
    fig.suptitle(f"{cond_name(cond)}: reply language (GlotLID)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- summary

def summary_md(meta, comp, dtable, gaps, base, lover, ttable, flags, charts):
    L = ["# Stage 2 summary", "",
         f"Generated {meta['created']} by `scripts/analyze.py`. Inputs: " + ", ".join(f"`{p}`" for p in meta["inputs"]) + ".",
         f"Bootstrap: {meta['n_boot']} resamples over conversations, seed {meta['seed']}. "
         f"Comparable turns: reached by >= {meta['min_convs']} conversations and by >= {meta.get('min_retention', 0):.0%} of each "
         f"language's conversations, in every language of the domain (unbroken from turn 1).",
         "Score: normalised projection (1 = default Assistant vector, 0 = mean role vector). "
         "Drift: score minus the same language's coding-domain mean.", ""]
    L += ["## Comparable turns", "", "| Condition | Domain | Turns |", "|---|---|---|"]
    for (c, d), ts in sorted(comp.items()):
        L.append(f"| {cond_name(c)} | {d} | {f'{ts[0]}-{ts[-1]} ({len(ts)})' if ts else 'none'} |")
    L += ["", "## Per-language coding baselines", "", "| Condition | Language | Coding mean score | Conversations |",
          "|---|---|---|---|"]
    for (c, l), b in sorted(base.items()):
        L.append(f"| {cond_name(c)} | {l} | {st.mean(b):.3f} | {len(b)} |")
    L += ["", "## Mean drift over comparable turns (95% CI)", "",
          "| Condition | Domain | Language | Conversations | Mean score | Drift |", "|---|---|---|---|---|---|"]
    for r in dtable:
        L.append(f"| {r['model']} {'capped' if r['capped'] else 'uncapped'} | {r['domain']} | {r['lang']} | "
                 f"{r['n_convs']} | {r['mean_score']:.3f} | {fmt(r['drift'])}{ci(r['drift_lo'], r['drift_hi'])} |")
    if gaps:
        L += ["", "## Drift gap vs English (language drift minus English drift, 95% CI)", "",
              "| Condition | Domain | Language | Gap |", "|---|---|---|---|"]
        for g in gaps:
            L.append(f"| {g['model']} {'capped' if g['capped'] else 'uncapped'} | {g['domain']} | {g['lang']} | "
                     f"{fmt(g['gap_vs_en'])}{ci(g['gap_lo'], g['gap_hi'])} |")
    if lover:
        L += ["", "## Reply language (all turns)", "",
              "| Condition | Language | Domain | Replies | In user's language | en / ha / yo / other |",
              "|---|---|---|---|---|---|"]
        for r in lover:
            L.append(f"| {r['model']} {'capped' if r['capped'] else 'uncapped'} | {r['lang']} | {r['domain']} | "
                     f"{r['n']} | {r['share_user_lang']:.1%} | {r['detected_en']} / {r['detected_ha']} / "
                     f"{r['detected_yo']} / {r['detected_other']} |")
    if ttable:
        L += ["", "## Token budget", "",
              "| Condition | Language | Domain | Mean reply tokens | Hit max_new_tokens | Context stops | Max prompt tokens |",
              "|---|---|---|---|---|---|---|"]
        for r in ttable:
            L.append(f"| {r['model']} {'capped' if r['capped'] else 'uncapped'} | {r['lang']} | {r['domain']} | "
                     f"{r['mean_reply_tokens']:.0f} | {r['share_hit_max_new_tokens']:.1%} | {r['context_stops']} | "
                     f"{r['max_prompt_tokens']} |")
    L += ["", "## Flags (check before interpreting)", ""] + ([f"- {f}" for f in flags] or ["- none"])
    if charts:
        L += ["", "## Charts", ""] + [f"- `{c}`" for c in charts]
    return "\n".join(L) + "\n"


def make_flags(base, lover, ttable, max_lang_baseline_gap=0.1, min_share=0.9):
    flags = []
    by_cond = defaultdict(dict)
    for (c, l), b in base.items():
        by_cond[c][l] = st.mean(b)
    for c, m in by_cond.items():
        if len(m) > 1 and max(m.values()) - min(m.values()) > max_lang_baseline_gap:
            flags.append(f"{cond_name(c)}: coding baselines differ across languages by "
                         f"{max(m.values()) - min(m.values()):.2f} ({', '.join(f'{k} {v:.2f}' for k, v in m.items())}); "
                         f"language alone shifts the projection, so compare drift, not raw scores.")
    for r in lover:
        if r["share_user_lang"] < min_share:
            flags.append(f"{r['model']} {'capped' if r['capped'] else 'uncapped'} {r['lang']}/{r['domain']}: only "
                         f"{r['share_user_lang']:.0%} of replies in the user's language.")
    for r in ttable:
        if r["context_stops"]:
            flags.append(f"{r['model']} {'capped' if r['capped'] else 'uncapped'} {r['lang']}/{r['domain']}: "
                         f"{r['context_stops']} conversations stopped at the context limit.")
        if r["share_hit_max_new_tokens"] and r["share_hit_max_new_tokens"] > 0.2:
            flags.append(f"{r['model']} {'capped' if r['capped'] else 'uncapped'} {r['lang']}/{r['domain']}: "
                         f"{r['share_hit_max_new_tokens']:.0%} of replies cut at max_new_tokens.")
    return flags


def analyze(proj_paths, langid_paths=(), replay_paths=(), out=RESULTS, min_convs=10, n_boot=2000, seed=0,
            plots=True, overwrite=False, prefix="stage2", min_retention=0.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for p in proj_paths for r in read_jsonl(p)]
    lrows = [r for p in langid_paths for r in read_jsonl(p)]
    replays = [r for p in replay_paths for r in read_jsonl(p)]
    rng = random.Random(seed)

    comp = comparable_turns(rows, min_convs, min_retention)
    cm = per_conv_means(rows, comp)
    base = baselines(cm)
    trows = turn_table(rows, comp, base, n_boot, rng)
    dtable = domain_table(cm, base, comp, n_boot, rng)
    gaps = gap_table(cm, base, n_boot, rng)
    lturn, lover = reply_language_tables(lrows, comp)
    ttable = token_table(replays)
    flags = make_flags(base, lover, ttable)

    files = {f"{prefix}_turn_scores.csv": trows, f"{prefix}_domain_drift.csv": dtable,
             f"{prefix}_gap_vs_en.csv": gaps, f"{prefix}_reply_language_by_turn.csv": lturn,
             f"{prefix}_reply_language.csv": lover, f"{prefix}_tokens.csv": ttable}
    conds = sorted({cond_of(r) for r in rows})
    charts = []
    if plots:
        charts += [f"{prefix}_drift_{c[0]}_{'capped' if c[1] else 'uncapped'}.png" for c in conds]
        charts += [f"{prefix}_capping_{m}.png" for m in sorted({c[0] for c in conds}) if {(m, True), (m, False)} <= set(conds)]
        charts += [f"{prefix}_reply_language_{c[0]}_{'capped' if c[1] else 'uncapped'}.png"
                   for c in sorted({cond_of(r) for r in lrows})]
    targets = [out / f for f, v in files.items() if v] + [out / f"{prefix}_summary.md"] + [out / c for c in charts]
    clash = [str(t) for t in targets if t.exists()]
    if clash and not overwrite:
        raise SystemExit("Refusing to overwrite existing results (pass --overwrite after checking):\n  " + "\n  ".join(clash))

    for f, v in files.items():
        write_csv(out / f, v)
    if plots and charts:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _style(plt)
        for c in conds:
            plot_drift(trows, c, out / f"{prefix}_drift_{c[0]}_{'capped' if c[1] else 'uncapped'}.png", plt)
        for m in sorted({c[0] for c in conds}):
            if {(m, True), (m, False)} <= set(conds):
                plot_capping(trows, m, out / f"{prefix}_capping_{m}.png", plt)
        for c in sorted({cond_of(r) for r in lrows}):
            plot_reply_language(lturn, c, out / f"{prefix}_reply_language_{c[0]}_{'capped' if c[1] else 'uncapped'}.png", plt)

    meta = {"created": datetime.now(timezone.utc).isoformat(), "n_boot": n_boot, "seed": seed, "min_convs": min_convs,
            "min_retention": min_retention,
            "inputs": [Path(p).name for p in (*proj_paths, *langid_paths, *replay_paths)]}
    models = sorted({(r["model"], r.get("model_revision"), r.get("seed")) for r in rows})
    md = summary_md(meta, comp, dtable, gaps, base, lover, ttable, flags, charts)
    md += "\n## Provenance\n\n" + "\n".join(f"- {m} revision {rev}, generation seed {s}" for m, rev, s in models)
    versions = {str(r.get("versions")) for r in rows}
    md += "\n" + "\n".join(f"- projection libraries: {v}" for v in sorted(versions)) + "\n"
    (out / f"{prefix}_summary.md").write_text(md, encoding="utf-8")
    return {"comparable": comp, "baselines": base, "turns": trows, "domains": dtable, "gaps": gaps,
            "reply_language_by_turn": lturn, "reply_language": lover, "tokens": ttable, "flags": flags,
            "files": [str(t) for t in targets]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proj", nargs="+", required=True)
    ap.add_argument("--langid", nargs="*", default=[])
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--min-convs", type=int, default=10)
    ap.add_argument("--min-retention", type=float, default=0.8,
                    help="a turn counts only if >= this share of each language's conversations reach it")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace existing stage2_* files in --out")
    args = ap.parse_args()
    res = analyze(args.proj, args.langid, args.replay, args.out, args.min_convs, args.n_boot, args.seed,
                  plots=not args.no_plots, overwrite=args.overwrite, min_retention=args.min_retention)
    print((Path(args.out) / "stage2_summary.md").read_text(encoding="utf-8"))
    print("Wrote:\n  " + "\n  ".join(res["files"]))


if __name__ == "__main__":
    main()
