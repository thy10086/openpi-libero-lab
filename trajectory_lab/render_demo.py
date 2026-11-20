"""Render a local HTML report from the data audit, training and simulator artifacts."""

import argparse
from html import escape
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for(result: dict, label: str) -> str:
    rows = []
    for item in result["settings"]:
        status = ", ".join(f"{key}×{value}" for key, value in sorted(item["outcomes"].items()))
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{item['execute_steps']}</td>"
            f"<td>{item['successes']} / {item['episodes']}</td><td>{item['mean_inference_calls']}</td>"
            f"<td>{item['inference_p95_ms']}</td><td>{escape(status)}</td></tr>"
        )
    return "".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    root = parser.parse_args().run
    audit = load(root / "audit.json")
    train = load(root / "bc" / "training.json")
    learned = load(root / "closed_loop" / "result.json")
    zero = load(root / "zero" / "result.json")
    toy = load(root / "toy" / "result.json")
    figures = []
    for folder, result, label in (("closed_loop", learned, "行为克隆"), ("zero", zero, "零动作对照")):
        for video in result["videos"]:
            image = escape(f"{folder}/{video}", quote=True)
            name = escape(Path(video).stem)
            figures.append(
                f'<figure><img src="{image}" alt="{label} {name} 模拟器回放" loading="lazy">'
                f'<figcaption>{label} · {name}</figcaption></figure>'
            )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenPI 具身策略实验台</title>
<style>
:root {{ color-scheme: light; font-family: "Inter","Microsoft YaHei",system-ui,sans-serif; color:#142820; background:#f3f6ef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; }}
main {{ max-width:1050px; margin:auto; padding:44px 24px; }}
h1 {{ font-size:clamp(26px,5vw,47px); margin:0 0 14px; letter-spacing:-.035em; }}
h2 {{ margin:0 0 12px; font-size:22px; }}
p {{ line-height:1.75; margin:8px 0; }}
a {{ color:#246349; text-underline-offset:3px; }}
section {{ background:#fff; border:1px solid #d7e1d7; border-radius:14px; margin:18px 0; padding:22px; }}
.metrics,.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:14px; }}
.metrics div {{ padding:14px 12px; border:1px solid #e1eadd; border-radius:9px; background:#f6f9f4; }}
strong {{ display:block; font-size:24px; }}
.scroll {{ overflow:auto; }}
table {{ border-collapse:collapse; width:100%; text-align:left; font-size:14px; }}
th,td {{ padding:11px 9px; border-bottom:1px solid #e0e7e0; white-space:nowrap; }}
th {{ color:#315647; }}
figure {{ margin:0; border:1px solid #dae5da; border-radius:10px; overflow:hidden; }}
figure img {{ display:block; width:100%; aspect-ratio:1; object-fit:contain; background:#dfe8df; }}
figcaption {{ padding:10px; font-size:13px; line-height:1.5; }}
.note {{ color:#42534b; font-size:14px; }}
</style>
</head>
<body><main>
<h1>OpenPI 具身策略实验台</h1>
<p>从公开示范轨迹、动作窗口和行为克隆训练，到固定初态下的 LIBERO 闭环执行。</p>
<p class="note">任务为抓起位于餐盘与小碗之间的黑碗，并将它放入餐盘。以下结果来自 CPU 小样本实验。</p>
<section>
<h2>轨迹审计与离线训练</h2>
<div class="metrics">
<div><strong>{audit['episodes_seen']}</strong>公开示范轨迹</div>
<div><strong>{audit['rows']}</strong>动作和状态采样</div>
<div><strong>{train['validation_mae']:.4f}</strong>模型验证 MAE</div>
<div><strong>{train['mean_action_validation_mae']:.4f}</strong>均值动作验证 MAE</div>
</div>
<p class="note">按 episode 划分训练和验证。MAE 在保留的示范动作上计算，闭环成功率见下一节。</p>
<p><a href="audit.json">数据审计 JSON</a> · <a href="bc/training.json">逐轮训练记录</a></p>
</section>
<section>
<h2>LIBERO 闭环</h2>
<div class="scroll"><table><thead><tr><th>策略</th><th>每次执行步数 K</th><th>成功 / 总数</th><th>平均调用</th><th>推理 P95 ms</th><th>结局</th></tr></thead><tbody>
{rows_for(learned, "行为克隆")}{rows_for(zero, "零动作")}
</tbody></table></div>
<p class="note">固定任务、初态和仿真种子。行为克隆仅使用机器人自身状态。π₀.₅ 权重评测通过 OpenPI 服务单独运行。</p>
<p><a href="closed_loop/episodes.csv">行为克隆逐轮记录</a> · <a href="zero/episodes.csv">零动作逐轮记录</a></p>
</section>
<section>
<h2>模拟器回放</h2>
<div class="gallery">{''.join(figures)}</div>
</section>
<section>
<h2>动作片段长度的受控对照</h2>
<div class="scroll"><table><thead><tr><th>策略</th><th>每次执行步数 K</th><th>成功 / 总数</th><th>平均调用</th><th>推理 P95 ms</th><th>结局</th></tr></thead><tbody>
{rows_for(toy, "确定性规划器")}
</tbody></table></div>
<p class="note">本节使用一维模拟环境验证配对评测，并包含目标中途移动的场景。</p>
<p><a href="toy/episodes.csv">受控对照逐轮记录</a></p>
</section>
</main></body></html>
"""
    (root / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
