"""Render V2 diagnostic artifacts without further model execution or training."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/dynamic-r-v2-20260917'
OLD = ROOT / 'reports/dynamic-depth-20260917'


def main():
    v20 = json.loads((OUT / 'V2.0.json').read_text())
    v21 = json.loads((OUT / 'V2.1.json').read_text())
    completion = json.loads((OUT / 'completion.json').read_text())
    budget = json.loads((OUT / 'budget.json').read_text())
    launcher = json.loads((OUT / 'launcher.json').read_text())
    search = [json.loads(line) for line in (OUT / 'calibration-search.jsonl').read_text().splitlines()]
    assert len(search) == 147
    assert all(sum(r['strategy'] == name for r in search) == 49 for name in ('entropy', 'confidence', 'margin'))
    assert not v21['gate']['pass_gate'] and v21['frozen_backbone_unchanged']
    assert completion['training_tokens'] == 0
    rows = []
    for name, result in v21['scoreboard'].items():
        rows.append(dict(strategy=name, status='evaluated', fixed_ce=result['fixed']['ce'], extra_ce=result['extra']['ce'],
                         average_depth=result['all']['average_depth'], **{f'k{k+1}_percent': p for k, p in enumerate(result['all']['depth_percent'])}))
    for name in ('Value V2', 'Value+R4'):
        rows.append(dict(strategy=name, status='not_run_gate_failed'))
    with (OLD / 'per-token-ce.csv').open() as f:
        cache = [r for r in csv.DictReader(f) if r['valid'] == 'True']
    oracle = {}
    for maximum in (3, 8):
        oracle[maximum] = {}
        for group in ('fixed', 'extra', 'all'):
            subset = [r for r in cache if group == 'all' or r['split'] == group]
            values = [[float(r[f'ce_d{k}']) for k in range(1, maximum+1)] for r in subset]
            depths = [min(range(maximum), key=lambda k: v[k]) + 1 for v in values]
            oracle[maximum][group] = dict(tokens=len(values), ce=sum(min(v) for v in values)/len(values),
                average_depth=sum(depths)/len(depths), depth_percent=[100*depths.count(k)/len(depths) for k in range(1, maximum+1)])
        r = oracle[maximum]
        rows.append(dict(strategy=f'Cached Oracle R1..R{maximum}', status='retrospective_not_executable',
            fixed_ce=r['fixed']['ce'], extra_ce=r['extra']['ce'], average_depth=r['all']['average_depth'],
            **{f'k{k+1}_percent': p for k, p in enumerate(r['all']['depth_percent'][:4])}))
        rows[-1].setdefault('k4_percent', 0.)
    with (OUT / 'scoreboard.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['strategy','status','fixed_ce','extra_ce','average_depth','k1_percent','k2_percent','k3_percent','k4_percent'])
        writer.writeheader(); writer.writerows(rows)
    (OUT / 'cached-oracle-reference.json').write_text(json.dumps(oracle, indent=2)+'\n')
    table = ['| Strategy | Fixed CE | Extra CE | Avg Depth | K1% | K2% | K3% | K4% |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status'] == 'not_run_gate_failed':
            table.append(f"| {r['strategy']} | 未执行：V2.1 未通过 | — | — | — | — | — | — |")
        else:
            table.append('| '+r['strategy']+' | '+' | '.join(f"{r[k]:.4f}" for k in ['fixed_ce','extra_ce','average_depth','k1_percent','k2_percent','k3_percent','k4_percent'])+' |')
    lines = ['# Dynamic-R V2 实验结果', '',
        'V2.0 已完成；V2.1 **FAIL / STOP**。三种启发式在 Fixed 和 Extra 上的真实混合路径 CE 均明显高于 Fixed R3。按方案停止，不执行 V2.2、V2.3、V2.4。新增训练 token 为 **0**，主模型全程冻结且未改变。', '',
        *table, '',
        'Avg Depth 和 K 分布按 Fixed 4,096 + Extra 12,288 个有效 token 加权。Cached Oracle 使用真实目标事后挑选，是缓存参考，不可作为可执行策略或 Gate 依据；R1..R8 超出本轮 K≤3 范围，其 K1–K4 百分比不合计 100%。Value 两行未训练、未评估。', '',
        '## V2.1 方法与停止依据', '',
        '沿用 A checkpoint 的真实解码路径 D(Δ)，固定 R1/R2/R3 已分别复现上一轮 CE（误差阈值 1e-5）。停止 token 的状态保持冻结，活跃 token 在混合上下文中继续 R，最终对混合 Δ 张量运行 Decoder。路由函数不接收目标 token。', '',
        '独立 calibration 使用 8 篇不同文档、4,096 token；文档 SHA256 桶分区为 validation=0、calibration=1..9、V2 train=10..999。三种策略各搜索 7×7=49 组阈值，共 147 组，均按真实 calibration CE 选择；全部阈值封存后，才评估策略的 Fixed / Extra 结果。', '',
        '| 策略 | q1 / q2 | τ1 / τ2 | ΔCE Fixed | ΔCE Extra |', '|---|---:|---:|---:|---:|']
    baseline = v21['scoreboard']['Fixed R3']
    for name, record in v21['thresholds'].items():
        r = v21['scoreboard'][name]
        lines.append(f"| {name} | {record['quantiles'][0]:.1f} / {record['quantiles'][1]:.1f} | {record['thresholds'][0]:.8g} / {record['thresholds'][1]:.8g} | {r['fixed']['ce']-baseline['fixed']['ce']:+.6f} | {r['extra']['ce']-baseline['extra']['ce']:+.6f} |")
    lines += ['', '三种策略的每个验证 batch 都实际调用了 3 次完整 R。平均停止深度下降不等于 GPU 加速，本轮不宣称计算加速。结果说明当前实现和这组校准策略未达到 Gate，不证明所有动态架构均不可行。Fixed R2 的两组 CE 均略低于 R3，但它属于固定深度基线，不满足本轮动态启发式 Gate。', '',
        '## V2.0：R3→R4 尾部分析', '', 'g34 = CE3 − CE4，正值表示 R4 收益。', '',
        '| 指标 | Fixed | Extra | 合并 |', '|---|---:|---:|---:|']
    for metric in v20['groups']['all']['probabilities']:
        lines.append('| P('+metric+') | '+' | '.join(f"{100*v20['groups'][g]['probabilities'][metric]:.3f}%" for g in ('fixed','extra','all'))+' |')
    for metric in v20['groups']['all']['quantiles']:
        lines.append('| '+metric+' | '+' | '.join(f"{v20['groups'][g]['quantiles'][metric]:.4f}" for g in ('fixed','extra','all'))+' |')
    lines += ['', '合并数据中，17.49% token 从 R4 获益，但 65.00% token 的 CE 恶化超过 1；平均 g34=-2.4594。收益组平均收益 0.8337，损害组平均 g34=-3.1576，损害占主导。', '',
        '| 特征均值（合并） | 收益组 | 损害组 |', '|---|---:|---:|']
    for key, val in v20['groups']['all']['groups']['benefit']['means'].items():
        lines.append(f"| {key} | {val:.7g} | {v20['groups']['all']['groups']['damage']['means'][key]:.7g} |")
    lines += ['', '较小的 Δ3 范数与 R4 收益相关：以 −||Δ3|| 预测收益的描述性 AUC 为 Fixed 0.7169、Extra 0.7046；entropy1 的 AUC 为 0.6285 / 0.5742。该信号未经独立校准或真实 R4 路径验证，不能据此开放 R4。CE1 和收益标签只作离线分析，不能输入推理路由。局部方差仅使用过去至多 8 个位置。', '',
        '| 输入 token 类型 | 收益组比例 | 损害组比例 |', '|---|---:|---:|']
    cohorts = v20['groups']['all']['groups']
    for kind in sorted(set(cohorts['benefit']['input_token_types']) | set(cohorts['damage']['input_token_types'])):
        lines.append('| '+kind+' | '+' | '.join(f"{100*cohorts[g]['input_token_types'].get(kind,0)/cohorts[g]['tokens']:.3f}%" for g in ('benefit','damage'))+' |')
    finish = completion['finished_at']
    lines += ['', '## 时间、验证与产物', '',
        f"实验进程运行 {(finish-launcher['started_at'])/60:.2f} 分钟；从预算记录开始至实验完成 {(finish-budget['started_at'])/60:.2f} 分钟（含准备）。V2.0 {v20['elapsed_seconds']:.1f} 秒，V2.1 {v21['elapsed_seconds']:.1f} 秒；均在阶段及 8 小时总预算内。训练 token=0。完成时间：{datetime.fromtimestamp(finish,timezone.utc).isoformat()}。",
        '', '原有 4 项 V2 单元测试已通过，覆盖 future-value 标签、概率特征、固定路径复现与因果性、文档分区和双验证集 Gate。实模型固定路径复现及参数冻结断言也通过。监督进程退出码为 0，已确认实验进程退出，GPU 无计算进程。', '',
        '原始证据：[V2.0](V2.0.json)、[V2.1](V2.1.json)、[scoreboard.csv](scoreboard.csv)、[阈值封存](frozen-thresholds.json)、[147 组搜索记录](calibration-search.jsonl)、[逐 token 尾部数据](r4-tail-per-token.csv)、[来源及哈希](manifest.json)、[预算](budget.json)、[完成记录](completion.json)。', '',
        '结论范围：8 篇 calibration 文档与现有 Fixed/Extra 验证规模有限，且 token 非独立样本。没有对架构失败原因作因果识别；按方案下一方向为研究混合状态的上下文兼容性，本轮到此停止。', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    print(json.dumps({'report':str(OUT/'README.md'), 'search_candidates':len(search), 'gate':v21['gate'], 'training_tokens':0},ensure_ascii=False))

if __name__ == '__main__':
    main()
