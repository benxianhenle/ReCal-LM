"""Human-readable progress and gate evidence; never promotes an unfinished stage."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/dynamic-depth-20260917'


def render():
    def read(name):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else None
    status=read('status.json') or {};g0=read('G0.json');g1=read('G1.json');g2=read('G2.json');g3=read('G3.json');done=read('completion.json')
    lines=['# 动态深度 Gate 实验（A 终点权重）','',
        '严格按 G0 → G1 → G2 → G3 → G4 推进。总上限 8 小时；零训练诊断最多 3 小时；Controller 最多 1 小时、3M tokens；联合训练只有 G3 通过才启动。','',
        f"当前阶段：{status.get('stage','准备')}；事件：{status.get('event',status.get('action','—'))}。",'',
        '主指标使用真实训练路径 D(Δk)，同时单独记录方案中的绝对状态 D(nk)。后者并非当前 Decoder 的训练输入，不混入主排名。',
        '全部 Fixed/Extra 缓存只用于验证；Controller 标签来自独立文档哈希划分的训练数据。','',
        '| Gate | 状态 |','|---|---|',
        f"| G0 基线 | {'通过' if g0 else '未完成'} |",f"| G1 深度扫描 | {'通过' if g1 and g1['gates']['g1_pass'] else '未完成或停止'} |",
        f"| G2 Oracle | {'通过' if g2 and g2['gates']['g2_pass'] else '未完成或停止'} |",
        f"| G3 Controller | {('通过' if g3['pass_gate'] else '未通过，停止') if g3 else ('运行中' if status.get('stage')=='G3' else '未启动')} |",
        '| G4 联合训练 | '+('G3 已通过，待执行' if g3 and g3['pass_gate'] else '尚未满足进入条件')+' |','']
    if g1:
        groups=g1['groups'];lines+=['## R1–R8 深度扫描','','| Depth | Fixed CE | Extra CE | Fixed 最优 token % | Extra 最优 token % |','|---:|---:|---:|---:|---:|']
        for k in range(8):lines.append(f"| {k+1} | {groups['fixed']['ce'][k]:.6f} | {groups['extra']['ce'][k]:.6f} | {groups['fixed']['best_depth_percent'][k]:.2f} | {groups['extra']['best_depth_percent'][k]:.2f} |")
        lines+=['','## Oracle 与消融','','| 指标 | Fixed | Extra |','|---|---:|---:|']
        for label,key in [('固定 R3 CE','fixed_r3_ce'),('Oracle CE','oracle_ce'),('Oracle Gain 对 R3','oracle_gain_r3'),('Oracle Gain 对最佳固定深度','oracle_gain_best_fixed'),('Oracle 平均深度','mean_optimal_depth')]:lines.append(f"| {label} | {groups['fixed'][key]:.6f} | {groups['extra'][key]:.6f} |")
        for depth in (2,3,4):
            for label,key in [('Zero Δ','delta_zero'),('Shuffle Δ','delta_shuffle')]:lines.append(f"| {label}{depth} CE | {groups['fixed']['ablations'][str(depth)][key]:.6f} | {groups['extra']['ablations'][str(depth)][key]:.6f} |")
        lines+=['','Oracle 读取真实下一个 token，仅用于离线诊断；不能在推理时使用。深度不同的 Decoder 有不同上下文，所以它是密集深度扫描的参考上限，不保证是改变上下文后的真实动态路径的上限。Shuffle 是可能混合未来位置的离线干预。','',
            '## 难度分桶','','| 难度分位 | Fixed 平均最优深度 | Extra 平均最优深度 |','|---|---:|---:|']
        for i in range(5):lines.append(f"| {i*20}–{(i+1)*20}% | {groups['fixed']['difficulty_buckets'][i]['mean_optimal_depth']:.4f} | {groups['extra']['difficulty_buckets'][i]['mean_optimal_depth']:.4f} |")
    label=read('oracle-label-analysis.json')
    if label:
        lines+=['','## 一步 Continue 标签的额外诊断','','即便用真实标签执行‘下一步改善就继续，否则停止’，缓存路径只能恢复全局 Oracle 收益的一部分。这反映一步标签与全局最优深度之间的差别；它不是所有可学习策略的严格上限。','', '| 验证集 | 一步标签 Oracle CE | Recovery | 平均深度 |','|---|---:|---:|---:|']
        for group in ('fixed','extra'):
            m=label[group];lines.append(f"| {group} | {m['greedy_label_oracle_ce']:.6f} | {m['greedy_label_oracle_recovery']:.2%} | {m['greedy_label_oracle_mean_depth']:.4f} |")
    checkpoints=sorted(OUT.glob('controller-validation-*.json'))
    if checkpoints:
        latest=json.loads(checkpoints[-1].read_text());tokens=int(checkpoints[-1].stem.split('-')[-1])
        lines+=['','## Controller 最新验证','',f'该验证点已训练 {tokens:,} tokens；阈值固定为 0.5。','',
            '| 指标 | Fixed | Extra |','|---|---:|---:|']
        for label,key in [('固定 R3 CE','fixed_r3_ce'),('缓存选择 CE','cached_ce'),('真实混合深度 CE','actual_ce'),('缓存 Recovery','cached_recovery'),('真实路径 Recovery','actual_recovery'),('真实平均停止深度','actual_average_depth'),('每批实际密集递归轮数','mean_dense_recurrence_calls')]:
            lines.append(f"| {label} | {latest['groups']['fixed'][key]:.6f} | {latest['groups']['extra'][key]:.6f} |")
        lines+=['','真实路径在 token 停止后冻结其 R 状态，最后对混合深度 Δ 解码。实现仍使用密集 GPU 核，平均停止深度较低不等同于已获得 GPU 加速。',
            '进入 G4 必须在 Fixed 和 Extra 上同时满足：真实与缓存 Recovery >50%、真实 CE 优于 R3、平均停止深度 ≤3。']
    if g3:lines+=['',f"G3 停止原因：{g3['stop_reason']}；Controller 训练 tokens：{g3['tokens']:,}；主模型冻结检查：{g3['frozen_backbone_unchanged']}。"]
    if done:lines+=['','## 最终执行状态','','```json',json.dumps(done,ensure_ascii=False,indent=2),'```']
    for name in ('failure.json','controller-failure.json'):
        fail=read(name)
        if fail:lines+=['',f'执行异常（{name}）：{fail}']
    lines+=['','验证仅包含固定 8 批、扩展 24 批，同组批次可能来自关联文档；本轮不把 token 当作完全独立样本来声称统计显著性。',
        '原始权重、上一轮 A/B/C 权重均保留。所有逐 token CE、状态/增量缓存、哈希和 Gate 判定均存于本目录。']
    temp=OUT/'README.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(OUT/'README.md')
    if g1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8));groups=g1['groups']
        for group in ('fixed','extra'):
            m=groups[group];axes[0,0].plot(range(1,9),m['ce'],marker='o',label=group)
            axes[0,1].plot(range(1,9),m['best_depth_percent'],marker='o',label=group)
            axes[1,0].plot([10,30,50,70,90],[b['mean_optimal_depth'] for b in m['difficulty_buckets']],marker='o',label=group)
            if checkpoints:
                history=[(int(p.stem.split('-')[-1]),json.loads(p.read_text())['groups'][group]) for p in checkpoints]
                axes[1,1].plot([t/1e6 for t,m in history],[m['actual_recovery']*100 for t,m in history],marker='o',label=group)
        for ax,title in zip(axes.flat,['CE by recurrent depth (delta decoding)','Per-token optimal depth (%)','Difficulty percentile vs optimal depth','Actual controller recovery (%)']):
            ax.set_title(title);ax.grid(alpha=.2)
            if ax.lines:ax.legend()
        axes[0,0].set_xlabel('Depth');axes[0,1].set_xlabel('Depth');axes[1,0].set_xlabel('Initial CE percentile');axes[1,1].set_xlabel('Controller training tokens (M)')
        fig.tight_layout();fig.savefig(OUT/'diagnostic-curves.png',dpi=160);fig.savefig(OUT/'diagnostic-curves.svg');plt.close(fig)


if __name__=='__main__':
    if '--watch' in sys.argv:
        deadline=json.loads((OUT/'budget.json').read_text())['overall_deadline']
        while time.time()<deadline+60:
            try:render()
            except Exception as exc:
                with (OUT/'report-errors.log').open('a') as f:f.write(repr(exc)+'\n')
            if any((OUT/n).exists() for n in ('completion.json','failure.json','controller-failure.json')):break
            time.sleep(30)
    else:render()
