"""Render evidence from completed or running first-round diagnostic arms."""
import argparse,csv,json,statistics
from pathlib import Path


def render(out):
    baseline_path=out/'baseline.json'
    if not baseline_path.exists():return
    base=json.loads(baseline_path.read_text());results={};histories={};flat=[]
    ended=(out/'watchdog-exit.json').exists() or (out/'completion.json').exists()
    for arm in 'ABCD':
        rp=out/arm/'result.json'
        if rp.exists():results[arm]=json.loads(rp.read_text())
        histories[arm]=[]
        for p in sorted((out/arm).glob('validation-*.json')):
            step=int(p.stem.split('-')[-1]);r=json.loads(p.read_text())
            histories[arm].append((step,r))
            for group,m in r['groups'].items():
                flat.append(dict(arm=arm,step=step,tokens=step*4096,group=group,**m))
    keys=['arm','step','tokens','group']+sorted(set().union(*(set(r)-{'arm','step','tokens','group'} for r in flat)))
    with (out/'validation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)
    def fmt(x):return '—' if x is None else f'{x:.6f}'
    lines=['# 33,291 checkpoint：6 小时第一轮诊断实验','',
           'A=Control；B=关闭 A alignment；C=B+0.01×[ReLU(CE2−CE1)+ReLU(CE3−CE2)]；D=C+R LR×1.5，仅实测更新量触发。', '',
           '各组相同母本、数据游标、seed、batch、cosine LR 进度及验证集。母本无 Adam 状态，所有组统一从空 Adam 开始。本轮磁盘容量只允许保存 FP32 权重和数据/RNG/调度信息，分支恢复需要重置 Adam。', '',
           'CE 是实际 D(delta_k) 文本路径；绝对状态 D(n_k) 不参与模型优劣排名。新增深度目标沿原 A→R→D 图反传，并非仅 R 参数接收该项梯度。','',
           '| 指标 | 母本 | A | B | C | D |','|---|---:|---:|---:|---:|---:|']
    plan=json.loads((out/'plan.json').read_text()) if (out/'plan.json').exists() else {}
    if plan.get('control_reused_from'):
        lines[2:2]=[f"本次因保留原截止时间，将各组对照终点缩为 {plan['tokens_per_arm']:,} tokens。A 复用此前已执行更新；B/C 从原始母本重新开始。这里的‘完成’仅指缩短后的等量对照，原定各 5M 方案仍未完成。",'']
    metrics=[('Fixed Val CE','fixed','ce_depth_3'),('Extra Val CE','extra','ce_depth_3')]+[(n,'all',k) for n,k in [('CE1','ce_depth_1'),('CE2','ce_depth_2'),('CE3','ce_depth_3'),('G12','depth_1_to_2_gain'),('G23','depth_2_to_3_gain'),('G13','depth_1_to_3_gain')]]
    metrics += [(f'Var n{k}','all',f'state_token_variance_{k}') for k in (1,2,3)]
    metrics += [(f'Var Δ{k}','all',f'delta_token_variance_{k}') for k in (1,2,3)]
    metrics += [('Zero Δ3 CE','all','zero_delta3_ce'),('Shuffle Δ3 CE','all','shuffle_delta3_ce'),('Shuffle n3 CE','all','shuffle_r3_state_ce')]
    latest={a:h[-1][1]['groups'] for a,h in histories.items() if h}
    for name,group,key in metrics:
        vals=[base['groups'][group].get(key)]+[latest.get(a,{}).get(group,{}).get(key) for a in 'ABCD']
        lines.append('| '+name+' | '+' | '.join(map(fmt,vals))+' |')
    for name,module in [('U_A','attention'),('U_R','core'),('U_D','decoder')]:
        vals=[]
        for a in 'ABCD':
            p=out/a/'steps.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
            u=[r['relative_update'][module] for r in rows if r.get('relative_update') and r['local_step']>=100]
            vals.append(fmt(statistics.median(u)) if u else '—')
        lines.append('| '+name+'（采样中位数） | — | '+' | '.join(vals)+' |')
    lines.append('| clip % | — | '+' | '.join(f"{100*results[a]['training_statistics']['clip_ratio']:.2f}%" if a in results else '—' for a in 'ABCD')+' |')
    lines.extend(['','## 完成情况与比较',''])
    for a in 'ABCD':
        if a in results:
            r=results[a];lines.append(f"- {a}：{'完成' if r['complete'] else '未完成'}；{r['finished_tokens']:,} tokens；训练 {r['train_seconds']/60:.1f} 分钟；分支总耗时 {r['elapsed_seconds']/60:.1f} 分钟；停止原因 {r['stop_reason']}。")
        elif histories[a]:lines.append(f"- {a}：{'已停止且未完成' if ended else '运行中'}，最新验证 {histories[a][-1][0]*4096:,} tokens。")
    gate=out/'D-gate.json'
    if gate.exists():lines.extend(['',f"D 触发判断：`{json.dumps(json.loads(gate.read_text()),ensure_ascii=False)}`"])
    candidates=[]
    for a,r in results.items():
        vals=[v for s,v in histories[a] if s>0]
        if not vals:continue
        final=vals[-1]['groups'];initial=base['groups'];recent=vals[-3:]
        depth= len(recent)>=3 and all(v['groups'][g]['depth_1_to_2_gain']>0 and v['groups'][g]['depth_2_to_3_gain']>0 for v in recent for g in ('fixed','extra'))
        improves=all(final[g]['ce_depth_3']<initial[g]['ce_depth_3'] for g in ('fixed','extra'))
        b=initial['all'];m=final['all'];ratio=(m['state_token_variance_3']/m['state_token_variance_1'])/(b['state_token_variance_3']/b['state_token_variance_1'])
        mean_g={k:statistics.mean(v['groups']['all'][k] for v in vals) for k in ('depth_1_to_2_gain','depth_2_to_3_gain','depth_1_to_3_gain')}
        lines.extend(['',f"{a}：相对母本 Fixed CE {final['fixed']['ce_depth_3']-initial['fixed']['ce_depth_3']:+.6f}；Extra CE {final['extra']['ce_depth_3']-initial['extra']['ce_depth_3']:+.6f}。最近 3 次验证在两组验证集均满足 CE3<CE2<CE1：{depth}。Var(n3)/Var(n1) 相对母本倍数：{ratio:.4f}。各验证点平均 G12/G23/G13：{mean_g['depth_1_to_2_gain']:.6f}/{mean_g['depth_2_to_3_gain']:.6f}/{mean_g['depth_1_to_3_gain']:.6f}。"])
        if r['complete'] and depth and improves and ratio>.1:candidates.append(a)
        if a!='A' and 'A' in results and r['complete'] and results['A']['complete'] and r['finished_tokens']==results['A']['finished_tokens']:
            control=histories['A'][-1][1]['groups']
            lines.append(f"相对等 tokens A：Fixed CE {final['fixed']['ce_depth_3']-control['fixed']['ce_depth_3']:+.6f}；Extra CE {final['extra']['ce_depth_3']-control['extra']['ce_depth_3']:+.6f}；G23 {m['depth_2_to_3_gain']-control['all']['depth_2_to_3_gain']:+.6f}。")
    lines.extend(['','满足本次预设的文本改善、连续深度收益与非严重方差退化筛选条件：'+(', '.join(candidates) if candidates else '暂时没有足够证据选出 winner')+'。','',
        '32 个验证批次可能来自关联文档，固定/扩展验证也不是新的独立大规模评测。Shuffle/均值干预可能混合未来 token，只用于诊断；差值不能直接解释为纯推理收益。本轮短程实验不自动启动主训练或扩展轮。'])
    cp=out/'completion.json'
    if cp.exists():lines.extend(['',f"执行状态：`{cp.read_text().strip()}`"])
    wp=out/'watchdog-exit.json'
    if wp.exists():lines.extend(['',f"监督器退出记录：`{wp.read_text().strip()}`"])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for a,h in histories.items():
            if not h:continue
            x=[s*4096/1e6 for s,r in h]
            for ax,group,key,label in [(axes[0,0],'fixed','ce_depth_3','Fixed CE3'),(axes[0,1],'extra','ce_depth_3','Extra CE3'),(axes[1,0],'all','depth_1_to_2_gain','G12'),(axes[1,1],'all','depth_2_to_3_gain','G23')]:
                ax.plot(x,[r['groups'][group][key] for s,r in h],marker='o',label=a);ax.set_title(label);ax.set_xlabel('Million tokens');ax.grid(alpha=.2)
        for ax in axes.flat:ax.legend()
        fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    except ImportError:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);render(p.parse_args().out)
