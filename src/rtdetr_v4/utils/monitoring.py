"""Monitoring artifact generation for local training runs.

This module writes the JSON snapshots and self-contained HTML dashboard that the
old project used for semi-live inspection during training.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def load_metrics_history(metrics_log_path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited metric snapshots from ``metrics.jsonl``."""
    path = Path(metrics_log_path)
    if not path.exists():
        return []

    history: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                history.append(payload)
    return history


def write_monitoring_artifacts(
    *,
    output_dir: str | Path,
    metrics_history: list[dict[str, Any]],
    latest_eval: dict[str, Any] | None,
    focus_classes: list[str] | None = None,
    refresh_seconds: int = 15,
) -> Path:
    """Write JSON summaries plus the dashboard HTML into ``output_dir``.

    Files produced:
    - ``monitoring/history_summary.json``
    - ``monitoring/latest_eval.json``
    - ``monitoring/eval_epoch_XXX.json`` when an epoch is present
    - ``monitoring/dashboard.html``
    """
    monitor_dir = Path(output_dir) / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    history_summary = _build_history_summary(metrics_history)
    latest_eval_payload = _build_latest_eval_payload(latest_eval)
    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "refresh_seconds": max(int(refresh_seconds), 0),
        "focus_classes": list(focus_classes or []),
    }

    # These files are intentionally plain JSON so they can be inspected by
    # humans or consumed by later automation.
    _write_json(monitor_dir / "history_summary.json", history_summary)
    if latest_eval_payload is not None:
        _write_json(monitor_dir / "latest_eval.json", latest_eval_payload)
        epoch = latest_eval_payload.get("epoch")
        if isinstance(epoch, int):
            _write_json(monitor_dir / f"eval_epoch_{epoch:03d}.json", latest_eval_payload)

    dashboard_path = monitor_dir / "dashboard.html"
    dashboard_path.write_text(
        _build_dashboard_html(
            history_summary=history_summary,
            latest_eval=latest_eval_payload,
            metadata=metadata,
        ),
        encoding="utf-8",
    )
    return dashboard_path


def _build_history_summary(metrics_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse raw metric history into one clean record per epoch."""
    latest_by_epoch: dict[int, dict[str, Any]] = {}
    for entry in metrics_history:
        epoch_value = entry.get("epoch")
        if isinstance(epoch_value, bool):
            continue
        if isinstance(epoch_value, float) and not epoch_value.is_integer():
            continue
        if not isinstance(epoch_value, (int, float)):
            continue

        epoch = int(epoch_value)
        summary: dict[str, Any] = {"epoch": epoch}
        for key, value in entry.items():
            if key == "epoch" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                summary[key] = float(value)
        latest_by_epoch[epoch] = summary
    return [latest_by_epoch[epoch] for epoch in sorted(latest_by_epoch)]


def _build_latest_eval_payload(latest_eval: dict[str, Any] | None) -> dict[str, Any] | None:
    """Attach convenience summaries to the raw evaluation payload."""
    if latest_eval is None:
        return None
    payload = json.loads(json.dumps(latest_eval, ensure_ascii=False))
    confusion = payload.get("confusion_matrix")
    if isinstance(confusion, dict):
        payload["confusion_summary"] = _build_confusion_summary(confusion)
    return payload


def _build_confusion_summary(confusion_matrix: dict[str, Any], limit: int = 10) -> dict[str, Any]:
    """Extract the most useful confusion matrix slices for dashboard display."""
    labels = confusion_matrix.get("labels") or []
    matrix = confusion_matrix.get("matrix") or []
    if not labels or not matrix:
        return {"top_confusions": [], "top_missed_gt": [], "top_background_fp": []}

    background_index = len(labels) - 1
    top_confusions: list[dict[str, Any]] = []
    top_missed_gt: list[dict[str, Any]] = []
    top_background_fp: list[dict[str, Any]] = []

    for gt_index, row in enumerate(matrix):
        if gt_index >= background_index:
            continue
        if background_index < len(row) and int(row[background_index]) > 0:
            top_missed_gt.append({"label": labels[gt_index], "count": int(row[background_index])})
        for pred_index, value in enumerate(row):
            if pred_index >= background_index or pred_index == gt_index or int(value) <= 0:
                continue
            top_confusions.append(
                {
                    "gt_label": labels[gt_index],
                    "pred_label": labels[pred_index],
                    "count": int(value),
                }
            )

    if background_index < len(matrix):
        for pred_index, value in enumerate(matrix[background_index]):
            if pred_index >= background_index or int(value) <= 0:
                continue
            top_background_fp.append({"label": labels[pred_index], "count": int(value)})

    top_confusions.sort(key=lambda item: item["count"], reverse=True)
    top_missed_gt.sort(key=lambda item: item["count"], reverse=True)
    top_background_fp.sort(key=lambda item: item["count"], reverse=True)
    return {
        "top_confusions": top_confusions[:limit],
        "top_missed_gt": top_missed_gt[:limit],
        "top_background_fp": top_background_fp[:limit],
    }


def _write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with indentation for operator readability."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_for_script(payload: Any) -> str:
    """Serialize JSON safely for inline embedding inside the dashboard HTML."""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def _build_dashboard_html(
    *,
    history_summary: list[dict[str, Any]],
    latest_eval: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> str:
    """Render the self-contained dashboard page.

    The HTML intentionally has no external dependencies so it can be opened
    directly from disk during training.
    """
    refresh_tag = (
        f'<meta http-equiv="refresh" content="{int(metadata["refresh_seconds"])}">'
        if int(metadata["refresh_seconds"]) > 0
        else ""
    )
    template = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">__REFRESH__<title>RT-DETR v4 Dashboard</title><style>
body{margin:0;font:14px/1.45 "Segoe UI","PingFang SC",sans-serif;background:#f5efe6;color:#1f2933}
.wrap{max-width:1450px;margin:0 auto;padding:24px}.hero{display:flex;justify-content:space-between;gap:16px;align-items:flex-end}.badge{border:1px solid #d8cfbf;background:#fffdfa;padding:8px 12px;border-radius:999px;color:#6b7280}.cards,.grid{display:grid;gap:14px}.cards{grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin:18px 0}.grid{grid-template-columns:repeat(12,1fr)}.panel,.card{background:#fffdfa;border:1px solid #d8cfbf;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.05)}.card{padding:14px}.label{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.08em}.value{font-size:28px;font-weight:700;margin-top:8px}.detail,.meta,.note{color:#6b7280}.panel{padding:14px}.s6{grid-column:span 6}.s4{grid-column:span 4}.s8{grid-column:span 8}.s12{grid-column:span 12}canvas{width:100%;height:auto;border:1px solid #ece3d6;border-radius:12px;background:#fff}.tags{display:flex;flex-wrap:wrap;gap:8px}.tag{padding:5px 8px;border:1px solid #d8cfbf;border-radius:10px;background:#fff;font-size:12px;color:#6b7280}table{width:100%;border-collapse:collapse}.tablewrap{max-height:560px;overflow:auto;border:1px solid #ece3d6;border-radius:12px;background:#fff}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #ece3d6;vertical-align:top}th{font-size:12px;color:#6b7280;text-transform:uppercase}.empty{min-height:120px;display:flex;align-items:center;justify-content:center;color:#6b7280;border:1px dashed #d8cfbf;border-radius:12px}@media(max-width:1100px){.hero{display:block}.s4,.s6,.s8{grid-column:span 12}}</style></head><body><div class="wrap">
<div class="hero"><div><h1 style="margin:0 0 6px">RT-DETR v4 Monitoring</h1><div class="meta">Semi-live local dashboard for training history and validation diagnostics.</div></div><div class="badge" id="stamp"></div></div>
<div class="cards" id="cards"></div>
<div class="grid">
<div class="panel s6"><h2>Loss History</h2><div class="meta">train_loss vs val_loss</div><canvas id="loss" width="920" height="320"></canvas></div>
<div class="panel s6"><h2>Metric History</h2><div class="meta">map50, best_f1, lr</div><canvas id="metric" width="920" height="320"></canvas></div>
<div class="panel s4"><h2>P-Curve</h2><div class="meta">precision vs confidence</div><canvas id="pcurve" width="620" height="280"></canvas></div>
<div class="panel s4"><h2>R-Curve</h2><div class="meta">recall vs confidence</div><canvas id="rcurve" width="620" height="280"></canvas></div>
<div class="panel s4"><h2>F1-Curve</h2><div class="meta">f1 vs confidence</div><canvas id="f1curve" width="620" height="280"></canvas></div>
<div class="panel s8"><h2>Confusion Matrix</h2><div class="meta" id="cmmeta">No evaluation snapshot yet</div><canvas id="cm" width="1100" height="920"></canvas><div class="tags" id="labels"></div></div>
<div class="panel s4"><h2>Evaluation Summary</h2><div id="summary"></div></div>
<div class="panel s6"><h2>Recent Epochs</h2><div id="recent"></div></div>
<div class="panel s6"><h2>Files</h2><div class="tags"><span class="tag">monitoring/dashboard.html</span><span class="tag">monitoring/history_summary.json</span><span class="tag">monitoring/latest_eval.json</span><span class="tag">metrics.jsonl</span></div><div class="note" style="margin-top:10px">This page is regenerated after each epoch. Leave it open in a browser and use auto-refresh for semi-live monitoring.</div></div>
<div class="panel s12"><h2>Per-Class Metrics</h2><div class="meta">AP50 and Recall50 at the best-F1 confidence threshold</div><div id="classmetrics"></div></div>
</div></div>
<script id="history" type="application/json">__HISTORY__</script><script id="eval" type="application/json">__EVAL__</script><script id="meta" type="application/json">__META__</script><script>
const history=JSON.parse(document.getElementById("history").textContent),latestEval=JSON.parse(document.getElementById("eval").textContent),meta=JSON.parse(document.getElementById("meta").textContent);
const C={a:"#c2410c",b:"#0f766e",c:"#a16207",d:"#b91c1c",grid:"#e9dfd1",axis:"#7b7280",ink:"#1f2933"};
const fmt=(v,d=4)=>typeof v==="number"&&Number.isFinite(v)?v.toFixed(d):"n/a";
const esc=(v)=>String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
document.getElementById("stamp").textContent=(meta.refresh_seconds>0?`Auto refresh ${meta.refresh_seconds}s`:"Auto refresh off")+` | ${meta.generated_at}`;
const last=history.length?history[history.length-1]:null,best=history.filter(x=>typeof x.map50==="number").reduce((p,x)=>!p||x.map50>=p.map50?x:p,null),curve=latestEval.confidence_curves||null;
const recallMap=latestEval.per_class_recall50||{},apMap=latestEval.per_class_ap50||{},precisionMap=latestEval.per_class_precision50||{},countMap=latestEval.per_class_counts||{},focusClasses=Array.isArray(meta.focus_classes)?meta.focus_classes:[];
const classMetricRows=Array.from(new Set([...Object.keys(apMap),...Object.keys(recallMap)])).map(name=>({name,ap50:apMap[name],recall50:recallMap[name],precision50:precisionMap[name],counts:countMap[name]||{}}));
const focusMetricRows=focusClasses.map(name=>classMetricRows.find(row=>row.name===name)).filter(Boolean);
document.getElementById("cards").innerHTML=[
["Latest Epoch",last?last.epoch:"n/a",last?`train_loss ${fmt(last.train_loss)}`:"No history yet"],
["Latest map50",last&&typeof last.map50==="number"?fmt(last.map50):"n/a",best?`best epoch ${best.epoch}`:"No val metric yet"],
["Best map50",best?fmt(best.map50):"n/a",best?`epoch ${best.epoch}`:"No val metric yet"],
["Latest val_loss",last&&typeof last.val_loss==="number"?fmt(last.val_loss):"n/a",last&&typeof last.lr==="number"?`lr ${fmt(last.lr,6)}`:"No lr yet"],
["Best F1",curve?fmt(curve.best_f1):"n/a",curve?`conf ${fmt(curve.best_threshold,3)}`:"Run validation first"],
["Precision / Recall",curve?fmt(curve.precision_at_best_f1):"n/a",curve?`recall ${fmt(curve.recall_at_best_f1)}`:"Run validation first"]
].map(([l,v,d])=>`<div class="card"><div class="label">${esc(l)}</div><div class="value">${esc(v)}</div><div class="detail">${esc(d)}</div></div>`).join("");
function chart(id,xs,series,opt={}){const c=document.getElementById(id),ctx=c.getContext("2d"),w=c.width,h=c.height,p={l:52,r:18,t:22,b:36};ctx.clearRect(0,0,w,h);const vals=[];for(const s of series)for(const v of s.data)if(Number.isFinite(v))vals.push(v);if(!xs.length||!vals.length){ctx.fillStyle=C.axis;ctx.font="16px Segoe UI";ctx.fillText("No data yet",24,40);return;}let min=opt.minY??Math.min(...vals),max=opt.maxY??Math.max(...vals);if(opt.zero)min=Math.min(0,min);if(min===max){const d=min===0?1:Math.abs(min)*.1;min-=d;max+=d;}const xr=(Math.max(...xs)-Math.min(...xs))||1,yr=(max-min)||1,px=w-p.l-p.r,py=h-p.t-p.b,toX=v=>p.l+((v-Math.min(...xs))/xr)*px,toY=v=>p.t+py-((v-min)/yr)*py;ctx.strokeStyle=C.grid;for(let i=0;i<5;i++){const y=p.t+(py/4)*i;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();}ctx.strokeStyle=C.axis;ctx.beginPath();ctx.moveTo(p.l,p.t);ctx.lineTo(p.l,h-p.b);ctx.lineTo(w-p.r,h-p.b);ctx.stroke();ctx.fillStyle=C.axis;ctx.font="12px Segoe UI";for(let i=0;i<5;i++){const y=p.t+(py/4)*i;ctx.fillText(fmt(min+(yr/4)*(4-i),min<1?3:2),8,y+4);}const ticks=Math.min(6,xs.length);for(let i=0;i<ticks;i++){const idx=Math.round((i/Math.max(ticks-1,1))*(xs.length-1));ctx.fillText(String(xs[idx]),toX(xs[idx])-8,h-12);}let off=0;for(const s of series){ctx.fillStyle=s.color;ctx.fillRect(p.l+off,8,14,6);ctx.fillStyle=C.ink;ctx.fillText(s.label,p.l+off+20,16);off+=110;ctx.strokeStyle=s.color;ctx.lineWidth=2.2;ctx.beginPath();let started=false;for(let i=0;i<s.data.length;i++){const yv=s.data[i];if(!Number.isFinite(yv))continue;const x=toX(xs[i]),y=toY(yv);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);}ctx.stroke();}}
function heatmap(id,payload){const c=document.getElementById(id),ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);if(!payload||!payload.normalized||!payload.normalized.length){ctx.fillStyle=C.axis;ctx.font="16px Segoe UI";ctx.fillText("No confusion matrix yet",24,40);return;}const labels=payload.labels||[],m=payload.normalized,n=m.length,p={l:170,r:24,t:120,b:150},pw=w-p.l-p.r,ph=h-p.t-p.b,cw=pw/n,ch=ph/n;const col=t=>`rgb(${Math.round(255-(255-156)*t)},${Math.round(247-(247-52)*t)},${Math.round(235-(235-18)*t)})`;ctx.strokeStyle="#efe5d8";for(let r=0;r<n;r++)for(let k=0;k<n;k++){const x=p.l+k*cw,y=p.t+r*ch,v=Number(m[r][k])||0;ctx.fillStyle=col(Math.max(0,Math.min(1,v)));ctx.fillRect(x,y,cw,ch);ctx.strokeRect(x,y,cw,ch);}ctx.fillStyle=C.ink;ctx.font="14px Segoe UI";ctx.fillText("Predicted class",p.l,34);ctx.save();ctx.translate(34,p.t+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText("Ground truth class",0,0);ctx.restore();const step=Math.max(1,Math.ceil(n/18));ctx.font="10px Segoe UI";ctx.fillStyle=C.axis;for(let i=0;i<n;i+=step){const x=p.l+i*cw+cw/2,y=p.t+i*ch+ch/2;ctx.save();ctx.translate(x,p.t-8);ctx.rotate(-Math.PI/4);ctx.fillText(`${i}:${labels[i]}`,0,0);ctx.restore();ctx.fillText(`${i}:${labels[i]}`,8,y+3);}}
const epochs=history.map(x=>x.epoch);chart("loss",epochs,[{label:"train_loss",color:C.a,data:history.map(x=>x.train_loss)},{label:"val_loss",color:C.b,data:history.map(x=>x.val_loss)}],{zero:true});chart("metric",epochs,[{label:"map50",color:C.b,data:history.map(x=>x.map50)},{label:"best_f1",color:C.c,data:history.map(x=>x.best_f1)},{label:"lr",color:C.d,data:history.map(x=>x.lr)}],{zero:true});
const th=curve?curve.thresholds:[];chart("pcurve",th,[{label:"precision",color:C.a,data:curve?curve.precision:[]}],{minY:0,maxY:1});chart("rcurve",th,[{label:"recall",color:C.b,data:curve?curve.recall:[]}],{minY:0,maxY:1});chart("f1curve",th,[{label:"f1",color:C.c,data:curve?curve.f1:[]}],{minY:0,maxY:1});heatmap("cm",latestEval.confusion_matrix||null);
if(latestEval.confusion_matrix){const cm=latestEval.confusion_matrix;document.getElementById("cmmeta").textContent=`IoU ${fmt(cm.iou_threshold,2)} | conf ${fmt(cm.conf_threshold,3)} | labels ${cm.labels.length}`;document.getElementById("labels").innerHTML=(cm.labels||[]).map((x,i)=>`<span class="tag">${i}: ${esc(x)}</span>`).join("");}
const box=[];if(latestEval.map50!==undefined){box.push(`<div class="note">map50=${fmt(latestEval.map50)} | val_loss=${fmt(latestEval.loss)} | eval_images=${fmt(latestEval.num_eval_images,0)}</div>`);if(focusMetricRows.length){box.push("<div class='note' style='margin-top:10px'><strong>Focus Classes</strong></div><table><thead><tr><th>Class</th><th>Recall50</th><th>AP50</th></tr></thead><tbody>");for(const row of focusMetricRows)box.push(`<tr><td>${esc(row.name)}</td><td>${fmt(row.recall50)}</td><td>${fmt(row.ap50)}</td></tr>`);box.push("</tbody></table>");}const worstRecall=classMetricRows.filter(row=>typeof row.recall50==="number").sort((a,b)=>a.recall50-b.recall50).slice(0,8);if(worstRecall.length){box.push("<div class='note' style='margin-top:10px'><strong>Lowest Recall Classes</strong></div><table><thead><tr><th>Class</th><th>Recall50</th><th>GT</th></tr></thead><tbody>");for(const row of worstRecall)box.push(`<tr><td>${esc(row.name)}</td><td>${fmt(row.recall50)}</td><td>${esc(row.counts.gt??"n/a")}</td></tr>`);box.push("</tbody></table>");}const s=latestEval.confusion_summary||{};for(const [title,items,fn] of [["Top Confusions",s.top_confusions||[],x=>`${esc(x.gt_label)} -> ${esc(x.pred_label)} (${x.count})`],["Top Missed GT",s.top_missed_gt||[],x=>`${esc(x.label)} (${x.count})`],["Top Background FP",s.top_background_fp||[],x=>`${esc(x.label)} (${x.count})`]]){box.push(`<div class="note" style="margin-top:10px"><strong>${title}</strong></div>`);box.push(items.length?`<div class="tags">${items.map(fn).map(x=>`<span class="tag">${x}</span>`).join("")}</div>`:`<div class="note">none</div>`);}}else box.push('<div class="empty">Validation has not produced a snapshot yet.</div>');document.getElementById("summary").innerHTML=box.join("");
const recent=history.slice(-8).reverse();document.getElementById("recent").innerHTML=recent.length?`<table><thead><tr><th>Epoch</th><th>train_loss</th><th>val_loss</th><th>map50</th><th>best_f1</th></tr></thead><tbody>${recent.map(x=>`<tr><td>${x.epoch}</td><td>${fmt(x.train_loss)}</td><td>${fmt(x.val_loss)}</td><td>${fmt(x.map50)}</td><td>${fmt(x.best_f1)}</td></tr>`).join("")}</tbody></table>`:'<div class="empty">No epoch data yet.</div>';
const allClassRows=classMetricRows.sort((a,b)=>{const ar=typeof a.recall50==="number"?a.recall50:999,br=typeof b.recall50==="number"?b.recall50:999;if(ar!==br)return ar-br;const aa=typeof a.ap50==="number"?a.ap50:999,ba=typeof b.ap50==="number"?b.ap50:999;return aa-ba;});document.getElementById("classmetrics").innerHTML=allClassRows.length?`<div class="tablewrap"><table><thead><tr><th>Class</th><th>Recall50</th><th>AP50</th><th>Precision50</th><th>GT</th><th>TP</th><th>FP</th></tr></thead><tbody>${allClassRows.map(row=>`<tr><td>${esc(row.name)}</td><td>${fmt(row.recall50)}</td><td>${fmt(row.ap50)}</td><td>${fmt(row.precision50)}</td><td>${esc(row.counts.gt??"n/a")}</td><td>${esc(row.counts.tp??"n/a")}</td><td>${esc(row.counts.fp??"n/a")}</td></tr>`).join("")}</tbody></table></div>`:'<div class="empty">No per-class validation metrics yet.</div>';
</script></body></html>"""
    return (
        template.replace("__REFRESH__", refresh_tag)
        .replace("__HISTORY__", _json_for_script(history_summary))
        .replace("__EVAL__", _json_for_script(latest_eval or {}))
        .replace("__META__", _json_for_script(metadata))
    )
