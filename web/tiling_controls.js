import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CONTROL_NAMES = new Set([
    "highres_tiling", "highres_tiling_mode", "tiling_mode", "tiling_tiles", "tiling_axis",
    "tiling_overlap", "spatial_anchor", "highres_token_chunk", "highres_weight_cache_gb",
]);
const SPATIAL_CONTROLS = new Set([
    "tiling_mode", "tiling_tiles", "tiling_axis", "tiling_overlap", "spatial_anchor",
]);
const GLOBAL_CONTROLS = new Set(["highres_token_chunk", "highres_weight_cache_gb"]);
const MODE_LABELS = {
    spatial: "空间分块（兼容模式）",
    global_stream: "全局流（推荐）",
    global: "全局（兼容模式）",
    global_ram: "全局内存（兼容模式）",
};
const originalVisibility = new WeakMap();

function seamColor(value) {
    if (value <= 0.10) return "#38c172";
    if (value <= 0.25) return "#f0ad4e";
    return "#e74c3c";
}

function clearSeamDiagram(node) {
    const canvas = node.selfliftSeamCanvas;
    const ctx = canvas?.getContext?.("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#888";
    ctx.font = "12px monospace";
    ctx.fillText("接缝诊断：运行测试节点后显示。", 10, 36);
}

function drawSeamDiagram(node, diag) {
    const canvas = node.selfliftSeamCanvas;
    const ctx = canvas?.getContext?.("2d");
    if (!ctx) return;
    const seams = diag?.seams || [];
    if (!seams.length) {
        clearSeamDiagram(node);
        return;
    }
    const axis = diag.axis === "H" ? "H" : "W";
    const length = Number(diag.length) || Math.max(
        ...(diag.regions || []).flatMap((r) => [r[0], r[1]]), 1);
    const maxValue = Math.max(...seams.map((s) => Number(s.relative_diff) || 0));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "11px monospace";
    ctx.fillStyle = "#bbb";
    ctx.fillText(`接缝诊断｜轴=${axis}｜最大差异=${maxValue.toFixed(4)}`, 10, 14);

    const regions = diag.regions || [];
    if (axis === "W") {
        const x0 = 10, y0 = 24, width = canvas.width - 20, height = 18;
        for (let i = 0; i < regions.length; i++) {
            const [start, end] = regions[i];
            const x = x0 + (start / length) * width;
            const w = Math.max(1, ((end - start) / length) * width);
            ctx.fillStyle = i % 2 ? "#30353c" : "#28323d";
            ctx.fillRect(x, y0, w, height);
        }
        for (const seam of seams) {
            const x = x0 + ((Number(seam.cut) || 0) / length) * width;
            ctx.strokeStyle = seamColor(Number(seam.relative_diff) || 0);
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, y0 - 4);
            ctx.lineTo(x, y0 + height + 4);
            ctx.stroke();
        }
        ctx.fillStyle = "#777";
        ctx.fillText("绿色 < .10   黄色 < .25   红色 >= .25", 10, 57);
    } else {
        const x0 = 12, y0 = 22, width = 18, height = canvas.height - 34;
        for (let i = 0; i < regions.length; i++) {
            const [start, end] = regions[i];
            const y = y0 + (start / length) * height;
            const h = Math.max(1, ((end - start) / length) * height);
            ctx.fillStyle = i % 2 ? "#30353c" : "#28323d";
            ctx.fillRect(x0, y, width, h);
        }
        for (const seam of seams) {
            const y = y0 + ((Number(seam.cut) || 0) / length) * height;
            ctx.strokeStyle = seamColor(Number(seam.relative_diff) || 0);
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x0 - 4, y);
            ctx.lineTo(x0 + width + 4, y);
            ctx.stroke();
        }
        ctx.fillStyle = "#777";
        ctx.fillText("绿色 < .10   黄色 < .25   红色 >= .25", 42, 38);
    }
}

function read(node, name, fallback) {
    return node.widgets?.find((w) => w.name === name)?.value ?? fallback;
}

function memoryEnabled(node) {
    return Boolean(read(node, "highres_tiling", false));
}

function memoryMode(node) {
    return read(node, "highres_tiling_mode", "spatial");
}

function choices(node) {
    if (!memoryEnabled(node)) return "设置：关闭（低噪声阶段省显存关闭）";
    const m = memoryMode(node);
    if (m === "spatial") {
        const mode = read(node, "tiling_mode", "auto");
        const axis = read(node, "tiling_axis", "auto");
        const anchor = read(node, "spatial_anchor", null);
        const overlap = read(node, "tiling_overlap", null);
        const axisLabel = { auto: "自动", width: "宽度", height: "高度" }[axis] ?? axis;
        const anchorLabel = { lowres_global: "高噪声全局锚点", off: "关闭" }[anchor] ?? anchor;
        const extras = [
            anchor != null ? `锚点=${anchorLabel}` : null,
            overlap != null ? `重叠=${overlap}` : null,
        ].filter(Boolean).join("｜");
        const suffix = extras ? `｜${extras}` : "";
        return `设置：空间分块｜${mode === "manual" ? "手动" : "自动"}｜轴=${axisLabel}｜分块=${mode === "manual" ? read(node, "tiling_tiles", 2) : "自动（1–8）"}${suffix}`;
    }
    const cacheValue = read(node, "highres_weight_cache_gb", 4);
    const cacheText = typeof cacheValue === "number" ? cacheValue.toFixed(1) : cacheValue;
    const cache = m === "global_stream" ? `｜权重缓存=${cacheText} GiB` : "";
    return `设置：${MODE_LABELS[m] ?? m}｜token 分块=${read(node, "highres_token_chunk", 4096)}${cache}`;
}

function actualCount(node) {
    if (!memoryEnabled(node)) return "实际分块数量：关闭，不分块";
    const plan = node.selfliftTilingPlan;
    if (plan?.kind === "global") {
        return `全局上下文：${plan.mode ?? memoryMode(node)}（无空间分块）`;
    }
    const count = plan?.tiles;
    if (!Number.isInteger(count) || count < 1) return "实际分块数量：待估算（低噪声阶段准备时确定）";
    return `实际分块数量：${count} 块${count === 1 ? "（整图，不拆分）" : ""}`;
}

function setHidden(widget, hidden) {
    if (!originalVisibility.has(widget)) {
        originalVisibility.set(widget, { type: widget.type, computeSize: widget.computeSize });
    }
    const original = originalVisibility.get(widget);
    if ("hidden" in widget) widget.hidden = hidden;
    else {
        widget.type = hidden ? "selflift-hidden" : original.type;
        widget.computeSize = hidden ? () => [0, -4] : original.computeSize;
    }
}

function refreshVisibility(node) {
    const on = memoryEnabled(node);
    const m = memoryMode(node);
    for (const widget of node.widgets || []) {
        if (!CONTROL_NAMES.has(widget.name) || widget.name === "highres_tiling") continue;
        // Converted widget inputs are owned by ComfyUI; do not alter their visibility.
        if (String(widget.type).startsWith("converted-widget")) continue;
        const linked = node.inputs?.some((slot) => slot.widget?.name === widget.name && slot.link != null);
        if (linked) {
            setHidden(widget, false);
            continue;
        }
        let hidden = false;
        if (!on) hidden = true;
        else if (m === "spatial") hidden = GLOBAL_CONTROLS.has(widget.name);
        else {
            hidden = SPATIAL_CONTROLS.has(widget.name)
                || (widget.name === "highres_weight_cache_gb" && m !== "global_stream");
        }
        setHidden(widget, hidden);
    }
    node.setDirtyCanvas?.(true, true);
}

function display(node, text, plan) {
    const box = node.selfliftTilingPanel;
    if (!box) return;
    if (text !== undefined) {
        node.selfliftTilingLast = String(text);
        // Explicit null resets a previous run; undefined supports older cached UI text.
        node.selfliftTilingPlan = plan ?? null;
        if (plan === undefined) {
            const match = String(text).match(/Effective \/ 实际:\s*(\d+)/);
            if (match) node.selfliftTilingPlan = { tiles: Number(match[1]) };
        }
        if (plan === null && node.selfliftSeamCanvas) {
            node.selfliftSeamData = null;
            clearSeamDiagram(node);
        }
    }
    refreshVisibility(node);
    box.value = actualCount(node) + "\n" + choices(node) + "\n" + (node.selfliftTilingLast || "尚未运行。实际方案会在低噪声阶段准备时显示。\n空间控制仅适用于空间模式；token 分块和权重缓存仅适用于全局模式。");
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "selflift-Avatar.TilingPlan",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["SelfLiftAvatarH3Sampler", "SelfLiftAvatarH3SpatialAnchoredSampler"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            const seamDiagramNode = [
                "SelfLiftAvatarH3Sampler", "SelfLiftAvatarH3SpatialAnchoredSampler",
            ].includes(nodeData.name);
            const box = document.createElement("textarea");
            box.readOnly = true;
            box.setAttribute("aria-label", "SelfLift-Avatar 低噪声阶段省显存方案");
            box.style.cssText = "width:100%;height:100%;box-sizing:border-box;resize:none;overflow:auto;font:12px/1.45 monospace;color:var(--input-text,#ddd);background:var(--comfy-input-bg,#222);border:1px solid var(--border-color,#555);padding:6px;";
            this.selfliftTilingPanel = box;
            this.addDOMWidget("selflift_tiling_plan", "selflift-tiling-status", box, { serialize: false });
            if (seamDiagramNode) {
                const canvas = document.createElement("canvas");
                canvas.width = 380;
                canvas.height = 66;
                canvas.style.cssText = "width:100%;height:66px;display:block;border:1px solid var(--border-color,#555);background:#1e1e1e;";
                this.selfliftSeamCanvas = canvas;
                this.selfliftSeamData = null;
                this.addDOMWidget("selflift_seam_diag", "selflift-seam-diag", canvas, { serialize: false });
                clearSeamDiagram(this);
            }
            for (const widget of this.widgets || []) {
                if (widget.name === "highres_tiling_mode" && widget.options) {
                    widget.options.getOptionLabel = (value) => MODE_LABELS[value] ?? value;
                }
                if (!CONTROL_NAMES.has(widget.name)) continue;
                const callback = widget.callback;
                const node = this;
                widget.callback = function () {
                    const value = callback?.apply(this, arguments);
                    display(node, "设置已修改。再次运行以计算新方案。");
                    return value;
                };
            }
            this.size[0] = Math.max(this.size[0], 370);
            this.size[1] = Math.max(this.size[1], this.computeSize()[1] + 170);
            display(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            display(this);
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            if (message?.selflift_tiling) display(this, message.selflift_tiling.join("\n"), message.selflift_tiling_plan?.[0]);
            if (message?.selflift_seam_diag?.[0]) {
                this.selfliftSeamData = message.selflift_seam_diag[0];
                drawSeamDiagram(this, this.selfliftSeamData);
            }
            return result;
        };
    },
    setup() {
        api.addEventListener("selflift-avatar-tiling", (event) => {
            const id = String(event.detail?.node_id ?? "");
            // Root-graph events use ordinary IDs. Subgraph UI still receives onExecuted.
            const node = app.graph?.getNodeById(id) ?? (/^\d+$/.test(id) ? app.graph?.getNodeById(Number(id)) : null);
            if (node?.selfliftTilingPanel) display(node, event.detail.text, event.detail.plan);
        });
    },
});
