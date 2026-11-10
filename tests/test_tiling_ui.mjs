// 不依赖 npm，以最小宿主替身覆盖本插件节点的显隐和状态事件。
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
let extension;
let node;
const events = new Map();
globalThis.testApp = {registerExtension(e){extension=e;}, graph:{getNodeById:id=>String(id)==='78'?node:null}};
globalThis.testApi = {addEventListener:(name, handler)=>events.set(name, handler)};
const context = {clearRect(){}, fillRect(){}, fillText(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}};
globalThis.document = {createElement:()=>({style:{}, setAttribute(){}, value:'', getContext:()=>context})};
let lastMenu;
class ContextMenu {
    constructor(values, options) {
        assert.ok(values.length > 0, '选项必须在构造菜单时完整传入');
        this.values = values;
        this.options = options;
        lastMenu = this;
    }
}
globalThis.LiteGraph = {ContextMenu};
const source = (await fs.readFile(new URL('../web/tiling_controls.js', import.meta.url), 'utf8'))
    .replace('import { app } from "../../scripts/app.js";', 'const app=globalThis.testApp;')
    .replace('import { api } from "../../scripts/api.js";', 'const api=globalThis.testApi;');
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
class Node {
    constructor(){
        this.size=[300,300];
        this.pos=[0,0];
        this.inputs=[];
        this.widgets=Object.entries({
            highres_tiling:true, tiling_mode:'auto', tiling_tiles:2, tiling_axis:'auto',
            highres_tiling_mode:'spatial', highres_token_chunk:4096, highres_weight_cache_gb:4,
            temporal_chunk_frames:124, temporal_overlap_frames:22, temporal_seam_lock:true,
        }).map(([name,value])=>({
            name,value,type:'number',options:{},
            onClick(){this.originalClicks=(this.originalClicks||0)+1;},
            setValue(next){this.value=next;this.callback?.(next);},
        }));
        for (const [name, values] of Object.entries({
            highres_tiling_mode:['spatial','temporal','global_stream','global','global_ram'],
            tiling_mode:['auto','manual'], tiling_axis:['auto','width','height'],
        })) {
            const widget=this.widgets.find(w=>w.name===name);
            widget.type='combo';
            widget.options.values=values;
        }
    }
    addDOMWidget(name,type,element,options){this.widgets.push({name,type,options});return this.widgets.at(-1);}
    computeSize(){return [300,300+this.widgets.filter(w=>w.type!=='selflift-hidden'&&!w.hidden).length*20];}
    setSize(size){this.size=[...size];}
    setDirtyCanvas(){}
}
await extension.beforeRegisterNodeDef(Node,{name:'SelfLiftAvatarH3Sampler'});
extension.setup();
node=new Node();
node.onNodeCreated();
const widget=name=>node.widgets.find(w=>w.name===name);
const change=(name,value)=>{widget(name).value=value;widget(name).callback();};
const hidden=name=>widget(name).type==='selflift-hidden';
const temporal=['temporal_chunk_frames','temporal_overlap_frames','temporal_seam_lock'];
const spatial=['tiling_mode','tiling_tiles','tiling_axis'];
assert.equal(node.selfliftTilingPanel.readOnly,true);
assert.equal(node.widgets.at(-1).options.serialize,false);
assert.ok(node.selfliftSeamCanvas);
assert.match(node.selfliftTilingPanel.value,/尚未运行/);
assert.ok(temporal.every(hidden));
assert.ok(spatial.every(name=>!hidden(name)));
assert.equal(widget('temporal_chunk_frames').label,'时间窗口帧数');
assert.equal(widget('highres_tiling').label,'低噪声省显存');
assert.equal(widget('highres_tiling').options.on,'开启');
assert.equal(widget('temporal_seam_lock').options.off,'关闭');
assert.equal(widget('highres_tiling_mode').label,'省显存策略');
const clickOptions={e:{canvasX:100},node,canvas:{ds:{scale:1}}};
const strategy=widget('highres_tiling_mode');
strategy.onClick(clickOptions);
assert.equal(lastMenu.values[1],'时间分块（实验）');
lastMenu.options.callback(lastMenu.values[1]);
assert.equal(strategy.value,'temporal');
assert.ok(temporal.every(name=>!hidden(name)));
assert.ok(spatial.every(hidden));
assert.match(node.selfliftTilingPanel.value,/设置：时间分块/);
strategy.onClick(clickOptions);
lastMenu.options.callback('空间分块（兼容模式）');
assert.equal(strategy.value,'spatial');
assert.ok(temporal.every(hidden));
strategy.onClick({...clickOptions,e:{canvasX:10}});
strategy.onClick({...clickOptions,e:{canvasX:node.size[0]-10}});
assert.equal(strategy.originalClicks,2);
const manual=widget('tiling_mode');
manual.onClick(clickOptions);
assert.equal(lastMenu.values[1],'手动');
lastMenu.options.callback({content:'手动'});
assert.equal(manual.value,'manual');
assert.equal(globalThis.LiteGraph.ContextMenu,ContextMenu);
manual.onClick(clickOptions);
lastMenu.options.callback({value:'auto'});
assert.equal(manual.value,'auto');
lastMenu.options.callback('未知选项');
assert.equal(manual.value,'auto');
change('tiling_mode','auto');
change('tiling_mode','manual');
assert.match(node.selfliftTilingPanel.value,/手动/);
assert.match(node.selfliftTilingPanel.value,/设置已修改/);
change('tiling_axis','height');
assert.match(node.selfliftTilingPanel.value,/轴=高度/);
node.onConfigure();
assert.match(node.selfliftTilingPanel.value,/设置已修改/);
events.get('selflift-avatar-tiling')({detail:{node_id:'79',text:'错误节点'}});
assert.doesNotMatch(node.selfliftTilingPanel.value,/错误节点/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'采样中',plan:{tiles:3}}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0],/3 块/);
assert.equal(widget('tiling_tiles').value,2);
node.onExecuted({selflift_tiling:['完成'],selflift_tiling_plan:[{tiles:3}]});
assert.match(node.selfliftTilingPanel.value,/完成/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'准备',plan:null}});
assert.match(node.selfliftTilingPanel.value,/待估算/);
change('highres_tiling_mode','temporal');
assert.ok(temporal.every(name=>!hidden(name)));
assert.ok(spatial.every(hidden));
assert.ok(hidden('highres_token_chunk'));
assert.ok(hidden('highres_weight_cache_gb'));
assert.equal(widget('highres_tiling_mode').options.getOptionLabel('temporal'),'时间分块（实验）');
assert.match(node.selfliftTilingPanel.value,/窗口=124 帧.*重叠=22 帧/);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'时间采样中',plan:{kind:'temporal',chunks:4,chunk_frames:124,overlap_frames:22}}});
assert.match(node.selfliftTilingPanel.value.split('\n')[0],/时间分块：4 段/);
const temporalHeight=node.size[1];
change('highres_tiling_mode','global_stream');
assert.ok(temporal.every(hidden));
assert.ok(spatial.every(hidden));
assert.ok(!hidden('highres_token_chunk'));
assert.ok(!hidden('highres_weight_cache_gb'));
assert.ok(node.size[1]<temporalHeight);
assert.match(node.selfliftTilingPanel.value,/全局流/);
change('highres_tiling_mode','global');
assert.ok(hidden('highres_weight_cache_gb'));
const globalHeight=node.size[1];
change('highres_tiling_mode','global');
assert.equal(node.size[1],globalHeight);
change('highres_tiling_mode','spatial');
assert.ok(temporal.every(hidden));
assert.ok(spatial.every(name=>!hidden(name)));
assert.equal(node.size[1],temporalHeight);
change('highres_tiling',false);
assert.ok(hidden('highres_tiling_mode'));
assert.ok(temporal.every(hidden));
assert.match(node.selfliftTilingPanel.value,/关闭/);
change('highres_tiling',true);
change('highres_tiling_mode','temporal');
node.inputs=[{widget:{name:'temporal_chunk_frames'},link:1}];
change('highres_tiling_mode','spatial');
assert.ok(!hidden('temporal_chunk_frames'));
assert.ok(hidden('temporal_overlap_frames'));
node.onExecuted({selflift_tiling:['接缝完成'],selflift_seam_diag:[{
    axis:'W',length:24,regions:[[0,12],[10,24]],
    seams:[{boundary:1,cut:11,overlap:2,relative_diff:0.12}],
}]});
assert.equal(node.selfliftSeamData.seams[0].relative_diff,0.12);
events.get('selflift-avatar-tiling')({detail:{node_id:'78',text:'失败',plan:null}});
assert.equal(node.selfliftSeamData,null);
assert.match(node.selfliftTilingPanel.value,/失败/);
console.log('前端状态与显隐检查通过：空间/时间/全局互斥、连接输入保留、状态事件、接缝重置。');
