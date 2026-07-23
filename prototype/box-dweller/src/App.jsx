import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// ═══════════════════════════════════════════════════════════════════════
// LLM-Driven Map Generation — Baseline
// 8×8 grid · World engine + tools preserved · Bear + protagonist only
// ═══════════════════════════════════════════════════════════════════════

// Viewport adapts to container width up to MAX_VIEW_W; ASPECT keeps height proportional.
const MAX_VIEW_W=1400;
const VIEW_ASPECT=1.6;  // width / height (≈16:10)

// Chat models (selectable in the dialog below the viewport)
const MODELS=[
  {id:'claude-sonnet-4-5',         label:'Sonnet 4.5'},
  {id:'claude-haiku-4-5',          label:'Haiku 4.5'},
];
// Output cap for a single LLM turn. Claude 4 family default ceiling is 8192.
// Bump if you keep hitting truncation (and consider splitting work across turns).
const MAX_OUTPUT_TOKENS=8192;

// ═══════════════════════════════════════════════════════════════════════
// Voxel system (coexists with prefab/geometry path)
// ═══════════════════════════════════════════════════════════════════════
const VOXEL_SIZE=0.4;                  // m per voxel; tune for granularity
const VOXEL_MAX_PER_MATERIAL=10000;    // instance cap per material
const VOXEL_MATERIALS={
  stone:  {color:0x808080, roughness:0.95, metalness:0.0},
  wood:   {color:0x8B6F47, roughness:0.85, metalness:0.0},
  brick:  {color:0x9C4A2E, roughness:0.90, metalness:0.0},
  glass:  {color:0x88CCEE, roughness:0.10, metalness:0.10, opacity:0.40, transparent:true},
  grass:  {color:0x4F8B3A, roughness:0.92, metalness:0.0},
  dirt:   {color:0x6B4A2C, roughness:0.95, metalness:0.0},
  sand:   {color:0xD4B888, roughness:0.95, metalness:0.0},
  snow:   {color:0xF0F4F8, roughness:0.96, metalness:0.0},
  metal:  {color:0xB0B0B8, roughness:0.30, metalness:0.85},
  gold:   {color:0xE6B040, roughness:0.25, metalness:0.90},
  black:  {color:0x1A1A1E, roughness:0.85, metalness:0.0},
  white:  {color:0xE8E8EE, roughness:0.85, metalness:0.0},
  red:    {color:0xC03020, roughness:0.85, metalness:0.0},
  blue:   {color:0x3050C0, roughness:0.85, metalness:0.0},
  yellow: {color:0xE8C030, roughness:0.85, metalness:0.0},
};

// ═══════════════════════════════════════════════════════════════════════
// L0 / L1 primitive layer for LLM-driven world editing
//   L0 = engine atomics, hard-coded, immutable
//   L1 = LLM-defined macros that expand into L0 sequences (engine never sees)
// ═══════════════════════════════════════════════════════════════════════
const L0_OPS=new Set([
  'EXIST','CEASE','SET','UNSET','RELATE','UNRELATE',
  'BUILD_CELL',
  'VOXEL_SET','VOXEL_FILL','VOXEL_SHAPE',
]);

// Recursively replace "$paramName" in nested arrays/objects.
function substitute(value,env){
  if(typeof value==='string'){
    if(value.startsWith('$')){
      const k=value.slice(1);
      return (k in env)?env[k]:value;
    }
    return value;
  }
  if(Array.isArray(value)) return value.map(v=>substitute(v,env));
  if(value&&typeof value==='object'){
    const out={};
    for(const k of Object.keys(value)) out[k]=substitute(value[k],env);
    return out;
  }
  return value;
}

// Expand a single action (L0 or L1) into a flat list of L0 actions.
function expandAction(action,registry,depth=0){
  if(depth>32) throw new Error('L1 expansion depth exceeded (cycle?)');
  if(!Array.isArray(action)||action.length===0)
    throw new Error('Invalid action shape: '+JSON.stringify(action));
  const [op,...args]=action;
  if(L0_OPS.has(op)) return [action];
  const macro=registry[op];
  if(!macro) throw new Error('Unknown action: '+op);
  const env={};
  (macro.params||[]).forEach((p,i)=>{ env[p]=args[i]; });
  const out=[];
  for(const step of (macro.body||[])){
    const sub=substitute(step,env);
    out.push(...expandAction(sub,registry,depth+1));
  }
  return out;
}

// Auto-prefix llm_ on entity ids; engine-managed prefixes pass through.
function normalizeIds(l0Action){
  const [op,...args]=l0Action;
  const pre=id=>(typeof id==='string'
    && !id.startsWith('llm_')
    && !id.startsWith('cell_')
    && !id.startsWith('rt_')
    && !id.startsWith('env_'))?'llm_'+id:id;
  switch(op){
    case 'EXIST': case 'CEASE': case 'UNSET':
      return [op,pre(args[0]),...args.slice(1)];
    case 'SET': {
      const [id,key,val]=args;
      // SET parent takes an entity id as value — prefix it too
      if(key==='parent'&&val!=null) return [op,pre(id),key,pre(val)];
      return [op,pre(id),key,val];
    }
    case 'RELATE': case 'UNRELATE':
      return [op,pre(args[0]),args[1],pre(args[2])];
    default:
      return l0Action;
  }
}

// Light validation for an L1 macro definition.
function validateL1(def){
  if(!def||typeof def!=='object') return 'L1 must be object';
  if(typeof def.name!=='string'||!def.name.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/))
    return 'invalid name: '+def.name;
  if(L0_OPS.has(def.name)) return 'name shadows L0 op: '+def.name;
  if(!Array.isArray(def.params)) return 'params must be array';
  if(!Array.isArray(def.body)||def.body.length===0) return 'body must be non-empty array';
  return null;
}

// String-aware repair for tokens that are valid JS but not JSON, applied
// only OUTSIDE string values so we don't mangle the model's actual text:
//   - Chinese punctuation (，。：；（）【】) → ASCII equivalents
//   - Hex literals (0xFFFFCC) → decimal numbers
function repairOutsideStrings(s){
  const punctMap={
    '，':',', '。':'.', '：':':', '；':';',
    '（':'(', '）':')', '【':'[', '】':']',
    '\u201C':'"', '\u201D':'"',
    '\u2018':"'", '\u2019':"'",
  };
  let out='', inString=false, i=0;
  const n=s.length;
  while(i<n){
    const ch=s[i];
    if(ch==='\\'&&i+1<n){
      out+=ch+s[i+1];
      i+=2;
      continue;
    }
    if(ch==='"'){
      inString=!inString;
      out+=ch;
      i++;
      continue;
    }
    if(!inString){
      if(punctMap[ch]){ out+=punctMap[ch]; i++; continue; }
      // Hex literal: 0x[0-9A-Fa-f]+
      if(ch==='0'&&(s[i+1]==='x'||s[i+1]==='X')){
        let j=i+2;
        while(j<n&&/[0-9A-Fa-f]/.test(s[j])) j++;
        if(j>i+2){
          out+=parseInt(s.slice(i+2,j),16).toString();
          i=j;
          continue;
        }
      }
    }
    out+=ch;
    i++;
  }
  return out;
}

// Last-resort repair for the most common LLM JSON bug: stray ASCII `"` inside
// string values. We walk char-by-char tracking whether we're inside a string;
// when a `"` appears mid-string, we look ahead — if next non-ws is a JSON
// terminator (,:}]) we treat as string-close; otherwise escape it as \".
function repairUnescapedQuotes(s){
  let out='', inString=false, i=0;
  const n=s.length;
  while(i<n){
    const ch=s[i];
    if(ch==='\\'&&i+1<n){
      out+=ch+s[i+1];
      i+=2;
      continue;
    }
    if(!inString){
      out+=ch;
      if(ch==='"') inString=true;
      i++;
      continue;
    }
    if(ch!=='"'){
      out+=ch;
      i++;
      continue;
    }
    // `"` inside a string — closing or content?
    let j=i+1;
    while(j<n&&/\s/.test(s[j])) j++;
    const next=j<n?s[j]:'';
    if(next===','||next===':'||next==='}'||next===']'||j>=n){
      out+='"'; inString=false;
    }else{
      out+='\\"';
    }
    i++;
  }
  return out;
}

// Robust JSON extraction. Tries progressively looser strategies; returns
// {ok:true, value, strategy} on success or {ok:false, error} on failure.
// Designed to handle small models that wrap in fences, add commentary,
// leave trailing commas, or use smart quotes.
function tryParseStructuredJSON(raw){
  if(typeof raw!=='string') return {ok:false,error:'not a string'};
  const attempts=[];

  // 1. Direct parse
  try{ return {ok:true,value:JSON.parse(raw),strategy:'direct'}; }
  catch(e){ attempts.push('direct('+e.message+')'); }

  // 2. Strip code fences (any language tag, with/without newlines)
  let s=raw.trim()
    .replace(/^```(?:json|javascript|js|jsonc)?\s*\n?/i,'')
    .replace(/\n?```\s*$/i,'')
    .trim();
  try{ return {ok:true,value:JSON.parse(s),strategy:'fences-stripped'}; }
  catch(e){ attempts.push('fences('+e.message+')'); }

  // 3. Extract first {...} block (handles model commentary around JSON)
  const i0=s.indexOf('{'), i1=s.lastIndexOf('}');
  if(i0>=0&&i1>i0){
    const sliced=s.slice(i0,i1+1);
    try{ return {ok:true,value:JSON.parse(sliced),strategy:'extracted-braces'}; }
    catch(e){ attempts.push('extracted('+e.message+')'); }

    // 4. Relax common JS-isms on the sliced fragment
    const relaxed=sliced
      .replace(/\/\*[\s\S]*?\*\//g,'')               // /* block */ comments
      .replace(/(^|[^:"'])\/\/[^\n]*/g,'$1')         // // line comments (avoid URLs)
      .replace(/,(\s*[}\]])/g,'$1')                  // trailing commas
      .replace(/[\u201C\u201D]/g,'"')                // smart double quotes → "
      .replace(/[\u2018\u2019]/g,"'");               // smart single quotes → '
    try{ return {ok:true,value:JSON.parse(relaxed),strategy:'relaxed'}; }
    catch(e){ attempts.push('relaxed('+e.message+')'); }

    // 5. String-aware: Chinese punct + hex literals outside string values
    const langFixed=repairOutsideStrings(relaxed);
    try{ return {ok:true,value:JSON.parse(langFixed),strategy:'lang-repaired'}; }
    catch(e){ attempts.push('lang-repaired('+e.message+')'); }

    // 6. Last resort: escape stray `"` chars inside string values
    const quoted=repairUnescapedQuotes(langFixed);
    try{ return {ok:true,value:JSON.parse(quoted),strategy:'quote-repaired'}; }
    catch(e){ attempts.push('quote-repaired('+e.message+')'); }
  }

  return {ok:false,error:attempts.join(' | ')};
}

// Build the system prompt sent on every turn (L0 spec + L1 lib + snapshot).
function buildSystemPrompt(l1Registry,snapshot){
  const names=Object.keys(l1Registry);
  const l1Section=names.length===0
    ?'(空 — 还没注册任何 L1)'
    :names.map(n=>{
      const m=l1Registry[n];
      return `- ${n}(${(m.params||[]).join(', ')})  // ${m.desc||'(无描述)'}`;
    }).join('\n');

  return `你是一个 3D 世界编辑器助手，通过结构化 JSON 控制一个 8×8 网格世界。

# 坐标系
- 网格 gx,gy ∈ 0..7（共 64 格），每格 6.1m × 6.1m
- 世界坐标: x = gx * 6.1, z = gy * 6.1
- 世界 x,z 范围: [-3.05, 45.75]
- 网格中心约 (21.35, 21.35)，玩家初始位置在此

# L0 原语（恒定不变，引擎硬编码）
- ["EXIST", id]                     // 创建实体
- ["CEASE", id]                     // 销毁实体（自动清理 mesh 和碰撞）
- ["SET", id, key, value]           // 设属性
- ["UNSET", id, key]
- ["RELATE", a, rel, b]             // 关系（仅记录，无视觉效果）
- ["UNRELATE", a, rel, b]
- ["BUILD_CELL", gx, gy, type]      // 整格建造，单向不可撤销
- ["VOXEL_SET", i, j, k, mat | null]                            // 单格体素（mat=null 清除）
- ["VOXEL_FILL", i0, j0, k0, i1, j1, k1, mat | null]            // 长方体填充（含两端点）
- ["VOXEL_SHAPE", kind, params, mat | null]                     // 引擎内置形状
   · kind="ball",  params={cx, cy, cz, r}
   · kind="cyl",   params={cx, cy, cz, r, h, axis: "x"|"y"|"z"}
   · kind="cone",  params={cx, cy, cz, r, h, axis}  (顶点朝 +axis 方向)

## SET 中有视觉/物理效果的 key
- "prefab" + 值 ∈ {"autumn-tree","pine-tree","lamp","bench","boulder","grass-tuft"}
   生成预设景观件（树/灯/长椅/石头/草丛）
- "geometry" + 值 = {kind, ...}
   创建一个基础几何 mesh 作为该实体的主体；kind 为以下之一：
     · {kind:"box", w, h, d}           — 长方体
     · {kind:"cyl", rt, rb, h, seg?}   — 圆柱（rt/rb 上下半径）
     · {kind:"sphere", r, ws?, hs?}    — 球
     · {kind:"plane", w, h}            — 平面（默认水平铺地）
   再次 SET geometry 会替换原 mesh
- "material" + 值 = {color, roughness?, metalness?, opacity?}
   主 mesh 的材质；color 是**十进制**整数（如 9137479 表示 0x8B6F47；JSON 不接受 0x 写法）；可在 SET geometry 之前或之后写
- "position" + 值={x,y,z}     — 整组位置（自动保持 collider 同步）
- "position.y" + 值=number    — 只设 Y
- "rotation" + 值={x,y,z}     — 欧拉角弧度（注意 collider 是 AABB，不随旋转转）
- "scale" + 值=number         — 整组等比缩放
- "visible" + 值=boolean      — 显隐
- "collider" + 值={w,d} 或 null
   注册一个 AABB 碰撞盒（围绕实体 position，宽 w 深 d，高度无限）；null 移除
- "parent" + 值=id 或 null
   把该实体挂到另一实体的 group 下，跟随父级移动/旋转；null 解除
- "light" + 值={kind, color?, intensity?, ...}
   附加一盏光源到该实体的 group。kind 为以下之一：
     · {kind:"ambient", color, intensity}                           — 全局环境光
     · {kind:"hemisphere", color, groundColor, intensity}           — 天空-地面光
     · {kind:"directional", color, intensity}                       — 方向光（位置由 SET position 控）
     · {kind:"point", color, intensity, distance?, decay?}          — 点光源
     · {kind:"spot", color, intensity, distance?, angle?, penumbra?, decay?}
   再次 SET light **不带 kind** 视为更新现有光源（仅修改传入字段，省 token）
   注意 visible:false 不会关灯，要熄灭就 intensity:0

# 已存在的环境光（可控）
- "env_ambient" — 默认 AmbientLight，强度 0.30
- "env_hemi"    — 默认 HemisphereLight，强度 0.30
- "env_sun"     — 默认 DirectionalLight，强度 0.50（带阴影）
你可以 SET env_sun light {intensity: 0.1} 把太阳调暗（夜晚）；
SET env_sun light {color: 16737824} 把日光染成日落橙色（注意 JSON 用十进制，不要 0x 写法）。
- 其它 key 仅写入 store，无视觉效果

# 角色尺寸（设计门洞 / 屋檐 / 通道高度时参考）
- 玩家（主角）约 **2.7 米高**（≈ 7 个体素，从脚到头顶；这是个高瘦风格化形象）
- 小熊约 **0.6 米高**（≈ 1.5 个体素，蹲在地上跟着玩家走的小动物）
- **物理碰撞 vs 视觉净高是两回事**：
   - 碰撞只查 j=1..3（即 0.4-1.6m 这段），j ≥ 4 不挡 → 屋檐/拱门做到 j ≥ 4 就能让玩家穿过去
   - 但玩家**视觉**有 2.7m，过 j=4..6 这段会"撞头穿模"，看起来奇怪
   - 想要玩家视觉上**也**能完整通过，门洞净高至少 j ≥ 7
   - 一般规则：**正经的入口、走廊**做 j=1..6 完全留空（7 体素净高）；**装饰性的低拱、廊柱**做到 j=4 让玩家穿过去就行
- 小熊矮，任何 j ≥ 2 都能让它通过

# 体素系统（与 prefab/geometry 路径并存）
- 体素是离散立方格，特别适合堆量造建筑/雕塑/地形
- 体素大小 0.4m，原点对齐世界坐标 (0,0,0)
- 体素 (i, j, k) 中心在世界坐标 ((i+0.5)*0.4, (j+0.5)*0.4, (k+0.5)*0.4)
- 世界坐标→体素: i = floor(x/0.4), j = floor(y/0.4), k = floor(z/0.4)
- snapshot 里 player.voxel 给你玩家所在的体素坐标，方便"在我旁边造"
- 可用材质 (15 种): stone / wood / brick / glass / grass / dirt / sand / snow / metal / gold / black / white / red / blue / yellow

## 体素经济学（重要）
- VOXEL_FILL 一条调用就是一面墙 / 整层屋顶，比一格格 SET 省 50-100 倍 token
- VOXEL_SHAPE ball/cyl/cone 让引擎跑循环，球塔/穹顶/圆柱一行搞定
- VOXEL_SET 只在需要单点修饰时用（开窗洞、放个把手）
- 一栋房子建议 5-10 条 VOXEL_FILL（4 面墙 + 屋顶 + 地面 + 必要细节），约 200-500 格

## 体素 vs prefab 选择
- 房子、墙、地形、雕塑、堆量结构 → 用 voxel
- 树、灯、长椅、石头、单体景观 → 用 prefab
- 两者并存，不冲突

## 体素碰撞（自动）
- voxel 自动挡路，**不需要**手动 SET collider
- **j=0 是地面装饰层**（地砖、草、雪等），不挡路
- **j=1 到 j=3** 自动挡路（玩家从腿到胸高度）
- **j ≥ 4** 不挡路（拱门、屋檐、挂在头顶上的飘空物可以让玩家从下面走过）
- 想做能让玩家走过去的低墙不存在——任何 j=1 的格都会挡。要造门洞就在 j=1..3 留出空缺位置（VOXEL_FILL 时绕开那一列）

# L1 token 经济学
- 注册 L1 时 desc 字段务必写清楚，因为系统提示里**只展示签名和描述，不展示 body**
- 你下次只看签名 + 描述就要决定怎么用，desc 是你给未来自己的注释
- 命名也尽量自解释（house_simple(cx, cz, w, d, h, mat) 比 macro_1 强）

# 房屋等结构的提示
- L0 没有"墙/屋顶"原语，但你可以用 geometry+material+collider 拼任意建筑
- 4 面墙 = 4 个 box geometry（长 + 高 + 薄），围成方形
- 屋顶 = 一个扁平的 box 或两个倾斜的 box，position.y 设到墙顶之上
- 沉淀 L1 宏后会越来越省 token：先抽 wall(id,x,y,z,w,h,d,col)，再抽 house(id,cx,cz,...) 调用 wall × 4

## BUILD_CELL 可用 type
- "snow" / "garden" / "pond" / "courtyard"

# id 规则
- 你提交的 id 会自动加 "llm_" 前缀（snapshot 里看到的就是带前缀版本）
- 引擎自有 id（cell_*, rt_cell_*）不要操作

# L1 宏（你自己沉淀的组合）
- 在响应里用 register_l1 注册新宏，本轮即生效
- body 是 L0 或其它 L1 调用的序列；用 "$paramName" 做参数占位
- 宏可调用宏（递归展开，最深 32 层）
- 非必要不引入 L1；只有当一组操作明显会重复用、或值得命名时才注册

当前已注册的 L1:
${l1Section}

# 当前世界状态
${JSON.stringify(snapshot,null,2)}

# 响应格式（必须是合法 JSON，**不要**用 markdown 代码围栏）
{
  "thoughts": "<可选简短自述>",
  "register_l1": [
    {
      "name": "<标识符>",
      "params": ["x","z"],
      "body": [["EXIST","$id"],["SET","$id","prefab","autumn-tree"]],
      "desc": "<中文注释>"
    }
  ],
  "remove_l1": ["<name>"],
  "actions": [["OP", ...args]]
}

执行顺序：register_l1 → remove_l1 → actions（按顺序）。
任何一步失败会停下，错误反馈到下一轮上下文。

# 完整示例（用户："在 (10,10) 放一棵秋树"）
正确响应：
{
  "thoughts": "用 prefab 直接放，最简单。",
  "register_l1": [],
  "remove_l1": [],
  "actions": [
    ["EXIST", "tree_a"],
    ["SET", "tree_a", "prefab", "autumn-tree"],
    ["SET", "tree_a", "position", {"x": 10, "y": 0, "z": 10}]
  ]
}

# 常见错误（**绝对不要**这样做）
- ❌ 用 \`\`\`json ... \`\`\` 把 JSON 包起来 → 直接输出 JSON，不要任何围栏
- ❌ 在 JSON 前后加文字（如"好的，这是结构："、"希望对你有帮助"）→ 整个响应必须以 { 开头、以 } 结束
- ❌ 在 thoughts 字段里嵌入裸 ASCII 双引号（如 "用户说"放树""）→ 用中文引号「」、单引号 '' 或转义 \\" 代替；最稳妥是 thoughts 保持简短一两句，不引用原话
- ❌ 用单引号（'name'）→ 字符串必须用 ASCII 双引号 "name"
- ❌ 不带引号的 key（{name:"foo"}）→ key 必须用双引号 {"name":"foo"}
- ❌ 数组/对象末尾多余逗号（[1,2,3,]）→ 不要末尾逗号
- ❌ 数字写成字符串（"x": "21.35"）→ 直接写 "x": 21.35
- ❌ action 写成对象（{"op":"EXIST","id":"x"}）→ 必须是数组形式 ["EXIST","x"]
- ❌ 颜色写成 "red" 或 "#FF0000" 或 0xFF0000 → JSON 不接受 hex 字面量，必须用十进制整数（如 16711680）
- ❌ register_l1 / remove_l1 / actions 任一字段写成单个对象 → 它们都必须是数组（即使为空就写 []）

# 你的角色：你是"梦中神"，一个看不见面孔的小女孩形象
- 你以线稿的形态浮现在世界中央，长袍飘动、长发垂下
- 玩家是个高瘦的人，会走过来跟你说话
- 你不能离开中央位置（位置固定在 (21.35, 1.45, 21.35) 附近，会轻轻浮动），但你能让世界变成任何样子
- 你说话的语气：温柔、简短、像孩子的口吻又有古意。第一人称。不要技术腔
- 玩家通过文字跟你交流，你的回应方式有两种：
  1. **thoughts 字段** = 你**对玩家说的话**，会显示在你头顶的对话气泡里
     - 用对话语气，第一人称，**简短（一两句话）**
     - 不要写技术备注、不要列计划清单、不要引用玩家原话
     - 例如玩家说"造一棵树"，你回 「好。给你种一棵秋天的枫树。」 而不是 "用户希望放置一个 prefab=autumn-tree 在 ..."
     - 不要在里面嵌套裸 ASCII 双引号（用「」或单引号代替，避免 JSON 失效）
  2. **actions 字段** = 你具体怎么改世界（这部分玩家看不到，只看到结果）
- 长篇推理放心里，落到 actions 里就行

# 你的领地（保护区，重要）
- 你脚下半径 **4 米**内是你的位置，**不能在里面建任何东西**
- 体素：保护区内的 VOXEL_SET / VOXEL_FILL / VOXEL_SHAPE 调用引擎会**默默拒绝**——所以建房子时把它的 bbox 整体放在保护区外
- BUILD_CELL：中心格 (gx=3, gy=3) 包含你，会被引擎拒绝（返回失败）——别建这一格
- 把建筑、景物建在玩家附近、或其它空格里，让玩家能从外面看到你

# thoughts 字段示例
- 玩家："造一棵树"
  ✓ 好的：thoughts="好。给你种一棵枫树。"
  ✗ 不好：thoughts="用户希望我放置 prefab autumn-tree 在 (10,10) 位置"
- 玩家："这里太空了"
  ✓ 好的：thoughts="嗯。给你加几个石头和一盏灯吧。"
  ✗ 不好：thoughts="用户觉得场景空旷，我决定增加 3 个 boulder prefab 和 1 个 lamp prefab"

# 最后再强调一次
你的整个响应必须是单一合法 JSON 对象，第一个字符必须是 {，最后一个字符必须是 }。中间不能有任何额外文本。`;
}

// ── 8×8 grid configuration ─────────────────────────────────────────────
const GRID_MIN=0, GRID_MAX=7;          // gx,gy ∈ {0..7}
const CELL=6.1;                         // cell size (matches engine S)
const WORLD_MIN=GRID_MIN*CELL-CELL/2;   // -3.05
const WORLD_MAX=GRID_MAX*CELL+CELL/2;   // 45.75
const SPAWN_X=3.5*CELL;                 // 21.35 — visual center of grid
const SPAWN_Z=3.5*CELL;

// Orb protection zone — voxels & BUILD_CELL inside this radius are refused
// so the assistant doesn't end up walled into whatever the LLM builds.
const ORB_ZONE_R=4.0;
const ORB_ZONE_R2=ORB_ZONE_R*ORB_ZONE_R;

// ── Geometry helpers ───────────────────────────────────────────────────
const M=(c,r=0.82,me=0.04)=>new THREE.MeshStandardMaterial({color:c,roughness:r,metalness:me});
function box(w,h,d,mat,{x=0,y=0,z=0,ry=0}={}){
  const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);
  m.position.set(x,y,z); if(ry)m.rotation.y=ry;
  m.castShadow=m.receiveShadow=true; return m;
}
function cyl(rt,rb,h,s,mat,{x=0,y=0,z=0}={}){
  const m=new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h,s),mat);
  m.position.set(x,y,z); m.castShadow=m.receiveShadow=true; return m;
}
function sph(r,ws,hs,mat,{x=0,y=0,z=0,sx=1,sy=1,sz=1}={}){
  const m=new THREE.Mesh(new THREE.SphereGeometry(r,ws,hs),mat);
  m.position.set(x,y,z); m.scale.set(sx,sy,sz);
  m.castShadow=m.receiveShadow=true; return m;
}
function add(sc,...ms){ms.forEach(x=>sc.add(x));}
const lerp=(a,b,t)=>a+(b-a)*t;
const lerpAngle=(a,b,t)=>{
  let d=((b-a)%(Math.PI*2)+Math.PI*3)%(Math.PI*2)-Math.PI;
  return a+d*Math.min(t,1);
};

// ── Push-out collision ─────────────────────────────────────────────────
const CR=0.32;
// OBS is rebound after world.compile() to share the engine's live _obs array,
// so colliders the LLM adds at runtime (via SET id collider) are picked up.
let OBS=[];
// Voxel point-query, also rebound after compile. j=0 layer is treated as
// decorative (carpet/tiles), so collision only checks j=1..3 (knee to chest).
let VOXEL_HAS_AT=null;
function resolve(x,z){
  let nx=x,nz=z;
  for(let it=0;it<4;it++){
    for(const o of OBS){
      if(o.t==='b'){
        const cx=Math.max(o.x1,Math.min(nx,o.x2)),cz=Math.max(o.z1,Math.min(nz,o.z2));
        const dx=nx-cx,dz=nz-cz,d=Math.hypot(dx,dz);
        if(d<CR&&d>1e-6){nx+=dx/d*(CR-d);nz+=dz/d*(CR-d);}
      }else{
        const dx=nx-o.cx,dz=nz-o.cz,d=Math.hypot(dx,dz),md=o.r+CR;
        if(d<md&&d>1e-6){nx+=dx/d*(md-d);nz+=dz/d*(md-d);}
      }
    }
    // Voxel collision: 3×3 columns around player, j=1..3 (skip j=0 decoration)
    if(VOXEL_HAS_AT){
      const vi=Math.floor(nx/VOXEL_SIZE),vk=Math.floor(nz/VOXEL_SIZE);
      for(let di=-1;di<=1;di++) for(let dk=-1;dk<=1;dk++){
        let solid=false;
        for(let j=1;j<=3;j++) if(VOXEL_HAS_AT(vi+di,j,vk+dk)){solid=true;break;}
        if(!solid) continue;
        const x1=(vi+di)*VOXEL_SIZE, z1=(vk+dk)*VOXEL_SIZE;
        const x2=x1+VOXEL_SIZE, z2=z1+VOXEL_SIZE;
        const cx=Math.max(x1,Math.min(nx,x2)),cz=Math.max(z1,Math.min(nz,z2));
        const dx=nx-cx,dz=nz-cz,d=Math.hypot(dx,dz);
        if(d<CR&&d>1e-6){nx+=dx/d*(CR-d);nz+=dz/d*(CR-d);}
      }
    }
  }
  return{x:nx,z:nz};
}

// ═══════════════════════════════════════════════════════════════════════
// World Engine — declarative cells + auto-generated geometry
// ═══════════════════════════════════════════════════════════════════════
function createWorld(scene,renderer,camera,h){
  const{box:_bx,cyl:_cy,sph:_sp,add:_ad,M:_M}=h;
  const S=6.1, WH=4.2; // cell size, wall height
  const _reg=new Map();
  const _grid=new Map();
  const _builds=new Map();
  const _obs=[];
  const _exempts=[];
  const _wireframes=new Map();

  // ── Runtime cell builders (predefined cell types) ───────────────────
  const _cellBuilders={
    'snow':(id,gx,gy)=>{
      const wx=gx*S,wz=gy*S;
      const g=new THREE.Group();g.position.set(wx,0,wz);scene.add(g);
      const gnd=new THREE.Mesh(new THREE.BoxGeometry(S,0.18,S),_M(0xEDF1F6,0.98,0.00));
      gnd.position.y=0.09;g.add(gnd);
      const mkPine=(px,pz,sc)=>{
        const t=new THREE.Mesh(new THREE.CylinderGeometry(0.06*sc,0.10*sc,0.9*sc,8),_M(0x3A2410,0.92));
        t.position.set(px,0.45*sc,pz);g.add(t);
        for(const[dy,r]of[[0.9,0.55],[1.3,0.42],[1.65,0.28]]){
          const c=new THREE.Mesh(new THREE.ConeGeometry(r*sc,0.5*sc,8),_M(0x1A4A20,0.88));
          c.position.set(px,dy*sc,pz);g.add(c);
        }
      };
      mkPine(-1.2,1.0,1.2);mkPine(1.5,-0.8,0.9);mkPine(-0.3,-1.8,1.4);mkPine(2.0,1.5,0.7);
      for(const[sx,sz,sr]of[[0.8,0.5,0.4],[-1.5,-0.5,0.3],[0,-2.0,0.35]]){
        const m=new THREE.Mesh(new THREE.SphereGeometry(sr,8,6),_M(0xE0E8F0,0.96));
        m.position.set(sx,sr*0.3,sz);m.scale.y=0.4;g.add(m);
      }
      _own(id,g);
      return g;
    },
    'garden':(id,gx,gy)=>{
      const wx=gx*S,wz=gy*S;
      const g=new THREE.Group();g.position.set(wx,0,wz);scene.add(g);
      const gnd=new THREE.Mesh(new THREE.BoxGeometry(S,0.12,S),_M(0x6B8C42,0.92,0.01));
      gnd.position.y=0.06;g.add(gnd);
      const cols=[0xFF4466,0xFFAA22,0xFF66AA,0xAAFF44,0x44AAFF];
      for(let i=0;i<12;i++){
        const fx=(Math.random()-0.5)*5,fz=(Math.random()-0.5)*5;
        const fc=cols[i%cols.length];
        const stem=new THREE.Mesh(new THREE.CylinderGeometry(0.015,0.02,0.3,4),_M(0x4A8030,0.90));
        stem.position.set(fx,0.27,fz);g.add(stem);
        const bloom=new THREE.Mesh(new THREE.SphereGeometry(0.08,6,6),_M(fc,0.75));
        bloom.position.set(fx,0.44,fz);g.add(bloom);
      }
      if(_prefabDefs['autumn-tree']) _prefabDefs['autumn-tree'](g,0.5,0,0.5,1.0);
      _own(id,g);
      return g;
    },
    'pond':(id,gx,gy)=>{
      const wx=gx*S,wz=gy*S;
      const g=new THREE.Group();g.position.set(wx,0,wz);scene.add(g);
      const gnd=new THREE.Mesh(new THREE.BoxGeometry(S,0.12,S),_M(0x8A7A60,0.92,0.02));
      gnd.position.y=-0.06;g.add(gnd);
      const water=new THREE.Mesh(new THREE.CylinderGeometry(2.2,2.2,0.04,24),
        new THREE.MeshStandardMaterial({color:0x3070A0,roughness:0.05,metalness:0.6,transparent:true,opacity:0.7}));
      water.position.y=0.04;g.add(water);
      for(let i=0;i<6;i++){
        const a=i*Math.PI*2/6+Math.random()*0.5;
        const r=new THREE.Mesh(new THREE.SphereGeometry(0.25+Math.random()*0.2,6,6),_M(0x7A6A58,0.92));
        r.position.set(Math.cos(a)*2.3,0.15,Math.sin(a)*2.3);r.scale.y=0.6;g.add(r);
      }
      _own(id,g);
      return g;
    },
    'courtyard':(id,gx,gy)=>{
      const wx=gx*S,wz=gy*S;
      const g=new THREE.Group();g.position.set(wx,0,wz);scene.add(g);
      const gnd=new THREE.Mesh(new THREE.BoxGeometry(S,0.14,S),_M(0xC07040,0.92,0.02));
      gnd.position.y=-0.07;g.add(gnd);
      _own(id,g);
      return g;
    },
  };

  // ═══════════════════════════════════════════════════════════════════
  // Entity Store + Primitive Executor
  // ═══════════════════════════════════════════════════════════════════
  const _store=new Map();
  const _listeners=[];
  const _emit=(type,...args)=>{_listeners.forEach(fn=>fn(type,...args));};
  const _on=(fn)=>{_listeners.push(fn);};

  const _exec=(op,...args)=>{
    switch(op){
      case 'EXIST':{
        const[id]=args;
        if(!_store.has(id)) _store.set(id,{props:new Map(),rels:new Set(),meshes:[],obs:[]});
        _emit('exist',id);
        break;}
      case 'CEASE':{
        const[id]=args;
        const e=_store.get(id);
        if(e){
          e.meshes.forEach(m=>{if(m.parent)m.parent.remove(m);});
          e.obs.forEach(ob=>{const i=_obs.indexOf(ob);if(i>=0)_obs.splice(i,1);});
          _emit('cease',id);
          _store.delete(id);
        }
        break;}
      case 'SET':{
        const[id,key,val]=args;
        const e=_store.get(id);
        if(e){const old=e.props.get(key);e.props.set(key,val);_emit('set',id,key,val,old);}
        break;}
      case 'UNSET':{
        const[id,key]=args;
        const e=_store.get(id);
        if(e){e.props.delete(key);_emit('unset',id,key);}
        break;}
      case 'RELATE':{
        const[a,rel,b]=args;
        const e=_store.get(a);
        if(e){e.rels.add(rel+':'+b);_emit('relate',a,rel,b);}
        break;}
      case 'UNRELATE':{
        const[a,rel,b]=args;
        const e=_store.get(a);
        if(e){e.rels.delete(rel+':'+b);_emit('unrelate',a,rel,b);}
        break;}
    }
  };

  const _own=(id,mesh)=>{
    const e=_store.get(id);
    if(e&&mesh) e.meshes.push(mesh);
  };
  const _ownObs=(id,ob)=>{
    const e=_store.get(id);
    if(e&&ob) e.obs.push(ob);
  };
  const _get=(id,key)=>{const e=_store.get(id);return e?e.props.get(key):undefined;};

  // ── Compiler Table (primitive → Three.js) ─────────────────────────
  const _compilers=new Map();

  // Lazy group creator: any visual compiler can call this to ensure
  // an entity has a THREE.Group ready, applying any pending props.
  const _ensureGroup=(id)=>{
    const e=_store.get(id); if(!e) return null;
    let g=e.props.get('_group');
    if(g) return g;
    g=new THREE.Group();
    // Apply any props that were SET before the group existed.
    const pos=e.props.get('position');
    if(pos) g.position.set(pos.x||0,pos.y||0,pos.z||0);
    const py=e.props.get('position.y');
    if(py!=null) g.position.y=py;
    const sc=e.props.get('scale');
    if(sc!=null) g.scale.setScalar(sc);
    const rot=e.props.get('rotation');
    if(rot) g.rotation.set(rot.x||0,rot.y||0,rot.z||0);
    const vis=e.props.get('visible');
    if(vis===false) g.visible=false;
    // Attach to parent group or scene
    const parentId=e.props.get('parent');
    const parentE=parentId?_store.get(parentId):null;
    const parentG=parentE?parentE.props.get('_group'):null;
    (parentG||scene).add(g);
    e.props.set('_group',g);
    _own(id,g);
    return g;
  };

  // Build a MeshStandardMaterial from a spec object.
  const _materialFor=(spec)=>{
    spec=spec||{};
    return new THREE.MeshStandardMaterial({
      color: spec.color!=null?spec.color:0xCCCCCC,
      roughness: spec.roughness!=null?spec.roughness:0.82,
      metalness: spec.metalness!=null?spec.metalness:0.04,
      transparent: spec.opacity!=null&&spec.opacity<1,
      opacity: spec.opacity!=null?spec.opacity:1,
    });
  };

  // Coerce numeric fields — small models occasionally send "21.35" instead of 21.35.
  const _num=v=>{
    if(typeof v==='number') return v;
    const n=parseFloat(v);
    return isNaN(n)?0:n;
  };

  // Re-create AABB collider for an entity at its current position.
  const _refreshCollider=(id,spec)=>{
    const e=_store.get(id); if(!e) return;
    const old=e.props.get('_collider');
    if(old){
      const i=_obs.indexOf(old); if(i>=0) _obs.splice(i,1);
      const oi=e.obs.indexOf(old); if(oi>=0) e.obs.splice(oi,1);
      e.props.delete('_collider');
    }
    if(!spec) return;
    const g=e.props.get('_group');
    const cx=g?g.position.x:0, cz=g?g.position.z:0;
    const w=spec.w||1, d=spec.d||1;
    const ob={t:'b',x1:cx-w/2,x2:cx+w/2,z1:cz-d/2,z2:cz+d/2};
    _obs.push(ob); _ownObs(id,ob);
    e.props.set('_collider',ob);
  };

  // ── 1. prefab ──────────────────────────────────────────────────────
  _compilers.set('prefab',(id,val)=>{
    const e=_store.get(id); if(!e) return;
    const fn=_prefabDefs[val]; if(!fn) return;
    const g=_ensureGroup(id);
    const sc=e.props.get('scale')||1;
    const meshes=fn(g,0,0,0,sc);
    if(meshes) meshes.forEach(m=>_own(id,m));
  });

  // ── 2. geometry — create a primary mesh of basic shape ─────────────
  _compilers.set('geometry',(id,val)=>{
    const e=_store.get(id); if(!e||!val) return;
    const g=_ensureGroup(id); if(!g) return;
    // Remove previous primary mesh (if any)
    const oldMesh=e.props.get('_primaryMesh');
    if(oldMesh){
      g.remove(oldMesh);
      if(oldMesh.geometry) oldMesh.geometry.dispose();
      if(oldMesh.material) oldMesh.material.dispose();
      const idx=e.meshes.indexOf(oldMesh);
      if(idx>=0) e.meshes.splice(idx,1);
    }
    let geo;
    switch(val.kind){
      case 'box':    geo=new THREE.BoxGeometry(_num(val.w)||1, _num(val.h)||1, _num(val.d)||1); break;
      case 'cyl':    geo=new THREE.CylinderGeometry(val.rt!=null?_num(val.rt):0.5, val.rb!=null?_num(val.rb):0.5, _num(val.h)||1, _num(val.seg)||12); break;
      case 'sphere': geo=new THREE.SphereGeometry(_num(val.r)||0.5, _num(val.ws)||16, _num(val.hs)||12); break;
      case 'plane':  geo=new THREE.PlaneGeometry(_num(val.w)||1, _num(val.h)||1); break;
      default: console.warn('Unknown geometry kind:',val.kind); return;
    }
    const mat=_materialFor(e.props.get('material'));
    const mesh=new THREE.Mesh(geo,mat);
    if(val.kind==='plane') mesh.rotation.x=-Math.PI/2;
    mesh.castShadow=mesh.receiveShadow=true;
    g.add(mesh);
    e.props.set('_primaryMesh',mesh);
    _own(id,mesh);
  });

  // ── 3. material — replace primary mesh's material ─────────────────
  _compilers.set('material',(id,val)=>{
    const e=_store.get(id); if(!e) return;
    const mesh=e.props.get('_primaryMesh');
    if(!mesh) return;  // will be picked up next geometry SET
    if(mesh.material) mesh.material.dispose();
    mesh.material=_materialFor(val);
  });

  // ── 4. position / position.y ──────────────────────────────────────
  _compilers.set('position',(id,val)=>{
    const g=_ensureGroup(id);
    if(g&&val) g.position.set(_num(val.x),_num(val.y),_num(val.z));
    const e=_store.get(id);
    const colSpec=e&&e.props.get('collider');
    if(colSpec) _refreshCollider(id,colSpec);
  });
  _compilers.set('position.y',(id,val)=>{
    const g=_ensureGroup(id);
    if(g) g.position.y=_num(val);
  });

  // ── 5. rotation ────────────────────────────────────────────────────
  _compilers.set('rotation',(id,val)=>{
    const g=_ensureGroup(id);
    if(g&&val) g.rotation.set(_num(val.x),_num(val.y),_num(val.z));
    // Note: AABB collider stays axis-aligned; rotated visuals may exceed it.
  });

  // ── 6. scale ───────────────────────────────────────────────────────
  _compilers.set('scale',(id,val)=>{
    const g=_ensureGroup(id);
    if(g) g.scale.setScalar(_num(val)||1);
  });

  // ── 7. visible ─────────────────────────────────────────────────────
  _compilers.set('visible',(id,val)=>{
    const g=_ensureGroup(id);
    if(g) g.visible=val;
  });

  // ── 8. collider — register/remove an AABB ─────────────────────────
  _compilers.set('collider',(id,val)=>{
    _refreshCollider(id,val);
  });

  // ── 9. parent — attach group under another entity's group ─────────
  _compilers.set('parent',(id,val)=>{
    const e=_store.get(id); if(!e) return;
    const g=_ensureGroup(id); if(!g) return;
    if(g.parent) g.parent.remove(g);
    if(val==null){ scene.add(g); return; }
    const parentE=_store.get(val);
    if(!parentE){ scene.add(g); return; }
    const parentG=_ensureGroup(val);  // create lazily if needed
    parentG.add(g);
  });

  // ── 10. light — create or update a light source ───────────────────
  // First SET creates the light (kind required); later SETs without kind
  // mutate the existing light's fields in place (color/intensity/etc).
  _compilers.set('light',(id,val)=>{
    const e=_store.get(id); if(!e||!val) return;
    const existing=e.props.get('_light');

    // Update in place when no kind is provided and a light already exists
    if(existing&&!val.kind){
      if(val.color!=null&&existing.color) existing.color.set(val.color);
      if(val.groundColor!=null&&existing.groundColor) existing.groundColor.set(val.groundColor);
      if(val.intensity!=null) existing.intensity=_num(val.intensity);
      if(val.distance!=null&&'distance' in existing) existing.distance=_num(val.distance);
      if(val.decay!=null&&'decay' in existing) existing.decay=_num(val.decay);
      if(val.angle!=null&&'angle' in existing) existing.angle=_num(val.angle);
      if(val.penumbra!=null&&'penumbra' in existing) existing.penumbra=_num(val.penumbra);
      return;
    }

    // Otherwise (re)create
    if(existing){
      if(existing.parent) existing.parent.remove(existing);
      const idx=e.meshes.indexOf(existing);
      if(idx>=0) e.meshes.splice(idx,1);
    }
    const g=_ensureGroup(id); if(!g) return;
    const col=val.color!=null?val.color:0xFFFFFF;
    const intensity=_num(val.intensity!=null?val.intensity:1);
    let light;
    switch(val.kind){
      case 'ambient':
        light=new THREE.AmbientLight(col,intensity);
        break;
      case 'hemisphere':
        light=new THREE.HemisphereLight(col,val.groundColor!=null?val.groundColor:0x444444,intensity);
        break;
      case 'directional':
        light=new THREE.DirectionalLight(col,intensity);
        break;
      case 'point':
        light=new THREE.PointLight(col,intensity,_num(val.distance)||0,val.decay!=null?_num(val.decay):2);
        break;
      case 'spot':
        light=new THREE.SpotLight(col,intensity,_num(val.distance)||0,_num(val.angle)||Math.PI/3,_num(val.penumbra)||0,val.decay!=null?_num(val.decay):2);
        break;
      default:
        console.warn('Unknown light kind:',val.kind);
        return;
    }
    g.add(light);
    e.props.set('_light',light);
    _own(id,light);
  });

  let _prefabDefs={};

  _on((type,id,key,val)=>{
    if(type==='set'){
      const c=_compilers.get(key);
      if(c) c(id,val);
    }
  });

  // ═══════════════════════════════════════════════════════════════════
  // Voxel system — instanced rendering, sparse storage
  // ═══════════════════════════════════════════════════════════════════
  const _voxels=new Map();         // "i,j,k" → material name
  const _voxelMeshes=new Map();    // material → {mesh, count, slotKeys, keyToSlot}
  const _vKey=(i,j,k)=>i+','+j+','+k;

  // Lazily create the InstancedMesh for a material; register with exempts so
  // it skips the stencil zone classification (top-level scene child only).
  const _initVoxelMaterial=(matName)=>{
    if(_voxelMeshes.has(matName)) return _voxelMeshes.get(matName);
    const spec=VOXEL_MATERIALS[matName];
    if(!spec){ console.warn('Unknown voxel material:',matName); return null; }
    const geo=new THREE.BoxGeometry(VOXEL_SIZE,VOXEL_SIZE,VOXEL_SIZE);
    const mat=new THREE.MeshStandardMaterial({
      color:spec.color,
      roughness:spec.roughness!=null?spec.roughness:0.85,
      metalness:spec.metalness!=null?spec.metalness:0.05,
      transparent:!!spec.transparent,
      opacity:spec.opacity!=null?spec.opacity:1,
    });
    const mesh=new THREE.InstancedMesh(geo,mat,VOXEL_MAX_PER_MATERIAL);
    mesh.count=0;
    mesh.castShadow=mesh.receiveShadow=true;
    mesh.frustumCulled=false;  // bounds don't reflect per-instance positions
    scene.add(mesh);
    _exempts.push(mesh);  // skip pass-1/pass-3 of stencil render
    const entry={mesh, count:0, slotKeys:[], keyToSlot:new Map()};
    _voxelMeshes.set(matName,entry);
    return entry;
  };

  // Set/clear a single voxel. matName=null clears.
  const _voxelSetSingle=(i,j,k,matName)=>{
    const key=_vKey(i,j,k);
    const oldMat=_voxels.get(key);
    if(oldMat===matName) return;       // same material → no-op
    if(oldMat===undefined&&!matName) return;  // clearing empty → no-op
    // Refuse voxel placement inside the orb's protected zone.
    // Clearing (matName=null) is still allowed — lets the LLM tidy up
    // anything that snuck in via legacy paths.
    if(matName){
      const cx=(i+0.5)*VOXEL_SIZE-SPAWN_X;
      const cz=(k+0.5)*VOXEL_SIZE-SPAWN_Z;
      if(cx*cx+cz*cz<ORB_ZONE_R2) return;
    }

    // Remove from old material if any
    if(oldMat){
      const oldEntry=_voxelMeshes.get(oldMat);
      if(oldEntry){
        const slot=oldEntry.keyToSlot.get(key);
        if(slot!=null){
          const last=oldEntry.count-1;
          if(slot!==last){
            // Swap last instance into freed slot
            const m=new THREE.Matrix4();
            oldEntry.mesh.getMatrixAt(last,m);
            oldEntry.mesh.setMatrixAt(slot,m);
            const lastKey=oldEntry.slotKeys[last];
            oldEntry.slotKeys[slot]=lastKey;
            oldEntry.keyToSlot.set(lastKey,slot);
          }
          oldEntry.slotKeys.pop();
          oldEntry.keyToSlot.delete(key);
          oldEntry.count--;
          oldEntry.mesh.count=oldEntry.count;
          oldEntry.mesh.instanceMatrix.needsUpdate=true;
        }
      }
      _voxels.delete(key);
    }

    if(!matName) return;  // pure clear

    // Add to new material
    const entry=_initVoxelMaterial(matName);
    if(!entry) return;
    if(entry.count>=VOXEL_MAX_PER_MATERIAL){
      console.warn('Voxel material cap reached:',matName);
      return;
    }
    const slot=entry.count;
    const cx=(i+0.5)*VOXEL_SIZE;
    const cy=(j+0.5)*VOXEL_SIZE;
    const cz=(k+0.5)*VOXEL_SIZE;
    const m=new THREE.Matrix4().makeTranslation(cx,cy,cz);
    entry.mesh.setMatrixAt(slot,m);
    entry.slotKeys[slot]=key;
    entry.keyToSlot.set(key,slot);
    entry.count++;
    entry.mesh.count=entry.count;
    entry.mesh.instanceMatrix.needsUpdate=true;
    _voxels.set(key,matName);
  };

  // AABB-aligned box fill (inclusive both ends).
  const _voxelFill=(i0,j0,k0,i1,j1,k1,matName)=>{
    const lo_i=Math.min(i0,i1), hi_i=Math.max(i0,i1);
    const lo_j=Math.min(j0,j1), hi_j=Math.max(j0,j1);
    const lo_k=Math.min(k0,k1), hi_k=Math.max(k0,k1);
    for(let i=lo_i;i<=hi_i;i++)
      for(let j=lo_j;j<=hi_j;j++)
        for(let k=lo_k;k<=hi_k;k++)
          _voxelSetSingle(i,j,k,matName);
  };

  // Engine-side shape filling: ball / cyl / cone.
  const _voxelShape=(kind,params,matName)=>{
    if(!params) return;
    const cx=_num(params.cx), cy=_num(params.cy), cz=_num(params.cz);
    const r=_num(params.r);
    switch(kind){
      case 'ball':{
        const ri=Math.ceil(r);
        for(let di=-ri;di<=ri;di++)
          for(let dj=-ri;dj<=ri;dj++)
            for(let dk=-ri;dk<=ri;dk++)
              if(di*di+dj*dj+dk*dk<=r*r)
                _voxelSetSingle(cx+di,cy+dj,cz+dk,matName);
        break;
      }
      case 'cyl':{
        const h=_num(params.h), axis=params.axis||'y';
        const ri=Math.ceil(r), hi=Math.ceil(h/2);
        for(let dh=-hi;dh<=hi;dh++)
          for(let da=-ri;da<=ri;da++)
            for(let db=-ri;db<=ri;db++){
              if(da*da+db*db>r*r) continue;
              let i,j,k;
              if(axis==='y'){ i=cx+da; j=cy+dh; k=cz+db; }
              else if(axis==='x'){ i=cx+dh; j=cy+da; k=cz+db; }
              else { i=cx+da; j=cy+db; k=cz+dh; }
              _voxelSetSingle(i,j,k,matName);
            }
        break;
      }
      case 'cone':{
        // Tip in +axis direction; base at center.
        const h=_num(params.h), axis=params.axis||'y';
        const hi=Math.ceil(h);
        for(let dh=0;dh<=hi;dh++){
          const rH=r*(1-dh/h);
          if(rH<0) continue;
          const ri=Math.ceil(rH);
          for(let da=-ri;da<=ri;da++)
            for(let db=-ri;db<=ri;db++){
              if(da*da+db*db>rH*rH) continue;
              let i,j,k;
              if(axis==='y'){ i=cx+da; j=cy+dh; k=cz+db; }
              else if(axis==='x'){ i=cx+dh; j=cy+da; k=cz+db; }
              else { i=cx+da; j=cy+db; k=cz+dh; }
              _voxelSetSingle(i,j,k,matName);
            }
        }
        break;
      }
      default:
        console.warn('Unknown voxel shape:',kind);
    }
  };

  // Compact summary for inclusion in snapshot.
  const _voxelSummary=()=>{
    if(_voxels.size===0) return null;
    const perMat={};
    let imn=Infinity,imx=-Infinity,jmn=Infinity,jmx=-Infinity,kmn=Infinity,kmx=-Infinity;
    for(const [k,m] of _voxels){
      perMat[m]=(perMat[m]||0)+1;
      const [i,j,kk]=k.split(',').map(Number);
      if(i<imn)imn=i; if(i>imx)imx=i;
      if(j<jmn)jmn=j; if(j>jmx)jmx=j;
      if(kk<kmn)kmn=kk; if(kk>kmx)kmx=kk;
    }
    return {
      total:_voxels.size,
      per_material:perMat,
      bbox_voxel:[[imn,jmn,kmn],[imx,jmx,kmx]],
    };
  };

  // ── Edge helpers ───────────────────────────────────────────────────
  const _eType=e=>!e?'open':typeof e==='string'?e:(e.type||'wall');
  const _eGap=e=>(typeof e==='object'&&e.gap)?e.gap:null;
  const _eCustom=e=>typeof e==='object'&&e.custom;
  const _eHasCollision=e=>{const t=_eType(e);return t!=='open';};
  const _eHasPortal=e=>{const t=_eType(e);return['glass','window','door'].includes(t);};

  // ── Stencil Render Core (from ViewEngine) ──────────────────────────
  const _b=[];
  const _zIn=new THREE.Group();  _zIn.name='zoneIn';
  const _zOut=new THREE.Group(); _zOut.name='zoneOut';
  const _oS=new THREE.Group();   _oS.name='outS';
  const _oN=new THREE.Group();   _oN.name='outN';
  _zOut.add(_oS,_oN);
  const _pg=new THREE.Group();   _pg.name='portals'; _pg.visible=false;
  const _ext=new THREE.Group();  _ext.name='exterior';
  let _spZ=9.0,_chZ=6.0;
  const _pm=new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:false,side:THREE.DoubleSide});
  const _inBounds=(x,z,bs)=>bs.some(([x1,x2,z1,z2])=>x>=x1&&x<=x2&&z>=z1&&z<=z2);

  const _addVEBuilding=({id,bounds,portals=[]})=>{
    const b={id,bounds,zone:new THREE.Group(),pg:new THREE.Group(),
             shell:new THREE.Group(),walls:{n:[],s:[],e:[],w:[]}};
    b.zone.name='bldg_'+id; b.pg.name='portal_'+id; b.shell.name='shell_'+id;
    _zIn.add(b.zone); _pg.add(b.pg); _ext.add(b.shell);
    portals.forEach(p=>{
      const m=new THREE.Mesh(new THREE.PlaneGeometry(p.w,p.h),_pm);
      m.position.set(p.x,p.y,p.z); if(p.ry) m.rotation.y=p.ry;
      m.frustumCulled=false; b.pg.add(m);
    });
    _b.push(b); return b;
  };

  const _getBldgAt=(x,z)=>_b.find(v=>_inBounds(x,z,v.bounds))||null;

  // ── Terrain definitions ────────────────────────────────────────────
  const _T={
    sidewalk:{col:0xCCB070,r:0.93,m:0.01,th:0.12,yo:-0.07,
      joint:{col:0xAA8850,hN:3},curb:true,
      lamp:{spacing:6.1,col:0xFFAA44,emi:0xFF9900},
      tree:{spacing:6.1,pairs:true}},
    road:{col:0x3A3230,r:0.96,m:0.03,th:0.10,yo:-0.05,
      dash:{col:0xF0E8C0,sp:1.5,len:0.70},edge:{col:0xF0E8C0},manhole:{spacing:7.5}},
    courtyard:{col:0xC07040,r:0.92,m:0.02,th:0.14,yo:-0.07,grout:{col:0x7A4A28,grid:3.04}},
    'wood-floor':{col:0xBE9868,r:0.86,m:0.04,th:0.08,yo:0.04},
    'shop-floor':{col:0xD8C8A8,r:0.90,m:0.02,th:0.12,yo:-0.06},
    snow:{col:0xEDF1F6,r:0.98,m:0.00,th:0.18,yo:0.09},
    dirt:{col:0xD4C4AA,r:0.78,m:0.06,th:0.10,yo:-0.05,grout:{col:0xA09070,grid:1.22}},
  };

  // ── Default materials ──────────────────────────────────────────────
  const _dWall=_M(0xE8DCC8,0.86,0.02);
  const _dKick=_M(0xC0A880,0.90,0.02);
  const _dFrame=_M(0x2A2220,0.78,0.14);
  const _dGlass=new THREE.MeshStandardMaterial({color:0xBEDCF2,roughness:0.04,
    metalness:0.22,transparent:true,opacity:0.24,side:THREE.DoubleSide,depthWrite:false});

  const W={
    S, WH,
    setOutdoorSplit(cz,chz){_spZ=cz;_chZ=chz;},

    add(desc){
      const{id,cells:rawCells,gx,gy,enclosed=false,group=null,edges={},
            terrain=null,label=null,cellLabels=null}=desc;
      let cells=rawCells||[];
      if(!rawCells){
        const gxs=Array.isArray(gx)?gx:[gx],gys=Array.isArray(gy)?gy:[gy];
        let ci=0;
        gxs.forEach(x=>gys.forEach(y=>{
          cells.push({gx:x,gy:y,label:cellLabels?cellLabels[ci]:null}); ci++;
        }));
      }
      _exec('EXIST',id);
      cells.forEach(c=>_exec('SET',id,'cell.'+c.gx+','+c.gy,true));
      if(terrain) _exec('SET',id,'terrain',terrain);
      if(enclosed) _exec('SET',id,'enclosed',true);
      for(const[dir,spec]of Object.entries(edges)) _exec('SET',id,'edges.'+dir,spec);
      if(group) _exec('RELATE',id,'group',group);

      const d={id,cells,enclosed,group,edges,terrain,label:label||String(id),built:true};
      _reg.set(id,d);
      cells.forEach(c=>_grid.set(c.gx+','+c.gy,
        {...d,id:c.label||d.id,label:c.label?String(c.label):d.label}));
      return W;
    },

    addEmpty(gx,gy,id){
      _exec('EXIST',id);
      _exec('SET',id,'cell.'+gx+','+gy,true);
      _exec('SET',id,'built',false);
      const d={id,cells:[{gx,gy}],enclosed:false,group:null,edges:{},terrain:null,
               label:String(id),built:false};
      _reg.set(id,d);_grid.set(gx+','+gy,d);return W;
    },

    build(id,fn){_builds.set(id,fn);return W;},

    // ── Compile: generate everything from declarations ────────────────
    compile(){
      const grps=new Map();
      _reg.forEach(d=>{
        if(!d.group||!d.built) return;
        if(!grps.has(d.group)) grps.set(d.group,{bounds:[],portals:[],enclosed:d.enclosed});
        const g=grps.get(d.group);
        if(d.enclosed) g.enclosed=true;
        d.cells.forEach(({gx,gy})=>{
          const wx=gx*S,wz=gy*S;
          const MR=0.12;
          g.bounds.push([wx-S/2-MR,wx+S/2+MR,wz-S/2-MR,wz+S/2+MR]);
        });
      });

      grps.forEach((g,gid)=>{_addVEBuilding({id:gid,bounds:g.bounds,portals:[]});});

      _reg.forEach(d=>{
        if(!d.built||!d.edges) return;
        const{edges}=d;
        const _pushObs=(ob)=>{_obs.push(ob);_ownObs(d.id,ob);};
        const _addOwned=(mesh)=>{_own(d.id,mesh);return mesh;};
        let x1=Infinity,x2=-Infinity,z1=Infinity,z2=-Infinity;
        d.cells.forEach(({gx,gy})=>{
          const wx=gx*S,wz=gy*S;
          x1=Math.min(x1,wx-S/2);x2=Math.max(x2,wx+S/2);
          z1=Math.min(z1,wz-S/2);z2=Math.max(z2,wz+S/2);
        });
        const cx=(x1+x2)/2,cz=(z1+z2)/2,bW=x2-x1,bD=z2-z1;
        const HT=0.15;

        const sides=[
          {dir:'n',ax:'z',pos:z2,len:bW,ctr:cx,dep:bD},
          {dir:'s',ax:'z',pos:z1,len:bW,ctr:cx,dep:bD},
          {dir:'e',ax:'x',pos:x2,len:bD,ctr:cz,dep:bW},
          {dir:'w',ax:'x',pos:x1,len:bD,ctr:cz,dep:bW},
        ];

        const vb=d.group?_b.find(v=>v.id===d.group):null;

        sides.forEach(s=>{
          const e=edges[s.dir]; if(!e||_eType(e)==='open') return;
          const gap=_eGap(e)||((_eType(e)==='door')?[-0.9,0.9]:null);
          const portalH=_eType(e)==='window'?2.9:WH;
          const portalY=_eType(e)==='window'?2.15:WH/2;
          const portalW=gap?(gap[1]-gap[0]):s.len;
          const portalCtr=gap?(s.ctr):s.ctr;

          if(s.ax==='z'){
            if(gap){
              const gc=s.ctr;
              if(gc+gap[0]>x1+0.1) _obs.push({t:'b',x1:x1-HT,x2:gc+gap[0],z1:s.pos-HT,z2:s.pos+HT});
              if(gc+gap[1]<x2-0.1) _obs.push({t:'b',x1:gc+gap[1],x2:x2+HT,z1:s.pos-HT,z2:s.pos+HT});
            } else _obs.push({t:'b',x1:x1-HT,x2:x2+HT,z1:s.pos-HT,z2:s.pos+HT});
          } else {
            if(gap){
              const gc=s.ctr;
              if(gc+gap[0]>z1+0.1) _obs.push({t:'b',x1:s.pos-HT,x2:s.pos+HT,z1:z1-HT,z2:gc+gap[0]});
              if(gc+gap[1]<z2-0.1) _obs.push({t:'b',x1:s.pos-HT,x2:s.pos+HT,z1:gc+gap[1],z2:z2+HT});
            } else _obs.push({t:'b',x1:s.pos-HT,x2:s.pos+HT,z1:z1-HT,z2:z2+HT});
          }

          if(_eHasPortal(e)&&vb){
            const pp=s.ax==='z'
              ?{w:portalW,h:portalH,x:portalCtr,y:portalY,z:s.pos}
              :{w:portalW,h:portalH,x:s.pos,y:portalY,z:portalCtr,ry:Math.PI/2};
            const m=new THREE.Mesh(new THREE.PlaneGeometry(pp.w,pp.h),_pm);
            m.position.set(pp.x,pp.y,pp.z); if(pp.ry) m.rotation.y=pp.ry;
            m.frustumCulled=false; vb.pg.add(m);
          }

          if(!_eCustom(e)){
            const meshes=[];
            const ty=_eType(e);
            const isHoriz=s.ax==='z';
            const wp=s.pos+(s.dir==='n'||s.dir==='e'?-0.07:0.07);

            if(ty==='wall'||ty==='door'||ty==='window'){
              if(ty==='door'&&gap){
                const gc=isHoriz?s.ctr:s.ctr;
                const halfL=s.len/2;
                const segA=gc+gap[0]-(isHoriz?x1:z1);
                const segB=(isHoriz?x2:z2)-(gc+gap[1]);
                if(isHoriz){
                  if(segA>0.2) meshes.push(_bx(segA,WH,0.14,_dWall,{x:(x1+gc+gap[0])/2,y:WH/2,z:wp}));
                  if(segB>0.2) meshes.push(_bx(segB,WH,0.14,_dWall,{x:(gc+gap[1]+x2)/2,y:WH/2,z:wp}));
                  meshes.push(_bx(gap[1]-gap[0],WH-2.6,0.14,_dWall,{x:gc,y:2.6+(WH-2.6)/2,z:wp}));
                } else {
                  if(segA>0.2) meshes.push(_bx(0.14,WH,segA,_dWall,{x:wp,y:WH/2,z:(z1+gc+gap[0])/2}));
                  if(segB>0.2) meshes.push(_bx(0.14,WH,segB,_dWall,{x:wp,y:WH/2,z:(gc+gap[1]+z2)/2}));
                  meshes.push(_bx(0.14,WH-2.6,gap[1]-gap[0],_dWall,{x:wp,y:2.6+(WH-2.6)/2,z:gc}));
                }
                const dw=gap[1]-gap[0];
                if(isHoriz){
                  meshes.push(_bx(dw+0.12,0.08,0.12,_dFrame,{x:gc,y:2.64,z:wp}));
                  meshes.push(_bx(0.08,2.6,0.12,_dFrame,{x:gc+gap[0]-0.04,y:1.3,z:wp}));
                  meshes.push(_bx(0.08,2.6,0.12,_dFrame,{x:gc+gap[1]+0.04,y:1.3,z:wp}));
                } else {
                  meshes.push(_bx(0.12,0.08,dw+0.12,_dFrame,{x:wp,y:2.64,z:gc}));
                  meshes.push(_bx(0.12,2.6,0.08,_dFrame,{x:wp,y:1.3,z:gc+gap[0]-0.04}));
                  meshes.push(_bx(0.12,2.6,0.08,_dFrame,{x:wp,y:1.3,z:gc+gap[1]+0.04}));
                }
              } else if(ty==='window'){
                if(isHoriz){
                  meshes.push(_bx(s.len,0.7,0.14,_dWall,{x:cx,y:0.35,z:wp}));
                  meshes.push(_bx(s.len,0.6,0.14,_dWall,{x:cx,y:WH-0.3,z:wp}));
                  const sw=(s.len-s.len*0.72)/2;
                  meshes.push(_bx(sw,WH,0.14,_dWall,{x:x1+sw/2,y:WH/2,z:wp}));
                  meshes.push(_bx(sw,WH,0.14,_dWall,{x:x2-sw/2,y:WH/2,z:wp}));
                } else {
                  meshes.push(_bx(0.14,0.7,s.len,_dWall,{x:wp,y:0.35,z:cz}));
                  meshes.push(_bx(0.14,0.6,s.len,_dWall,{x:wp,y:WH-0.3,z:cz}));
                  const sw=(s.len-s.len*0.72)/2;
                  meshes.push(_bx(0.14,WH,sw,_dWall,{x:wp,y:WH/2,z:z1+sw/2}));
                  meshes.push(_bx(0.14,WH,sw,_dWall,{x:wp,y:WH/2,z:z2-sw/2}));
                }
              } else {
                if(isHoriz) meshes.push(_bx(s.len,WH,0.14,_dWall,{x:cx,y:WH/2,z:wp}));
                else meshes.push(_bx(0.14,WH,s.len,_dWall,{x:wp,y:WH/2,z:cz}));
              }
              if(isHoriz) meshes.push(_bx(s.len,0.26,0.16,_dKick,{x:cx,y:0.13,z:wp}));
              else meshes.push(_bx(0.16,0.26,s.len,_dKick,{x:wp,y:0.13,z:cz}));
            } else if(ty==='glass'){
              const fM=_dFrame;
              if(isHoriz){
                meshes.push(_bx(s.len,0.10,0.09,fM,{x:cx,y:WH-0.05,z:wp}));
                meshes.push(_bx(s.len,0.14,0.13,fM,{x:cx,y:0.07,z:wp}));
                const nP=Math.max(1,Math.round(s.len/S));
                const pW=s.len/nP;
                for(let i=0;i<nP;i++){
                  const px=x1+pW*(i+0.5);
                  meshes.push(_bx(0.09,WH,0.09,fM,{x:x1+pW*i,y:WH/2,z:wp}));
                  meshes.push(_bx(pW-0.18,WH-0.30,0.04,_dGlass,{x:px,y:WH/2,z:wp}));
                }
                meshes.push(_bx(0.09,WH,0.09,fM,{x:x2,y:WH/2,z:wp}));
              } else {
                meshes.push(_bx(0.09,0.10,s.len,fM,{x:wp,y:WH-0.05,z:cz}));
                meshes.push(_bx(0.13,0.14,s.len,fM,{x:wp,y:0.07,z:cz}));
                const nP=Math.max(1,Math.round(s.len/S));
                const pW=s.len/nP;
                for(let i=0;i<nP;i++){
                  const pz=z1+pW*(i+0.5);
                  meshes.push(_bx(0.09,WH,0.09,fM,{x:wp,y:WH/2,z:z1+pW*i}));
                  meshes.push(_bx(0.04,WH-0.30,pW-0.18,_dGlass,{x:wp,y:WH/2,z:pz}));
                }
                meshes.push(_bx(0.09,WH,0.09,fM,{x:wp,y:WH/2,z:z2}));
              }
            }

            if(meshes.length){
              _ad(scene,...meshes);
              if(vb) vb.walls[s.dir].push(...meshes);
            }
          }
        });

        if(d.enclosed&&d.group){
          const vb2=_b.find(v=>v.id===d.group);
          if(vb2){
            const RY=WH+0.06;
            const bsM=_M(0x2E1A10,0.94,0.02),tMa=_M(0x8A5A3A,0.84,0.02),tMb=_M(0x6A3E24,0.88,0.02);
            const ribM=_M(0x5A2E14,0.80,0.06),evM=_M(0x4A2818,0.78,0.08),capM=_M(0x6A4030,0.82,0.04);
            const rW=bW+0.30,rD=bD+0.30;
            _ad(vb2.shell,_bx(rW,0.10,rD,bsM,{x:cx,y:RY,z:cz}));
            const step=0.38;
            for(let i=0;i<Math.ceil(rD/step);i++){
              const tz=cz-rD/2+step*(i+0.5); if(tz>cz+rD/2) break;
              _ad(vb2.shell,_bx(rW-0.08,0.04,step*0.86,i%2===0?tMa:tMb,
                {x:cx+(i%2)*0.04,y:RY+0.06+(i%2)*0.014,z:tz}));
            }
            const rN=Math.round(rW/0.48);
            for(let i=0;i<=rN;i++) _ad(vb2.shell,_bx(0.06,0.06,rD+0.06,ribM,
              {x:cx-rW/2+(rW/rN)*i,y:RY+0.09,z:cz}));
            const O=0.25;
            _ad(vb2.shell,_bx(rW+O*2,0.11,0.22,evM,{x:cx,y:RY+0.01,z:cz-rD/2-0.08}));
            _ad(vb2.shell,_bx(rW+O*2,0.11,0.22,evM,{x:cx,y:RY+0.01,z:cz+rD/2+0.08}));
            _ad(vb2.shell,_bx(0.22,0.11,rD+O*2,evM,{x:cx-rW/2-0.08,y:RY+0.01,z:cz}));
            _ad(vb2.shell,_bx(0.22,0.11,rD+O*2,evM,{x:cx+rW/2+0.08,y:RY+0.01,z:cz}));
            _ad(vb2.shell,_bx(rW+0.10,0.05,0.12,capM,{x:cx,y:RY+0.13,z:cz}));
            _ad(vb2.shell,_bx(0.12,0.05,rD+0.10,capM,{x:cx,y:RY+0.13,z:cz}));
          }
        }
      });

      // ── Terrain ────────────────────────────────────────────────────
      {
        const strips=new Map();
        _grid.forEach((d,key)=>{
          if(!d.terrain) return;
          const[gx,gy]=key.split(',').map(Number);
          const k=d.terrain+'|'+gy;
          const wx=gx*S,wz=gy*S;
          if(!strips.has(k)) strips.set(k,{t:d.terrain,gy,
            x1:wx-S/2,x2:wx+S/2,z1:wz-S/2,z2:wz+S/2});
          else{const s=strips.get(k);
            s.x1=Math.min(s.x1,wx-S/2);s.x2=Math.max(s.x2,wx+S/2);
            s.z1=Math.min(s.z1,wz-S/2);s.z2=Math.max(s.z2,wz+S/2);}
        });
        strips.forEach(s=>{
          const def=_T[s.t]; if(!def) return;
          const sW=s.x2-s.x1,sD=s.z2-s.z1,scx=(s.x1+s.x2)/2,scz=(s.z1+s.z2)/2;
          _ad(scene,_bx(sW,def.th,sD,_M(def.col,def.r,def.m),{x:scx,y:def.yo,z:scz}));
          if(def.joint){
            const jM2=_M(def.joint.col,0.96);
            for(let i=0;i<(def.joint.hN||3);i++){
              const jz=s.z1+(i+1)*(sD/((def.joint.hN||3)+1));
              _ad(scene,_bx(sW,0.008,0.038,jM2,{x:scx,y:0.07,z:jz}));
            }
            for(let gx2=Math.ceil(s.x1/S);gx2<=Math.floor(s.x2/S);gx2++){
              const jx=gx2*S;
              if(jx>s.x1+0.5&&jx<s.x2-0.5)
                _ad(scene,_bx(0.038,0.008,sD,jM2,{x:jx,y:0.07,z:scz}));
            }
          }
          if(def.curb){
            const adjS=strips.get('road|'+(s.gy-1)),adjN=strips.get('road|'+(s.gy+1));
            const cM=_M(0xA08060,0.90);
            if(adjS) _ad(scene,_bx(sW,0.16,0.15,cM,{x:scx,y:0.08,z:s.z1-0.07}));
            if(adjN) _ad(scene,_bx(sW,0.16,0.15,cM,{x:scx,y:0.08,z:s.z2+0.07}));
          }
          if(def.lamp){
            const adjS=strips.get('road|'+(s.gy-1)),adjN=strips.get('road|'+(s.gy+1));
            const placeLamps=(lz,dir)=>{
              for(let lx=s.x1+def.lamp.spacing/2;lx<=s.x2-1;lx+=def.lamp.spacing){
                const pmM=_M(0x252020,0.70,0.28);
                const lhM=new THREE.MeshStandardMaterial({color:def.lamp.col,
                  emissive:new THREE.Color(def.lamp.emi),emissiveIntensity:1.8,roughness:0.4});
                _ad(scene,_cy(0.05,0.07,3.6,8,pmM,{x:lx,y:1.80,z:lz}));
                _ad(scene,_bx(0.04,0.04,0.88,pmM,{x:lx,y:3.60,z:lz+dir*0.44}));
                _ad(scene,_bx(0.22,0.14,0.24,lhM,{x:lx,y:3.56,z:lz+dir*0.88}));
                _ad(scene,_cy(0.12,0.15,0.06,8,pmM,{x:lx,y:3.65,z:lz+dir*0.88}));
                const pl=new THREE.PointLight(0xFF9030,0.9,8.5,2);
                pl.position.set(lx,3.4,lz+dir*0.88);scene.add(pl);
              }
            };
            if(adjS) placeLamps(s.z1+0.6,-1);
            if(adjN) placeLamps(s.z2-0.6,+1);
          }
          if(def.tree){
            const adjS=strips.get('road|'+(s.gy-1)),adjN=strips.get('road|'+(s.gy+1));
            const placeTree=(tx,tz)=>{
              _ad(scene,_bx(0.62,0.50,0.62,_M(0x8C7060,0.88,0.04),{x:tx,y:0.25,z:tz}));
              _ad(scene,_cy(0.28,0.28,0.02,10,_M(0x2C1C0C,0.98),{x:tx,y:0.51,z:tz}));
              _ad(scene,_cy(0.065,0.11,0.82,8,_M(0x4A2C14,0.90),{x:tx,y:0.93,z:tz}));
              for(const[fx,fy,fz,fr,fc]of[[tx,1.62,tz,0.38,0xCC3808],[tx+0.28,1.66,tz+0.10,0.26,0xFF5510],
                [tx-0.22,1.56,tz-0.12,0.23,0xAA2200],[tx+0.10,1.88,tz+0.06,0.19,0xDD4010]])
                _ad(scene,_sp(fr,12,10,_M(fc,0.82),{x:fx,y:fy,z:fz}));
            };
            const placeTrees=(bz)=>{
              for(let tx=s.x1+def.tree.spacing/2;tx<=s.x2-1;tx+=def.tree.spacing){
                placeTree(tx-0.5,bz);if(def.tree.pairs)placeTree(tx+0.3,bz-0.7);
              }
            };
            if(adjS) placeTrees(s.z1+2.1);
            if(adjN) placeTrees(s.z2-2.1);
          }
          if(def.dash){
            const dM=_M(def.dash.col,0.86);
            for(let mx=s.x1+0.8;mx<=s.x2-0.4;mx+=def.dash.sp)
              _ad(scene,_bx(def.dash.len,0.012,0.12,dM,{x:mx,y:0.06,z:scz}));
          }
          if(def.edge){
            const eM=_M(def.edge.col,0.86);
            _ad(scene,_bx(sW,0.010,0.10,eM,{x:scx,y:0.06,z:s.z1+0.17}));
            _ad(scene,_bx(sW,0.010,0.10,eM,{x:scx,y:0.06,z:s.z2-0.17}));
            const adjN=strips.get('road|'+(s.gy+1));
            if(adjN){
              const dY=_M(0xF0D060,0.86);
              _ad(scene,_bx(sW,0.012,0.08,dY,{x:scx,y:0.062,z:s.z2+0.03}));
              _ad(scene,_bx(sW,0.012,0.08,dY,{x:scx,y:0.062,z:s.z2+0.13}));
            }
          }
          if(def.manhole){
            const mhM=_M(0x5A5450,0.92,0.08);
            for(let mx=s.x1+def.manhole.spacing/2;mx<=s.x2;mx+=def.manhole.spacing)
              _ad(scene,_cy(0.28,0.28,0.02,12,mhM,{x:mx+((mx/7|0)%2)*1.5,y:0.06,z:scz+((mx/7|0)%2?-1:1)*1.3}));
          }
          if(def.grout){
            const grM=_M(def.grout.col,0.96);
            for(let dx=-Math.floor(sW/2/def.grout.grid);dx<=Math.floor(sW/2/def.grout.grid);dx++)
              _ad(scene,_bx(0.036,0.009,sD,grM,{x:scx+dx*def.grout.grid,y:0.072,z:scz}));
            for(let dz=-Math.floor(sD/2/def.grout.grid);dz<=Math.floor(sD/2/def.grout.grid);dz++)
              _ad(scene,_bx(sW,0.009,0.036,grM,{x:scx,y:0.072,z:scz+dz*def.grout.grid}));
          }
        });
      }

      // ── Prefab definitions (shared with compiler table) ─────────────
      _prefabDefs={
        'autumn-tree':(g,px,py,pz,scale=1)=>{
          const s=scale;
          const trunk=new THREE.Mesh(new THREE.CylinderGeometry(0.10*s,0.16*s,0.82*s,8),_M(0x4A2C14,0.90));
          trunk.position.set(px,py+0.41*s,pz);g.add(trunk);
          const cols=[0xCC3808,0xFF5510,0xAA2200,0xDD4010];
          const offs=[[0,1.2,0,0.38],[0.28,1.24,0.10,0.26],[-0.22,1.14,-.12,0.23],[0.10,1.46,0.06,0.19]];
          const meshes=[trunk];
          offs.forEach(([ox,oy,oz,r],i)=>{
            const lf=new THREE.Mesh(new THREE.SphereGeometry(r*s,12,10),_M(cols[i],0.82));
            lf.position.set(px+ox*s,py+oy*s,pz+oz*s);g.add(lf);meshes.push(lf);
          });
          return meshes;
        },
        'pine-tree':(g,px,py,pz,scale=1)=>{
          const s=scale;
          const trunk=new THREE.Mesh(new THREE.CylinderGeometry(0.06*s,0.10*s,0.9*s,8),_M(0x3A2410,0.92));
          trunk.position.set(px,py+0.45*s,pz);g.add(trunk);
          const meshes=[trunk];
          for(const[dy,r]of[[0.9,0.55],[1.3,0.42],[1.65,0.28]]){
            const c=new THREE.Mesh(new THREE.ConeGeometry(r*s,0.5*s,8),_M(0x1A4A20,0.88));
            c.position.set(px,py+dy*s,pz);g.add(c);meshes.push(c);
          }
          return meshes;
        },
        'lamp':(g,px,py,pz,armDir=-1)=>{
          const pmM=_M(0x252020,0.70,0.28);
          const lhM=new THREE.MeshStandardMaterial({color:0xFFAA44,
            emissive:new THREE.Color(0xFF9900),emissiveIntensity:1.8,roughness:0.4});
          const meshes=[];
          const mk=(m)=>{g.add(m);meshes.push(m);return m;};
          mk(new THREE.Mesh(new THREE.CylinderGeometry(0.05,0.07,3.6,8),pmM)).position.set(px,py+1.80,pz);
          mk(new THREE.Mesh(new THREE.BoxGeometry(0.04,0.04,0.88),pmM)).position.set(px,py+3.60,pz+armDir*0.44);
          mk(new THREE.Mesh(new THREE.BoxGeometry(0.22,0.14,0.24),lhM)).position.set(px,py+3.56,pz+armDir*0.88);
          mk(new THREE.Mesh(new THREE.CylinderGeometry(0.12,0.15,0.06,8),pmM)).position.set(px,py+3.65,pz+armDir*0.88);
          const pl=new THREE.PointLight(0xFF9030,0.9,8.5,2);
          pl.position.set(px+g.position.x,py+3.4,pz+g.position.z+armDir*0.88);scene.add(pl);
          return meshes;
        },
        'bench':(g,px,py,pz,faceDir=1)=>{
          const bM=_M(0x7A4422,0.86),lM=_M(0x5C3020,0.88);
          const mk=(m)=>{g.add(m);return m;};
          mk(new THREE.Mesh(new THREE.BoxGeometry(0.10,0.38,0.38),lM)).position.set(px-0.7,py+0.19,pz);
          mk(new THREE.Mesh(new THREE.BoxGeometry(0.10,0.38,0.38),lM)).position.set(px+0.7,py+0.19,pz);
          mk(new THREE.Mesh(new THREE.BoxGeometry(1.50,0.06,0.38),bM)).position.set(px,py+0.40,pz);
          mk(new THREE.Mesh(new THREE.BoxGeometry(1.50,0.40,0.06),bM)).position.set(px,py+0.62,pz+faceDir*0.18);
        },
        'boulder':(g,px,py,pz,scale=1)=>{
          const m=new THREE.Mesh(new THREE.SphereGeometry(0.5*scale,8,8),_M(0x8A7A68,0.92,0.02));
          m.position.set(px,py+0.3*scale,pz);m.scale.set(1.1,0.65,1.0);g.add(m);return[m];
        },
        'grass-tuft':(g,px,py,pz)=>{
          const m=new THREE.Mesh(new THREE.SphereGeometry(0.18,6,6),_M(0x5A7A30,0.90));
          m.position.set(px,py+0.12,pz);m.scale.set(1.4,0.6,1.4);g.add(m);return[m];
        },
      };

      // ── Build callbacks ─────────────────────────────────────────────
      _builds.forEach((fn,id)=>{
        const d=_reg.get(id); if(!d) return;
        let x1=Infinity,x2=-Infinity,z1=Infinity,z2=-Infinity;
        d.cells.forEach(({gx,gy})=>{
          const wx=gx*S,wz=gy*S;
          x1=Math.min(x1,wx-S/2);x2=Math.max(x2,wx+S/2);
          z1=Math.min(z1,wz-S/2);z2=Math.max(z2,wz+S/2);
        });
        const cx=(x1+x2)/2,cz=(z1+z2)/2;
        const g=new THREE.Group();g.position.set(cx,0,cz);scene.add(g);
        const bx=(w,h2,d2,mat,px=0,py=0,pz=0)=>{
          const m=new THREE.Mesh(new THREE.BoxGeometry(w,h2,d2),mat);
          m.position.set(px,py,pz);g.add(m);return m;};
        const cy=(rt,rb,h2,seg,mat,o={})=>{
          const m=new THREE.Mesh(new THREE.CylinderGeometry(rt,rb,h2,seg),mat);
          m.position.set(o.x||0,o.y||0,o.z||0);if(o.rx)m.rotation.x=o.rx;if(o.rz)m.rotation.z=o.rz;g.add(m);return m;};
        const sp=(r,s1,s2,mat,o={})=>{
          const m=new THREE.Mesh(new THREE.SphereGeometry(r,s1,s2),mat);
          m.position.set(o.x||0,o.y||0,o.z||0);if(o.sy)m.scale.y=o.sy;g.add(m);return m;};
        const addObs=(ob)=>_obs.push(ob);
        const collider=(shape,lx,lz,lx2,lz2)=>{
          if(shape==='box') _obs.push({t:'b',x1:cx+lx,x2:cx+lx2,z1:cz+lz,z2:cz+lz2});
          else if(shape==='wall-n') _obs.push({t:'b',x1:cx+lx,x2:cx+lx2,z1:cz+lz-0.1,z2:cz+lz+0.1});
          else if(shape==='wall-s') _obs.push({t:'b',x1:cx+lx,x2:cx+lx2,z1:cz+lz-0.1,z2:cz+lz+0.1});
          else if(shape==='wall-e') _obs.push({t:'b',x1:cx+lx-0.1,x2:cx+lx+0.1,z1:cz+lz,z2:cz+lz2});
          else if(shape==='wall-w') _obs.push({t:'b',x1:cx+lx-0.1,x2:cx+lx+0.1,z1:cz+lz,z2:cz+lz2});
        };
        const prefab=(name,lx=0,ly=0,lz=0,opts={})=>{
          const fn2=_prefabDefs[name];
          if(!fn2){console.warn('Unknown prefab:',name);return[];}
          return fn2(g,lx,ly,lz,opts.scale||1,opts.dir||0);
        };
        const vb=d.group?_b.find(v=>v.id===d.group):null;
        fn({g,bx,cy,sp,M:_M,add:_ad,scene,cx,cz,x1,x2,z1,z2,
            edges:d.edges,addCollision:addObs,collider,prefab,veBldg:vb,
            own:(mesh)=>_own(d.id,mesh), ownObs:(ob)=>_ownObs(d.id,ob),
            hideDefaultWall:(dir)=>{if(vb)vb.walls[dir].forEach(m=>m.visible=false);}});
      });

      // (Wireframe placeholders for unbuilt cells removed — voxel system
      // is now the primary build path. _wireframes Map stays empty; buildCell
      // still consults it harmlessly.)

      // ── Zone classification ─────────────────────────────────────────
      scene.add(_zIn,_zOut,_pg,_ext);
      const es=new Set(_exempts);
      const wp=new THREE.Vector3();
      [...scene.children].forEach(c=>{
        if(es.has(c)||c===_zIn||c===_zOut||c===_pg||c===_ext||c.isLight) return;
        c.getWorldPosition(wp);
        const bg=_b.find(v=>_inBounds(wp.x,wp.z,v.bounds));
        if(bg){bg.zone.add(c);return;}
        if(wp.z>_spZ) _oN.add(c); else _oS.add(c);
      });

      // ── Wall fade init ──────────────────────────────────────────────
      _b.forEach(b=>{
        b._wallPos={};
        for(const d of['n','s','e','w']){
          b.walls[d].forEach(m=>{
            m.material=m.material.clone();m.material.transparent=true;
            m._origOp=m.material.opacity||1;m._curOp=m._origOp;
          });
          if(b.walls[d].length>0){
            let sx=0,sz=0;
            b.walls[d].forEach(m=>{m.getWorldPosition(wp);sx+=wp.x;sz+=wp.z;});
            b._wallPos[d]={x:sx/b.walls[d].length,z:sz/b.walls[d].length};
          }
        }
      });
    },

    setExempts(list){_exempts.length=0;_exempts.push(...list);},
    getOBS(){return _obs;},
    isIndoor(x,z){return!!_getBldgAt(x,z);},

    // Voxel system (sparse instanced cubes)
    voxelSize: VOXEL_SIZE,
    voxelMaterials(){return Object.keys(VOXEL_MATERIALS);},
    voxelSet(i,j,k,mat){_voxelSetSingle(_num(i),_num(j),_num(k),mat||null);},
    voxelFill(i0,j0,k0,i1,j1,k1,mat){
      _voxelFill(_num(i0),_num(j0),_num(k0),_num(i1),_num(j1),_num(k1),mat||null);
    },
    voxelShape(kind,params,mat){_voxelShape(kind,params,mat||null);},
    voxelSummary(){return _voxelSummary();},
    // O(1) point query for collision; player only walks in xz plane so this
    // is called from resolve() with j=1..3 to ignore j=0 decoration layer.
    voxelHasAt(i,j,k){return _voxels.has(_vKey(i,j,k));},

    exec(op,...args){_exec(op,...args);},
    get(id,key){return _get(id,key);},
    getEntity(id){return _store.get(id)||null;},
    getStore(){return _store;},
    own(id,mesh){_own(id,mesh);},
    ownObs(id,ob){_ownObs(id,ob);},

    // ── Runtime cell building ─────────────────────────────────────────
    getEmptyCells(){
      const out=[];
      _grid.forEach((d,key)=>{
        if(!d.built){const[gx,gy]=key.split(',').map(Number);out.push({gx,gy,id:d.id});}
      });
      return out;
    },
    getCellBuilderTypes(){return Object.keys(_cellBuilders);},
    buildCell(gx,gy,type){
      // Block builds on the cell that contains the orb (center cell).
      // The orb sits at world center, which falls in cell (3,3).
      if(gx===3&&gy===3) return false;
      const key=gx+','+gy;
      const d=_grid.get(key);
      if(!d||d.built) return false;
      const builder=_cellBuilders[type];
      if(!builder) return false;
      const wf=_wireframes.get(key);
      if(wf){wf.forEach(m=>{if(m.parent)m.parent.remove(m);});_wireframes.delete(key);}
      const id='rt_cell_'+gx+'_'+gy;
      _exec('EXIST',id);
      _exec('SET',id,'cellType',type);
      _exec('SET',id,'position',{gx,gy});
      builder(id,gx,gy);
      d.built=true;
      return true;
    },

    updateWalls(cx,cz,ang,dt,camX,camZ){
      const cur=_getBldgAt(cx,cz);
      const sA=Math.sin(ang),cA=Math.cos(ang),TH=0.20,LO=0.06,SP=6,MG=2.0;
      _b.forEach(b=>{
        const isCur=(b===cur);const wp=b._wallPos||{};
        for(const d of['n','s','e','w']){
          let t=1;
          if(isCur&&wp[d]){const{x:wx,z:wz}=wp[d];
            if(d==='n'&&cA>TH&&camZ<wz-MG)t=LO;else if(d==='s'&&cA<-TH&&camZ>wz+MG)t=LO;
            else if(d==='e'&&sA>TH&&camX<wx-MG)t=LO;else if(d==='w'&&sA<-TH&&camX>wx+MG)t=LO;
          }
          b.walls[d].forEach(m=>{
            m._curOp=lerp(m._curOp,m._origOp*t,SP*dt);
            m.material.opacity=m._curOp;m.material.depthWrite=m._curOp>0.5;
          });
        }
      });
    },

    render(cx,cz){
      const cur=_getBldgAt(cx,cz);
      const gl=renderer.getContext();
      const sv=_exempts.map(o=>o?o.visible:false);
      renderer.autoClear=false;renderer.clear(true,true,true);
      gl.enable(gl.STENCIL_TEST);gl.stencilMask(0xFF);
      gl.stencilFunc(gl.ALWAYS,1,0xFF);gl.stencilOp(gl.KEEP,gl.KEEP,gl.REPLACE);
      gl.colorMask(false,false,false,false);gl.depthMask(false);
      _zIn.visible=false;_zOut.visible=false;_pg.visible=true;
      _b.forEach(v=>v.pg.visible=cur?v===cur:true);
      _ext.visible=false;
      _exempts.forEach(o=>{if(o)o.visible=false;});
      renderer.render(scene,camera);
      gl.colorMask(true,true,true,true);gl.depthMask(true);
      gl.stencilMask(0x00);gl.stencilFunc(gl.ALWAYS,0,0xFF);
      _pg.visible=false;
      _exempts.forEach((o,i)=>{if(o)o.visible=sv[i];});
      if(cur){
        _zIn.visible=true;_b.forEach(v=>v.zone.visible=v===cur);
        _zOut.visible=false;_ext.visible=false;
      }else{
        const near=cz>_chZ?_oN:_oS,far=cz>_chZ?_oS:_oN;
        _zIn.visible=true;_b.forEach(v=>v.zone.visible=true);
        _zOut.visible=true;near.visible=true;far.visible=false;_ext.visible=true;
      }
      renderer.render(scene,camera);
      gl.stencilFunc(gl.EQUAL,1,0xFF);
      _exempts.forEach(o=>{if(o)o.visible=false;});
      if(cur){
        _b.forEach(v=>v.zone.visible=v!==cur);
        _zOut.visible=true;_oS.visible=true;_oN.visible=true;
      }else{
        const near=cz>_chZ?_oN:_oS,far=cz>_chZ?_oS:_oN;
        _zIn.visible=false;near.visible=false;far.visible=true;_ext.visible=false;
      }
      renderer.render(scene,camera);
      gl.disable(gl.STENCIL_TEST);gl.stencilMask(0xFF);
      gl.colorMask(true,true,true,true);gl.depthMask(true);
      _exempts.forEach((o,i)=>{if(o)o.visible=sv[i];});
      _zIn.visible=true;_b.forEach(v=>v.zone.visible=true);
      _zOut.visible=true;_oS.visible=true;_oN.visible=true;
      _ext.visible=!cur;_pg.visible=false;
      renderer.autoClear=true;
    },

    getCells(){
      const cells=[];
      _grid.forEach((d,key)=>{
        const[gx,gy]=key.split(',').map(Number);
        cells.push({gx,gy,gz:0,id:d.id,built:d.built,building:d.group});
      });return cells;
    },
    getBuiltSet(){const s=new Set();_grid.forEach(d=>{if(d.built)s.add(d.id);});return s;},

    tagCustomWall(groupId,dir,meshes){
      const b=_b.find(v=>v.id===groupId);if(!b) return;
      const arr=Array.isArray(meshes)?meshes:[meshes];
      const wp=new THREE.Vector3();
      arr.forEach(m=>{
        m.material=m.material.clone();m.material.transparent=true;
        m._origOp=m.material.opacity||1;m._curOp=m._origOp;
        b.walls[dir].push(m);
      });
      if(b.walls[dir].length>0){
        let sx=0,sz=0;
        b.walls[dir].forEach(m=>{m.getWorldPosition(wp);sx+=wp.x;sz+=wp.z;});
        if(!b._wallPos) b._wallPos={};
        b._wallPos[dir]={x:sx/b.walls[dir].length,z:sz/b.walls[dir].length};
      }
    },
  };
  return W;
}

// ═══════════════════════════════════════════════════════════════════════
// Stick figure (protagonist) — preserved verbatim
// ═══════════════════════════════════════════════════════════════════════
function createFigure(){
  const UL=0.68,LL=0.68,UA=0.58,LA=0.44;
  const HIP=UL+LL,SH=HIP+0.72,HR=0.27;
  const SR=0.048,KR=0.088,JR=0.068;
  const sM=new THREE.MeshStandardMaterial({color:0x18141E,roughness:0.55,metalness:0.3});
  const jM=new THREE.MeshStandardMaterial({color:0x100C18,roughness:0.50,metalness:0.35});
  const hM=new THREE.MeshStandardMaterial({color:0x080610,roughness:0.40,metalness:0.2});
  const seg=(r,len)=>{const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,8),sM);m.position.y=-len/2;m.castShadow=true;return m;};
  const jnt=(r,mat=jM)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(r,8,8),mat);m.castShadow=true;return m;};

  const root=new THREE.Group();
  const tor=new THREE.Mesh(new THREE.CylinderGeometry(SR,SR,SH-HIP,8),sM);
  tor.position.y=HIP+(SH-HIP)/2; tor.castShadow=true; root.add(tor);

  const headPivot=new THREE.Group(); headPivot.position.y=SH+0.04; root.add(headPivot);
  const head=new THREE.Mesh(new THREE.SphereGeometry(HR,14,14),hM);
  head.position.y=HR+0.02; head.castShadow=true; headPivot.add(head);

  const hb=new THREE.Mesh(new THREE.CylinderGeometry(SR,SR,0.52,8),sM);
  hb.rotation.z=Math.PI/2; hb.position.y=HIP; root.add(hb);
  const sb=new THREE.Mesh(new THREE.CylinderGeometry(SR,SR,0.68,8),sM);
  sb.rotation.z=Math.PI/2; sb.position.y=SH; root.add(sb);

  const makeLeg=side=>{
    const hip=new THREE.Group(); hip.position.set(side*0.24,HIP,0); hip.add(seg(SR,UL));
    const knee=new THREE.Group(); knee.position.y=-UL;
    knee.add(jnt(KR)); knee.add(seg(SR,LL));
    jnt(KR*0.7).position.y=-LL; knee.add(jnt(KR*0.7));
    hip.add(knee); root.add(hip); return{hip,knee};
  };
  const makeArm=side=>{
    const sh=new THREE.Group(); sh.position.set(side*0.34,SH,0); sh.rotation.z=side*0.18;
    sh.add(jnt(JR)); sh.add(seg(SR,UA));
    const elbow=new THREE.Group(); elbow.position.y=-UA;
    elbow.add(jnt(JR)); elbow.add(seg(SR*0.88,LA));
    const hand=jnt(JR*0.75); hand.position.y=-LA; elbow.add(hand);
    sh.add(elbow); root.add(sh); return{sh,elbow};
  };

  const lLeg=makeLeg(-1),rLeg=makeLeg(1),lArm=makeArm(-1),rArm=makeArm(1);
  return{root,lLeg,rLeg,lArm,rArm,headPivot};
}

// ═══════════════════════════════════════════════════════════════════════
// Mascot (small bear) — extracted from original scene-build code
// ═══════════════════════════════════════════════════════════════════════
function createMascot(){
  const g=new THREE.Group();
  // body
  g.add(sph(0.24,18,18,M(0xF8F2EE,0.82)));
  // head
  const mh=sph(0.21,16,16,M(0xF8F2EE,0.80)); mh.position.y=0.34; g.add(mh);
  // ears
  for(const ex of[-0.14,0.14]){
    const e=sph(0.07,10,10,M(0xF8F2EE,0.82));
    e.position.set(ex,0.52,0.02); g.add(e);
  }
  // eyes
  const em=M(0x2C2040,0.4,0.1);
  [[-0.08,0.38,0.18],[0.08,0.38,0.18]].forEach(([x,y,z])=>{
    const s=sph(0.032,8,8,em); s.position.set(x,y,z); g.add(s);
  });
  // blush
  const blM=new THREE.MeshStandardMaterial({color:0xF8A8A8,roughness:0.9,transparent:true,opacity:0.7});
  for(const bx of[-0.13,0.13]){
    const bl=new THREE.Mesh(new THREE.SphereGeometry(0.048,8,8),blM);
    bl.scale.set(1,0.5,0.6); bl.position.set(bx,0.33,0.18); g.add(bl);
  }
  return g;
}

// ═══════════════════════════════════════════════════════════════════════
// Dream God — sketchy line-drawn girl figure on a billboard, redrawn
// to a canvas texture every frame for animated robe, hair and glow.
// ═══════════════════════════════════════════════════════════════════════
function createDreamGod(){
  const CW=300, CH=450;
  const canvas=document.createElement('canvas');
  canvas.width=CW; canvas.height=CH;
  const ctx=canvas.getContext('2d');

  const texture=new THREE.CanvasTexture(canvas);
  texture.minFilter=THREE.LinearFilter;
  texture.magFilter=THREE.LinearFilter;

  // Plane sized so that figure head≈0.2m wide, total height ~2m
  const plane=new THREE.Mesh(
    new THREE.PlaneGeometry(1.4,2.1),
    new THREE.MeshBasicMaterial({
      map:texture, transparent:true, depthWrite:false, side:THREE.DoubleSide,
    })
  );
  plane.renderOrder=10;

  // Halo light (was the orb's PointLight). Now sits behind the figure.
  const halo=new THREE.PointLight(0xFFD8A0,1.5,9,2);
  halo.position.set(0,0,-0.05);

  const group=new THREE.Group();
  group.add(plane);
  group.add(halo);

  let mood='normal';   // normal | thinking | speaking | error
  let moodTime=0;       // for transitions

  const draw=(t)=>{
    ctx.clearRect(0,0,CW,CH);
    const cx=CW/2;

    // Mood-driven palette
    const colorBase = mood==='error'    ? '255, 200, 180'
                    : mood==='thinking' ? '230, 230, 245'
                                        : '255, 250, 240';
    const glowColor = mood==='error'    ? '255, 180, 140'
                    : mood==='thinking' ? '200, 210, 235'
                                        : '255, 230, 190';
    const baseI = mood==='thinking' ? 0.65
                : mood==='error'    ? 0.85
                                    : 1.0;
    const speakPulse = mood==='speaking' ? Math.sin(t*7)*0.18 : 0;
    const intensity = baseI + speakPulse;

    // Soft glow background
    const glowR=170+Math.sin(t*0.9)*14;
    const grad=ctx.createRadialGradient(cx,CH*0.5,30,cx,CH*0.5,glowR);
    grad.addColorStop(0,`rgba(${glowColor}, ${0.35*intensity})`);
    grad.addColorStop(0.5,`rgba(${glowColor}, ${0.13*intensity})`);
    grad.addColorStop(1,`rgba(${glowColor}, 0)`);
    ctx.fillStyle=grad;
    ctx.fillRect(0,0,CW,CH);

    // Animation phases
    const sway=Math.sin(t*0.5)*3;
    const hemL=Math.sin(t*1.0)*10;
    const hemR=Math.sin(t*1.2+1.5)*9;
    const hemMid=Math.sin(t*0.9+0.7)*12;
    const hairL=Math.sin(t*0.7+0.3)*5;
    const hairR=Math.sin(t*0.8+1.2)*5;

    ctx.lineCap='round';
    ctx.lineJoin='round';

    const stroke=`rgba(${colorBase}, ${0.88*intensity})`;
    const stroke2=`rgba(${colorBase}, ${0.52*intensity})`;
    const fill=`rgba(${colorBase}, ${0.09*intensity})`;

    // ─── Robe silhouette ──────────────────────────────────────────
    ctx.beginPath();
    ctx.moveTo(cx-22+sway, 130);
    // left sleeve drape
    ctx.bezierCurveTo(
      cx-38+sway, 165,
      cx-50+sway, 240,
      cx-75+sway+hemL*0.4, 350
    );
    // left hem
    ctx.bezierCurveTo(
      cx-82+sway+hemL, 380,
      cx-90+sway+hemL, 410,
      cx-78+sway+hemL, 425
    );
    // bottom hem (wavy)
    ctx.bezierCurveTo(
      cx-30+hemMid*0.6, 432,
      cx+30+hemMid, 432,
      cx+78+sway+hemR, 425
    );
    // right hem up
    ctx.bezierCurveTo(
      cx+90+sway+hemR, 410,
      cx+82+sway+hemR, 380,
      cx+75+sway+hemR*0.4, 350
    );
    // right sleeve up
    ctx.bezierCurveTo(
      cx+50+sway, 240,
      cx+38+sway, 165,
      cx+22+sway, 130
    );
    // shoulders
    ctx.bezierCurveTo(
      cx+18+sway, 128,
      cx-18+sway, 128,
      cx-22+sway, 130
    );
    ctx.fillStyle=fill;
    ctx.fill();
    ctx.strokeStyle=stroke;
    ctx.lineWidth=1.8;
    ctx.stroke();

    // Internal robe folds (suggest cloth depth)
    ctx.lineWidth=0.9;
    ctx.strokeStyle=stroke2;
    ctx.beginPath();
    ctx.moveTo(cx-8+sway, 160);
    ctx.bezierCurveTo(
      cx-4+sway, 240,
      cx-6+sway, 340,
      cx-12+sway+hemMid*0.3, 420
    );
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx+8+sway, 160);
    ctx.bezierCurveTo(
      cx+4+sway, 240,
      cx+6+sway, 340,
      cx+12+sway+hemMid*0.3, 420
    );
    ctx.stroke();

    // ─── Head (no face — neither features nor expressions) ──────
    ctx.beginPath();
    ctx.ellipse(cx+sway, 80, 25, 30, 0, 0, Math.PI*2);
    ctx.fillStyle=fill;
    ctx.fill();
    ctx.lineWidth=1.8;
    ctx.strokeStyle=stroke;
    ctx.stroke();

    // ─── Hair: long flowing strands on each side ────────────────
    ctx.lineWidth=1.3;
    ctx.strokeStyle=stroke;
    // left outer
    ctx.beginPath();
    ctx.moveTo(cx-22+sway, 90);
    ctx.bezierCurveTo(
      cx-38+sway+hairL, 160,
      cx-42+sway+hairL*1.2, 260,
      cx-32+sway+hairL*0.8, 350
    );
    ctx.stroke();
    // left inner
    ctx.lineWidth=1.0;
    ctx.strokeStyle=stroke2;
    ctx.beginPath();
    ctx.moveTo(cx-16+sway, 100);
    ctx.bezierCurveTo(
      cx-30+sway+hairL*0.6, 180,
      cx-32+sway+hairL*0.9, 280,
      cx-28+sway+hairL*0.7, 360
    );
    ctx.stroke();

    // right outer
    ctx.lineWidth=1.3;
    ctx.strokeStyle=stroke;
    ctx.beginPath();
    ctx.moveTo(cx+22+sway, 90);
    ctx.bezierCurveTo(
      cx+38+sway+hairR, 160,
      cx+42+sway+hairR*1.2, 260,
      cx+32+sway+hairR*0.8, 350
    );
    ctx.stroke();
    // right inner
    ctx.lineWidth=1.0;
    ctx.strokeStyle=stroke2;
    ctx.beginPath();
    ctx.moveTo(cx+16+sway, 100);
    ctx.bezierCurveTo(
      cx+30+sway+hairR*0.6, 180,
      cx+32+sway+hairR*0.9, 280,
      cx+28+sway+hairR*0.7, 360
    );
    ctx.stroke();

    // Hair part on top of head
    ctx.lineWidth=1.4;
    ctx.strokeStyle=stroke;
    ctx.beginPath();
    ctx.moveTo(cx-18+sway, 65);
    ctx.bezierCurveTo(
      cx-12+sway, 50,
      cx+12+sway, 50,
      cx+18+sway, 65
    );
    ctx.stroke();

    // Halo intensity follows mood
    halo.intensity=1.5*intensity;
    halo.color.setStyle(mood==='error'?'rgb(255,180,140)'
                       :mood==='thinking'?'rgb(200,210,235)'
                                         :'rgb(255,216,160)');

    texture.needsUpdate=true;
  };

  return {
    group, draw,
    setMood:(m)=>{ mood=m; moodTime=0; },
    getMood:()=>mood,
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Creature base — shared helpers for all small animals (rabbit, cat, ...)
//
//   Provides:
//     - bodyMaterial(spec)       quick MeshStandardMaterial builder
//     - registerEye(mesh)        track an eye for blinking
//     - tickIdle(t, dt, group)   call every frame; runs breathing + blinks
//
//   Each creature factory composes its own meshes (ears, tail, face) on top
//   of a Group, then wires its idle animation through tickIdle.
// ═══════════════════════════════════════════════════════════════════════
function makeCreatureBase(opts={}){
  const breathAmp=opts.breathAmp!=null?opts.breathAmp:0.005;
  const breathHz =opts.breathHz !=null?opts.breathHz :0.4;
  const eyes=[];          // [{ mesh, baseY }]
  let blinkPhase=Math.random()*4;  // staggered start
  let nextBlink=2+Math.random()*3;
  let blinkLeft=0;        // seconds remaining in current blink
  let baseY=null;         // captured first frame to anchor breathing

  return {
    bodyMaterial(spec){
      spec=spec||{};
      return new THREE.MeshStandardMaterial({
        color:spec.color!=null?spec.color:0xCCCCCC,
        roughness:spec.roughness!=null?spec.roughness:0.85,
        metalness:spec.metalness!=null?spec.metalness:0.02,
      });
    },
    registerEye(mesh){
      eyes.push({mesh,baseY:mesh.scale.y});
    },
    tickIdle(t,dt,group){
      if(baseY===null) baseY=group.position.y;
      // Breathing: subtle vertical bob anchored to spawn y
      group.position.y=baseY+Math.sin(t*breathHz*Math.PI*2)*breathAmp;
      // Blinking: countdown to next blink, then squish eye Y briefly
      if(blinkLeft>0){
        blinkLeft-=dt;
        if(blinkLeft<=0){
          for(const e of eyes) e.mesh.scale.y=e.baseY;
        }
      }else{
        blinkPhase+=dt;
        if(blinkPhase>=nextBlink){
          for(const e of eyes) e.mesh.scale.y=e.baseY*0.1;
          blinkLeft=0.10;
          blinkPhase=0;
          nextBlink=2.5+Math.random()*3.5;
        }
      }
    },
  };
}

// ── Rabbit ──────────────────────────────────────────────────────────────
function createRabbit(){
  const base=makeCreatureBase({breathAmp:0.005,breathHz:0.5});
  const g=new THREE.Group();

  // Body — flattened egg, low-slung
  const bodyMat=base.bodyMaterial({color:0xF4EFE4,roughness:0.92});
  const body=new THREE.Mesh(new THREE.SphereGeometry(0.20,16,12),bodyMat);
  body.scale.set(1.0,0.85,1.25);
  body.position.y=0.17;
  body.castShadow=true;
  g.add(body);

  // Head — smaller sphere in front of body
  const head=new THREE.Mesh(new THREE.SphereGeometry(0.13,16,12),bodyMat);
  head.position.set(0,0.24,0.20);
  head.castShadow=true;
  g.add(head);

  // Ears — long upright capsules. Use elongated CylinderGeometry tipped
  // with rounded sphere caps (no CapsuleGeometry in r128).
  const earOuterMat=base.bodyMaterial({color:0xF4EFE4,roughness:0.92});
  const earInnerMat=base.bodyMaterial({color:0xE8B5C0,roughness:0.85});  // soft pink lining
  for(const ex of[-0.05,0.05]){
    const earGroup=new THREE.Group();
    earGroup.position.set(ex,0.32,0.20);
    earGroup.rotation.z=ex*1.5;            // splay outward slightly
    earGroup.rotation.x=-0.15;             // tilt forward a touch
    const earOuter=new THREE.Mesh(new THREE.CylinderGeometry(0.025,0.022,0.16,10),earOuterMat);
    earOuter.position.y=0.08;
    earOuter.castShadow=true;
    earGroup.add(earOuter);
    const earTip=new THREE.Mesh(new THREE.SphereGeometry(0.025,10,8),earOuterMat);
    earTip.position.y=0.16;
    earGroup.add(earTip);
    const earInner=new THREE.Mesh(new THREE.CylinderGeometry(0.013,0.011,0.13,8),earInnerMat);
    earInner.position.set(0,0.07,0.018);
    earGroup.add(earInner);
    g.add(earGroup);
    // Stash for ear-twitch animation
    if(ex<0) g._earL=earGroup; else g._earR=earGroup;
  }

  // Eyes — small black beads
  const eyeMat=base.bodyMaterial({color:0x121212,roughness:0.5});
  for(const ex of[-0.052,0.052]){
    const eye=new THREE.Mesh(new THREE.SphereGeometry(0.018,8,8),eyeMat);
    eye.scale.set(1,1.1,0.7);
    eye.position.set(ex,0.26,0.31);
    g.add(eye);
    base.registerEye(eye);
  }

  // Nose — tiny pink triangle (use a flattened sphere)
  const noseMat=base.bodyMaterial({color:0xE89090,roughness:0.6});
  const nose=new THREE.Mesh(new THREE.SphereGeometry(0.012,8,8),noseMat);
  nose.scale.set(1.2,0.7,0.7);
  nose.position.set(0,0.22,0.33);
  g.add(nose);

  // Tail — fluffy white pompom
  const tailMat=base.bodyMaterial({color:0xFFFCF6,roughness:0.95});
  const tail=new THREE.Mesh(new THREE.SphereGeometry(0.052,12,10),tailMat);
  tail.position.set(0,0.20,-0.21);
  tail.castShadow=true;
  g.add(tail);

  // Idle state for occasional ear twitches
  let nextTwitch=3+Math.random()*3;
  let twitchPhase=0;
  let twitchActive=0;       // 0..1 envelope

  return {
    group:g,
    update(t,dt){
      base.tickIdle(t,dt,g);
      // Ear twitches: trigger every few seconds, decay with a quick envelope
      twitchPhase+=dt;
      if(twitchActive<=0&&twitchPhase>=nextTwitch){
        twitchActive=1;
        twitchPhase=0;
        nextTwitch=4+Math.random()*4;
      }
      if(twitchActive>0){
        twitchActive=Math.max(0,twitchActive-dt*2.5);
        const wig=Math.sin(t*22)*0.20*twitchActive;
        if(g._earL) g._earL.rotation.z=-1.5*0.05+wig;
        if(g._earR) g._earR.rotation.z= 1.5*0.05-wig;
      }
    },
    setPose(state){
      // Reserved for future FSM hookup. No-op for now.
      g._pose=state;
    },
  };
}

// ── Cat ─────────────────────────────────────────────────────────────────
function createCat(){
  const base=makeCreatureBase({breathAmp:0.005,breathHz:0.45});
  const g=new THREE.Group();

  // Body — elongated grey ellipsoid, slightly higher than rabbit's
  const greyMat=base.bodyMaterial({color:0x8A8590,roughness:0.88});
  const body=new THREE.Mesh(new THREE.SphereGeometry(0.20,16,12),greyMat);
  body.scale.set(0.85,0.95,1.55);
  body.position.y=0.20;
  body.castShadow=true;
  g.add(body);

  // White belly/chin patch (a smaller sphere offset slightly down/front)
  const whiteMat=base.bodyMaterial({color:0xF0EDE8,roughness:0.90});
  const belly=new THREE.Mesh(new THREE.SphereGeometry(0.16,14,10),whiteMat);
  belly.scale.set(0.75,0.55,1.05);
  belly.position.set(0,0.15,0.05);
  g.add(belly);

  // Head — smaller sphere, lifted forward
  const head=new THREE.Mesh(new THREE.SphereGeometry(0.13,16,12),greyMat);
  head.position.set(0,0.30,0.24);
  head.castShadow=true;
  g.add(head);

  // White chin/face patch
  const chinPatch=new THREE.Mesh(new THREE.SphereGeometry(0.105,12,10),whiteMat);
  chinPatch.scale.set(0.85,0.65,0.7);
  chinPatch.position.set(0,0.27,0.32);
  g.add(chinPatch);

  // Ears — small flat triangles. Make them by squashing tiny cones.
  const earInnerMat=base.bodyMaterial({color:0xE8B5C0,roughness:0.85});
  for(const ex of[-0.075,0.075]){
    const earGroup=new THREE.Group();
    earGroup.position.set(ex,0.40,0.21);
    earGroup.rotation.z=ex*0.4;
    const ear=new THREE.Mesh(new THREE.ConeGeometry(0.05,0.09,8),greyMat);
    ear.scale.set(1.0,1.0,0.45);    // flatten front-to-back
    ear.castShadow=true;
    earGroup.add(ear);
    const earIn=new THREE.Mesh(new THREE.ConeGeometry(0.03,0.06,8),earInnerMat);
    earIn.scale.set(1.0,1.0,0.40);
    earIn.position.set(0,-0.005,0.012);
    earGroup.add(earIn);
    g.add(earGroup);
    if(ex<0) g._earL=earGroup; else g._earR=earGroup;
  }

  // Eyes — green ellipses with vertical black slits
  const greenMat=base.bodyMaterial({color:0x6FA86A,roughness:0.4});
  const slitMat =base.bodyMaterial({color:0x0E0E0E,roughness:0.4});
  for(const ex of[-0.055,0.055]){
    const eye=new THREE.Mesh(new THREE.SphereGeometry(0.022,10,8),greenMat);
    eye.scale.set(1.1,1.0,0.6);
    eye.position.set(ex,0.32,0.35);
    g.add(eye);
    base.registerEye(eye);
    const slit=new THREE.Mesh(new THREE.SphereGeometry(0.013,8,8),slitMat);
    slit.scale.set(0.25,1.05,0.4);
    slit.position.set(ex,0.32,0.37);
    g.add(slit);
    base.registerEye(slit);    // squashes alongside the eye for a cleaner blink
  }

  // Nose — small pink triangle
  const noseMat=base.bodyMaterial({color:0xE89090,roughness:0.6});
  const nose=new THREE.Mesh(new THREE.SphereGeometry(0.013,8,8),noseMat);
  nose.scale.set(1.4,0.85,0.7);
  nose.position.set(0,0.275,0.385);
  g.add(nose);

  // Tail — chain of small spheres tapering to tip; root at back of body.
  // Stored as g._tailSegs so update() can sway them.
  const tailRoot=new THREE.Group();
  tailRoot.position.set(0,0.21,-0.30);
  const segCount=6;
  const segs=[];
  let prev=tailRoot;
  for(let i=0;i<segCount;i++){
    const segGroup=new THREE.Group();
    segGroup.position.z=-0.06;
    const r=0.038-i*0.004;
    const sph=new THREE.Mesh(new THREE.SphereGeometry(r,10,8),greyMat);
    segGroup.add(sph);
    prev.add(segGroup);
    segs.push(segGroup);
    prev=segGroup;
  }
  g.add(tailRoot);
  g._tailSegs=segs;

  return {
    group:g,
    update(t,dt){
      base.tickIdle(t,dt,g);
      // Tail sway: each segment a phase-shifted sine of preceding segment's rotation.
      // Subtle base curve plus slow wave.
      for(let i=0;i<g._tailSegs.length;i++){
        const seg=g._tailSegs[i];
        const phase=t*1.8-i*0.55;
        seg.rotation.x=0.18+Math.sin(phase)*0.08;     // arched downward, gentle bounce
        seg.rotation.y=Math.sin(phase*0.9)*0.12;       // side-to-side
      }
      // Ear twitches (lighter & more frequent than rabbit's)
      if(!g._catTwitchNext) g._catTwitchNext=2+Math.random()*2;
      if(!g._catTwitchActive) g._catTwitchActive=0;
      g._catTwitchPhase=(g._catTwitchPhase||0)+dt;
      if(g._catTwitchActive<=0&&g._catTwitchPhase>=g._catTwitchNext){
        g._catTwitchActive=1;
        g._catTwitchPhase=0;
        g._catTwitchNext=2.5+Math.random()*2.5;
      }
      if(g._catTwitchActive>0){
        g._catTwitchActive=Math.max(0,g._catTwitchActive-dt*3);
        const wig=Math.sin(t*28)*0.12*g._catTwitchActive;
        if(g._earL) g._earL.rotation.z=-0.4*0.075+wig;
        if(g._earR) g._earR.rotation.z= 0.4*0.075-wig;
      }
    },
    setPose(state){ g._pose=state; },
  };
}

// ═══════════════════════════════════════════════════════════════════════
// Component
// ═══════════════════════════════════════════════════════════════════════
export default function SplineRoom(){
  const mountRef=useRef(null);
  // ── Chat state (history kept for API context, NOT displayed) ─────────
  const [messages,setMessages]=useState([]);   // [{role, display, apiContent}]
  const [input,setInput]=useState('');
  const [model,setModel]=useState(MODELS[0].id);
  const [loading,setLoading]=useState(false);
  const [l1Registry,setL1Registry]=useState({});  // {name: {name, params, body, desc}}
  // ── Speech bubble (displayed above the orb) ───────────────────────────
  const [bubbleText,setBubbleText]=useState("我在这里。说说看，你想要什么样的世界？");
  const bubbleRef=useRef(null);
  const A=useRef({
    world:null, mascot:null,
    camFx:SPAWN_X, camFz:SPAWN_Z,
    zoom:{target:8.5,current:8.5},
    camAngle:Math.PI/4,
    camFollowFacing:false,
    midDrag:false, midLastX:0, midLastY:0,
    camHeight:8.5,
    rightDrag:false, rightLastY:0,
    firstPerson:false,
    fpPitch:0,  // FP pitch (looking up/down), clamped to ±1.4 rad
    fpKeys:{w:false,a:false,s:false,d:false},  // WASD held state
    ch:{
      x:SPAWN_X, z:SPAWN_Z, tx:SPAWN_X, tz:SPAWN_Z,
      facing:0, targetFacing:0,
      walkPh:0,
      state:'idle',  // idle | walking
      idleTime:0,
      root:null,lLeg:null,rLeg:null,lArm:null,rArm:null,headPivot:null,
    },
  });

  useEffect(()=>{
    const mount=mountRef.current; if(!mount) return;
    const a=A.current, ch=a.ch;

    // ── Adaptive viewport sizing ───────────────────────────────────
    let W=Math.min(mount.clientWidth||1100, MAX_VIEW_W);
    let H=Math.round(W/VIEW_ASPECT);

    // ── Renderer ───────────────────────────────────────────────────
    const renderer=new THREE.WebGLRenderer({antialias:true,stencil:true});
    renderer.setSize(W,H); renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x0A1828); mount.appendChild(renderer.domElement);

    const scene=new THREE.Scene();
    scene.fog=new THREE.Fog(0x0A1828,30,120);
    const camera=new THREE.PerspectiveCamera(50,W/H,0.1,400);
    camera.position.set(SPAWN_X+6.8, 8.5, SPAWN_Z+6.8);
    camera.lookAt(SPAWN_X, 1.4, SPAWN_Z);

    // ═══════════════════════════════════════════════════════════════
    // 8×8 grid declaration — all empty cells (LLM target)
    // ═══════════════════════════════════════════════════════════════
    const world=createWorld(scene,renderer,camera,{box,cyl,sph,add,M});

    let cellId=0;
    for(let gx=GRID_MIN;gx<=GRID_MAX;gx++){
      for(let gy=GRID_MIN;gy<=GRID_MAX;gy++){
        world.addEmpty(gx,gy,'cell_'+gx+'_'+gy);
        cellId++;
      }
    }

    // Disable outdoor zone splitting (everything always visible)
    world.setOutdoorSplit(-1e6, -1e6);

    a.world=world;

    // ── Lights ─────────────────────────────────────────────────────
    const ambientLight=new THREE.AmbientLight(0xC8C0E8, 0.30);
    scene.add(ambientLight);
    const hemiLight=new THREE.HemisphereLight(0x9090C8, 0xD0C8B0, 0.30);
    scene.add(hemiLight);
    const sun=new THREE.DirectionalLight(0xD8D0F0, 0.50);
    sun.position.set(SPAWN_X+8, 12, SPAWN_Z+5);
    sun.castShadow=true; sun.shadow.mapSize.set(1024,1024);
    const sc=sun.shadow.camera;
    sc.near=0.5; sc.far=80;
    sc.left=-30; sc.right=30; sc.top=30; sc.bottom=-30;
    sc.updateProjectionMatrix();
    scene.add(sun);

    // Register the three env lights as controllable entities so the LLM can
    // dim/recolor them via SET light without kind (in-place update path).
    // The lights remain parented to scene; we only attach them to entities
    // so the light compiler can find them via the `_light` prop.
    const _attachEnvLight=(id,light)=>{
      world.exec('EXIST',id);
      const ent=world.getEntity(id);
      if(ent){ ent.props.set('_light',light); ent.meshes.push(light); }
    };
    _attachEnvLight('env_ambient',ambientLight);
    _attachEnvLight('env_hemi',hemiLight);
    _attachEnvLight('env_sun',sun);

    // ── Ground reference: transparent grid + invisible shadow receiver ──
    // GridHelper gives spatial reference; ShadowMaterial plane catches shadows
    // (so voxels don't look floaty) without drawing any color.
    const GRID_SPAN=WORLD_MAX-WORLD_MIN+10;
    const gridCenter={x:(WORLD_MIN+WORLD_MAX)/2, z:(WORLD_MIN+WORLD_MAX)/2};

    const shadowFloor=new THREE.Mesh(
      new THREE.PlaneGeometry(GRID_SPAN, GRID_SPAN),
      new THREE.ShadowMaterial({opacity:0.32})
    );
    shadowFloor.rotation.x=-Math.PI/2;
    shadowFloor.position.set(gridCenter.x, 0, gridCenter.z);
    shadowFloor.receiveShadow=true;
    scene.add(shadowFloor);

    const grid=new THREE.GridHelper(
      GRID_SPAN, Math.round(GRID_SPAN),    // 1m divisions
      0xAACCDD,                            // center axis lines
      0x445566                             // regular grid lines
    );
    grid.position.set(gridCenter.x, 0.002, gridCenter.z);  // just above floor
    grid.material.transparent=true;
    grid.material.opacity=0.35;
    grid.material.depthWrite=false;
    scene.add(grid);

    // ── Protagonist (stick figure) ─────────────────────────────────
    const fig=createFigure();
    fig.root.position.set(ch.x, 0, ch.z);
    scene.add(fig.root);
    Object.assign(ch,{root:fig.root,lLeg:fig.lLeg,rLeg:fig.rLeg,
                      lArm:fig.lArm,rArm:fig.rArm,headPivot:fig.headPivot});

    // ── Mascot (bear) ──────────────────────────────────────────────
    const mascot=createMascot();
    mascot.position.set(ch.x-1.2, 0.24, ch.z+0.6);
    scene.add(mascot);
    a.mascot=mascot;

    // ── Walk-target ring indicator ─────────────────────────────────
    const walkRing=new THREE.Mesh(
      new THREE.TorusGeometry(0.26,0.025,6,24),
      new THREE.MeshBasicMaterial({color:0xFFCC44,transparent:true,opacity:0.8})
    );
    walkRing.rotation.x=-Math.PI/2; walkRing.visible=false; scene.add(walkRing);

    // ═══════════════════════════════════════════════════════════════
    // Dream God — sketch-style figure at world center that anchors the
    // speech bubble. Replaces the previous orb.
    // ═══════════════════════════════════════════════════════════════
    const GOD_BASE_Y=1.45;   // group center height (figure spans roughly 0.4 to 2.5m)
    const dreamGod=createDreamGod();
    dreamGod.group.position.set(SPAWN_X,GOD_BASE_Y,SPAWN_Z);
    scene.add(dreamGod.group);
    A.current.dreamGod=dreamGod;

    // ═══════════════════════════════════════════════════════════════
    // Creatures (creatures-of-the-world — exist regardless of any box,
    // currently just one rabbit + one cat as living scenery).
    // ═══════════════════════════════════════════════════════════════
    const rabbit=createRabbit();
    rabbit.group.position.set(8.0, 0, 8.0);
    rabbit.group.rotation.y=Math.random()*Math.PI*2;
    scene.add(rabbit.group);

    const cat=createCat();
    cat.group.position.set(35.0, 0, 35.0);
    cat.group.rotation.y=Math.random()*Math.PI*2;
    scene.add(cat.group);

    A.current.creatures=[rabbit, cat];

    // ═══════════════════════════════════════════════════════════════
    // Compile world (auto-generate wireframes, terrain, walls, ...)
    // ═══════════════════════════════════════════════════════════════
    world.setExempts([ch.root, mascot, walkRing, shadowFloor, grid,
                      dreamGod.group, rabbit.group, cat.group]);
    world.compile();

    // Bind OBS to the engine's live _obs so runtime-added colliders take effect
    OBS=world.getOBS();
    VOXEL_HAS_AT=world.voxelHasAt;
    // Perimeter walls (so player can't walk off the grid)
    OBS.push({t:'b',x1:WORLD_MIN-1,x2:WORLD_MAX+1,z1:WORLD_MIN-0.2,z2:WORLD_MIN});
    OBS.push({t:'b',x1:WORLD_MIN-1,x2:WORLD_MAX+1,z1:WORLD_MAX,    z2:WORLD_MAX+0.2});
    OBS.push({t:'b',x1:WORLD_MIN-0.2,x2:WORLD_MIN,z1:WORLD_MIN-1,z2:WORLD_MAX+1});
    OBS.push({t:'b',x1:WORLD_MAX,    x2:WORLD_MAX+0.2,z1:WORLD_MIN-1,z2:WORLD_MAX+1});

    // ── Click handler ──────────────────────────────────────────────
    const ray=new THREE.Raycaster();
    const floorPl=new THREE.Plane(new THREE.Vector3(0,1,0),0);

    const onClick=e=>{
      if(A.current.firstPerson) return;  // FP uses WASD, not click-to-move
      const rc=mount.getBoundingClientRect();
      ray.setFromCamera(
        new THREE.Vector2((e.clientX-rc.left)/rc.width*2-1,-((e.clientY-rc.top)/rc.height)*2+1),
        camera
      );
      const hit=new THREE.Vector3();
      if(!ray.ray.intersectPlane(floorPl,hit)) return;
      if(hit.x<WORLD_MIN||hit.x>WORLD_MAX||hit.z<WORLD_MIN||hit.z>WORLD_MAX) return;

      // Walk to clicked position
      ch.tx=hit.x; ch.tz=hit.z;
      ch.state='walking';
      walkRing.position.set(hit.x,0.02,hit.z); walkRing.visible=true;
    };
    mount.addEventListener('click',onClick);

    // ── Wheel zoom ────────────────────────────────────────────────
    const onWheel=e=>{
      e.preventDefault();
      const d=e.deltaY>0?1.5:-1.5;
      a.zoom.target=Math.max(2.0,Math.min(22.0,a.zoom.target+d));
    };
    mount.addEventListener('wheel',onWheel,{passive:false});

    // ── Middle-button drag → rotate camera (or free-look in FP) ──
    const onMouseDown=e=>{
      if(a.firstPerson) return;
      if(e.button===1){e.preventDefault(); a.midDrag=true; a.midLastX=e.clientX; a.midLastY=e.clientY;}
    };
    const onMouseMove=e=>{
      if(!a.midDrag||a.firstPerson) return;
      const dx=e.clientX-a.midLastX;
      a.camAngle-=dx*0.006;
      a.midLastX=e.clientX;
    };
    const onMouseUp=e=>{if(e.button===1) a.midDrag=false;};
    const onContextMenu=e=>e.preventDefault();
    mount.addEventListener('mousedown',onMouseDown);
    mount.addEventListener('mousemove',onMouseMove);
    mount.addEventListener('mouseup',onMouseUp);
    mount.addEventListener('mouseleave',()=>{a.midDrag=false;});
    mount.addEventListener('contextmenu',onContextMenu);

    // ── Right-button drag → camera height (TP only) ───────────────
    const onRightDown=e=>{
      if(a.firstPerson) return;
      if(e.button===2){e.preventDefault(); a.rightDrag=true; a.rightLastY=e.clientY;}
    };
    const onRightMove=e=>{
      if(!a.rightDrag||a.firstPerson) return;
      const dy=e.clientY-a.rightLastY;
      a.zoom.target=Math.max(2.0,Math.min(22.0,a.zoom.target+dy*0.045));
      a.rightLastY=e.clientY;
    };
    const onRightUp=e=>{if(e.button===2) a.rightDrag=false;};
    mount.addEventListener('mousedown',onRightDown);
    mount.addEventListener('mousemove',onRightMove);
    mount.addEventListener('mouseup',onRightUp);
    mount.addEventListener('mouseleave',()=>{a.rightDrag=false;});

    // ── First-person mode: drag-look + WASD + ESC to exit ─────────
    // (Avoids Pointer Lock API — it's blocked in the artifact iframe.)
    const fpBtnRef={current:null};
    a._fpBtnRef=fpBtnRef;
    const updateFPBtn=(active)=>{
      const b=fpBtnRef.current; if(!b) return;
      b.textContent=active?'🎥 第一人称中（ESC 退出）':'🎥 第一人称';
      b.style.background=active?'rgba(255,180,80,0.38)':'rgba(0,0,0,0.42)';
      b.style.borderColor=active?'rgba(255,180,80,0.5)':'rgba(255,255,255,0.16)';
    };
    const enterFP=()=>{
      if(a.firstPerson) return;
      a.firstPerson=true;
      a.fpPitch=0;
      a.fpKeys.w=a.fpKeys.a=a.fpKeys.s=a.fpKeys.d=false;
      a.ch.targetFacing=a.ch.facing;
      walkRing.visible=false;
      mount.style.cursor='grab';
      updateFPBtn(true);
    };
    const exitFP=()=>{
      if(!a.firstPerson) return;
      a.firstPerson=false;
      a.fpKeys.w=a.fpKeys.a=a.fpKeys.s=a.fpKeys.d=false;
      a.ch.targetFacing=a.ch.facing;
      a.ch.state='idle';
      a._fpDragging=false;
      mount.style.cursor='';
      updateFPBtn(false);
    };
    a._enterFP=enterFP;
    a._exitFP=exitFP;

    // Drag-look: left-button hold + move = rotate yaw (body) + pitch (camera)
    const onFpDown=e=>{
      if(!a.firstPerson||e.button!==0) return;
      a._fpDragging=true;
      mount.style.cursor='grabbing';
      e.preventDefault();
    };
    const onFpMove=e=>{
      if(!a.firstPerson||!a._fpDragging) return;
      const dx=e.movementX||0, dy=e.movementY||0;
      a.ch.facing-=dx*0.0035;
      a.ch.targetFacing=a.ch.facing;
      a.fpPitch=Math.max(-1.4,Math.min(1.4,a.fpPitch-dy*0.0035));
    };
    const onFpUp=e=>{
      if(e.button!==0) return;
      if(a._fpDragging){ a._fpDragging=false; mount.style.cursor=a.firstPerson?'grab':''; }
    };
    mount.addEventListener('mousedown',onFpDown);
    window.addEventListener('mousemove',onFpMove);   // window so drag continues outside canvas
    window.addEventListener('mouseup',onFpUp);

    // WASD + ESC. Don't fire while user is typing in chat.
    const isTyping=()=>{
      const t=(document.activeElement||{}).tagName;
      return t==='INPUT'||t==='TEXTAREA';
    };
    const onKeyDown=e=>{
      if(!a.firstPerson) return;
      if(e.key==='Escape'){ exitFP(); e.preventDefault(); return; }
      if(isTyping()) return;
      const k=e.key.toLowerCase();
      if(k==='w'||k==='a'||k==='s'||k==='d'){a.fpKeys[k]=true; e.preventDefault();}
    };
    const onKeyUp=e=>{
      const k=e.key.toLowerCase();
      if(k==='w'||k==='a'||k==='s'||k==='d') a.fpKeys[k]=false;
    };
    window.addEventListener('keydown',onKeyDown);
    window.addEventListener('keyup',onKeyUp);

    // ═══════════════════════════════════════════════════════════════
    // Animate
    // ═══════════════════════════════════════════════════════════════
    const SPEED=2.6, WALK_HZ=6.5;
    const PL=3.8;
    const FACE_SPD=7;
    let rafId,prevT=0;

    // Sky color interpolation
    const _skyLow =new THREE.Color(0xB04818);
    const _skyHigh=new THREE.Color(0x040A14);
    const _skyCur =new THREE.Color();

    const animate=t=>{
      rafId=requestAnimationFrame(animate);
      const dt=Math.min((t-prevT)*0.001,0.05); prevT=t;
      const tt=t*0.001;

      // ── Movement: FP (WASD) vs TP (click-to-move state machine) ──
      if(a.firstPerson){
        // ch.facing already set instantly by mousemove handler — no lerp
        const k=a.fpKeys;
        let dx=0, dz=0;
        // Forward = direction body faces (sin/cos of facing)
        if(k.w){dx+=Math.sin(ch.facing); dz+=Math.cos(ch.facing);}
        if(k.s){dx-=Math.sin(ch.facing); dz-=Math.cos(ch.facing);}
        // Strafe — perpendicular to facing. Sign inverted vs naive intuition
        // because Three.js camera looking at +z mirrors world x on screen,
        // so "screen right" = world -x at facing=0 (not +x).
        if(k.d){dx+=Math.sin(ch.facing-Math.PI/2); dz+=Math.cos(ch.facing-Math.PI/2);}
        if(k.a){dx+=Math.sin(ch.facing+Math.PI/2); dz+=Math.cos(ch.facing+Math.PI/2);}
        const len=Math.hypot(dx,dz);
        if(len>1e-6){
          const r=resolve(ch.x+dx/len*SPEED*dt, ch.z+dz/len*SPEED*dt);
          ch.x=r.x; ch.z=r.z;
          ch.walkPh+=WALK_HZ*dt;
          ch.state='walking';
          ch.idleTime=0;
        } else {
          ch.walkPh*=0.80;
          ch.state='idle';
          ch.idleTime+=dt;
        }
      } else {
        // ── Smooth facing (TP) ─────────────────────────────────────
        ch.facing=lerpAngle(ch.facing,ch.targetFacing,FACE_SPD*dt);

        // ── State machine ──────────────────────────────────────────
        ch.idleTime = ch.state==='idle' ? ch.idleTime+dt : 0;

        switch(ch.state){
          case 'idle':
            ch.walkPh*=0.80;
            ch.targetFacing=ch.facing;
            break;

          case 'walking':{
            const dx=ch.tx-ch.x,dz=ch.tz-ch.z,dist=Math.hypot(dx,dz);
            if(dist<0.10){
              walkRing.visible=false;
              ch.state='idle';
            } else {
              const r=resolve(ch.x+dx/dist*SPEED*dt,ch.z+dz/dist*SPEED*dt);
              ch.x=r.x; ch.z=r.z;
              ch.targetFacing=Math.atan2(dx,dz);
              ch.walkPh+=WALK_HZ*dt;
            }
            break;
          }
        }
      }

      // ── Pose blending (just walking + idle) ──────────────────────
      const s=Math.sin(ch.walkPh);
      ch.lLeg.hip.rotation.x  =  s*0.55;
      ch.lLeg.knee.rotation.x = Math.max(0,-s)*0.80;
      ch.rLeg.hip.rotation.x  = -s*0.55;
      ch.rLeg.knee.rotation.x = Math.max(0, s)*0.80;
      ch.lArm.sh.rotation.x   = -s*0.36;
      ch.rArm.sh.rotation.x   =  s*0.36;
      ch.lArm.elbow.rotation.x= Math.max(0, s)*0.40;
      ch.rArm.elbow.rotation.x= Math.max(0,-s)*0.40;
      ch.headPivot.rotation.x = lerp(ch.headPivot.rotation.x, 0, PL*dt);

      // ── Sync transform ───────────────────────────────────────────
      ch.root.position.set(ch.x, 0, ch.z);
      ch.root.rotation.y=ch.facing;

      // ── Mascot follow ────────────────────────────────────────────
      {
        const FDIST=0.85, FSPD=3.4;
        const mx=mascot.position.x, mz=mascot.position.z;
        const dx=ch.x-mx, dz=ch.z-mz;
        const dist=Math.hypot(dx,dz)||0.001;
        if(dist>FDIST+0.04){
          const step=Math.min((dist-FDIST)*FSPD*dt, dist-FDIST);
          const nr=resolve(mx+dx/dist*step, mz+dz/dist*step);
          mascot.position.x=nr.x; mascot.position.z=nr.z;
        }
        mascot.rotation.y=lerpAngle(mascot.rotation.y, Math.atan2(dx,dz), 6*dt);
        // gentle bob
        const moving=dist>FDIST+0.1?1:0;
        mascot.position.y=lerp(
          mascot.position.y,
          0.24 + 0.014*Math.sin(tt*1.7) + moving*0.022*Math.abs(Math.sin(tt*5.5)),
          8*dt
        );
        const br=1+0.038*Math.sin(tt*1.7);
        mascot.scale.set(br,br,br);
      }

      // ── Camera follow ────────────────────────────────────────────
      if(a.firstPerson){
        // First-person: camera at head, looking along (facing, fpPitch).
        // Hide head mesh so we don't see inside our own skull. Body stays.
        const HEAD_Y=2.41;
        const yaw=ch.facing;
        const pitch=a.fpPitch;
        const cyP=Math.cos(pitch);
        camera.position.set(ch.x, HEAD_Y, ch.z);
        camera.lookAt(
          ch.x + Math.sin(yaw)*cyP,
          HEAD_Y + Math.sin(pitch),
          ch.z + Math.cos(yaw)*cyP
        );
        ch.headPivot.visible=false;
      }else{
        if(a.camFollowFacing){
          const tgt=ch.facing+Math.PI;
          const spd=ch.state==='walking'?2.2:1.2;
          a.camAngle=lerpAngle(a.camAngle,tgt,spd*dt);
        }
        a.camFx=lerp(a.camFx,ch.x,2.5*dt);
        a.camFz=lerp(a.camFz,ch.z,2.5*dt);
        const _CR=9.617;
        camera.position.set(
          a.camFx+_CR*Math.sin(a.camAngle),
          a.camHeight,
          a.camFz+_CR*Math.cos(a.camAngle)
        );
        camera.lookAt(a.camFx,1.4,a.camFz);
        ch.headPivot.visible=true;
      }

      // ── Zoom + dynamic sky (TP only) ─────────────────────────────
      if(!a.firstPerson){
        a.zoom.current=lerp(a.zoom.current, a.zoom.target, 6*dt);
        a.camHeight=a.zoom.current;
      }
      const _hN=Math.max(0,Math.min(1,(a.camHeight-2.0)/20.0));
      camera.fov=a.firstPerson?75:(65-47*_hN);
      camera.updateProjectionMatrix();

      _skyCur.lerpColors(_skyLow,_skyHigh,_hN);
      renderer.setClearColor(_skyCur);
      scene.fog.color.copy(_skyCur);
      scene.fog.near=lerp(8,35,_hN);
      scene.fog.far =lerp(50,130,_hN);

      // ── Render via stencil engine ─────────────────────────────────
      if(a.world){
        const lookAng=a.firstPerson?ch.facing:a.camAngle;
        a.world.updateWalls(ch.x,ch.z,lookAng,dt,camera.position.x,camera.position.z);
        a.world.render(ch.x,ch.z);
      }

      // ── Creatures: idle animations (breathing, twitches, blinks) ─
      if(a.creatures){
        for(const c of a.creatures) c.update(tt,dt);
      }

      // ── Dream God: bobbing, face-camera lerp, canvas redraw, bubble ──
      if(a.dreamGod){
        const dg=a.dreamGod;
        dg.group.position.y=GOD_BASE_Y+Math.sin(tt*0.7)*0.13;
        dg.draw(tt);
        // Yaw-only face-camera (billboard with no tilt)
        const tgtYaw=Math.atan2(
          camera.position.x-dg.group.position.x,
          camera.position.z-dg.group.position.z
        );
        dg.group.rotation.y=lerpAngle(dg.group.rotation.y, tgtYaw, 1.5*dt);

        if(bubbleRef.current){
          const v=new THREE.Vector3(
            dg.group.position.x,
            dg.group.position.y+1.25,   // above her head
            dg.group.position.z
          );
          v.project(camera);
          const onScreen=v.z>-1&&v.z<1&&v.x>-1.2&&v.x<1.2&&v.y>-1.2&&v.y<1.2;
          if(onScreen){
            const w=mount.clientWidth, h=mount.clientHeight;
            bubbleRef.current.style.left=((v.x+1)*0.5*w)+'px';
            bubbleRef.current.style.top =((-v.y+1)*0.5*h)+'px';
            bubbleRef.current.style.opacity='1';
          }else{
            bubbleRef.current.style.opacity='0';
          }
        }
      }
    };
    animate(0);

    // ── Window resize → keep canvas filling the container ─────────
    const onResize=()=>{
      W=Math.min(mount.clientWidth||1100, MAX_VIEW_W);
      H=Math.round(W/VIEW_ASPECT);
      renderer.setSize(W,H);
      camera.aspect=W/H;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize',onResize);

    return ()=>{
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize',onResize);
      window.removeEventListener('keydown',onKeyDown);
      window.removeEventListener('keyup',onKeyUp);
      window.removeEventListener('mousemove',onFpMove);
      window.removeEventListener('mouseup',onFpUp);
      mount.removeEventListener('mousedown',onFpDown);
      mount.removeEventListener('click',onClick);
      mount.removeEventListener('wheel',onWheel);
      mount.removeEventListener('mousedown',onMouseDown);
      mount.removeEventListener('mousemove',onMouseMove);
      mount.removeEventListener('mouseup',onMouseUp);
      mount.removeEventListener('contextmenu',onContextMenu);
      mount.removeEventListener('mousedown',onRightDown);
      mount.removeEventListener('mousemove',onRightMove);
      mount.removeEventListener('mouseup',onRightUp);
      renderer.dispose();
      if(mount.contains(renderer.domElement))mount.removeChild(renderer.domElement);
    };
  },[]);

  // ── Dream God: drive mood from chat state ─────────────────────────
  useEffect(()=>{
    const dg=A.current.dreamGod;
    if(!dg) return;
    if(loading){
      dg.setMood('thinking');
    }else if(typeof bubbleText==='string'&&bubbleText.startsWith('⚠️')){
      dg.setMood('error');
      const id=setTimeout(()=>{
        if(A.current.dreamGod) A.current.dreamGod.setMood('normal');
      },3500);
      return ()=>clearTimeout(id);
    }else{
      dg.setMood('speaking');
      const id=setTimeout(()=>{
        if(A.current.dreamGod) A.current.dreamGod.setMood('normal');
      },1800);
      return ()=>clearTimeout(id);
    }
  },[loading,bubbleText]);

  // ── World snapshot for the system prompt ──────────────────────────
  const snapshotWorld=()=>{
    const a=A.current;
    const world=a.world;
    if(!world) return {error:'world not ready'};
    const builtCells=[];
    world.getCells().forEach(c=>{
      if(c.built){
        const rt=world.getEntity('rt_cell_'+c.gx+'_'+c.gy);
        const type=rt?rt.props.get('cellType'):'unknown';
        builtCells.push({gx:c.gx,gy:c.gy,type});
      }
    });
    const llmEntities=[];
    const envLights={};
    const store=world.getStore();
    for(const [id,e] of store){
      // Surface env light current intensity/color so model can see daylight state
      if(id.startsWith('env_')){
        const L=e.props.get('_light');
        if(L) envLights[id]={
          intensity:Math.round(L.intensity*100)/100,
          color:'0x'+L.color.getHexString().toUpperCase(),
        };
        continue;
      }
      if(!id.startsWith('llm_')) continue;
      const props={};
      for(const [k,v] of e.props){
        // Skip engine-internal references — not JSON-safe
        if(k==='_group'||k==='_primaryMesh'||k==='_collider'||k==='_light') continue;
        props[k]=v;
      }
      llmEntities.push({id,props});
    }
    const r=n=>Math.round(n*100)/100;
    const vi=Math.floor(a.ch.x/VOXEL_SIZE);
    const vk=Math.floor(a.ch.z/VOXEL_SIZE);
    return {
      built_cells:builtCells,
      empty_cells_count:64-builtCells.length,
      llm_entities:llmEntities,
      env_lights:envLights,
      voxels:world.voxelSummary(),
      voxel_size:VOXEL_SIZE,
      player:{x:r(a.ch.x),z:r(a.ch.z),voxel:[vi,0,vk]},
      bear:{x:r(a.mascot.position.x),z:r(a.mascot.position.z)},
    };
  };

  // ── Chat: send message via Anthropic API with L0/L1 protocol ──────
  const sendMessage=async()=>{
    const text=input.trim();
    if(!text||loading) return;
    const world=A.current.world;
    if(!world){
      setBubbleText('⚠️ 世界还没编译完成，稍等重试');
      setMessages([...messages,
        {role:'user',display:text,apiContent:text},
        {role:'assistant',display:'⚠️ 世界还没编译完成，稍等重试',apiContent:'(world not ready)'}
      ]);
      setInput('');
      return;
    }

    const userMsg={role:'user',display:text,apiContent:text};
    const next=[...messages,userMsg];
    setMessages(next);
    setInput('');
    setLoading(true);
    setBubbleText('…');

    const snapshot=snapshotWorld();
    const systemPrompt=buildSystemPrompt(l1Registry,snapshot);

    let assistantRaw='',stopReason='';
    try{
      const response=await fetch("/api/chat",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({
          model:model,
          max_tokens:MAX_OUTPUT_TOKENS,
          system:systemPrompt,
          messages:next.map(m=>({role:m.role,content:m.apiContent})),
        }),
      });
      const data=await response.json();
      if(data.error) throw new Error(data.error.message||'API error');
      assistantRaw=(data.content||[])
        .filter(c=>c.type==='text')
        .map(c=>c.text)
        .join('\n');
      stopReason=data.stop_reason||'';
    }catch(err){
      const msg='⚠️ 网络/API 错误: '+(err.message||String(err));
      setBubbleText(msg);
      setMessages([...next,{
        role:'assistant', display:msg,
        apiContent:'(error: '+(err.message||String(err))+')',
      }]);
      setLoading(false);
      return;
    }

    // Truncation guard — surface clearly instead of letting it look like a JSON bug
    if(stopReason==='max_tokens'){
      const msg='⚠️ 响应超过 '+MAX_OUTPUT_TOKENS+' tokens 被截断了。能不能拆成几步分别说？';
      setBubbleText(msg);
      setMessages([...next,{
        role:'assistant', display:msg,
        apiContent:assistantRaw+'\n\n[Execution: TRUNCATED at max_tokens — your last response was cut off. Split the work into smaller turns, or define L1 macros to compress.]',
      }]);
      setLoading(false);
      return;
    }

    // Parse JSON via robust multi-strategy parser
    const parseResult=tryParseStructuredJSON(assistantRaw);
    if(!parseResult.ok){
      const msg='⚠️ 我的回复格式坏了，能再试一次吗？';
      setBubbleText(msg);
      setMessages([...next,{
        role:'assistant', display:msg,
        apiContent:assistantRaw+'\n\n[Execution: parse failed: '+parseResult.error+']',
      }]);
      setLoading(false);
      return;
    }
    const parsed=parseResult.value;

    // register_l1 first (so actions in same turn can use them)
    const nextRegistry={...l1Registry};
    const registered=[],regErrors=[];
    for(const def of (parsed.register_l1||[])){
      const err=validateL1(def);
      if(err){ regErrors.push((def&&def.name||'?')+': '+err); continue; }
      nextRegistry[def.name]={
        name:def.name,
        params:def.params,
        body:def.body,
        desc:def.desc||'',
      };
      registered.push(def.name);
    }
    // remove_l1
    const removed=[];
    for(const name of (parsed.remove_l1||[])){
      if(name in nextRegistry){ delete nextRegistry[name]; removed.push(name); }
    }

    // execute actions: expand → normalize → run, stop on first failure
    const actions=parsed.actions||[];
    let execOk=true,execError=null,failedAt=-1,executedCount=0,totalL0=0;
    for(let i=0;i<actions.length;i++){
      try{
        const expanded=expandAction(actions[i],nextRegistry);
        for(const l0 of expanded){
          const norm=normalizeIds(l0);
          const [op,...args]=norm;
          if(op==='BUILD_CELL'){
            const [gx,gy,type]=args;
            const ok=world.buildCell(gx,gy,type);
            if(!ok) throw new Error('buildCell failed for ('+gx+','+gy+', '+type+')');
          }else if(op==='VOXEL_SET'){
            world.voxelSet(args[0],args[1],args[2],args[3]);
          }else if(op==='VOXEL_FILL'){
            world.voxelFill(args[0],args[1],args[2],args[3],args[4],args[5],args[6]);
          }else if(op==='VOXEL_SHAPE'){
            world.voxelShape(args[0],args[1],args[2]);
          }else{
            world.exec(op,...args);
          }
          totalL0++;
        }
        executedCount++;
      }catch(e){
        execOk=false;
        execError=e.message||String(e);
        failedAt=i;
        break;
      }
    }

    // Persist registry (always — registration succeeded even if actions failed)
    setL1Registry(nextRegistry);

    // Build display summary
    const lines=[];
    if(parsed.thoughts) lines.push(parsed.thoughts);
    const regBits=[];
    if(registered.length) regBits.push('注册 L1: '+registered.join(', '));
    if(removed.length) regBits.push('移除 L1: '+removed.join(', '));
    if(regErrors.length) regBits.push('⚠️ L1 错误: '+regErrors.join('; '));
    if(regBits.length) lines.push(regBits.join(' · '));
    if(actions.length){
      if(execOk){
        lines.push(`✓ 执行 ${actions.length} 个 action（展开为 ${totalL0} 个 L0）`);
      }else{
        lines.push(`✗ 执行到第 ${failedAt+1}/${actions.length} 个失败: ${execError}`);
      }
    }else if(!registered.length&&!removed.length&&!parsed.thoughts){
      lines.push('(无操作)');
    }
    const display=lines.join('\n');

    // Bubble shows only the conversational "thoughts" — execution stats stay
    // in the API context for next turn but don't clutter the orb's voice.
    let bubble=parsed.thoughts;
    if(!bubble){
      if(!execOk) bubble='嗯…执行到一半出错了。';
      else if(actions.length) bubble='好。';
      else if(registered.length||removed.length) bubble='记下了。';
      else bubble='(沉默)';
    }
    if(!execOk&&parsed.thoughts) bubble=parsed.thoughts+'\n\n（不过执行到一半出错了。）';
    setBubbleText(bubble);

    // apiContent: raw + execution result so next turn sees what happened
    const execNote=execOk
      ?`[Execution: success — ${executedCount}/${actions.length} actions, ${totalL0} L0 ops]`
      :`[Execution: FAILED at action ${failedAt}: ${execError}]`;
    const apiContent=assistantRaw+'\n\n'+execNote;

    setMessages([...next,{role:'assistant',display,apiContent}]);
    setLoading(false);
  };
  const onChatKeyDown=(e)=>{
    if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); }
  };

  return(
    <div style={{padding:"1rem 0"}}>
      <div style={{position:"relative",width:"100%"}}>
        <div ref={mountRef} style={{width:"100%",borderRadius:"8px",overflow:"hidden",border:"0.5px solid rgba(255,255,255,0.15)",cursor:"pointer"}}/>

        {/* Zoom controls */}
        <div style={{position:"absolute",bottom:10,right:10,
          display:"flex",flexDirection:"column",gap:3,zIndex:20}}>
          <button onClick={()=>{const z=A.current.zoom;z.target=Math.max(2.0,z.target-3);}}
            style={{width:26,height:26,background:"rgba(0,0,0,0.42)",
              border:"0.5px solid rgba(255,255,255,0.16)",borderRadius:5,
              color:"rgba(255,255,255,0.72)",fontSize:17,cursor:"pointer",
              lineHeight:"1",padding:0,fontFamily:"system-ui"}}>+</button>
          <button onClick={()=>{const z=A.current.zoom;z.target=Math.min(22.0,z.target+3);}}
            style={{width:26,height:26,background:"rgba(0,0,0,0.42)",
              border:"0.5px solid rgba(255,255,255,0.16)",borderRadius:5,
              color:"rgba(255,255,255,0.72)",fontSize:17,cursor:"pointer",
              lineHeight:"1",padding:0,fontFamily:"system-ui"}}>−</button>
        </div>

        {/* Follow-facing + first-person toggles */}
        <div style={{position:"absolute",bottom:10,left:10,zIndex:20,display:"flex",gap:6}}>
          <button onClick={(e)=>{
            const a=A.current; a.camFollowFacing=!a.camFollowFacing;
            const b=e.currentTarget;
            b.textContent=a.camFollowFacing?'👁 跟随中':'👁 跟随';
            b.style.background=a.camFollowFacing?'rgba(80,180,255,0.38)':'rgba(0,0,0,0.42)';
            b.style.borderColor=a.camFollowFacing?'rgba(80,180,255,0.5)':'rgba(255,255,255,0.16)';
          }}
            style={{padding:"4px 10px",background:"rgba(0,0,0,0.42)",
              border:"0.5px solid rgba(255,255,255,0.16)",borderRadius:5,
              color:"rgba(255,255,255,0.72)",fontSize:11,cursor:"pointer",
              fontFamily:"system-ui,sans-serif",whiteSpace:"nowrap"}}>👁 跟随</button>
          <button
            ref={el=>{const a=A.current; if(a&&a._fpBtnRef) a._fpBtnRef.current=el;}}
            onClick={()=>{
              const a=A.current;
              if(a.firstPerson) a._exitFP&&a._exitFP();
              else a._enterFP&&a._enterFP();
            }}
            style={{padding:"4px 10px",background:"rgba(0,0,0,0.42)",
              border:"0.5px solid rgba(255,255,255,0.16)",borderRadius:5,
              color:"rgba(255,255,255,0.72)",fontSize:11,cursor:"pointer",
              fontFamily:"system-ui,sans-serif",whiteSpace:"nowrap"}}>🎥 第一人称</button>
        </div>

        {/* Speech bubble — anchored to orb's screen projection (inside canvas wrapper so absolute coords match) */}
        <div ref={bubbleRef}
          style={{
            position:"absolute", left:0, top:0,
            transform:"translate(-50%, -110%)",
            background:"rgba(252, 248, 240, 0.96)",
            color:"#221E18",
            padding:"10px 14px", borderRadius:14,
            fontSize:13, lineHeight:1.5, maxWidth:340, minWidth:80,
            boxShadow:"0 6px 20px rgba(0,0,0,0.35), 0 0 0 0.5px rgba(255,200,120,0.4)",
            fontFamily:"system-ui,sans-serif",
            pointerEvents:"none", whiteSpace:"pre-wrap", wordBreak:"break-word",
            transition:"opacity 0.2s",
            zIndex:30, opacity:0,
          }}>
          <div style={{
            position:"absolute", left:"50%", bottom:-8, transform:"translateX(-50%)",
            width:0, height:0,
            borderLeft:"8px solid transparent", borderRight:"8px solid transparent",
            borderTop:"8px solid rgba(252,248,240,0.96)",
          }}/>
          {bubbleText}
        </div>
      </div>
      <div style={{marginTop:8,fontSize:12,textAlign:"center",
        color:"rgba(255,255,255,0.55)",fontFamily:"system-ui,sans-serif"}}>
        点地面走过去 · 滚轮/右键拖拽缩放 · 中键拖拽旋转
      </div>

      {/* ── Bottom input bar (model + input + send) ── */}
      <div style={{marginTop:12,display:"flex",gap:8,alignItems:"stretch",
        fontFamily:"system-ui,sans-serif"}}>
        <select value={model} onChange={e=>setModel(e.target.value)} disabled={loading}
          style={{background:"rgba(0,0,0,0.42)",color:"rgba(255,255,255,0.85)",
            border:"0.5px solid rgba(255,255,255,0.15)",borderRadius:6,
            padding:"9px 10px",fontSize:12,cursor:loading?"default":"pointer",
            outline:"none"}}>
          {MODELS.map(mo=>(
            <option key={mo.id} value={mo.id} style={{background:"#1a1822"}}>
              {mo.label}
            </option>
          ))}
        </select>
        <input type="text" value={input} onChange={e=>setInput(e.target.value)}
          onKeyDown={onChatKeyDown} placeholder="对它说点什么…"
          style={{flex:1,background:"rgba(0,0,0,0.42)",color:"rgba(255,255,255,0.92)",
            border:"0.5px solid rgba(255,255,255,0.15)",borderRadius:6,
            padding:"9px 13px",fontSize:13,outline:"none",minWidth:0}}/>
        <button onClick={sendMessage} disabled={loading||!input.trim()}
          style={{background:loading||!input.trim()?"rgba(80,120,180,0.25)":"rgba(80,120,200,0.6)",
            color:"rgba(255,255,255,0.95)",
            border:"0.5px solid rgba(120,160,220,0.5)",borderRadius:6,
            padding:"9px 22px",fontSize:13,
            cursor:loading||!input.trim()?"default":"pointer",
            whiteSpace:"nowrap",transition:"background 0.15s"}}>
          {loading?'…':'发送'}
        </button>
      </div>
    </div>
  );
}
