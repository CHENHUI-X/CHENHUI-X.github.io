/** Curated source + fresh-build navigation and canonical-topic regression checks. */
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";
const root = fileURLToPath(new URL("../", import.meta.url));
const read = (name) => readFileSync(path.join(root, name), "utf8");
async function load(name) {
    const js = ts.transpileModule(read(name), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
    return import(`data:text/javascript;base64,${Buffer.from(js).toString("base64")}`);
}
const { KNOWLEDGE_MAP_SLUG, learningPathEntries: entries, learningMainline: mainline, getLearningPathNavigation: nav } = await load("src/constants/learning-path.ts");
const { articleTopics: topics, getArticleTopic } = await load("src/constants/article-topics.ts");
const { memorySeriesEntries: series, getMemorySeriesNavigation: seriesNav } = await load("src/constants/memory-series.ts");
let assertions = 0;
function check(value, msg) { assert.ok(value, msg); assertions++; }
function equal(a,b,msg) { assert.deepEqual(a,b,msg); assertions++; }
const expected = ["2026-05-16-llm-tokenization-guide","2024-04-10-perplexity","2024-04-13-text-generator-with-transformer-decoder","2026-05-16-llm-rope-rotary-position-embedding","2026-05-16-llm-decoding-strategies","2026-06-12-llm-mem-opt-1-fundamentals","2026-05-16-llm-quantization-guide","2026-06-16-llm-mem-opt-5-sft","2026-06-13-llm-mem-opt-2-inference"];
const removed = ["2026-05-16-llm-memory-breakdown","2025-02-06-a-series-on-llm-inference-ii","2026-06-14-llm-mem-opt-3-quantization"];
const sources = new Map(readdirSync(path.join(root,"src/content/posts")).filter(f=>f.endsWith(".md")).map(f=>[f.slice(0,-3).replace(/\s+/g,"-").replace(/[()]/g,"").toLowerCase(),read(`src/content/posts/${f}`)]));
equal(sources.size,28,"28 canonical articles; three redundant pages remain removed");
equal(mainline.map(x=>x.slug),expected,"nine-step beginner route");
equal(entries.filter(x=>x.kind==="branch").length,5,"five independent advanced topics");
equal(entries.filter(x=>x.kind==="foundation").length,4,"four optional foundations");
equal(entries.length,18,"curated route entry count");
equal(nav(KNOWLEDGE_MAP_SLUG).next.slug,expected[0],"map starts route");
check(/入门路线共 \$\{navigation\.total\} 篇/.test(read("src/components/LearningPathNav.astro")),"map introduction derives its count from the route");
const expectedSeries = ["2026-06-12-llm-mem-opt-1-fundamentals", "2026-06-15-llm-mem-opt-4-zero", "2026-06-16-llm-mem-opt-5-sft", "2026-06-13-llm-mem-opt-2-inference", "2026-06-14-llm-mem-opt-3-flashattention"];
equal(series.map(e=>e.slug),expectedSeries,"five-part memory series order independent of mainline");
equal(topics.find(t=>t.id==="llm-memory").articles,series,"directory and series metadata agree");
for(const [i,e] of series.entries()) {
 const n=seriesNav(e.slug);
 equal(n.position,i+1,`${e.slug}: series position`);
 equal(n.previous?.slug,expectedSeries[i-1],`${e.slug}: series previous`);
 equal(n.next?.slug,expectedSeries[i+1],`${e.slug}: series next`);
 equal(entries.find(x=>x.slug===e.slug).title,e.title,`${e.slug}: learning title agrees`);
}
for(const slug of [expectedSeries[1],expectedSeries[4]]) {
 check(!mainline.some(e=>e.slug===slug),`${slug}: optional rather than mainline`);
 check(!entries.find(e=>e.slug===expectedSeries[2]).prerequisites.includes(slug),`${slug}: not a single-card SFT prerequisite`);
}
check(!seriesNav("2026-05-16-llm-quantization-guide"),"quantization remains independent");
const topicSlugs = topics.flatMap(t=>t.articles.map(a=>a.slug));
equal(new Set(topicSlugs).size,topicSlugs.length,"each article one topic home");
equal([...topicSlugs].sort(),[...sources.keys()].sort(),"all canonical posts indexed exactly once");
for (const [slug,body] of sources) {
 check(getArticleTopic(slug),`${slug}: topic exists`);
 for(const old of removed)check(!body.includes(`/posts/${old}/`),`${slug}: no links via obsolete pages`);
}
for(const old of removed) {
 check(!sources.has(old),`${old}: no duplicate content page`);
 check(read("astro.config.mjs").includes(`"/posts/${old}/":`),`${old}: explicit route redirect`);
}
const map = sources.get(KNOWLEDGE_MAP_SLUG);
const legacyMapAnchors = ["0-这张图怎么用", "0-阅读对象与使用方法", "1-入门主线先把流程走通", "2-深入专题按问题选择", "2-长上下文优化器与对齐专题", "3-基础与应用需要时再进入", "3-数学基础应用开发与部署", "4-知识版图已经覆盖什么还缺什么", "4-各领域的内容覆盖情况", "一数据表示-representation", "二模型架构-architecture", "三推理-inference", "四训练原理-training-mechanics", "五分布式训练-distributed-training", "六微调与对齐-fine-tuning-and-alignment", "七数学与评估基础-foundations", "八工具与框架-ecosystem", "按目标选择下一步", "用小实验检查理解"];
for(const id of legacyMapAnchors)check(map.includes(`id="${id}"`),`legacy map anchor ${id}`);
for(const id of ["llm-start", "llm-deep", "applications", "foundations", "causal", "guides"])check(topics.some(t=>t.id===id),`legacy topic anchor ${id}`);
for(const entry of entries) {
 check(sources.has(entry.slug),`${entry.slug}: source exists`);
 check(map.includes(`/posts/${entry.slug}/`),`${entry.slug}: map link`);
 const n=nav(entry.slug), i=expected.indexOf(entry.slug);
 equal(n.position,i+1,`${entry.slug}: position`);
 equal(n.previous?.slug,i>0?expected[i-1]:undefined,`${entry.slug}: previous`);
 equal(n.next?.slug,i>=0?expected[i+1]:undefined,`${entry.slug}: next`);
 if(i<0)check(entries.some(x=>x.slug===entry.returnSlug) && entry.returnSlug!==entry.slug,`${entry.slug}: related return target`);
 for(const dep of entry.prerequisites)check(entries.some(e=>e.slug===dep),`${entry.slug}: prerequisite source`);
}
function walk(slug,visiting=new Set()) {
 check(!visiting.has(slug),`${slug}: acyclic dependencies`);
 const copy=new Set(visiting).add(slug);
 for(const dep of entries.find(e=>e.slug===slug).prerequisites)walk(dep,copy);
}
for(const e of entries)walk(e.slug);
for(const part of ["一、文本怎样变成模型输入","二、模型由哪些部分组成","三、模型怎样生成文本","四、怎样训练模型","五、多张 GPU 怎样一起训练","六、微调与对齐","七、数学基础与模型评估","八、工具、框架和部署"])check(map.includes(`## ${part}`),`roadmap preserves ${part}`);
check(read("src/config.ts").includes('url: "/topics/"'),"navbar has sitewide topic entry");
check(read("src/pages/[...page].astro").includes('page.currentPage === 1'),"home guide only on first page");
if(!process.argv.includes("--source-only")) {
 function attrs(html,name){return [...html.matchAll(/<a\b([^>]*)>/g)].filter(m=>new RegExp(`\\b${name}(?:\\s|=|$)`).test(m[1])).map(m=>m[1].match(/href="([^"]+)"/)?.[1]);}
 for(const [slug] of sources) {
  const html=read(`dist/posts/${slug}/index.html`);
  check(html.includes("data-article-context"),`${slug}: visible scope context`);
  const n=nav(slug);
  if(n) {
   const regions=[...html.matchAll(/<nav\b[^>]*\bdata-learning-path(?:\s|=|>)[\s\S]*?<\/nav>/g)];
   equal(regions.length,1,`${slug}: one learning nav`);
   const h=regions[0][0];
   const sn=seriesNav(slug);
   equal(attrs(h,"data-path-next"),!sn && n.next?[`/posts/${n.next.slug}/`]:[],`${slug}: next HTML`);
   equal(attrs(h,"data-path-previous"),!sn && n.previous?[`/posts/${n.previous.slug}/`]:[],`${slug}: previous HTML`);
   equal(attrs(h,"data-path-return"),!sn && n.returnTo?[`/posts/${n.returnTo.slug}/`]:[],`${slug}: return HTML`);
   equal(attrs(h,"data-path-prerequisite"),n.prerequisites.map(x=>`/posts/${x.slug}/`),`${slug}: prerequisites HTML`);
   if(sn) {
    equal(attrs(h,"data-series-next"),sn.next?[`/posts/${sn.next.slug}/`]:[],`${slug}: series next HTML`);
    equal(attrs(h,"data-series-previous"),sn.previous?[`/posts/${sn.previous.slug}/`]:[],`${slug}: series previous HTML`);
    equal(attrs(h,"data-series-directory"),["/topics/#llm-memory"],`${slug}: back to series directory`);
    check(h.includes(`${sn.position} / 5`),`${slug}: series progress HTML`);
    check(html.includes("data-series-position"),`${slug}: series location above article`);
   }
   if(n.entry?.kind==="mainline")check(h.includes(`主线 ${n.position} / 9`),`${slug}: progress HTML`);
  } else check(html.includes("data-topic-navigation"),`${slug}: topical rather than chronological fallback`);
 }
 const mapHtml=read(`dist/posts/${KNOWLEDGE_MAP_SLUG}/index.html`);
 check(mapHtml.replace(/<[^>]*>/g, "").includes(`入门路线共 ${mainline.length} 篇`),"map introduction displays the actual mainline count");
 for(const id of legacyMapAnchors)check(mapHtml.includes(`id="${id}"`),`rendered legacy map anchor ${id}`);
 const catalog=read("dist/topics/index.html");
 equal(attrs(catalog,"data-topic-article").length,28,"28 actual directory links");
 for(const t of topics)check(catalog.includes(`id="${t.id}"`),`${t.id}: directory anchor`);
 for(const old of removed) {
  const html=read(`dist/posts/${old}/index.html`);
  check(/http-equiv="refresh"/i.test(html),`${old}: actual static redirect`);
  check(!/data-pagefind-body/.test(html),`${old}: no duplicate article index body`);
  check(!read("dist/rss.xml").includes(`/posts/${old}/`),`${old}: absent RSS`);
 }
 check(existsSync(path.join(root,"dist/pagefind")),"search index built");
}
console.log(`PASS: ${assertions} consolidation/navigation assertions (${process.argv.includes("--source-only")?"source only":"source + generated HTML"})`);
