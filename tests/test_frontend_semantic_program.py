"""Execute the actual task UI functions, including repeated step identities."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = [
    "isSemanticScenario", "semanticStatusLabel", "submitSemanticTask", "pointOfInterestLabel",
    "semanticElement", "semanticConditionLabel", "semanticProgramNodes", "semanticProgramLabel",
    "semanticEvidenceForStep", "renderSemanticMission", "evidenceResultLabel",
]


class FrontendSemanticProgramTest(unittest.TestCase):
    def run_ui(self, test):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required for frontend behavior checks")
        source = (ROOT / "uav_llm_demo/static/app.js").read_text(encoding="utf-8")
        declarations = []
        for name in FUNCTIONS:
            match = re.search(rf"^(?:async )?function {name}\([^\n]*\) \{{.*?^\}}", source, re.M | re.S)
            self.assertIsNotNone(match, name)
            declarations.append(match.group())
        for name in ("SEMANTIC_ACTION_LABELS", "STEP_STATUS_LABELS"):
            match = re.search(rf"^const {name} = \{{.*?^\}};", source, re.M | re.S)
            self.assertIsNotNone(match, name)
            declarations.append(match.group())
        script = "const assert = require('node:assert/strict');\n(async () => {\n"
        script += "\n".join(declarations) + "\n" + r'''
class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.hidden = false; this.value = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this.ownText = ''; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, callback) { this[type] = callback; }
  focus() {}
  set textContent(value) { this.ownText = String(value); this.children = []; }
  get textContent() { return (this.ownText || '') + this.children.map(child => child.textContent).join(''); }
}
const elements = new Map();
const document = {
  createElement: tag => new Element(tag),
  getElementById: id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
};
const isRecord = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const setText = (node, value) => { node.textContent = value; };
const semanticPanel = new Element(), semanticProgram = new Element(), semanticSteps = new Element();
const taskForm = new Element(), taskInstruction = new Element(), submitTaskBtn = new Element();
const recommendedTasks = new Element(), flightDiagnostics = new Element();
let commandPending = false, commandError = '', renderedMissionMode = null, recommendationKey = '', semanticStepsKey = '';
let state = {scenario_id:'semantic_campus', deepseek_key_configured:true, flight:{running:true},
  scenario:{mission_mode:'semantic', points_of_interest:[{id:'lab',label:'实验楼'}, {id:'library',label:'图书馆'}]},
  semantic_mission:{mission_id:'m', vehicle_id:'uav-1', status:'running', revision:1, instruction:'初始任务'}};
taskInstruction.value = '温度超过 40 度就复查';
''' + test + "\n})().catch(error => {console.error(error); process.exit(1)});"
        result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_task_changes_can_be_submitted_while_running_or_interpreting(self):
        self.run_ui(r'''
let posts = [], opened = 0;
const postState = async (url, body) => posts.push({url,body});
const runCommand = async (action, options) => { assert.equal(options.semanticValidation,true); await action(); return true; };
const openAgentDialog = () => opened++;
for (const status of ['running','interpreting','needs_clarification','completed']) {
  state.semantic_mission.status = status;
  renderSemanticMission();
  assert.equal(submitTaskBtn.disabled, false);
  assert.equal(submitTaskBtn.textContent, '修改任务');
  assert.equal(await submitSemanticTask(), true);
}
assert.equal(posts.length, 4);
assert.deepEqual(posts[0], {url:'/api/mission/task',body:{instruction:taskInstruction.value}});
commandPending = true;
renderSemanticMission(); assert.equal(submitTaskBtn.disabled,true);
assert.equal(await submitSemanticTask(), false);
commandPending = false;
taskInstruction.value = '  '; renderSemanticMission(); assert.equal(submitTaskBtn.disabled,true);
assert.equal(await submitSemanticTask(), false);
taskInstruction.value = '返航'; state.deepseek_key_configured = false;
renderSemanticMission(); assert.equal(submitTaskBtn.disabled,true);
assert.equal(await submitSemanticTask(),false); assert.equal(opened,1);
assert.equal(posts.length,4);
''')

    def test_custom_condition_text_preserves_operators_units_and_composition(self):
        self.run_ui(r'''
const mission = {steps:[{id:'scan',target_id:'lab'}], program:[], evidence:[{step_id:'old',target_id:'library'}]};
const condition = {all:[
  {step_id:'scan',field:'temperature_c',op:'>',value:40},
  {any:[{source:'telemetry',field:'battery_percent',op:'>=',value:25},
    {not:{source:'telemetry',field:'elapsed_seconds',op:'>',value:120}}]},
]};
const text = semanticConditionLabel(condition,mission);
assert.match(text,/实验楼 · 温度超过 40 °C/);
assert.match(text,/且/); assert.match(text,/或/); assert.match(text,/电量至少 25%/);
assert.match(text,/不满足（任务用时超过 120 秒）/);
assert.doesNotMatch(text,/图书馆/); // A missing evidence_id must not pick the first record.
assert.equal(semanticConditionLabel({step_id:'scan',field:'anomaly',op:'==',value:false}, mission),'实验楼 · 异常等于 否');
const revisions = {revision:2,steps:[{id:'scan',revision:1,target_id:'library'},
  {id:'scan',revision:2,target_id:'lab'}],program:[{id:'scan',action:'thermal_scan',target_id:'lab'}]};
assert.match(semanticConditionLabel({step_id:'scan',field:'temperature_c',op:'>',value:40},revisions),/^实验楼/);
assert.match(semanticConditionLabel({step_id:'scan',field:'temperature_c',op:'>',value:40},revisions,1),/^图书馆/);
''')

    def test_repeated_logical_ids_match_only_their_execution_and_revision(self):
        self.run_ui(r'''
const makeStep = (execution_id, revision) => ({id:'scan',execution_id,revision,action:'thermal_scan',target_id:'lab',status:'completed'});
const result = (execution_id, revision, temperature_c) => ({mission_id:'m',vehicle_id:'uav-1',step_id:'scan',
  execution_id,revision,action:'thermal_scan',target_id:'lab',result:{temperature_c}});
const mission = {mission_id:'m',vehicle_id:'uav-1',evidence:[result('scan-1',1,68),result('scan-2',1,42),result('scan-3',2,29)]};
assert.equal(semanticEvidenceForStep(makeStep('scan-2',1),mission).result.temperature_c,42);
assert.equal(semanticEvidenceForStep(makeStep('scan-3',2),mission).result.temperature_c,29);
assert.equal(semanticEvidenceForStep(makeStep('scan-3',1),mission),undefined);
assert.equal(semanticEvidenceForStep({...makeStep('scan-1',1),status:'cancelled'},mission),null);
assert.equal(semanticEvidenceForStep({id:'scan',action:'thermal_scan',target_id:'lab',status:'completed'},mission),undefined);
''')

    def test_execution_view_keeps_revision_history_and_unexecuted_branch_as_plan(self):
        self.run_ui(r'''
state.semantic_mission = {mission_id:'m',vehicle_id:'uav-1',status:'running',revision:2,instruction:'改为复查两次',current_step_id:'scan',current_execution_id:'scan-r2-2',
  program:[{id:'loop',action:'repeat',count:2,until:{step_id:'scan',field:'temperature_c',op:'<=',value:40},steps:[{id:'scan',action:'thermal_scan',target_id:'lab'}]},
    {id:'branch',action:'branch',condition:{source:'telemetry',field:'battery_percent',op:'>=',value:25},then:[{id:'sample',action:'capture',target_id:'lab'}],else:[]},
    {id:'home',action:'return_home',target_id:'home'}],
  loops:[{id:'loop',revision:2,iteration:2,count:2,status:'active'}],
  steps:[{id:'scan',execution_id:'scan-r1',revision:1,action:'thermal_scan',target_id:'lab',status:'completed'},
    {id:'old',execution_id:'old-r1',revision:1,action:'visit',target_id:'library',status:'cancelled'},
    {id:'scan',execution_id:'scan-r2-1',revision:2,loop_iterations:{loop:1},action:'thermal_scan',target_id:'lab',status:'completed'},
    {id:'scan',execution_id:'scan-r2-2',revision:2,loop_iterations:{loop:2},action:'thermal_scan',target_id:'lab',status:'active'}],
  evidence:[{mission_id:'m',vehicle_id:'uav-1',step_id:'scan',execution_id:'scan-r1',revision:1,action:'thermal_scan',target_id:'lab',result:{temperature_c:68,anomaly:true}},
    {mission_id:'m',vehicle_id:'uav-1',step_id:'scan',execution_id:'scan-r2-1',revision:2,action:'thermal_scan',target_id:'lab',result:{temperature_c:42,anomaly:false}}]};
renderSemanticMission();
assert.match(semanticProgram.textContent,/循环 2 次 · 第 2 次/);
assert.match(semanticProgram.textContent,/结束条件：实验楼 · 温度不超过 40 °C/);
assert.match(semanticProgram.textContent,/条件分支 · 电量至少 25%/);
assert.match(semanticProgram.textContent,/满足时：模拟采样/);
assert.match(semanticSteps.textContent,/任务记录 · 第 1 版/);
assert.match(semanticSteps.textContent,/当前任务 · 第 2 版/);
assert.match(semanticSteps.textContent,/任务修改后取消/);
assert.match(semanticSteps.textContent,/68.0 °C/); assert.match(semanticSteps.textContent,/42.0 °C/);
assert.match(semanticSteps.textContent,/循环第 2 次/);
const active = semanticSteps.children.filter(child => child.attributes['aria-current'] === 'step');
assert.equal(active.length,1); assert.equal(active[0].dataset.status,'active');
assert.doesNotMatch(semanticSteps.textContent,/模拟采样/); // The untaken branch is a plan, not an execution record.
assert.match(document.getElementById('plannerProgress').textContent,/当前任务已完成 1 步/);
state.semantic_mission.plan_revision = 2;
state.semantic_mission.revision = 3;
state.semantic_mission.status = 'needs_clarification';
renderSemanticMission();
assert.match(semanticSteps.textContent,/当前任务 · 第 2 版/);
assert.match(semanticProgram.textContent,/循环 2 次 · 第 2 次/);
assert.match(document.getElementById('plannerProgress').textContent,/当前任务已完成 1 步/);
''')

    def test_pause_clarification_and_initial_submission_have_distinct_labels(self):
        self.run_ui(r'''
state.semantic_mission = {status:'idle'}; renderSemanticMission();
assert.equal(submitTaskBtn.textContent,'提交任务');
state.semantic_mission = {status:'interpreting',instruction:'检查实验楼'};
renderSemanticMission(); assert.equal(document.getElementById('semanticStatus').textContent,'解析中');
assert.equal(submitTaskBtn.disabled,false);
state.flight.running = false; renderSemanticMission();
assert.equal(document.getElementById('semanticStatus').textContent,'已暂停');
state.semantic_mission = {status:'needs_clarification',instruction:'检查',message:'请指定巡检地点。'};
renderSemanticMission(); assert.equal(document.getElementById('semanticStatus').textContent,'待补充');
assert.equal(document.getElementById('semanticMessage').textContent,'请指定巡检地点。');
assert.equal(submitTaskBtn.disabled,false);
state.semantic_mission = {status:'safety_wait',revision:1}; renderSemanticMission();
assert.equal(document.getElementById('semanticStatus').textContent,'等待处理');
assert.match(document.getElementById('semanticMessage').textContent,/修改任务或提交返航/);
assert.equal(submitTaskBtn.disabled,false);
''')


if __name__ == "__main__":
    unittest.main()
