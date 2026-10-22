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
    "semanticMotionSummary", "semanticMotionProgressLabel", "semanticStepLabel",
    "semanticControlSummary", "semanticGoalMessage",
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

    def test_generation_failure_can_retry_unchanged_instruction(self):
        self.run_ui(r'''
const original = taskInstruction.value;
let sent;
const postState = async (url, body) => { sent = {url, body}; };
const runCommand = async action => { await action(); return true; };
const openAgentDialog = () => { throw new Error('Key is already configured'); };
state.semantic_mission.status = 'generation_failed';
state.flight.running = false;
renderSemanticMission();
assert.equal(document.getElementById('semanticStatus').textContent, '生成失败');
assert.equal(submitTaskBtn.textContent, '重试任务');
assert.equal(submitTaskBtn.disabled, false);
assert.equal(document.getElementById('semanticMessage').dataset.error, 'true');
assert.doesNotMatch(document.getElementById('semanticMessage').textContent, /补充|调整/);
assert.equal(await submitSemanticTask(), true);
assert.deepEqual(sent, {url:'/api/mission/task',body:{instruction:original}});
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

    def test_motion_progress_uses_confirmed_path_progress_and_updates_without_step_changes(self):
        self.run_ui(r'''
const motion = {path:{kind:'boundary',direction:'clockwise',laps:2},speed_mps:2,
  heading:{mode:'face_target',target_id:'library'}};
const step = {id:'flight',execution_id:'flight-2',revision:2,action:'motion',target_id:'library',status:'active',motion};
state.semantic_mission = {mission_id:'m',vehicle_id:'uav-1',status:'running',revision:2,plan_revision:2,
  current_step_id:'flight',current_execution_id:'flight-2',steps:[step],
  program:[{id:'flight',action:'motion',target_id:'library',motion}],
  motion_progress:{execution_id:'flight-2',phase:'transit',distance_m:0,total_length_m:150,progress:0}};
renderSemanticMission();
assert.match(semanticSteps.textContent,/轨迹飞行 · 图书馆/);
assert.match(semanticSteps.textContent,/顺时针 2 圈 · 2 m\/s · 朝向图书馆/);
assert.match(semanticSteps.textContent,/前往轨迹起点/);
assert.doesNotMatch(semanticSteps.textContent,/NaN|undefined|停留/);
state.semantic_mission.motion_progress = {execution_id:'flight-2',phase:'tracking',distance_m:75.8,total_length_m:150.2,progress:0.504};
renderSemanticMission();
assert.match(semanticSteps.textContent,/轨迹进度 50% · 75 \/ 151 m/);
assert.doesNotMatch(semanticSteps.textContent,/已完成 .*圈/);
state.semantic_mission.motion_progress.progress = 1;
renderSemanticMission();
assert.match(semanticSteps.textContent,/轨迹进度 99%/);
assert.doesNotMatch(semanticSteps.textContent,/已完成规定轨迹/);
state.semantic_mission.motion_progress.phase = 'complete';
renderSemanticMission();
assert.match(semanticSteps.textContent,/轨迹已通过，正在确认完成/);
step.status='completed';
state.semantic_mission.evidence=[{mission_id:'m',vehicle_id:'uav-1',step_id:'flight',execution_id:'flight-2',revision:2,
  action:'motion',target_id:'library',result:{completed:true,total_length_m:150.2,distance_m:150.2}}];
renderSemanticMission();
assert.match(semanticSteps.textContent,/已完成规定轨迹 · 已覆盖 150.2 m/);
assert.doesNotMatch(semanticSteps.textContent,/正在确认|轨迹进度/);
''')

    def test_motion_progress_does_not_attach_to_prior_execution_or_infer_completion(self):
        self.run_ui(r'''
const step={id:'flight',execution_id:'flight-2',revision:2,action:'motion',status:'active'};
const mission={current_execution_id:'flight-2',current_step_id:'flight',revision:2,
  motion_progress:{execution_id:'flight-1',phase:'tracking',progress:0.7}};
assert.equal(semanticMotionProgressLabel(step,mission),'');
mission.motion_progress.execution_id='flight-2';
assert.equal(semanticMotionProgressLabel({...step,execution_id:'flight-1'},mission),'');
mission.motion_progress={phase:'tracking',distance_m:null,total_length_m:0,progress:null};
assert.equal(semanticMotionProgressLabel(step,mission),'轨迹进度');
mission.motion_progress={phase:'tracking',distance_m:100,total_length_m:200,progress:0.5,completed_laps:1,laps:2};
assert.match(semanticMotionProgressLabel(step,mission),/已完成 1 \/ 2 圈/);
assert.equal(evidenceResultLabel({action:'motion',result:{visited:true,completed:false}}),'轨迹完成尚未确认。');
assert.equal(evidenceResultLabel({action:'motion',result:{completed:true}}),'已完成规定轨迹');
assert.equal(semanticMotionSummary({motion:{path:{kind:'curve'},heading:{mode:'fixed',yaw_deg:90}}}),'三维曲线 · 固定朝向 90°');
assert.equal(semanticMotionSummary({motion:{path:{kind:'polyline'}}}),'指定路径');
assert.equal(semanticMotionSummary({}), '');
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

    def test_control_steps_show_landing_hover_and_actual_results_without_return_assumption(self):
        self.run_ui(r'''
state.scenario.points_of_interest.push({id:'field',label:'北区足球场'});
const land = {id:'land',execution_id:'land-1',revision:1,action:'control',target_id:'field',status:'active',
  control:{anchor:'center',offset_m:[0,0,0],state:'landed',altitude_m:null,duration_seconds:15,yaw_deg:0}};
state.semantic_mission={mission_id:'m',vehicle_id:'uav-1',status:'running',revision:1,steps:[land],program:[land]};
renderSemanticMission();
assert.match(semanticSteps.textContent,/降落停留 · 北区足球场/);
assert.match(semanticSteps.textContent,/场地中心 · 停留 15 秒 · 朝向 0°/);
assert.equal((semanticSteps.textContent.match(/停留 15 秒/g)||[]).length,1);
assert.doesNotMatch(semanticSteps.textContent,/返航|undefined|null|NaN/);
assert.equal(semanticProgramLabel(land,state.semantic_mission),'降落停留 · 北区足球场 · 场地中心 · 停留 15 秒 · 朝向 0°');
const hover={...land,control:{anchor:'arrival',offset_m:[5,-3,0],state:'airborne',altitude_m:20,duration_seconds:0,yaw_deg:null}};
assert.equal(semanticStepLabel(hover),'定点停留 · 北区足球场');
assert.equal(semanticControlSummary(hover),'向东 5 m · 向北 3 m · 高度 20 m');
assert.equal(semanticControlSummary({control:null}),'');
assert.equal(evidenceResultLabel({action:'control',result:{completed:true,state:'landed',landed:true}}),'遥测已确认降落并停驻。');
assert.equal(evidenceResultLabel({action:'control',result:{completed:true,state:'airborne',landed:false}}),'遥测已确认到达并保持位置。');
for (const result of [{completed:false,landed:true,state:'landed'}, {completed:true,state:'landed'},
  {completed:true,state:'landed',landed:false}, {completed:true,state:'airborne',landed:true}]) {
  assert.equal(evidenceResultLabel({action:'control',result}),'目标状态尚未确认。');
}
land.status='completed';
state.semantic_mission.evidence=[{mission_id:'m',vehicle_id:'uav-1',step_id:'land',execution_id:'land-1',revision:1,
  action:'control',target_id:'field',result:{completed:true,state:'landed',landed:true}}];
renderSemanticMission();
assert.match(semanticSteps.textContent,/遥测已确认降落并停驻。/);
''')

    def test_terminal_message_uses_verified_goal_and_feedback_without_extra_panel(self):
        self.run_ui(r'''
state.scenario.points_of_interest.push({id:'field',label:'北区足球场'});
state.semantic_mission={status:'completed',goal_state:{target_id:'field',state:'landed',satisfied:true},
  execution_feedback:[{message:'正在执行降落。'}],message:'任务已完成。'};
renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'任务已完成，已在北区足球场降落停驻。');
assert.doesNotMatch(document.getElementById('semanticMessage').textContent,/返航/);
state.semantic_mission.goal_state.state='airborne'; renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'任务已完成，正在北区足球场上空保持位置。');
state.semantic_mission.goal_state.satisfied=false; renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'任务结束状态尚未确认。');
state.semantic_mission={status:'running',execution_feedback:[null,{message:'旧反馈'}, {message:'正在依据执行结果评估后续任务。'}, {message:' '}]};
renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'正在依据执行结果评估后续任务。');
state.semantic_mission.message='正在执行新任务。'; renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'正在执行新任务。');
commandError='通信失败'; renderSemanticMission();
assert.equal(document.getElementById('semanticMessage').textContent,'通信失败');
''')

    def test_noop_completion_does_not_invent_a_destination(self):
        self.run_ui(r'''
assert.equal(semanticGoalMessage({status:"completed",message:"任务无需执行动作，已保持当前位置。",
  goal_state:{target_id:null,state:"landed",satisfied:true}}),"任务无需执行动作，已保持当前位置。");
''')

    def test_review_is_displayed_as_assessment_without_an_invented_destination(self):
        self.run_ui(r'''
const review={id:'review',action:'review',instruction:'依据观测结果安排后续巡检。',status:'active'};
state.semantic_mission={status:'running',steps:[review],program:[review]};
renderSemanticMission();
assert.match(semanticSteps.textContent,/执行评估/);
assert.match(semanticSteps.textContent,/依据观测结果安排后续巡检。/);
assert.doesNotMatch(semanticSteps.textContent,/指定地点|undefined|返航/);
assert.equal(semanticProgramLabel(review,state.semantic_mission),'执行评估');
assert.equal(evidenceResultLabel({action:'review',result:{reviewed:true}}),'执行反馈已评估。');
assert.equal(evidenceResultLabel({action:'review',result:{}}),'正在评估执行反馈。');
''')


if __name__ == "__main__":
    unittest.main()
